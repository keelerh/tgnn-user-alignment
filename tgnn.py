import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric_temporal.nn.recurrent import GConvGRU


class TemporalGNN(nn.Module):
    def __init__(self, node_features, hidden_dim, embedding_dim, dropout_rate=0.5):
        super(TemporalGNN, self).__init__()

        self.edge_conv = GConvGRU(node_features, hidden_dim, hidden_dim)
        self.gat_conv = GATConv(hidden_dim, hidden_dim)
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.edge_classifier = nn.Linear(2 * embedding_dim, 1)

    def forward(self, x, edge_idx, return_embeddings=False):
        h = self.edge_conv(x, edge_idx)
        h = F.relu(self.gat_conv(h, edge_idx))
        h = self.dropout(h)
        embeddings = self.embedding(h)

        if return_embeddings:
            return embeddings

        # Edge Classification
        source_nodes = embeddings[edge_idx[0]]
        target_nodes = embeddings[edge_idx[1]]
        edge_features = torch.cat([source_nodes, target_nodes], dim=1)
        logits = self.edge_classifier(edge_features).squeeze()

        return logits


def weighted_bce_with_logits_loss(predictions, targets, weights, bce_loss_function):
    bce_loss = bce_loss_function(predictions, targets)
    weighted_loss = bce_loss * weights
    return weighted_loss.mean()


def train(model, pos_training_data, neg_training_data, optimizer, pos_weight, neg_weight, device):
    model.train()

    total_loss = 0
    bce_loss_func = nn.BCEWithLogitsLoss(reduction='none')

    for idx, pos_snapshot in enumerate(pos_training_data):
        optimizer.zero_grad()

        neg_snapshot = neg_training_data[idx]

        x = pos_snapshot.x.to(device)  # Node features the same for positive and negative samples

        pos_edge_idx = pos_snapshot.edge_index.to(device)
        neg_edge_idx = neg_snapshot.edge_index.to(device)
        edge_idx = torch.cat([pos_edge_idx, neg_edge_idx], dim=1)

        edge_predictions = model(x, edge_idx)

        pos_targets = pos_snapshot.y.to(device)
        neg_targets = neg_snapshot.y.to(device)
        y = torch.cat([pos_targets, neg_targets], dim=0)

        weights = torch.where(y > 0, pos_weight, neg_weight)

        bce_loss = weighted_bce_with_logits_loss(edge_predictions, y, weights, bce_loss_func)

        bce_loss.backward()
        optimizer.step()

        total_loss += bce_loss.item()

    return total_loss / pos_training_data.snapshot_count


def val(model, pos_validation_data, neg_validation_data, pos_weight, neg_weight, device):
    model.eval()

    total_loss = 0
    bce_loss_func = nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for idx, pos_snapshot in enumerate(pos_validation_data):
            neg_snapshot = neg_validation_data[idx]

            x = pos_snapshot.x.to(device)  # Node features the same for positive and negative samples

            pos_edge_idx = pos_snapshot.edge_index.to(device)
            neg_edge_idx = neg_snapshot.edge_index.to(device)
            edge_idx = torch.cat([pos_edge_idx, neg_edge_idx], dim=1)

            edge_predictions = model(x, edge_idx)

            pos_targets = pos_snapshot.y.to(device)
            neg_targets = neg_snapshot.y.to(device)
            y = torch.cat([pos_targets, neg_targets], dim=0)

            weights = torch.where(y > 0, pos_weight, neg_weight)

            bce_loss = weighted_bce_with_logits_loss(edge_predictions, y, weights, bce_loss_func)

            total_loss += bce_loss.item()

    return total_loss / pos_validation_data.snapshot_count
