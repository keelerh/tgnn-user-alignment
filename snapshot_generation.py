import numpy as np
import torch
from torch_geometric_temporal import DynamicGraphTemporalSignal


def generate_snapshots(data, node_idx_map, node_features_map, num_features, start_day, end_day, is_negative):
    edge_indices = []
    node_features = []
    targets = []

    data['nidx_from'] = data['unique_from'].apply(lambda a: node_idx_map[a]).astype(int)
    data['nidx_to'] = data['unique_to'].apply(lambda a: node_idx_map[a]).astype(int)

    data['label'] = 0 if is_negative else 1

    address_count = list(node_idx_map.keys())
    node_count = len(address_count)

    for day in range(start_day, end_day):
        df = data[data['day'] == day].copy()

        # Duplicate edge indices, labels, and targets for both directions (similarity network is undirected)
        edges_forward = df[['nidx_from', 'nidx_to']].values
        edges_backward = edges_forward[:, [1, 0]]  # Swap columns to reverse directions
        edges = np.vstack([edges_forward, edges_backward])

        edge_idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
        targets_tensor = torch.tensor(np.concatenate([df['label'].values, df['label'].values]), dtype=torch.float)

        x = torch.zeros([node_count, num_features], dtype=torch.float)
        for node_addr in address_count:
            node_idx = node_idx_map[node_addr]
            feature_vector = node_features_map[day].get(node_addr, [0] * num_features)
            x[node_idx] = torch.tensor(feature_vector, dtype=torch.float)

        edge_indices.append(edge_idx.numpy())
        node_features.append(x.numpy())
        targets.append(targets_tensor.numpy())

    edge_weights = np.zeros(len(edge_indices))  # Unused
    signal = DynamicGraphTemporalSignal(
        edge_indices=edge_indices, edge_weights=edge_weights, features=node_features, targets=targets)
    return signal
