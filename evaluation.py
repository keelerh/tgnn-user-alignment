from collections import defaultdict

import numpy as np
import torch


def get_eval_inputs(df, node_features_map, address_to_node_idx_map, start_day, end_day, feature_count):
    edge_indices = []
    features = []

    df['nidx_from'] = df['unique_from'].apply(lambda a: address_to_node_idx_map[a]).astype(int)
    df['nidx_to'] = df['unique_to'].apply(lambda a: address_to_node_idx_map[a]).astype(int)

    all_addresses = address_to_node_idx_map.keys()

    for day in range(start_day, end_day):
        df = df[df['day'] == day].copy()

        # Add edges for both directions
        edge_idx = torch.tensor(df[['nidx_from', 'nidx_to']].values, dtype=torch.long)
        edge_idx_reversed = torch.tensor(df[['nidx_to', 'nidx_from']].values, dtype=torch.long)
        edge_idx = torch.cat((edge_idx, edge_idx_reversed), dim=0).t().contiguous()

        x = torch.zeros([len(all_addresses), feature_count], dtype=torch.float)
        for node_addr in all_addresses:
            node_idx = address_to_node_idx_map[node_addr]
            x[node_idx] = torch.tensor(node_features_map[day].get(node_addr, [0] * feature_count), dtype=torch.float)

        edge_indices.append(edge_idx.numpy())
        features.append(x.numpy())

    return edge_indices, features


def get_edge_probabilities(
        df, node_features, node_idx_to_address_map, model, device, start_day, end_day, feature_count, batch_size=1000):
    edge_indices, features = get_eval_inputs(
        df, node_features, node_idx_to_address_map, start_day, end_day, feature_count)

    model.eval()
    edge_probabilities = defaultdict(list)

    with torch.no_grad():
        for day in range(len(features)):
            node_features_tensor = torch.tensor(features[day], dtype=torch.float).to(device)
            day_edge_indices = edge_indices[day]

            for batch_start in range(0, day_edge_indices.shape[1], batch_size):  # Process in batches
                batch_end = batch_start + batch_size
                batch_edge_indices = torch.tensor(day_edge_indices[:, batch_start:batch_end], dtype=torch.long).to(
                    device)

                logits = model(node_features_tensor, batch_edge_indices)
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
                edge_list = batch_edge_indices.t().cpu().numpy()

                for (i, j), prob in zip(edge_list, probabilities):
                    address_i = node_idx_to_address_map[int(i)]
                    address_j = node_idx_to_address_map[int(j)]
                    edge_key = tuple(sorted((address_i, address_j)))
                    edge_probabilities[edge_key].append(prob)

    avg_edge_probabilities = {edge: np.mean(probs) for edge, probs in edge_probabilities.items()}
    sorted_classifier_results = sorted(avg_edge_probabilities.items(), key=lambda x: x[1], reverse=True)
    classifier_results = [(u, v, avg_prob) for (u, v), avg_prob in sorted_classifier_results]

    return classifier_results
