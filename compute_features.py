import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from data_io import read_transactions_subset, write_transaction_features_to_file


def compute_graph_features_for_subgraph(G):
    G_undirected = G.to_undirected()

    if len(G) > 1000:
        # For large graphs, sample nodes to approximate betweenness centrality
        sample_size = min(100, len(G))  # Adjust the sample size as needed
        betweenness = nx.betweenness_centrality(G, k=sample_size, normalized=True)
    else:
        # For smaller graphs, compute exact betweenness centrality
        betweenness = nx.betweenness_centrality(G, normalized=True)

    clustering = nx.clustering(G_undirected)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)

    return clustering, closeness, betweenness, pagerank


def compute_transaction_features(directory, valid_nodes, start_day, end_day, start_date, total_edges):
    transaction_df = read_transactions_subset(directory, start_day, end_day)
    all_node_features = {}

    for day in range(start_day, end_day):
        day_df = transaction_df[transaction_df['day'] == day]

        grouped = day_df.groupby('token_symbol')
        day_features = {node: [] for node in valid_nodes[day]}

        i = 0
        for token_symbol, group in grouped:
            G = nx.from_pandas_edgelist(group, 'unique_from', 'unique_to', create_using=nx.DiGraph())

            features = compute_graph_features_for_subgraph(G)

            for node in valid_nodes[day].intersection(G.nodes()):
                node_features = [
                    group['unique_to'].value_counts().get(node, 0),
                    group['unique_from'].value_counts().get(node, 0),
                    group.groupby('unique_to').size().get(node, 0),
                    group.groupby('unique_from').size().get(node, 0),
                    features[0].get(node, 0),  # Clustering
                    features[1].get(node, 0),  # Closeness
                    features[2].get(node, 0),  # Betweenness
                    features[3].get(node, 0),  # Pagerank
                ]
                day_features[node] = node_features
            i += 1

        all_node_features[day] = day_features

        output_filename = (f'data/preprocessed/transaction_features/{start_date}/'
                           f'top_{total_edges}_node_features_day_{day}.json')
        write_transaction_features_to_file(day_features, output_filename)

    return all_node_features


def read_silences(directory, valid_nodes, start_day, end_day):
    sorted_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

    distributions = {}
    for day in range(start_day, end_day):
        idx = day - 1
        filename = sorted_files[idx]
        file_path = os.path.join(directory, filename)

        df = pd.read_json(file_path)
        df['pair'] = df['add'] + '-' + df['tok']
        df = df[df['pair'].isin(valid_nodes[day])]
        daily_dist_by_address = pd.Series(df['silence_arr'].values, index=df['pair']).to_dict()
        distributions[day] = daily_dist_by_address

    return distributions


def compute_silence_distribution_features(silences, log_bins, linear_bins):
    features = {}
    for node, silence_arr in silences.items():
        silence_arr = np.array(silence_arr)

        num_txs = len(silence_arr) + 1
        mean_val = np.mean(silence_arr)
        variance_val = np.var(silence_arr)
        std_dev_val = np.std(silence_arr)
        range_val = np.ptp(silence_arr)
        iqr_val = np.percentile(silence_arr, 75) - np.percentile(silence_arr, 25)
        log_binned_counts, _ = np.histogram(silence_arr, bins=log_bins)
        lin_binned_counts, _ = np.histogram(silence_arr, bins=linear_bins)

        features[node] = np.concatenate(([num_txs, mean_val, variance_val,
                                          std_dev_val, range_val, iqr_val], log_binned_counts, lin_binned_counts))

    return features


def get_silence_dist_features(all_silences, valid_nodes, start_day, end_day, bin_count=11):
    for day, silences in all_silences.items():
        all_silences[day] = {k: v for k, v in silences.items() if k in valid_nodes}

    concatenated_silences = np.concatenate([np.array(x) for inner_dict in all_silences.values()
                                            for x in inner_dict.values() if len(x) > 0])
    log_bins = np.logspace(np.log10(np.min(concatenated_silences)),
                           np.log10(np.max(concatenated_silences)),
                           bin_count)  # Define global bins
    lin_bins = np.linspace(np.min(concatenated_silences), np.max(concatenated_silences), bin_count)

    daily_silence_distribution_features = {
        day: compute_silence_distribution_features(all_silences[day], log_bins, lin_bins)
        for day in range(start_day, end_day)
    }
    return daily_silence_distribution_features


def get_scaled_node_features(silence_feats, tx_node_feats, start_day, end_day):
    daily_node_features = {}
    missing_nodes = set()

    all_features = []
    for day in range(start_day, end_day):
        features = {}
        for node, silence_dist_feats in silence_feats[day].items():
            if node not in tx_node_feats[day]:
                print(f'Day {day}: Transaction Features - Missing node: {node}')
                missing_nodes.add(node)
                continue
            else:
                tx_feats = tx_node_feats[day][node]

            features[node] = np.concatenate([tx_feats, silence_dist_feats])

        daily_node_features[day] = features
        all_features.extend(features.values())

    print(f'Missing Node Count: {len(missing_nodes)}')

    scaler = RobustScaler()
    all_features_np = np.array(all_features)
    scaled_features = scaler.fit_transform(all_features_np)

    idx = 0
    scaled_daily_node_features = {}
    for day in range(start_day, end_day):
        scaled_daily_node_features[day] = {}
        for node in daily_node_features[day]:
            scaled_daily_node_features[day][node] = scaled_features[idx].tolist()
            idx += 1

    return scaled_daily_node_features
