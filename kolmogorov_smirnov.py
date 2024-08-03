from collections import defaultdict
from statistics import mean

import pandas as pd


def get_ks_nodes(file_path, start_day, end_day):
    chunksize = 10000
    all_nodes = {day: set() for day in range(start_day, end_day)}

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        for day in range(start_day, end_day):
            day_df = chunk[chunk['day'] == day]
            nodes = set(day_df['unique_from']).union(day_df['unique_to'])
            all_nodes[day].update(nodes)

    return all_nodes


def get_top_n_edges(n, file_path):
    chunksize = 10000
    ks_results = defaultdict(list)

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        edge_dist_tuples = chunk.apply(
            lambda row: (tuple(sorted([row['unique_from'], row['unique_to']])), row['ks_dist']), axis=1
        ).tolist()

        for normalized_edge, ks_dist in edge_dist_tuples:
            ks_results[normalized_edge].append(ks_dist)

    sorted_edges = sorted(
        [(pair[0], pair[1], mean(dists)) for pair, dists in ks_results.items()],
        key=lambda x: x[2]
    )
    return sorted_edges[:n]
