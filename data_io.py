import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def read_transaction_file(file_path, day):
    df = pd.read_csv(file_path, sep='\t', compression='gzip', usecols=['sender', 'recipient', 'token_symbol', 'time'])
    df['unique_from'] = df['sender'] + '-' + df['token_symbol']
    df['unique_to'] = df['recipient'] + '-' + df['token_symbol']
    df['day'] = day
    df.dropna(inplace=True)
    return df


def read_transactions_subset(directory, start_day, end_day):
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('.tsv.gz')])
    data = []

    tasks = []
    with ProcessPoolExecutor() as executor:
        for day in range(start_day, end_day):
            idx = day - 1
            filename = all_files[idx]
            file_path = os.path.join(directory, filename)
            tasks.append(executor.submit(read_transaction_file, file_path, day))

            print(f'Transactions: Processing day {day} - {file_path}')

    for future in tasks:
        data.append(future.result())

    concatenated_data = pd.concat(data, ignore_index=True)
    return concatenated_data


def write_transaction_features_to_file(features, filename):
    def convert(item):
        if isinstance(item, np.integer):
            return int(item)
        elif isinstance(item, np.floating):
            return float(item)
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert(v) for v in item]
        else:
            return item

    features_converted = convert(features)

    with open(filename, 'w') as f:
        json.dump(features_converted, f, indent=4)


def write_results(tgnn_results, ks_results, low_ks_threshold, tgnn_filename_csv, ks_filename_csv):
    tgnn_results_to_write = [(u.split('-')[0], u.split('-')[1], v.split('-')[0], v.split('-')[1], prob)
                             for u, v, prob in tgnn_results]
    tgnn_results_df = pd.DataFrame(
        tgnn_results_to_write, columns=['address_from', 'domain_from', 'address_to', 'domain_to', 'prob'])
    tgnn_results_df.to_csv(tgnn_filename_csv, index=False)

    split_ks_results = [(u.split('-')[0], u.split('-')[1], v.split('-')[0], v.split('-')[1], score)
                        for u, v, score in ks_results if score > low_ks_threshold]
    filtered_ks_df = pd.DataFrame(
        split_ks_results, columns=['address_from', 'domain_from', 'address_to', 'domain_to', 'ks_dist'])

    filtered_ks_df.to_csv(ks_filename_csv, index=False)
