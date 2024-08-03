import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def matching_addresses(address, other_address):
    eth_address = address.split('-')[0]
    other_eth_address = other_address.split('-')[0]
    return eth_address == other_eth_address


def find_tie_ranges(scores, is_correct):
    sorted_indices = np.argsort(-np.array(scores))
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_is_correct = np.array(is_correct)[sorted_indices]

    precisions_min = [0] * len(scores)
    precisions_max = [0] * len(scores)
    cumulative_correct = 0
    tie_start = 0

    for i in range(len(sorted_scores)):
        if sorted_is_correct[i] == 1:
            cumulative_correct += 1

        if i == len(sorted_scores) - 1 or not sorted_scores[i] == sorted_scores[i + 1]:
            tie_end = i + 1
            precisions_min[tie_start:tie_end] = [cumulative_correct / tie_end] * (tie_end - tie_start)
            precisions_max[tie_start:tie_end] = [cumulative_correct / tie_end] * (tie_end - tie_start)
            tie_start = tie_end

        if sorted_is_correct[i] == 0:
            for j in range(tie_start, i + 1):
                precisions_max[j] = cumulative_correct / (j + 1)

    return precisions_min, precisions_max


def process_results(final_classifier_results, top_n_edges, valid_edges):
    tgnn_min, tgnn_avg, tgnn_max = process_tgnn_results(final_classifier_results)
    ks_min, ks_avg, ks_max = process_ks_results(top_n_edges, valid_edges)

    cutoff = min(len(tgnn_min), len(ks_min))
    tgnn_min, tgnn_avg, tgnn_max = tgnn_min[:cutoff], tgnn_avg[:cutoff], tgnn_max[:cutoff]
    ks_min, ks_avg, ks_max = ks_min[:cutoff], ks_avg[:cutoff], ks_max[:cutoff]

    best_min, best_avg, best_max = process_theoretical_best_results(cutoff, top_n_edges, valid_edges)

    return tgnn_min, tgnn_avg, tgnn_max, ks_min, ks_avg, ks_max, best_min, best_avg, best_max


def process_tgnn_results(final_classifier_results):
    tgnn_scores = [score for _, _, score in final_classifier_results]
    tgnn_is_correct = [1 if matching_addresses(u, v) else 0 for u, v, _ in final_classifier_results]
    tgnn_precisions_min, tgnn_precisions_max = find_tie_ranges(tgnn_scores, tgnn_is_correct)
    tgnn_precisions_avg = (np.array(tgnn_precisions_min) + np.array(tgnn_precisions_max)) / 2

    return tgnn_precisions_min, tgnn_precisions_avg, tgnn_precisions_max


def process_ks_results(top_n_edges, valid_edges,):
    ks_scores = [1 - score for u, v, score in top_n_edges if (u, v) in valid_edges]
    ks_is_correct = [1 if matching_addresses(u, v) else 0 for u, v, score in top_n_edges if (u, v) in valid_edges]
    ks_precisions_min, ks_precisions_max = find_tie_ranges(ks_scores, ks_is_correct)
    ks_precisions_avg = (np.array(ks_precisions_min) + np.array(ks_precisions_max)) / 2

    return ks_precisions_min, ks_precisions_avg, ks_precisions_max


def process_theoretical_best_results(cutoff, top_n_edges, valid_edges):
    total_correct = sum(1 if matching_addresses(u, v) else 0 for u, v, _ in top_n_edges if (u, v) in valid_edges)
    perfect_scores = [1 - i for i in range(cutoff)]
    perfect_is_correct = [1 if i < total_correct else 0 for i in range(cutoff)]
    perfect_precisions_min, perfect_precisions_max = find_tie_ranges(perfect_scores, perfect_is_correct)
    perfect_precisions_avg = (np.array(perfect_precisions_min) + np.array(perfect_precisions_max)) / 2

    return perfect_precisions_min, perfect_precisions_avg, perfect_precisions_max


def plot_precision_at_top_k(final_classifier_results, top_n_edges, valid_edges, tgnn_filename_png):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.labelpad'] = 20  # Space between the label and the axis

    tgnn_min, tgnn_avg, tgnn_max, ks_min, ks_avg, ks_max, best_min, best_avg, best_max = process_results(
        final_classifier_results, top_n_edges, valid_edges)

    cutoff = len(tgnn_min)
    x_values = range(1, cutoff + 1)

    # Plotting the precision curves and fill between for confidence intervals
    plt.figure(figsize=(12, 6))
    plt.fill_between(x_values, best_min, best_max, color='blue', alpha=0.2)
    plt.fill_between(x_values, tgnn_min, tgnn_max, color='orangered', alpha=0.2)
    plt.fill_between(x_values, ks_min, ks_max, color='gold', alpha=0.2)

    # Plot the average precision line in the middle of the confidence intervals
    plt.plot(x_values, best_avg, color='blue', linewidth=3, linestyle='-', label='Theoretical Best')
    plt.plot(x_values, tgnn_avg, color='orangered', linewidth=3, linestyle='-', label='TGNN Average')
    plt.plot(x_values, ks_avg, color='gold', linewidth=3, linestyle='-', label='KS Average')

    plt.axhline(y=0.02, color='grey', linestyle='-', linewidth=2, label='Baseline')

    plt.title(f'Precision at Top K Ranked Pairs for TGNN vs. KS Distances', fontsize=18)
    plt.legend(frameon=False, fontsize=14, loc='upper right')
    plt.ylim([0.0, 1.05])
    plt.xlabel('Top N', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xticks([10, 50, 100, 500, 1000], ['10', '50', '100', '500', '1,000'], fontsize=14)
    plt.minorticks_off()
    plt.yticks(fontsize=14)

    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, color='#CCCCCC')
    plt.tight_layout()

    plt.savefig(tgnn_filename_png, dpi=300)

    plt.show()


def plot_similarity_networks(positive_data, start_day, end_day):
    background_color = '#f4f4f4'

    positive_data['from'] = positive_data['unique_from'].str.split("-", n=1, expand=True)[0]
    positive_data['to'] = positive_data['unique_to'].str.split("-", n=1, expand=True)[0]
    unique_addresses = pd.concat([positive_data['from'], positive_data['to']]).unique()

    colors = sns.color_palette(cc.glasbey_dark, n_colors=len(unique_addresses))
    color_map = {node: color for node, color in zip(unique_addresses, colors)}

    plt.figure(figsize=(12, 24))

    for idx in range(start_day, end_day):
        ax = plt.subplot(5, 3, idx)

        day_low_ks_df = positive_data[positive_data['day'] == idx].reset_index(drop=True)
        G_pos = nx.from_pandas_edgelist(day_low_ks_df, 'unique_from', 'unique_to')

        for _, row in day_low_ks_df.iterrows():
            G_pos.nodes[row['unique_from']]['address'] = row['from']
            G_pos.nodes[row['unique_to']]['address'] = row['to']

        day_edges = [(u, v) for u, v, d in G_pos.edges(data=True)]
        day_nodes = list(set(u for u, v in day_edges) | set(v for u, v in day_edges))
        print(f'Day {idx}: {len(day_nodes)} nodes, {len(day_edges)} edges')

        pos = nx.spring_layout(G_pos.subgraph(day_nodes), k=0.6, iterations=200, seed=42)

        nx.draw_networkx_nodes(G_pos, pos=pos, ax=ax, nodelist=day_nodes,
                               node_color=[color_map[G_pos.nodes[node]['address']] for node in day_nodes],
                               node_size=30)

        nx.draw_networkx_edges(G_pos, pos=pos, ax=ax, edgelist=day_edges)

        V = len(day_nodes)
        E = len(day_edges)

        # Calculate the true positive rate (TPR) and false positive rate (FPR)
        true_positives = day_low_ks_df['correct'].sum()
        false_positives = len(day_low_ks_df) - true_positives
        tpr = true_positives / len(day_low_ks_df)
        fpr = false_positives / len(day_low_ks_df)

        label_text = f"G(V={V}, E={E})\nTPR: {tpr:.2f}, FPR: {fpr:.2f}"
        ax.text(0.04, 0.04, label_text, transform=ax.transAxes, fontsize=12,
                ha='left', va='bottom', bbox=dict(facecolor='none', edgecolor='none'))

        ax.set_title(f"Day {idx}", fontsize=16, pad=16)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor(background_color)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('results/similarity_networks.png', dpi=300)
    plt.show()
