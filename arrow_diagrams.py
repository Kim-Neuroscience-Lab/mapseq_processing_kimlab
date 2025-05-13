import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from scipy.stats import binom
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser(description="Generate Extended Data Fig. 10 panels f and g")
parser.add_argument("--matrix_file", type=str, required=True, help="Path to normalized matrix CSV")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output")
parser.add_argument("--output_prefix", type=str, default="Fig10f_g", help="Prefix for output files")
args = parser.parse_args()

# Load normalized projection matrix
df = pd.read_csv(args.matrix_file)
regions = df.columns.to_list()
matrix = df.to_numpy()

# Thresholding to determine presence/absence
binary_matrix = (matrix > 0).astype(int)
n_cells = binary_matrix.shape[0]

# Count dedicated projection neurons (only one target area)
proj_counts = binary_matrix.sum(axis=1)
only_one_proj = (proj_counts == 1)
dedicated_cells = binary_matrix[only_one_proj]

dedicated_counts = dedicated_cells.sum(axis=0)

# Plot panel f: dedicated projection histogram
plt.figure(figsize=(8,5))
plt.bar(regions, dedicated_counts, color='gray')
plt.ylabel("Number of Dedicated Neurons")
plt.title("Fig 10f: Dedicated Projection Neurons")
plt.xticks(rotation=45)
plt.tight_layout()
f_path = os.path.join(args.out_dir, f"{args.output_prefix}_panel_f_dedicated.svg")
plt.savefig(f_path, format='svg')
print(f"✅ Saved Fig 10f to {f_path}")

# Panel g: network diagram of observed projection pairs
multi_proj = binary_matrix[proj_counts > 1]

def get_pair_counts(matrix):
    pair_dict = {}
    for row in matrix:
        targets = np.where(row > 0)[0]
        for pair in combinations(targets, 2):
            key = tuple(sorted(pair))
            pair_dict[key] = pair_dict.get(key, 0) + 1
    return pair_dict

pair_counts = get_pair_counts(multi_proj)

# Compute expected probabilities assuming independence
region_probs = binary_matrix.mean(axis=0)
expected_pairs = {
    pair: binom.pmf(count, n_cells, region_probs[pair[0]] * region_probs[pair[1]])
    for pair, count in pair_counts.items()
}

# Create graph
G = nx.Graph()
for i, label in enumerate(regions):
    G.add_node(label)

# Add edges with width = observed count, color = over/under expected
max_count = max(pair_counts.values())
for (i, j), count in pair_counts.items():
    region_i = regions[i]
    region_j = regions[j]
    observed = count
    expected = region_probs[i] * region_probs[j] * n_cells
    significance = observed - expected
    color = 'blue' if significance > 0 else 'red'
    width = 1 + 9 * (observed / max_count)
    G.add_edge(region_i, region_j, weight=width, color=color)

# Layout and draw
pos = nx.circular_layout(G)
edges = G.edges(data=True)
colors = [e[2]['color'] for e in edges]
widths = [e[2]['weight'] for e in edges]

plt.figure(figsize=(8,8))
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, edge_color=colors, width=widths)
plt.title("Fig 10g: Broadcasting Neurons - Projection Motifs")

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Overrepresented'),
    Line2D([0], [0], color='red', lw=2, label='Underrepresented')
]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

plt.tight_layout()
g_path = os.path.join(args.out_dir, f"{args.output_prefix}_panel_g_broadcasting.svg")
plt.savefig(g_path, format='svg')
print(f"✅ Saved Fig 10g to {g_path}")

