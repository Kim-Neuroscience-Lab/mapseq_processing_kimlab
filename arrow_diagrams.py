import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from scipy.stats import binom
from statsmodels.stats.multitest import fdrcorrection
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser(description="Generate Extended Data Fig. 10 panels f and g with motif significance analysis")
parser.add_argument("--matrix_file", type=str, required=True, help="Path to normalized matrix CSV")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output")
parser.add_argument("--output_prefix", type=str, default="Fig10f_g", help="Prefix for output files")
args = parser.parse_args()

# Load normalized projection matrix
df = pd.read_csv(args.matrix_file, sep=None, engine='python')
manual_region_order = ["RSP", "PM", "AM", "A", "RL", "AL", "LM"]
regions = [r for r in manual_region_order if r in df.columns]
matrix = df[regions].apply(pd.to_numeric, errors='coerce').to_numpy()

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

# Panel g: motif frequency analysis
region_probs = binary_matrix.mean(axis=0)

# Store all motifs and counts
motif_stats = []
observed_counts = {}
for r in [2, 3, 4]:
    for combo in combinations(range(len(regions)), r):
        combo_mask = binary_matrix[:, combo].sum(axis=1) == r
        count = np.sum(combo_mask)
        observed_counts[combo] = count
        expected_prob = np.prod(region_probs[list(combo)])
        expected_count = expected_prob * n_cells
        pval = binom.sf(count - 1, n_cells, expected_prob) if expected_prob > 0 else 1.0
        motif_stats.append({
            "motif": '+'.join(sorted([regions[i] for i in combo])),
            "combo": combo,
            "size": r,
            "observed": count,
            "expected": expected_count,
            "pval": pval,
            "observed_minus_expected": count - expected_count
        })

# Multiple hypothesis correction
pvals = [m['pval'] for m in motif_stats]
rejected, corrected_pvals = fdrcorrection(pvals, alpha=0.05)
for i, val in enumerate(motif_stats):
    val['FDR_pval'] = corrected_pvals[i]
    val['significant'] = rejected[i]

# Save motif analysis
motif_df = pd.DataFrame(motif_stats)
motif_out_path = os.path.join(args.out_dir, f"{args.output_prefix}_motif_significance.csv")
motif_df.to_csv(motif_out_path, index=False)
print(f"✅ Saved motif significance table to {motif_out_path}")

# NetworkX visualization of significant 2-region motifs
pair_counts = {k: v for k, v in observed_counts.items() if len(k) == 2}

G = nx.Graph()
for label in regions:
    G.add_node(label)

max_count = max(pair_counts.values())
for (i, j), count in pair_counts.items():
    region_i = regions[i]
    region_j = regions[j]
    motif_label = '+'.join(sorted([region_i, region_j]))
    row = motif_df[(motif_df['motif'] == motif_label) & (motif_df['size'] == 2)]
    if not row.empty:
        observed = row['observed'].values[0]
        expected = row['expected'].values[0]
        diff = observed - expected
        color = 'red' if row['significant'].values[0] and diff > 0 else ('blue' if row['significant'].values[0] and diff < 0 else 'black')
        width = 1 + 9 * (observed / max_count)
        G.add_edge(region_i, region_j, weight=width, color=color)

# Layout and draw for bifurcations
angle_step = 2 * np.pi / len(regions)
pos = {
    region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
    for i, region in enumerate(manual_region_order)
}
edges = G.edges(data=True)
colors = [e[2]['color'] for e in edges]
widths = [e[2]['weight'] for e in edges]

plt.figure(figsize=(8,8))
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, edge_color=colors, width=widths)
plt.title("Fig 10g: Broadcasting Neurons - Significant 2-Region Motifs")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Overrepresented'),
    Line2D([0], [0], color='blue', lw=2, label='Underrepresented'),
    Line2D([0], [0], color='black', lw=2, label='Not Significant')
]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

plt.tight_layout()
g_path = os.path.join(args.out_dir, f"{args.output_prefix}_panel_g_broadcasting.svg")
plt.savefig(g_path, format='svg')
print(f"✅ Saved Fig 10g to {g_path}")

# NetworkX visualization for significant 3- and 4-region motifs
print("\n--- Generating 3- and 4-region motif graphs ---")
for r in [3, 4]:
    print(f"Checking {r}-way motifs...")
    filtered = motif_df[(motif_df['size'] == r)]
    if not filtered.empty:
        print(f"→ Found {len(filtered)} {r}-way motifs. Building graph...")
        # Determine max count for current size
        max_count_r = filtered['observed'].max()
        G_multi = nx.Graph()
        all_nodes = set()

        for _, row in filtered.iterrows():
            motif_nodes = row['motif'].split('+')
            observed = row['observed']
            expected = row['expected']
            diff = observed - expected
            color = 'red' if row['significant'] and diff > 0 else ('blue' if row['significant'] and diff < 0 else 'black')
            width = 1 + 4 * (observed / max(1, max_count_r))

            for i, j in combinations(motif_nodes, 2):
                if G_multi.has_edge(i, j):
                    # If already exists, keep thickest line
                    if G_multi[i][j]['weight'] < width:
                        G_multi[i][j]['color'] = color
                        G_multi[i][j]['weight'] = width
                else:
                    G_multi.add_edge(i, j, color=color, weight=width)
                all_nodes.update([i, j])

        angle_step = 2 * np.pi / len(manual_region_order)
        pos = {
            region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
            for i, region in enumerate(manual_region_order)
            if region in G_multi.nodes
        }
        edge_colors = [e[2]['color'] for e in G_multi.edges(data=True)]
        edge_weights = [e[2]['weight'] for e in G_multi.edges(data=True)]

        plt.figure(figsize=(8,8))
        nx.draw_networkx(G_multi, pos, with_labels=True, node_size=1000, edge_color=edge_colors, width=edge_weights)
        plt.title(f"Fig 10g Extension: {r}-Region Motifs")

        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Overrepresented'),
            Line2D([0], [0], color='blue', lw=2, label='Underrepresented'),
            Line2D([0], [0], color='black', lw=2, label='Not Significant')
        ]
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, f"{args.output_prefix}_panel_g_{r}way_broadcasting.svg")
        try:
            plt.savefig(fig_path, format='svg')
            plt.close()
            if os.path.exists(fig_path):
                print(f"✅ Saved {r}-way broadcasting motif network to {fig_path}")
            else:
                print(f"❌ File was not saved: {fig_path}")
        except Exception as e:
            print(f"❌ Failed to save {r}-way plot: {e}")
        print(f"✅ Saved {r}-way broadcasting motif network to {fig_path}")


for r in [3, 4]:
    filtered = motif_df[(motif_df['size'] == r) & (motif_df['significant'])]
    if not filtered.empty:
        txt_path = os.path.join(args.out_dir, f"{args.output_prefix}_motif_{r}way_significant.txt")
        with open(txt_path, 'w') as f:
            for _, row in filtered.iterrows():
                f.write(f"{row['motif']}: Observed={row['observed']}, Expected={row['expected']:.2f}, p={row['FDR_pval']:.3g}\n")
        print(f"✅ Saved significant {r}-way motifs to {txt_path}")
