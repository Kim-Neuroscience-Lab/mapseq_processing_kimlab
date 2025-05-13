import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
import argparse

# ---------------------
# Argument Parsing
# ---------------------
parser = argparse.ArgumentParser(description="Generate Extended Data Figure 10 style SVG from normalized matrix.")
parser.add_argument("--matrix_file", type=str, required=True, help="Path to the normalized matrix CSV")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the output SVG")
parser.add_argument("--k_clusters", type=int, default=8, help="Number of k-means clusters (default=8)")
parser.add_argument("--sort_labels", nargs="*", help="Custom order for brain area labels (optional)")
parser.add_argument("--output_name", type=str, default="ExtendedDataFig10_Recreation", help="Output SVG filename without extension")
args = parser.parse_args()

# ---------------------
# Load Data
# ---------------------
df = pd.read_csv(args.matrix_file)

# Optional: Reorder columns if sort_labels provided
if args.sort_labels:
    df = df[args.sort_labels]

# ---------------------
# K-Means Clustering
# ---------------------
kmeans = KMeans(n_clusters=args.k_clusters, random_state=42)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

# ---------------------
# Plotting Setup
# ---------------------
scolors = ['white', 'red']
scm = LinearSegmentedColormap.from_list('white_to_red', scolors, N=256)

fig, ax = plt.subplots(figsize=(12, 8))

im = ax.imshow(centroids, aspect='auto', cmap=scm, vmin=0, vmax=1)

# Axis setup
ax.set_title("Projection Motif Clusters (Recreated from Extended Data Fig. 10)", fontsize=14)
ax.set_xlabel("Target Regions", fontsize=12)
ax.set_ylabel("Cluster ID", fontsize=12)
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha='right')
ax.set_yticks(range(args.k_clusters))
ax.set_yticklabels([f"Cluster {i+1}" for i in range(args.k_clusters)])

# Clean up plot
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(top=False, bottom=True, left=True, right=False)

# Colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.set_label('Normalized Projection Strength', rotation=270, labelpad=15)

# ---------------------
# Save SVG
# ---------------------
os.makedirs(args.out_dir, exist_ok=True)
svg_path = os.path.join(args.out_dir, f"{args.output_name}.svg")
fig.tight_layout()
fig.savefig(svg_path, format='svg')
print(f"✅ SVG figure saved to: {svg_path}")

