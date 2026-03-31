#written by Adam to do clustering analysis on the upsetplot.csv data that we used in the trajectories.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
    set_link_color_palette,
    fcluster,
)
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import argparse


def prepare_clustering_data(df):
    """
    Reshape data for clustering: rows = motifs, columns = stages
    """
    # Remove empty motif labels
    df_clean = df[df["Motif_Label"] != ""].copy()

    # Pivot the data to get motifs as rows and stages as columns
    pivot_df = df_clean.pivot_table(
        index="Motif_Label",
        columns="Stage",
        values="Effect Size",
        aggfunc="first",  # In case of duplicates, take the first
    )

    # Ensure stages are in the correct order
    stage_order = ["P3", "P12", "P20", "P60"]
    pivot_df = pivot_df.reindex(columns=stage_order)

    # Handle missing values by filling with 0 (or you could use interpolation)
    pivot_df = pivot_df.fillna(0)

    print(f"Data shape for clustering: {pivot_df.shape}")
    print(f"Motifs: {len(pivot_df.index)}")
    print(f"Stages: {list(pivot_df.columns)}")

    return pivot_df


def perform_clustering(data_matrix):
    """
    Perform hierarchical clustering using Ward's method with Euclidean distance
    with proper dendrogram orientation to place most distant motifs at extremes

    Parameters:
    - data_matrix: DataFrame with motifs as rows and stages as columns
    """
    print("Using Euclidean Distance for clustering...")

    # Use built-in Euclidean distance from scipy
    condensed_distances = pdist(data_matrix.values, metric="euclidean")
    distance_matrix = squareform(condensed_distances)
    method_name = "Euclidean Distance"

    # Perform Ward linkage clustering with optimal ordering
    linkage_matrix = linkage(condensed_distances, method="ward", optimal_ordering=False)

    return linkage_matrix, distance_matrix, method_name


def extract_dendrogram_order(linkage_matrix, data_matrix):
    """
    Extract the linear order of motifs from the hierarchical clustering dendrogram
    """
    from scipy.cluster.hierarchy import dendrogram

    # Get dendrogram leaf order without plotting
    dend = dendrogram(linkage_matrix, no_plot=True)
    linear_order = [data_matrix.index[i] for i in dend["leaves"]]

    return linear_order


def create_distance_based_ordering(data_matrix, distance_matrix):
    """
    Create a linear ordering where most distant motifs are at extremes
    and there's a smooth progression of distances between them
    """
    print("Creating distance-based linear ordering...")

    # Find the pair of motifs with maximum distance
    max_dist = 0
    max_pair = (0, 0)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                max_pair = (i, j)

    motif1 = data_matrix.index[max_pair[0]]
    motif2 = data_matrix.index[max_pair[1]]

    print(f"Most distant motifs: '{motif1}' and '{motif2}' (distance: {max_dist:.3f})")

    # Use first principal component to create linear ordering
    from sklearn.decomposition import PCA

    pca = PCA(n_components=1)
    projected = pca.fit_transform(data_matrix.values)

    # Create ordering based on first principal component
    pc1_order = np.argsort(projected[:, 0])
    ordered_motifs = [data_matrix.index[i] for i in pc1_order]

    # Check if most distant motifs are at extremes, if not reverse
    pos1 = ordered_motifs.index(motif1)
    pos2 = ordered_motifs.index(motif2)

    # If they're not at extremes, we might need to reverse
    if not (
        (pos1 == 0 and pos2 == len(ordered_motifs) - 1)
        or (pos2 == 0 and pos1 == len(ordered_motifs) - 1)
    ):
        print(f"PC1 ordering: '{motif1}' at {pos1+1}, '{motif2}' at {pos2+1}")
        print("Most distant motifs not at extremes, this may be expected with PC1")
    else:
        print(f"✓ PC1 ordering places most distant motifs at extremes")

    print(f"PC1-based order: {ordered_motifs[:3]}...{ordered_motifs[-3:]}")

    return ordered_motifs


def plot_heatmap_ordered(data_matrix, motif_order, output_file=None):
    """
    Create a heatmap with motifs ordered according to the hierarchical clustering
    """
    # Reorder the data matrix according to the dendrogram order
    ordered_matrix = data_matrix.reindex(motif_order)

    # Create the heatmap without clustering
    plt.figure(figsize=(8, 12))

    # Note: seaborn heatmap displays with first row at top by default
    # Let's also reverse the y-axis to make sure ordering is clear
    ax = sns.heatmap(
        ordered_matrix,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Effect Size\nlog2(Observed / Expected)"},
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
    )

    plt.title(
        "Motif Effect Size Heatmap\n(Ordered by Distance Relationships)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Developmental Stage", fontsize=12, fontweight="bold")
    plt.ylabel("Motif (Distance-Based Order)", fontsize=12, fontweight="bold")

    # Adjust tick parameters
    plt.tick_params(axis="y", labelsize=8)
    plt.tick_params(axis="x", labelsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Ordered heatmap saved to: {output_file}")

    plt.close()


def plot_dendrogram_only(linkage_matrix, labels, output_file=None):
    """
    Plot just the dendrogram for detailed view with Set1 colors using natural clustering order
    """

    plt.figure(figsize=(15, 8))

    # Set color palette to Set1
    set_link_color_palette(
        [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
        ]
    )

    # Create dendrogram
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="top",
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7
        * max(linkage_matrix[:, 2]),  # Color threshold for Set1 colors
    )

    plt.title(
        "Motif Clustering Dendrogram\n(Ward's Method, Euclidean Distance)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Motifs", fontsize=12, fontweight="bold")
    plt.ylabel("Distance", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Dendrogram saved to: {output_file}")

    plt.close()


def evaluate_optimal_clusters(
    data_matrix, linkage_matrix, max_clusters=15, output_file=None
):
    """
    Evaluate different numbers of clusters using multiple metrics
    """
    cluster_range = range(2, min(max_clusters + 1, len(data_matrix)))
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []

    print("Evaluating cluster validation metrics...")
    print("Clusters\tSilhouette\tCalinski-H\tDavies-B")
    print("-" * 50)

    for n_clusters in cluster_range:
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

        # Calculate validation metrics
        sil_score = silhouette_score(data_matrix.values, cluster_labels)
        cal_score = calinski_harabasz_score(data_matrix.values, cluster_labels)
        db_score = davies_bouldin_score(data_matrix.values, cluster_labels)

        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)

        print(f"{n_clusters}\t\t{sil_score:.3f}\t\t{cal_score:.2f}\t\t{db_score:.3f}")

    # Plot the metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Silhouette Score (higher is better)
    axes[0, 0].plot(cluster_range, silhouette_scores, "bo-")
    axes[0, 0].set_title("Silhouette Score\n(Higher is Better)")
    axes[0, 0].set_xlabel("Number of Clusters")
    axes[0, 0].set_ylabel("Silhouette Score")
    axes[0, 0].grid(True, alpha=0.3)
    best_sil = cluster_range[np.argmax(silhouette_scores)]
    axes[0, 0].axvline(x=best_sil, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].text(
        best_sil,
        max(silhouette_scores),
        f"  Best: {best_sil}",
        verticalalignment="top",
        color="red",
    )

    # Calinski-Harabasz Index (higher is better)
    axes[0, 1].plot(cluster_range, calinski_scores, "go-")
    axes[0, 1].set_title("Calinski-Harabasz Index\n(Higher is Better)")
    axes[0, 1].set_xlabel("Number of Clusters")
    axes[0, 1].set_ylabel("Calinski-Harabasz Score")
    axes[0, 1].grid(True, alpha=0.3)
    best_cal = cluster_range[np.argmax(calinski_scores)]
    axes[0, 1].axvline(x=best_cal, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].text(
        best_cal,
        max(calinski_scores),
        f"  Best: {best_cal}",
        verticalalignment="top",
        color="red",
    )

    # Davies-Bouldin Index (lower is better)
    axes[1, 0].plot(cluster_range, davies_bouldin_scores, "ro-")
    axes[1, 0].set_title("Davies-Bouldin Index\n(Lower is Better)")
    axes[1, 0].set_xlabel("Number of Clusters")
    axes[1, 0].set_ylabel("Davies-Bouldin Score")
    axes[1, 0].grid(True, alpha=0.3)
    best_db = cluster_range[np.argmin(davies_bouldin_scores)]
    axes[1, 0].axvline(x=best_db, color="red", linestyle="--", alpha=0.7)
    axes[1, 0].text(
        best_db,
        min(davies_bouldin_scores),
        f"  Best: {best_db}",
        verticalalignment="bottom",
        color="red",
    )

    # Combined normalized scores
    # Normalize scores to [0, 1] range
    sil_norm = np.array(silhouette_scores)
    sil_norm = (sil_norm - sil_norm.min()) / (sil_norm.max() - sil_norm.min())

    cal_norm = np.array(calinski_scores)
    cal_norm = (cal_norm - cal_norm.min()) / (cal_norm.max() - cal_norm.min())

    db_norm = np.array(davies_bouldin_scores)
    db_norm = 1 - (db_norm - db_norm.min()) / (
        db_norm.max() - db_norm.min()
    )  # Invert since lower is better

    # Combined score (average of normalized metrics)
    combined_scores = (sil_norm + cal_norm + db_norm) / 3

    axes[1, 1].plot(cluster_range, combined_scores, "mo-")
    axes[1, 1].set_title("Combined Normalized Score\n(Higher is Better)")
    axes[1, 1].set_xlabel("Number of Clusters")
    axes[1, 1].set_ylabel("Combined Score")
    axes[1, 1].grid(True, alpha=0.3)
    best_combined = cluster_range[np.argmax(combined_scores)]
    axes[1, 1].axvline(x=best_combined, color="red", linestyle="--", alpha=0.7)
    axes[1, 1].text(
        best_combined,
        max(combined_scores),
        f"  Best: {best_combined}",
        verticalalignment="top",
        color="red",
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Cluster evaluation plot saved to: {output_file}")

    # Recommendations
    print("\nCluster Number Recommendations:")
    print("-" * 40)
    print(f"Best by Silhouette Score: {best_sil}")
    print(f"Best by Calinski-Harabasz: {best_cal}")
    print(f"Best by Davies-Bouldin: {best_db}")
    print(f"Best by Combined Score: {best_combined}")

    # Find the most common recommendation
    recommendations = [best_sil, best_cal, best_db, best_combined]
    from collections import Counter

    counter = Counter(recommendations)
    most_common = counter.most_common(1)[0][0]
    print(f"\nMost Frequently Recommended: {most_common} clusters")

    plt.close()
    return best_combined, {
        "silhouette": (cluster_range, silhouette_scores, best_sil),
        "calinski": (cluster_range, calinski_scores, best_cal),
        "davies_bouldin": (cluster_range, davies_bouldin_scores, best_db),
        "combined": (cluster_range, combined_scores, best_combined),
    }


def parse_motif_components(motif_labels):
    """
    Parse motif labels into individual components (AL, AM, LM, PM, RSP)
    Returns a binary matrix where rows=motifs, columns=components
    """
    all_components = ["AL", "AM", "LM", "PM", "RSP"]
    component_matrix = np.zeros((len(motif_labels), len(all_components)))

    for i, motif in enumerate(motif_labels):
        if motif and motif != "<null>" and motif != "<parse_error>":
            components = motif.split("+")
            for comp in components:
                if comp in all_components:
                    comp_idx = all_components.index(comp)
                    component_matrix[i, comp_idx] = 1

    return component_matrix, all_components


def plot_component_ordering_analysis(
    data_matrix, component_df, ordering_positions, clustered_data, output_prefix
):
    """
    Create visualizations for component contributions to spectrum and clustering
    """
    component_names = component_df.columns.tolist()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Motif Component Contributions to Clustering Spectrum",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Component presence along linear ordering
    ax1 = axes[0, 0]
    ordering_pos = [ordering_positions[motif] for motif in data_matrix.index]

    # Plot each component as scatter points
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, comp in enumerate(component_names):
        comp_presence = component_df[comp].values
        x_positions = [
            ordering_positions[motif]
            for motif in data_matrix.index
            if component_df.loc[motif, comp] == 1
        ]

        # Create y-positions with small jitter for visibility
        n_points = len(x_positions)
        if n_points > 0:
            y_positions = [i] * n_points + np.random.normal(0, 0.05, n_points)
        else:
            y_positions = []

        ax1.scatter(
            x_positions,
            y_positions,
            color=colors[i % len(colors)],
            alpha=0.7,
            s=60,
            label=comp,
        )

    ax1.set_xlabel("Linear Position (Distance-Based Order)")
    ax1.set_ylabel("Component")
    ax1.set_yticks(range(len(component_names)))
    ax1.set_yticklabels(component_names)
    ax1.set_title("Component Presence Along Linear Ordering")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Correlation with linear position
    ax2 = axes[0, 1]
    correlations = []
    for comp in component_names:
        comp_presence = component_df[comp].values.astype(float)
        if comp_presence.sum() > 0:
            correlation = np.corrcoef(comp_presence, ordering_pos)[0, 1]
        else:
            correlation = 0
        correlations.append(correlation)

    bars = ax2.bar(component_names, correlations, color=colors[: len(component_names)])
    ax2.set_ylabel("Correlation with Linear Position")
    ax2.set_title("Component-Position Correlations")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.02 if height >= 0 else -0.05),
            f"{corr:.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 3. Component distribution across clusters
    ax3 = axes[1, 0]
    n_clusters = clustered_data["Cluster"].nunique()
    cluster_ids = sorted(clustered_data["Cluster"].unique())

    x_pos = np.arange(len(component_names))
    width = 0.8 / n_clusters

    for i, cluster_id in enumerate(cluster_ids):
        cluster_motifs = clustered_data[clustered_data["Cluster"] == cluster_id].index
        cluster_comp_props = []

        for comp in component_names:
            if len(cluster_motifs) > 0:
                prop = component_df.loc[cluster_motifs, comp].sum() / len(
                    cluster_motifs
                )
            else:
                prop = 0
            cluster_comp_props.append(prop)

        ax3.bar(
            x_pos + i * width,
            cluster_comp_props,
            width,
            label=f"Cluster {cluster_id}",
            alpha=0.8,
        )

    ax3.set_xlabel("Component")
    ax3.set_ylabel("Proportion in Cluster")
    ax3.set_title("Component Distribution Across Clusters")
    ax3.set_xticks(x_pos + width * (n_clusters - 1) / 2)
    ax3.set_xticklabels(component_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Chi-square contributions
    ax4 = axes[1, 1]
    chi_squares = []

    for comp in component_names:
        if component_df[comp].sum() > 0:
            cluster_comp_counts = []
            cluster_sizes = []

            for cluster_id in cluster_ids:
                cluster_motifs = clustered_data[
                    clustered_data["Cluster"] == cluster_id
                ].index
                comp_count = component_df.loc[cluster_motifs, comp].sum()
                cluster_size = len(cluster_motifs)
                cluster_comp_counts.append(comp_count)
                cluster_sizes.append(cluster_size)

            # Expected vs observed component distribution
            total_comp = component_df[comp].sum()
            total_motifs = len(component_df)

            expected_dist = [
                size * (total_comp / total_motifs) for size in cluster_sizes
            ]
            observed_dist = cluster_comp_counts

            # Chi-square statistic
            chi_sq = sum(
                (obs - exp) ** 2 / (exp + 1e-6)
                for obs, exp in zip(observed_dist, expected_dist)
            )
        else:
            chi_sq = 0
        chi_squares.append(chi_sq)

    bars = ax4.bar(component_names, chi_squares, color=colors[: len(component_names)])
    ax4.set_ylabel("χ² Statistic")
    ax4.set_title(
        "Component Clustering Contribution\n(Higher = More Important for Separation)"
    )
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, chi_sq in zip(bars, chi_squares):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{chi_sq:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the figure
    output_file = f"{output_prefix}_ordering_analysis.png" if isinstance(output_prefix, str) else str(output_prefix / "motif_clustering_ordering_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(
        f"Linear ordering analysis plot saved to: {output_file}"
    )
    plt.close()

    return fig


def analyze_linear_ordering(data_matrix, motif_order, distance_matrix, clustered_data):
    """
    Analyze motif linear ordering using the provided distance-based order
    """
    # Find the pair of motifs with maximum distance
    max_dist = 0
    max_pair = (0, 0)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i, j] > max_dist:
                max_dist = distance_matrix[i, j]
                max_pair = (i, j)

    motif1 = data_matrix.index[max_pair[0]]
    motif2 = data_matrix.index[max_pair[1]]

    print(f"Most distant motifs: '{motif1}' and '{motif2}' (distance: {max_dist:.3f})")

    # Find positions of the most distant motifs in the linear order
    pos1 = motif_order.index(motif1)
    pos2 = motif_order.index(motif2)

    total_motifs = len(motif_order)
    print(
        f"Linear order positions: '{motif1}' at {pos1+1}/{total_motifs}, '{motif2}' at {pos2+1}/{total_motifs}"
    )

    # Create ordering positions for analysis
    ordering_positions = {motif: i for i, motif in enumerate(motif_order)}

    print(f"\nLinear Ordering (distance-based):")
    for i, motif in enumerate(motif_order):
        cluster_id = clustered_data.loc[motif, "Cluster"]
        print(f"{i+1:2d}. {motif:<20} (Cluster {cluster_id})")

    return ordering_positions


def analyze_motif_ordering(data_matrix, motif_order, clustered_data, output_prefix):
    """
    Analyze motif linear ordering and component contributions using the provided distance-based order
    """
    print("\n" + "=" * 60)
    print("MOTIF LINEAR ORDERING ANALYSIS")
    print("=" * 60)

    # Calculate distance matrix for verification
    condensed_distances = pdist(data_matrix.values, metric="euclidean")
    distance_matrix = squareform(condensed_distances)

    # Analyze ordering using the provided motif order
    ordering_positions = analyze_linear_ordering(
        data_matrix, motif_order, distance_matrix, clustered_data
    )

    print(f"Linear ordering spans from '{motif_order[0]}' to '{motif_order[-1]}'")

    # Parse motif components
    component_matrix, component_names = parse_motif_components(data_matrix.index)
    component_df = pd.DataFrame(
        component_matrix, index=data_matrix.index, columns=component_names
    )

    print(f"\nLinear Ordering (distance-based):")
    for i, motif in enumerate(motif_order):
        cluster_id = clustered_data.loc[motif, "Cluster"]
        print(f"{i+1:2d}. {motif:<20} (Cluster {cluster_id})")

    # Analyze component contributions to linear position
    print(f"\nComponent Contributions to Linear Position:")
    print("-" * 50)

    ordering_pos = [ordering_positions[motif] for motif in data_matrix.index]

    for comp in component_names:
        # Correlation between having this component and linear position
        comp_presence = component_df[comp].values.astype(float)
        if comp_presence.sum() > 0:  # Only analyze if component exists
            correlation = np.corrcoef(comp_presence, ordering_pos)[0, 1]

            # Average linear position for motifs with/without component
            with_comp = np.mean(
                [
                    ordering_positions[motif]
                    for motif in data_matrix.index
                    if component_df.loc[motif, comp] == 1
                ]
            )
            without_comp = np.mean(
                [
                    ordering_positions[motif]
                    for motif in data_matrix.index
                    if component_df.loc[motif, comp] == 0
                ]
            )

            print(
                f"{comp:>3}: r={correlation:6.3f}, "
                f"With: pos {with_comp:5.1f}, Without: pos {without_comp:5.1f}"
            )

    # Analyze component contributions to cluster separation
    print(f"\nComponent Contributions to Cluster Separation:")
    print("-" * 55)

    n_clusters = clustered_data["Cluster"].nunique()
    for comp in component_names:
        if component_df[comp].sum() > 0:
            # Chi-square like analysis - how unequally distributed is this component across clusters?
            cluster_comp_counts = []
            cluster_sizes = []

            for cluster_id in range(1, n_clusters + 1):
                cluster_motifs = clustered_data[
                    clustered_data["Cluster"] == cluster_id
                ].index
                comp_count = component_df.loc[cluster_motifs, comp].sum()
                cluster_size = len(cluster_motifs)
                cluster_comp_counts.append(comp_count)
                cluster_sizes.append(cluster_size)

            # Expected vs observed component distribution
            total_comp = component_df[comp].sum()
            total_motifs = len(component_df)

            expected_dist = [
                size * (total_comp / total_motifs) for size in cluster_sizes
            ]
            observed_dist = cluster_comp_counts

            # Chi-square statistic (rough measure of clustering contribution)
            chi_sq = sum(
                (obs - exp) ** 2 / (exp + 1e-6)
                for obs, exp in zip(observed_dist, expected_dist)
            )

            print(f"{comp:>3}: χ²={chi_sq:6.2f}, Distribution: {observed_dist}")

    # Create visualizations
    plot_component_ordering_analysis(
        data_matrix, component_df, ordering_positions, clustered_data, output_prefix
    )

    return motif_order, component_df, ordering_positions


def analyze_clusters(data_matrix, linkage_matrix, n_clusters=5):
    """
    Analyze and describe the clusters
    """
    from scipy.cluster.hierarchy import fcluster

    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    # Add cluster labels to the data
    clustered_data = data_matrix.copy()
    clustered_data["Cluster"] = cluster_labels

    print(f"\nCluster Analysis ({n_clusters} clusters):")
    print("=" * 50)

    for cluster_id in range(1, n_clusters + 1):
        cluster_motifs = clustered_data[clustered_data["Cluster"] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_motifs)} motifs):")
        print("-" * 30)

        # Show motifs in this cluster
        motif_list = cluster_motifs.index.tolist()
        print("Motifs:", ", ".join(motif_list))

        # Show average effect size trajectory
        avg_trajectory = cluster_motifs.drop("Cluster", axis=1).mean()
        print("Average trajectory:")
        for stage, value in avg_trajectory.items():
            print(f"  {stage}: {value:.3f}")

    return clustered_data


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Cluster motif effect size trajectories"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input CSV file with motif data (default: combined_effect_sizes.csv from motif_significange_trajectories output)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="motif_clustering",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for analysis (if not specified, optimal number will be determined automatically)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing output from 07_motif_significange_trajectories.py (default: helpers/outputs/07_motif_significange_trajectories)",
    )
    parser.add_argument(
        "--helper_output_dir",
        type=str,
        default=None,
        help="Directory for helper script outputs (default: helpers/outputs/08_motif_clustering)",
    )

    args = parser.parse_args()
    
    # Set output directory
    script_dir = Path(__file__).parent
    if args.helper_output_dir:
        output_dir = Path(args.helper_output_dir)
    else:
        output_dir = script_dir.parent / "outputs" / "08_motif_clustering"
    os.makedirs(output_dir, exist_ok=True)

    # Process all three models separately
    models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                     'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                     'bayesian_hierarchical', 'ml_nonparametric', 'proportional_effectsize']
    
    # Check if input_dir contains model subdirectories
    input_base_dir = Path(args.input_dir) if args.input_dir else script_dir.parent / "outputs" / "07_motif_significange_trajectories"
    
    for model_type in models_to_process:
        print("\n" + "="*80)
        print(f"Processing {model_type.upper()} MODEL")
        print("="*80)
        
        # Determine input file path for this model
        if args.input_file is None:
            # Check for model-specific file first, then fall back to main directory
            model_input_file = input_base_dir / model_type / f"combined_effect_sizes_{model_type}.csv"
            if not model_input_file.exists():
                # Fall back to main directory (backward compatibility)
                model_input_file = input_base_dir / "combined_effect_sizes.csv"
        else:
            # If input_file is specified, use it for first model only, then skip others
            if model_type == 'uniform':
                model_input_file = Path(args.input_file)
            else:
                # For region_specific, try to find corresponding file
                input_file_path = Path(args.input_file)
                model_input_file = input_file_path.parent / model_type / f"combined_effect_sizes_{model_type}.csv"
                if not model_input_file.exists():
                    print(f"Skipping {model_type} model - input file not found")
                    continue
        
        if not model_input_file.exists():
            print(f"Warning: Input file not found for {model_type} model: {model_input_file}")
            print(f"  Please run motif_significange_trajectories.py first or specify --input_file")
            continue
        
        # Create model-specific output directory
        model_output_dir = output_dir / model_type
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load and prepare data
        print(f"Loading {model_type} model data from {model_input_file}...")
        df = pd.read_csv(model_input_file)

        print(f"Preparing data for clustering ({model_type} model)...")
        data_matrix = prepare_clustering_data(df)

        # Perform clustering
        print(f"Performing hierarchical clustering ({model_type} model)...")
        linkage_matrix, distance_matrix, method_name = perform_clustering(data_matrix)

        # Evaluate optimal number of clusters
        print(f"Evaluating optimal number of clusters ({model_type} model)...")
        optimal_clusters, metrics_data = evaluate_optimal_clusters(
            data_matrix,
            linkage_matrix,
            max_clusters=15,
            output_file=str(model_output_dir / "motif_clustering_evaluation.png"),
        )

        # Use the optimal number of clusters if not specified
        if args.n_clusters is None:
            final_n_clusters = optimal_clusters
            print(f"Using optimal number of clusters: {final_n_clusters}")
        else:
            final_n_clusters = args.n_clusters
            print(f"Using user-specified number of clusters: {final_n_clusters}")

        # Analyze clusters to get cluster assignments
        print(f"Analyzing clusters ({model_type} model)...")
        clustered_data = analyze_clusters(data_matrix, linkage_matrix, final_n_clusters)

        # Extract the dendrogram order for tree visualization
        print(f"Extracting dendrogram order ({model_type} model)...")
        dendrogram_order = extract_dendrogram_order(linkage_matrix, data_matrix)

        # Create distance-based linear ordering for heatmap
        print(f"Creating distance-based ordering ({model_type} model)...")
        from scipy.spatial.distance import pdist, squareform

        condensed_distances = pdist(data_matrix.values, metric="euclidean")
        distance_matrix = squareform(condensed_distances)
        distance_order = create_distance_based_ordering(data_matrix, distance_matrix)

        # Create visualizations using appropriate orderings
        print(f"Creating dendrogram ({model_type} model)...")
        plot_dendrogram_only(
            linkage_matrix,
            data_matrix.index,
            str(model_output_dir / "motif_clustering_dendrogram.png"),
        )

        print(f"Creating distance-ordered heatmap ({model_type} model)...")
        plot_heatmap_ordered(
            data_matrix,
            distance_order,
            str(model_output_dir / "motif_clustering_heatmap.png"),
        )

        # Analyze motif linear ordering using distance-based ordering
        print(f"Analyzing motif linear ordering ({model_type} model)...")
        motif_order, component_df, ordering_positions = analyze_motif_ordering(
            data_matrix, distance_order, clustered_data, str(model_output_dir / "motif_clustering")
        )
        
        print(f"✅ Completed processing for {model_type} model")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("   - Uniform model: {}/uniform/".format(output_dir))
    print("   - Region-specific model: {}/region_specific/".format(output_dir))


if __name__ == "__main__":
    main()
