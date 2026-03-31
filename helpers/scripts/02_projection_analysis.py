import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent opening windows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import os
import glob
import time
from pathlib import Path
from itertools import combinations

# Set font to Helvetica with Arial fallback, and ensure SVG text is editable
plt.rcParams["font.family"] = ['Helvetica', 'Arial']
plt.rcParams["svg.fonttype"] = "none"


def analyze_projection_data(df, metadata=None, comparison_name="", save_dir=None):
    """
    Analyze projection data following Klingler et al. 2018 methods
    Each row represents a barcode with projection strengths to target regions
    """

    # Create save directory if specified
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Data preprocessing
    print("=== Data Preprocessing ===")
    print(f"Data shape: {df.shape}")

    # Replace NaN with 0 and remove zero-sum rows
    df_clean = df.fillna(0)
    row_sums = df_clean.sum(axis=1)
    valid_mask = row_sums > 0
    df_clean = df_clean[valid_mask].reset_index(drop=True)

    # Clean metadata to match cleaned data
    if metadata is not None:
        metadata_clean = metadata[valid_mask].reset_index(drop=True)
    else:
        metadata_clean = None

    # Normalize each row to sum to 1 (compositional normalization)
    df_norm = df_clean.div(df_clean.sum(axis=1), axis=0)

    # Reverse column order for display
    df_norm = df_norm[df_norm.columns[::-1]]

    print(f"Final data shape: {df_norm.shape}")
    print(f"Data normalized to sum to 1 per barcode")

    # 2. Principal Component Analysis
    print("\n=== Principal Component Analysis ===")

    # For compositional data, max components = n_features - 1
    n_components = len(df_norm.columns) - 1
    print(f"Using {n_components} components (n_features - 1)")

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_norm)

    # Create loadings dataframe
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df_norm.columns,
    )

    # 3. Visualization
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"PCA Analysis: {comparison_name}", fontsize=16, fontweight="bold")

    # First compute hierarchical clustering for ordering (like the paper)
    try:
        linkage_matrix = linkage(df_norm, method="ward")
        # Check if linkage matrix is valid
        if linkage_matrix.size == 0 or len(linkage_matrix) == 0:
            print("  Warning: Empty linkage matrix, using default order")
            cluster_order = np.arange(len(df_norm))
        else:
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro["leaves"]
    except Exception as e:
        print(f"  Warning: Clustering failed: {e}, using default order")
        cluster_order = np.arange(len(df_norm))

    # PC heatmap (barcodes x PCs) - sorted by clustering
    ax1 = plt.subplot(2, 2, 1)

    pca_clustered = pca_result[cluster_order]

    sns.heatmap(
        pca_clustered,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "PC Score"},
        ax=ax1,
        yticklabels=50,
        xticklabels=[f"PC{i+1}" for i in range(n_components)],
    )
    ax1.set_title(f"Principal Component Scores\n{comparison_name}")
    ax1.set_ylabel("Barcode Index (clustered order)")
    ax1.set_xlabel("Principal Components")

    # Variance explained with cumulative line
    ax2 = plt.subplot(2, 2, 2)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    bars = ax2.bar(pc_labels, pca.explained_variance_ratio_, alpha=0.7, color="skyblue")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Variance Explained", color="blue")
    ax2.set_title(f"Variance Explained\n{comparison_name}")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", labelcolor="blue")

    # Add cumulative line
    ax2_twin = ax2.twinx()
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    ax2_twin.plot(range(len(cumulative_var)), cumulative_var, "ro-", linewidth=2)
    ax2_twin.set_ylabel("Cumulative Variance", color="red")
    ax2_twin.tick_params(axis="y", labelcolor="red")
    ax2_twin.set_ylim(0, 1.0)

    # PCA scatter
    ax3 = plt.subplot(2, 2, 3)
    if metadata_clean is not None:
        # Color points by age group
        age_groups = metadata_clean["age_group"].unique()
        colors = ["blue", "red", "green"]

        for i, age_group in enumerate(sorted(age_groups)):
            age_mask = metadata_clean["age_group"] == age_group
            age_pc1 = pca_result[age_mask, 0]
            age_pc2 = pca_result[age_mask, 1]

            # Plot individual points
            ax3.scatter(
                age_pc1,
                age_pc2,
                alpha=0.6,
                label=f"{age_group} data",
                color=colors[i % len(colors)],
                linewidth=0,
                s=20,
            )

            # Calculate and plot centroid
            centroid_pc1 = np.mean(age_pc1)
            centroid_pc2 = np.mean(age_pc2)

            # Plot centroid with distinct marker
            ax3.scatter(
                centroid_pc1,
                centroid_pc2,
                marker="s",
                s=100,
                alpha=0.5,
                color=colors[i % len(colors)],
                edgecolors="black",
                linewidth=0,
                label=f"{age_group} centroid",
                zorder=10,
            )

        ax3.legend(loc="best")
    else:
        ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)

    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax3.set_title(f"PC1 vs PC2\n{comparison_name}")

    # Overall projection heatmap - sorted by clustering
    ax4 = plt.subplot(2, 2, 4)
    df_clustered = df_norm.iloc[cluster_order]

    sns.heatmap(
        df_clustered,
        cmap="binary",
        cbar_kws={"label": "Projection Strength"},
        ax=ax4,
        yticklabels=50,
        xticklabels=df_norm.columns,
    )
    ax4.set_title(f"All Projection Patterns\n{comparison_name}")
    ax4.set_ylabel("Barcode Index (clustered order)")
    ax4.set_xlabel("Target Regions")

    plt.tight_layout()

    # Save main PCA figure
    if save_dir:
        filename = f"PCA_Analysis_{comparison_name.replace(' vs ', '_vs_').replace(' ', '_')}.svg"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, format="svg", bbox_inches="tight")
        print(f"Saved: {filepath}")
        plt.close()  # Close figure to free memory

    # 3b. Age group specific heatmaps
    if metadata_clean is not None:
        print(f"\n=== Age Group Specific Heatmaps: {comparison_name} ===")

        age_groups = metadata_clean["age_group"].unique()
        n_ages = len(age_groups)

        fig, axes = plt.subplots(1, n_ages, figsize=(5 * n_ages, 10))
        fig.suptitle(
            f"Age Group Comparison: {comparison_name}", fontsize=14, fontweight="bold"
        )
        if n_ages == 1:
            axes = [axes]

        for i, age_group in enumerate(sorted(age_groups)):
            age_mask = metadata_clean["age_group"] == age_group
            age_data = df_norm[age_mask]

            # Hierarchical clustering within this age group (like the paper)
            if len(age_data) > 1:
                try:
                    age_linkage = linkage(age_data, method="ward")
                    if age_linkage.size == 0 or len(age_linkage) == 0:
                        age_cluster_order = np.arange(len(age_data))
                    else:
                        age_dendro = dendrogram(age_linkage, no_plot=True)
                        age_cluster_order = age_dendro["leaves"]
                    age_sorted = age_data.iloc[age_cluster_order]
                except Exception as e:
                    print(f"  Warning: Age clustering failed: {e}, using default order")
                    age_sorted = age_data
            else:
                age_sorted = age_data

            sns.heatmap(
                age_sorted,
                cmap="binary",
                cbar_kws={"label": "Projection"},
                ax=axes[i],
                yticklabels=max(1, len(age_sorted) // 10),
                xticklabels=True,
            )
            axes[i].set_title(f"{age_group.upper()}\n({len(age_data)} barcodes)")
            axes[i].set_ylabel("Barcode Index (clustered)")
            axes[i].set_xlabel("Target Regions")

        plt.tight_layout()

        # Save age group heatmaps
        if save_dir:
            filename = f"Age_Group_Heatmaps_{comparison_name.replace(' vs ', '_vs_').replace(' ', '_')}.svg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format="svg", bbox_inches="tight")
            print(f"Saved: {filepath}")
            plt.close()  # Close figure to free memory

        # 3c. Mean projection patterns by age group - GROUPED BAR PLOT WITH STATISTICS
        print(f"\n=== Mean Projection Patterns: {comparison_name} ===")

        # Calculate means and SEMs for each age group
        age_groups_sorted = sorted(age_groups)
        target_regions = df_norm.columns.tolist()

        mean_data = {}
        sem_data = {}
        raw_data = {}

        for age_group in age_groups_sorted:
            age_mask = metadata_clean["age_group"] == age_group
            age_values = df_norm[age_mask]

            # Store raw data for statistical tests
            raw_data[age_group] = age_values

            # Calculate means and SEMs
            means = age_values.mean()
            sems = age_values.sem()  # Standard error of mean

            mean_data[age_group] = means
            sem_data[age_group] = sems

        # Perform statistical tests between age groups for each target region
        print("Performing statistical comparisons...")
        p_values = {}

        for target_region in target_regions:
            # Get data for this target region from both age groups
            group1_data = raw_data[age_groups_sorted[0]][target_region].values
            group2_data = raw_data[age_groups_sorted[1]][target_region].values

            # Perform independent samples t-test
            statistic, p_value = stats.ttest_ind(group1_data, group2_data)
            p_values[target_region] = p_value

            print(f"  {target_region}: t={statistic:.3f}, p={p_value:.4f}")

        # Apply Bonferroni correction for multiple comparisons
        n_comparisons = len(target_regions)
        alpha = 0.05
        bonferroni_alpha = alpha / n_comparisons

        print(f"\nBonferroni corrected alpha: {bonferroni_alpha:.4f}")

        # Create significance markers
        sig_markers = {}
        for target_region, p_val in p_values.items():
            if p_val < 0.001:
                sig_markers[target_region] = "***"
            elif p_val < 0.01:
                sig_markers[target_region] = "**"
            elif p_val < bonferroni_alpha:
                sig_markers[target_region] = "*"
            else:
                sig_markers[target_region] = "ns"

        # Create grouped bar plot with error bars
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up bar positions
        n_groups = len(target_regions)
        n_bars = len(age_groups_sorted)
        bar_width = 0.35

        # Generate positions for each age group
        x_pos = np.arange(n_groups)
        positions = {}
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, age_group in enumerate(age_groups_sorted):
            positions[age_group] = x_pos + (i - 0.5) * bar_width

        # Plot bars for each age group with error bars
        for i, age_group in enumerate(age_groups_sorted):
            means = [mean_data[age_group][region] for region in target_regions]
            sems = [sem_data[age_group][region] for region in target_regions]

            bars = ax.bar(
                positions[age_group],
                means,
                width=bar_width,
                label=age_group.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                yerr=sems,  # Add SEM error bars
                capsize=4,
                error_kw={"linewidth": 1.5, "capthick": 1.5},
            )

        # Add significance markers
        max_heights = []
        for region in target_regions:
            max_val = 0
            for age_group in age_groups_sorted:
                mean_val = mean_data[age_group][region]
                sem_val = sem_data[age_group][region]
                max_val = max(max_val, mean_val + sem_val)
            max_heights.append(max_val)

        # Add significance markers above bars
        for i, region in enumerate(target_regions):
            max_height = max_heights[i]
            marker_height = max_height + 0.02  # Offset above error bars

            # Add significance marker
            sig_text = sig_markers[region]
            if sig_text != "ns":
                ax.text(
                    x_pos[i],
                    marker_height,
                    sig_text,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

                # Add horizontal line connecting the bars being compared
                left_pos = positions[age_groups_sorted[0]][i]
                right_pos = positions[age_groups_sorted[1]][i]
                line_height = marker_height - 0.01

                ax.plot(
                    [left_pos, right_pos], [line_height, line_height], "k-", linewidth=1
                )

        # Customize the plot
        ax.set_xlabel("Target Regions", fontsize=14)
        ax.set_ylabel("Mean Projection Strength ± SEM", fontsize=14)
        ax.set_title(
            f"Mean Projection Strength by Age Group\n{comparison_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_regions, rotation=0, ha="center")
        ax.legend(title="Age Group", loc="upper right", fontsize=12)

        plt.tight_layout()

        # Save mean projection figure
        if save_dir:
            filename = f"Mean_Projection_Stats_{comparison_name.replace(' vs ', '_vs_').replace(' ', '_')}.svg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format="svg", bbox_inches="tight")
            print(f"Saved: {filepath}")
            plt.close()  # Close figure to free memory

        # Print statistical summary
        print(f"\nStatistical Summary for {comparison_name}:")
        for region in target_regions:
            p_val = p_values[region]
            sig = sig_markers[region]
            print(f"  {region}: p = {p_val:.4f} ({sig})")

        significant_regions = [r for r in target_regions if sig_markers[r] != "ns"]
        if significant_regions:
            print(
                f"\nSignificant differences found in: {', '.join(significant_regions)}"
            )
        else:
            print(f"\nNo significant differences found after Bonferroni correction.")

        # 3d. Proportion comparisons - PRIMARY PROJECTION ANALYSIS
        print(f"\n=== Primary Projection Proportions: {comparison_name} ===")

        # For each barcode, find which target region has the highest projection
        primary_projections = {}
        proportion_data = {}

        for age_group in age_groups_sorted:
            age_mask = metadata_clean["age_group"] == age_group
            age_values = df_norm[age_mask]

            # Find primary projection for each barcode (region with max value)
            primary_regions = age_values.idxmax(axis=1)
            primary_projections[age_group] = primary_regions

            # Calculate proportions for each target region
            proportions = {}
            n_barcodes = len(age_values)

            for region in target_regions:
                count = (primary_regions == region).sum()
                proportion = count / n_barcodes
                proportions[region] = proportion

            proportion_data[age_group] = proportions

            print(f"{age_group.upper()}: {n_barcodes} barcodes")
            for region in target_regions:
                count = (primary_regions == region).sum()
                print(f"  {region}: {count}/{n_barcodes} ({proportions[region]:.3f})")

        # Statistical tests for proportions (Chi-square test)
        print("\nPerforming chi-square tests for proportion differences...")
        proportion_p_values = {}

        for region in target_regions:
            # Create contingency table for this region
            group1_primary = (primary_projections[age_groups_sorted[0]] == region).sum()
            group1_total = len(primary_projections[age_groups_sorted[0]])
            group1_other = group1_total - group1_primary

            group2_primary = (primary_projections[age_groups_sorted[1]] == region).sum()
            group2_total = len(primary_projections[age_groups_sorted[1]])
            group2_other = group2_total - group2_primary

            # Contingency table: [[primary, other], [primary, other]]
            contingency = np.array(
                [[group1_primary, group1_other], [group2_primary, group2_other]]
            )

            # Chi-square test - handle zero values that cause errors
            try:
                # Check if contingency table has any zero margins that would cause issues
                if contingency.sum(axis=0).min() == 0 or contingency.sum(axis=1).min() == 0:
                    # Skip chi-square test if there are zero margins
                    chi2, p_value, dof = np.nan, np.nan, 0
                    expected = contingency
                else:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            except ValueError as e:
                # Handle cases where chi-square test fails
                chi2, p_value, dof = np.nan, np.nan, 0
                expected = contingency
            proportion_p_values[region] = p_value

            if not np.isnan(chi2):
                print(f"  {region}: χ² = {chi2:.3f}, p = {p_value:.4f}")
            else:
                print(f"  {region}: χ² test skipped (zero margins), p = NaN")

        # Apply Bonferroni correction
        bonferroni_alpha_prop = alpha / len(target_regions)
        print(
            f"\nBonferroni corrected alpha for proportions: {bonferroni_alpha_prop:.4f}"
        )

        # Create significance markers for proportions
        prop_sig_markers = {}
        for region, p_val in proportion_p_values.items():
            if p_val < 0.001:
                prop_sig_markers[region] = "***"
            elif p_val < 0.01:
                prop_sig_markers[region] = "**"
            elif p_val < bonferroni_alpha_prop:
                prop_sig_markers[region] = "*"
            else:
                prop_sig_markers[region] = "ns"

        # Calculate confidence intervals for proportions (Wilson score interval)
        def wilson_confidence_interval(count, n, confidence=0.95):
            """Calculate Wilson score confidence interval for proportions"""
            if n == 0:
                return 0, 0

            z = stats.norm.ppf((1 + confidence) / 2)
            p = count / n

            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

            return max(0, centre - margin), min(1, centre + margin)

        # Create proportion bar plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up bar positions (same as before)
        x_pos = np.arange(len(target_regions))
        bar_width = 0.35
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        positions = {}
        for i, age_group in enumerate(age_groups_sorted):
            positions[age_group] = x_pos + (i - 0.5) * bar_width

        # Plot bars with confidence intervals
        for i, age_group in enumerate(age_groups_sorted):
            proportions = [
                proportion_data[age_group][region] for region in target_regions
            ]

            # Calculate confidence intervals
            n_barcodes = len(primary_projections[age_group])
            ci_lower = []
            ci_upper = []

            for region in target_regions:
                count = (primary_projections[age_group] == region).sum()
                lower, upper = wilson_confidence_interval(count, n_barcodes)
                ci_lower.append(proportions[target_regions.index(region)] - lower)
                ci_upper.append(upper - proportions[target_regions.index(region)])

            # Asymmetric error bars
            yerr = [ci_lower, ci_upper]

            bars = ax.bar(
                positions[age_group],
                proportions,
                width=bar_width,
                label=age_group.upper(),
                color=colors[i % len(colors)],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                yerr=yerr,
                capsize=4,
                error_kw={"linewidth": 1.5, "capthick": 1.5},
            )

        # Add significance markers (same approach as before)
        max_heights = []
        for region in target_regions:
            max_val = 0
            for age_group in age_groups_sorted:
                prop_val = proportion_data[age_group][region]
                n_barcodes = len(primary_projections[age_group])
                count = (primary_projections[age_group] == region).sum()
                _, upper_ci = wilson_confidence_interval(count, n_barcodes)
                max_val = max(max_val, upper_ci)
            max_heights.append(max_val)

        # Add significance markers above bars
        for i, region in enumerate(target_regions):
            max_height = max_heights[i]
            marker_height = max_height + 0.02

            sig_text = prop_sig_markers[region]
            if sig_text != "ns":
                ax.text(
                    x_pos[i],
                    marker_height,
                    sig_text,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

                # Add horizontal line
                left_pos = positions[age_groups_sorted[0]][i]
                right_pos = positions[age_groups_sorted[1]][i]
                line_height = marker_height - 0.01

                ax.plot(
                    [left_pos, right_pos], [line_height, line_height], "k-", linewidth=1
                )

        # Customize the plot
        ax.set_xlabel("Target Regions", fontsize=14)
        ax.set_ylabel("Proportion of Barcodes with Primary Projection", fontsize=14)
        ax.set_title(
            f"Primary Projection Proportions by Age Group\n{comparison_name}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_regions, rotation=0, ha="center")
        ax.legend(title="Age Group", loc="upper right", fontsize=12)
        ax.set_ylim(0, max(max_heights) + 0.05)

        plt.tight_layout()

        # Save proportion figure
        if save_dir:
            filename = f"Primary_Projection_Proportions_{comparison_name.replace(' vs ', '_vs_').replace(' ', '_')}.svg"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, format="svg", bbox_inches="tight")
            print(f"Saved: {filepath}")
            plt.close()  # Close figure to free memory

        # Print proportion statistical summary
        print(f"\nPrimary Projection Statistical Summary for {comparison_name}:")
        for region in target_regions:
            p_val = proportion_p_values[region]
            sig = prop_sig_markers[region]
            print(f"  {region}: χ² p = {p_val:.4f} ({sig})")

        significant_prop_regions = [
            r for r in target_regions if prop_sig_markers[r] != "ns"
        ]
        if significant_prop_regions:
            print(
                f"\nSignificant proportion differences found in: {', '.join(significant_prop_regions)}"
            )
        else:
            print(
                f"\nNo significant proportion differences found after Bonferroni correction."
            )

    # 4. Summary
    print(f"\nSUMMARY for {comparison_name}:")
    print(
        f"First 2 PCs explain {np.sum(pca.explained_variance_ratio_[:2]):.1%} of variance"
    )

    return {
        "pca": pca,
        "pca_result": pca_result,
        "loadings": loadings_df,
        "normalized_data": df_norm,
        "variance_explained": pca.explained_variance_ratio_,
        "metadata_clean": metadata_clean,
    }


def load_and_analyze_directory(base_directory, save_dir=None):
    """
    Load all projection data files and run pairwise analysis
    """

    print(f"Scanning directory: {base_directory}")

    # Create results directory if specified
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created results directory: {save_dir}")

    # Determine if we're in a parameterization-specific directory
    # Extract parameterization name from helper_output_dir if available, or search in base_directory
    base_path = Path(base_directory)
    search_dir = base_directory
    filter_type = None
    parameterization_filter = None
    
    # Try to extract parameterization name and filter type from helper_output_dir path if available
    if save_dir:
        save_path = Path(save_dir)
        # Check if save_dir contains a parameterization name (e.g., .../01.minimal_filter_parameters_..._helpers/...)
        for part in save_path.parts:
            if part.startswith(('01.', '02.', '03.', '04.', '05.')):
                # Extract parameterization name (before _helpers if present)
                if '_helpers' in part:
                    parameterization_filter = part.split('_helpers')[0]
                else:
                    parameterization_filter = part
                # Extract filter type from parameterization name
                if 'minimal' in part:
                    filter_type = 'minimal'
                elif 'medium' in part:
                    filter_type = 'medium'
                elif 'strict' in part:
                    filter_type = 'strict'
                elif 'extreme' in part:
                    filter_type = 'extreme'
                elif 'HAN' in part:
                    filter_type = 'HAN'
                break
    
    # If not found in save_dir, search for parameterization directories in base_directory
    if filter_type is None:
        # Search for parameterization directories in base_directory
        for age_dir in ['p3', 'p12', 'p20', 'p60']:
            age_path = Path(base_directory) / age_dir
            if age_path.exists():
                for param_dir in age_path.iterdir():
                    if param_dir.is_dir() and param_dir.name.startswith(('01.', '02.', '03.', '04.', '05.')):
                        # Extract filter type
                        if 'minimal' in param_dir.name:
                            filter_type = 'minimal'
                        elif 'medium' in param_dir.name:
                            filter_type = 'medium'
                        elif 'strict' in param_dir.name:
                            filter_type = 'strict'
                        elif 'extreme' in param_dir.name:
                            filter_type = 'extreme'
                        elif 'HAN' in param_dir.name:
                            filter_type = 'HAN'
                        break
                if filter_type:
                    break
    
    if filter_type:
        print(f"Detected filter type: {filter_type}")
    if parameterization_filter:
        print(f"Filtering by parameterization: {parameterization_filter}")
    print(f"Searching in: {search_dir}")
    
    # Find CSV files matching the pattern: {age}_ALL_{filter_type}_filters_Filtered_Matrix.csv
    # If filter_type is specified, only search for that type; otherwise search for all types
    filter_types = [filter_type] if filter_type else ['minimal', 'medium', 'strict', 'extreme', 'HAN']
    csv_files = []
    
    for ft in filter_types:
        if ft is None:
            continue
        # Try multiple case variations for "ALL" and age prefixes
        # IMPORTANT: Filter by parameterization directory if specified
        if parameterization_filter:
            # Only search within the specific parameterization directories
            patterns = []
            for age_dir in ['p3', 'p12', 'p20', 'p60']:
                param_path = Path(search_dir) / age_dir / parameterization_filter
                if param_path.exists():
                    patterns.extend([
                        str(param_path / "**" / f"*_ALL_{ft}_filters_Filtered_Matrix.csv"),
                        str(param_path / "**" / f"*_alL_{ft}_filters_Filtered_Matrix.csv"),
                        str(param_path / "**" / f"*_All_{ft}_filters_Filtered_Matrix.csv"),
                        str(param_path / "**" / f"*_aLl_{ft}_filters_Filtered_Matrix.csv"),
                    ])
        else:
            # Search all parameterizations (old behavior)
            patterns = [
                os.path.join(search_dir, "**", f"*_ALL_{ft}_filters_Filtered_Matrix.csv"),
                os.path.join(search_dir, "**", f"*_alL_{ft}_filters_Filtered_Matrix.csv"),
                os.path.join(search_dir, "**", f"*_All_{ft}_filters_Filtered_Matrix.csv"),
                os.path.join(search_dir, "**", f"*_aLl_{ft}_filters_Filtered_Matrix.csv"),
                # Also try case variations for age prefix (p3, P3, p12, P12, etc.)
                os.path.join(search_dir, "**", f"p*_ALL_{ft}_filters_Filtered_Matrix.csv"),
                os.path.join(search_dir, "**", f"P*_ALL_{ft}_filters_Filtered_Matrix.csv"),
            ]
        for pattern in patterns:
            csv_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    csv_files = list(set(csv_files))

    print(f"Found {len(csv_files)} data files")

    # Load all data first
    age_data = {}

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Extract age group from filename (e.g., "p12_ALL_HAN..." -> "p12")
        age_group = filename.split("_")[0]

        print(f"Loading {age_group}...")

        # Load CSV (depth bins as rows, regions as columns - already in correct format)
        df_raw = pd.read_csv(csv_file)

        # Reset index to avoid duplicate index issues
        df_raw = df_raw.reset_index(drop=True)

        print(f"  {age_group}: {len(df_raw)} depth bins, {len(df_raw.columns)} regions")

        # Store data for this age group
        age_data[age_group] = df_raw

    # Get all age groups and create pairwise combinations
    age_groups = sorted(age_data.keys())
    print(f"\nAge groups found: {age_groups}")

    # Generate all pairwise combinations
    age_pairs = list(combinations(age_groups, 2))

    print(f"Running {len(age_pairs)} pairwise comparisons:")
    for pair in age_pairs:
        print(f"  {pair[0]} vs {pair[1]}")

    # Store results for each pairwise comparison
    all_results = {}

    for age1, age2 in age_pairs:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {age1.upper()} vs {age2.upper()}")
        print(f"{'='*60}")

        # Get data for this pair
        data1 = age_data[age1]
        data2 = age_data[age2]

        # Create metadata
        metadata1 = pd.DataFrame(
            {"age_group": [age1] * len(data1), "depth_bin": range(len(data1))}
        )

        metadata2 = pd.DataFrame(
            {"age_group": [age2] * len(data2), "depth_bin": range(len(data2))}
        )

        # Combine this pair
        combined_data = pd.concat([data1, data2], ignore_index=True)
        combined_metadata = pd.concat([metadata1, metadata2], ignore_index=True)

        print(f"Combined data shape: {combined_data.shape}")
        print(f"  {age1}: {len(data1)} depth bins")
        print(f"  {age2}: {len(data2)} depth bins")

        # Balance datasets for this pair
        min_size = min(len(data1), len(data2))
        print(f"Balancing to {min_size} depth bins per age group")

        balanced_indices = []
        np.random.seed(137)  # For reproducibility

        for age_group in [age1, age2]:
            age_mask = combined_metadata["age_group"] == age_group
            age_indices = np.where(age_mask)[0]

            if len(age_indices) > min_size:
                # Randomly subsample
                selected_indices = np.random.choice(
                    age_indices, size=min_size, replace=False
                )
            else:
                # Use all if already at or below min_size
                selected_indices = age_indices

            balanced_indices.extend(selected_indices)

        # Apply balanced sampling
        projection_data_balanced = combined_data.iloc[balanced_indices].reset_index(
            drop=True
        )
        metadata_balanced = combined_metadata.iloc[balanced_indices].reset_index(
            drop=True
        )

        print(f"Final balanced dataset: {len(projection_data_balanced)} depth bins")
        print(f"Target regions: {list(projection_data_balanced.columns)}")

        # Run analysis on this pair
        comparison_name = f"{age1.upper()} vs {age2.upper()}"
        pair_results = analyze_projection_data(
            projection_data_balanced, metadata_balanced, comparison_name, save_dir
        )
        pair_results["metadata"] = pair_results["metadata_clean"]
        pair_results["age_pair"] = (age1, age2)

        # Store results
        pair_name = f"{age1}_vs_{age2}"
        all_results[pair_name] = pair_results

        print(f"\nCompleted analysis for {age1} vs {age2}")
        print(f"Figures generated for: {comparison_name}")
        print(f"{'-'*40}")

        # Add a small delay to separate figure generation
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"ALL PAIRWISE ANALYSES COMPLETED")
    print(f"{'='*60}")
    print(f"Results available for: {list(all_results.keys())}")

    return all_results


# Usage:
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Analyze projection data")
parser.add_argument('--base_output_dir', type=str, default=None,
                   help='Base output directory for processing results (default: REPO_ROOT/02_output)')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/02_projection_analysis)')
args = parser.parse_args()

# Get the repository root (assuming helpers/ is a subdirectory)
REPO_ROOT = Path(__file__).parent.parent.parent
if args.base_output_dir:
    DEFAULT_DATA_DIR = Path(args.base_output_dir)
else:
    DEFAULT_DATA_DIR = REPO_ROOT / "02_output"

if args.helper_output_dir:
    DEFAULT_SAVE_DIR = Path(args.helper_output_dir)
else:
    DEFAULT_SAVE_DIR = Path(__file__).parent.parent / "outputs" / "02_projection_analysis"

# Use 02_output as the base directory, it will search for *_ALL_HAN_filters_Filtered_Matrix.csv files
results = load_and_analyze_directory(
    str(DEFAULT_DATA_DIR), save_dir=str(DEFAULT_SAVE_DIR)
)

# Access individual pairwise results (only if they exist):
if results:
    print("\nAvailable pairwise comparisons:")
    for key in results.keys():
        print(f"  - {key}")
    # Example access (uncomment if needed):
    # if "p12_vs_p20" in results:
    #     p12_vs_p20_results = results["p12_vs_p20"]
    # if "p12_vs_p60" in results:
    #     p12_vs_p60_results = results["p12_vs_p60"]
    # if "p20_vs_p60" in results:
    #     p20_vs_p60_results = results["p20_vs_p60"]
else:
    print("\nNo pairwise comparisons available - no matching data files found.")

# Expected files in directory:
# - p12_ALL_HAN_filters_Filtered_Matrix.csv
# - p20_ALL_HAN_filters_Filtered_Matrix.csv
# - p60_ALL_HAN_filters_Filtered_Matrix.csv
