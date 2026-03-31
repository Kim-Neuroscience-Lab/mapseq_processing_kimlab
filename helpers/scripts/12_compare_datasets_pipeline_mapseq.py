#!/usr/bin/env python3
"""
Enhanced Modular Dataset Comparison Pipeline with MapSeq Integration

This script provides a comprehensive pipeline for comparing Allen, VSV, and MapSeq
datasets with area combining and consistent methodology.

Usage:
    python compare_datasets_pipeline_mapseq.py --allen <allen_file> --vsv <vsv_file> --mapseq <mapseq_file> [options]

Features:
- Loads Allen, VSV, and MapSeq datasets
- Supports area combining (e.g., LM+Li, RSPagl+RSPd)
- Three-way statistical comparisons
- Applies log-softmax normalization consistently
- Generates publication-ready visualizations
- Saves detailed results and summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
from scipy import stats
from pathlib import Path
import argparse
import sys
import itertools

# Set matplotlib parameters for publication-ready figures
plt.rcParams["font.family"] = ['Helvetica', 'Arial']
plt.rcParams["svg.fonttype"] = "none"  # Make text in SVG editable
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14


def log_transform_and_softmax(values, epsilon=1e-12):
    """Apply log transformation followed by softmax normalization"""
    # Add epsilon to handle zeros
    log_values = np.log(np.array(values) + epsilon)
    return softmax(log_values)


def load_dataset(filepath, dataset_type="standard"):
    """Load a dataset CSV file with appropriate handling for different types"""
    try:
        if dataset_type == "mapseq":
            # MapSeq has Sample column instead of index
            df = pd.read_csv(filepath, index_col=0)
        else:
            # Standard datasets (Allen, VSV) have first column as index
            df = pd.read_csv(filepath, index_col=0)

        print(f"Loaded {filepath}: {df.shape[0]} subjects x {df.shape[1]} regions")
        print(f"  Regions: {list(df.columns)}")
        print(f"  Value range: {df.values.min():.2e} to {df.values.max():.2e}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def preprocess_mapseq_data(df_mapseq):
    """
    Preprocess MapSeq data to create combined regions and standardize naming

    MapSeq columns: UMISum_RSP, UMISum_PM, UMISum_AM, UMISum_AL, UMISum_LM
    Target regions: RSPagl+RSPd, PM, AM, AL, LM+Li
    """
    print("\nPreprocessing MapSeq data...")

    # Create a new dataframe with standardized region names
    processed_df = pd.DataFrame(index=df_mapseq.index)

    # Map UMI columns to target regions
    region_mapping = {
        "UMISum_RSP": "RSPagl+RSPd",
        "UMISum_PM": "PM",
        "UMISum_AM": "AM",
        "UMISum_AL": "AL",
        "UMISum_LM": "LM+Li",
    }

    for umi_col, target_region in region_mapping.items():
        if umi_col in df_mapseq.columns:
            processed_df[target_region] = df_mapseq[umi_col]
        else:
            print(f"Warning: {umi_col} not found in MapSeq data")
            processed_df[target_region] = 0

    print(f"MapSeq processed regions: {list(processed_df.columns)}")
    return processed_df


def create_combined_regions_with_mapping(df, target_regions, dataset_type="allen_vsv"):
    """
    Create combined regions for datasets using proper region mappings

    Args:
        df: DataFrame with individual regions
        target_regions: List of target regions to create
        dataset_type: Type of dataset ("allen_vsv" or "mapseq")

    Returns:
        DataFrame with mapped and combined regions
    """
    print(f"\nCreating combined regions for {dataset_type} dataset: {target_regions}")

    # Define region mappings based on dataset type
    if dataset_type == "allen_vsv":
        # Mapping from target regions to Allen/VSV regions
        region_mapping = {
            "LM+Li": ["VISl", "VISli"],  # Lateral visual areas
            "AL": (
                ["VISal"] if "VISal" in df.columns else []
            ),  # Anterolateral (may not exist)
            "AM": ["VISam"],  # Anteromedial visual area
            "PM": ["VISpm"],  # Posteromedial visual area
            "RSPagl+RSPd": ["RSPagl", "RSPd"],  # Retrosplenial areas
        }
    else:  # mapseq - already processed in preprocess_mapseq_data
        # Direct mapping for MapSeq (already done in preprocessing)
        region_mapping = {region: [region] for region in target_regions}

    new_df = pd.DataFrame(index=df.index)

    for target_region in target_regions:
        if target_region in region_mapping:
            source_regions = region_mapping[target_region]
            available_regions = [r for r in source_regions if r in df.columns]

            if available_regions:
                if len(available_regions) == 1:
                    new_df[target_region] = df[available_regions[0]]
                    print(f"  {target_region} <- {available_regions[0]}")
                else:
                    new_df[target_region] = df[available_regions].sum(axis=1)
                    print(f"  {target_region} <- {' + '.join(available_regions)}")
            else:
                print(
                    f"  {target_region}: No matching regions found in dataset, setting to 0"
                )
                new_df[target_region] = 0
        else:
            print(f"  {target_region}: No mapping defined, setting to 0")
            new_df[target_region] = 0

    return new_df


def find_common_regions_three_way(df_allen, df_vsv, df_mapseq):
    """Find regions common to all three datasets"""
    regions_allen = set(df_allen.columns)
    regions_vsv = set(df_vsv.columns)
    regions_mapseq = set(df_mapseq.columns)

    # Find intersection of all three
    common_regions = list(regions_allen & regions_vsv & regions_mapseq)

    print(f"\nThree-way region overlap analysis:")
    print(f"  Allen regions: {len(regions_allen)} - {list(regions_allen)}")
    print(f"  VSV regions: {len(regions_vsv)} - {list(regions_vsv)}")
    print(f"  MapSeq regions: {len(regions_mapseq)} - {list(regions_mapseq)}")
    print(f"  Common regions: {len(common_regions)} - {common_regions}")

    if len(common_regions) == 0:
        print("ERROR: No common regions found across all three datasets!")
        sys.exit(1)

    return common_regions


def compare_three_datasets(
    df_allen, df_vsv, df_mapseq, common_regions, region_order=None
):
    """Compare three datasets using log-softmax normalization"""

    if region_order:
        # Filter and reorder regions according to specified order
        common_regions = [r for r in region_order if r in common_regions]
        print(f"Using specified region order: {common_regions}")

    print(f"\n=== THREE-WAY COMPARISON: Allen vs VSV vs MapSeq ===")
    print(f"Analyzing {len(common_regions)} common regions")
    print(f"Using log-softmax normalization for all datasets")

    # Extract common data
    data_allen = df_allen[common_regions]
    data_vsv = df_vsv[common_regions]
    data_mapseq = df_mapseq[common_regions]

    # STEP 1: Compute averages before normalization
    avg_allen_raw = data_allen.mean(axis=0)
    avg_vsv_raw = data_vsv.mean(axis=0)
    avg_mapseq_raw = data_mapseq.mean(axis=0)

    print(f"\nRaw averages before normalization:")
    print(f"Allen mean: {np.mean(avg_allen_raw):.2e}, std: {np.std(avg_allen_raw):.2e}")
    print(f"VSV mean: {np.mean(avg_vsv_raw):.2e}, std: {np.std(avg_vsv_raw):.2e}")
    print(
        f"MapSeq mean: {np.mean(avg_mapseq_raw):.2e}, std: {np.std(avg_mapseq_raw):.2e}"
    )

    # STEP 2: Apply log-softmax normalization
    avg_allen_norm = log_transform_and_softmax(avg_allen_raw)
    avg_vsv_norm = log_transform_and_softmax(avg_vsv_raw)
    avg_mapseq_norm = log_transform_and_softmax(avg_mapseq_raw)

    # STEP 3: Compute error bars from individually normalized data
    allen_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in data_allen.iterrows()]
    )
    vsv_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in data_vsv.iterrows()]
    )
    mapseq_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in data_mapseq.iterrows()]
    )

    sem_allen = stats.sem(allen_individual_norm, axis=0)
    sem_vsv = stats.sem(vsv_individual_norm, axis=0)
    sem_mapseq = stats.sem(mapseq_individual_norm, axis=0)

    print(f"Allen SEM range: {np.min(sem_allen):.2e} to {np.max(sem_allen):.2e}")
    print(f"VSV SEM range: {np.min(sem_vsv):.2e} to {np.max(sem_vsv):.2e}")
    print(f"MapSeq SEM range: {np.min(sem_mapseq):.2e} to {np.max(sem_mapseq):.2e}")

    # STEP 4: Statistical testing (pairwise comparisons)
    results = {
        "region": [],
        "allen_mean": [],
        "vsv_mean": [],
        "mapseq_mean": [],
        "allen_vs_vsv_p": [],
        "allen_vs_mapseq_p": [],
        "vsv_vs_mapseq_p": [],
        "allen_vs_vsv_cohens_d": [],
        "allen_vs_mapseq_cohens_d": [],
        "vsv_vs_mapseq_cohens_d": [],
    }

    def compute_cohens_d(vals1, vals2):
        """Compute Cohen's d effect size"""
        pooled_std = np.sqrt(
            ((len(vals1) - 1) * np.var(vals1) + (len(vals2) - 1) * np.var(vals2))
            / (len(vals1) + len(vals2) - 2)
        )
        return (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0

    for i, region in enumerate(common_regions):
        vals_allen = np.asarray(allen_individual_norm[:, i])
        vals_vsv = np.asarray(vsv_individual_norm[:, i])
        vals_mapseq = np.asarray(mapseq_individual_norm[:, i])

        # Pairwise statistical tests
        _, p_allen_vsv = stats.ttest_ind(vals_allen, vals_vsv, equal_var=False)
        _, p_allen_mapseq = stats.ttest_ind(vals_allen, vals_mapseq, equal_var=False)
        _, p_vsv_mapseq = stats.ttest_ind(vals_vsv, vals_mapseq, equal_var=False)

        # Effect sizes
        d_allen_vsv = compute_cohens_d(vals_allen, vals_vsv)
        d_allen_mapseq = compute_cohens_d(vals_allen, vals_mapseq)
        d_vsv_mapseq = compute_cohens_d(vals_vsv, vals_mapseq)

        # Store results
        results["region"].append(region)
        results["allen_mean"].append(avg_allen_norm[i])
        results["vsv_mean"].append(avg_vsv_norm[i])
        results["mapseq_mean"].append(avg_mapseq_norm[i])
        results["allen_vs_vsv_p"].append(float(p_allen_vsv))
        results["allen_vs_mapseq_p"].append(float(p_allen_mapseq))
        results["vsv_vs_mapseq_p"].append(float(p_vsv_mapseq))
        results["allen_vs_vsv_cohens_d"].append(float(d_allen_vsv))
        results["allen_vs_mapseq_cohens_d"].append(float(d_allen_mapseq))
        results["vsv_vs_mapseq_cohens_d"].append(float(d_vsv_mapseq))

    # Jensen-Shannon Divergences
    js_allen_vsv = jensenshannon(avg_allen_norm, avg_vsv_norm)
    js_allen_mapseq = jensenshannon(avg_allen_norm, avg_mapseq_norm)
    js_vsv_mapseq = jensenshannon(avg_vsv_norm, avg_mapseq_norm)

    print(f"\nJensen-Shannon Divergences:")
    print(f"  Allen vs VSV: {js_allen_vsv:.6f}")
    print(f"  Allen vs MapSeq: {js_allen_mapseq:.6f}")
    print(f"  VSV vs MapSeq: {js_vsv_mapseq:.6f}")

    return {
        "common_regions": common_regions,
        "allen_avg_norm": avg_allen_norm,
        "vsv_avg_norm": avg_vsv_norm,
        "mapseq_avg_norm": avg_mapseq_norm,
        "allen_sem": sem_allen,
        "vsv_sem": sem_vsv,
        "mapseq_sem": sem_mapseq,
        "results": results,
        "js_divergences": {
            "allen_vs_vsv": js_allen_vsv,
            "allen_vs_mapseq": js_allen_mapseq,
            "vsv_vs_mapseq": js_vsv_mapseq,
        },
    }


def create_three_way_plot(comparison_results, output_prefix):
    """Create publication-ready three-way comparison plot"""

    # Define simplified labels for display
    region_label_mapping = {
        "RSPagl+RSPd": "RSP",
        "PM": "PM",
        "AM": "AM",
        "AL": "AL",
        "LM+Li": "LM+LI",
    }

    # Reverse the order of regions for display (consistent with original)
    regions = comparison_results["common_regions"][::-1]
    # Create simplified labels for plotting
    region_labels = [
        region_label_mapping.get(region, region)
        for region in regions
        if region is not None
    ]
    allen_avg = comparison_results["allen_avg_norm"][::-1]
    vsv_avg = comparison_results["vsv_avg_norm"][::-1]
    mapseq_avg = comparison_results["mapseq_avg_norm"][::-1]
    allen_sem = comparison_results["allen_sem"][::-1]
    vsv_sem = comparison_results["vsv_sem"][::-1]
    mapseq_sem = comparison_results["mapseq_sem"][::-1]

    n_regions = len(regions)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    x_pos = np.arange(n_regions)
    width = 0.25

    # Create bars for three datasets
    bars_allen = ax.bar(
        x_pos - width,
        allen_avg,
        width,
        yerr=allen_sem,
        label="Allen Brain Institute",
        color="steelblue",
        alpha=0.8,
        capsize=3,
    )
    bars_vsv = ax.bar(
        x_pos,
        vsv_avg,
        width,
        yerr=vsv_sem,
        label="VSV",
        color="lightcoral",
        alpha=0.8,
        capsize=3,
    )
    bars_mapseq = ax.bar(
        x_pos + width,
        mapseq_avg,
        width,
        yerr=mapseq_sem,
        label="MapSeq",
        color="lightgreen",
        alpha=0.8,
        capsize=3,
    )

    ax.set_xlabel("Brain Regions")
    ax.set_ylabel("Normalized Density (Log-Softmax)")

    js_info = comparison_results["js_divergences"]
    title = (
        f"Allen vs VSV vs MapSeq Comparison\n"
        f'JS Divergences: Allen-VSV: {js_info["allen_vs_vsv"]:.4f}, '
        f'Allen-MapSeq: {js_info["allen_vs_mapseq"]:.4f}, '
        f'VSV-MapSeq: {js_info["vsv_vs_mapseq"]:.4f}'
    )
    ax.set_title(title)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_labels)
    ax.legend()
    # ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # Save plot
    plot_file = f"{output_prefix}_three_way_comparison.svg"
    plt.savefig(plot_file, format="svg", dpi=300, bbox_inches="tight")
    print(f"Three-way plot saved: {plot_file}")

    return plot_file


def save_three_way_results(comparison_results, output_prefix):
    """Save comprehensive three-way results to text file"""

    results_file = f"{output_prefix}_three_way_results.txt"

    with open(results_file, "w") as f:
        f.write("THREE-WAY DATASET COMPARISON ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASETS COMPARED:\n")
        f.write("Allen Brain Institute, VSV, MapSeq\n")
        f.write(
            f"Common regions analyzed: {len(comparison_results['common_regions'])}\n"
        )
        f.write(f"Normalization method: Log-Softmax\n\n")

        f.write("OVERALL SIMILARITIES (Jensen-Shannon Divergences):\n")
        js = comparison_results["js_divergences"]
        f.write(f"Allen vs VSV: {js['allen_vs_vsv']:.6f}\n")
        f.write(f"Allen vs MapSeq: {js['allen_vs_mapseq']:.6f}\n")
        f.write(f"VSV vs MapSeq: {js['vsv_vs_mapseq']:.6f}\n")
        f.write(
            "(Lower values = more similar, 0 = identical, 1 = maximally different)\n\n"
        )

        f.write("DETAILED REGIONAL ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(
            "Region      Allen (norm)  VSV (norm)    MapSeq (norm) Allen SEM    VSV SEM      MapSeq SEM\n"
        )
        f.write("-" * 80 + "\n")

        # Use specified region order for display
        regions = comparison_results["common_regions"]
        allen_norm = comparison_results["allen_avg_norm"]
        vsv_norm = comparison_results["vsv_avg_norm"]
        mapseq_norm = comparison_results["mapseq_avg_norm"]
        allen_sem = comparison_results["allen_sem"]
        vsv_sem = comparison_results["vsv_sem"]
        mapseq_sem = comparison_results["mapseq_sem"]

        for i, region in enumerate(regions):
            f.write(
                f"{region:<11} {allen_norm[i]:<12.6f} {vsv_norm[i]:<12.6f} "
                f"{mapseq_norm[i]:<12.6f} {allen_sem[i]:<12.6f} "
                f"{vsv_sem[i]:<12.6f} {mapseq_sem[i]:<12.6f}\n"
            )

        f.write("\nNOTE: Values are normalized using log-softmax transformation\n")
        f.write("SEM = Standard Error of the Mean\n")

    print(f"Three-way detailed results saved: {results_file}")
    return results_file


def save_three_way_csv(comparison_results, output_prefix):
    """Save three-way results as CSV for further analysis"""

    csv_file = f"{output_prefix}_three_way_results.csv"
    results_df = pd.DataFrame(comparison_results["results"])
    results_df.to_csv(csv_file, index=False)
    print(f"Three-way CSV results saved: {csv_file}")
    return csv_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare Allen, VSV, and MapSeq datasets with area combining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_datasets_pipeline_mapseq.py --allen standardized_datasets/abi_projection-density_standardized.csv --vsv standardized_datasets/vsv_density_standardized.csv --mapseq "Summary_data_07012025 - mapseq_data.csv"
  python compare_datasets_pipeline_mapseq.py --allen allen.csv --vsv vsv.csv --mapseq mapseq.csv --output my_three_way_comparison
        """,
    )

    parser.add_argument("--allen", required=True, help="Path to Allen dataset CSV file")
    parser.add_argument("--vsv", required=True, help="Path to VSV dataset CSV file")
    parser.add_argument(
        "--mapseq", required=True, help="Path to MapSeq dataset CSV file"
    )
    parser.add_argument(
        "--output",
        default="three_way_comparison",
        help="Output file prefix (default: three_way_comparison)",
    )

    return parser.parse_args()


def perform_three_way_comparison(allen_data, vsv_data, mapseq_data, common_regions):
    """Perform three-way statistical comparison"""

    print(f"\n=== THREE-WAY COMPARISON: Allen vs VSV vs MapSeq ===")
    print(f"Analyzing {len(common_regions)} common regions")
    print(f"Using log-softmax normalization for all datasets")

    # Filter to common regions
    allen_common = allen_data[common_regions]
    vsv_common = vsv_data[common_regions]
    mapseq_common = mapseq_data[common_regions]

    # Calculate raw averages and ranges
    allen_avg_raw = allen_common.mean()
    vsv_avg_raw = vsv_common.mean()
    mapseq_avg_raw = mapseq_common.mean()

    print(f"\nRaw averages before normalization:")
    print(f"Allen mean: {allen_avg_raw.mean():.2e}, std: {allen_avg_raw.std():.2e}")
    print(f"VSV mean: {vsv_avg_raw.mean():.2e}, std: {vsv_avg_raw.std():.2e}")
    print(f"MapSeq mean: {mapseq_avg_raw.mean():.2e}, std: {mapseq_avg_raw.std():.2e}")

    # Apply log-softmax normalization to averages
    allen_avg_norm = log_transform_and_softmax(allen_avg_raw.values)
    vsv_avg_norm = log_transform_and_softmax(vsv_avg_raw.values)
    mapseq_avg_norm = log_transform_and_softmax(mapseq_avg_raw.values)

    # Calculate normalized individual samples for error bars
    allen_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in allen_common.iterrows()]
    )
    vsv_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in vsv_common.iterrows()]
    )
    mapseq_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in mapseq_common.iterrows()]
    )

    # Calculate SEM from normalized data
    allen_sem = np.std(allen_individual_norm, axis=0) / np.sqrt(
        len(allen_individual_norm)
    )
    vsv_sem = np.std(vsv_individual_norm, axis=0) / np.sqrt(len(vsv_individual_norm))
    mapseq_sem = np.std(mapseq_individual_norm, axis=0) / np.sqrt(
        len(mapseq_individual_norm)
    )

    print(f"Allen SEM range: {allen_sem.min():.2e} to {allen_sem.max():.2e}")
    print(f"VSV SEM range: {vsv_sem.min():.2e} to {vsv_sem.max():.2e}")
    print(f"MapSeq SEM range: {mapseq_sem.min():.2e} to {mapseq_sem.max():.2e}")

    # Calculate Jensen-Shannon divergences
    js_allen_vs_vsv = jensenshannon(allen_avg_norm, vsv_avg_norm)
    js_allen_vs_mapseq = jensenshannon(allen_avg_norm, mapseq_avg_norm)
    js_vsv_vs_mapseq = jensenshannon(vsv_avg_norm, mapseq_avg_norm)

    print(f"\nJensen-Shannon Divergences:")
    print(f"  Allen vs VSV: {js_allen_vs_vsv:.6f}")
    print(f"  Allen vs MapSeq: {js_allen_vs_mapseq:.6f}")
    print(f"  VSV vs MapSeq: {js_vsv_vs_mapseq:.6f}")

    return {
        "common_regions": common_regions,
        "allen_avg_norm": allen_avg_norm,
        "vsv_avg_norm": vsv_avg_norm,
        "mapseq_avg_norm": mapseq_avg_norm,
        "allen_sem": allen_sem,
        "vsv_sem": vsv_sem,
        "mapseq_sem": mapseq_sem,
        "js_divergences": {
            "allen_vs_vsv": js_allen_vs_vsv,
            "allen_vs_mapseq": js_allen_vs_mapseq,
            "vsv_vs_mapseq": js_vsv_vs_mapseq,
        },
    }


def main():
    """Main comparison function"""
    args = parse_arguments()

    # Define target regions (custom order: RSPagl+RSPd, PM, AM, AL, LM+Li)
    # This follows the requested order: POR, LI, LM, AL, RL, A, AM, PM, RSPagl, RSPd, RSPv
    target_regions = ["RSPagl+RSPd", "PM", "AM", "AL", "LM+Li"]

    print("ENHANCED DATASET COMPARISON PIPELINE WITH MAPSEQ")
    print("=" * 70)
    print(f"Allen dataset: {args.allen}")
    print(f"VSV dataset: {args.vsv}")
    print(f"MapSeq dataset: {args.mapseq}")
    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / "compare_datasets_pipeline_mapseq"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / args.output)

    print(f"Output prefix: {output_prefix}")

    # Load datasets
    allen_data = load_dataset(args.allen, "Allen")
    vsv_data = load_dataset(args.vsv, "VSV")
    mapseq_data = load_dataset(args.mapseq, "MapSeq")

    # Preprocess MapSeq data
    print("\nPreprocessing MapSeq data...")
    mapseq_processed = preprocess_mapseq_data(mapseq_data.copy())
    print(f"MapSeq processed regions: {list(mapseq_processed.columns)}")

    # Create combined regions for Allen and VSV datasets
    print(f"\nCreating combined regions for allen_vsv dataset: {target_regions}")
    allen_combined = create_combined_regions_with_mapping(
        allen_data, target_regions, "allen_vsv"
    )
    print(f"\nCreating combined regions for allen_vsv dataset: {target_regions}")
    vsv_combined = create_combined_regions_with_mapping(
        vsv_data, target_regions, "allen_vsv"
    )

    # Find common regions across all three datasets
    allen_regions = set(allen_combined.columns)
    vsv_regions = set(vsv_combined.columns)
    mapseq_regions = set(mapseq_processed.columns)

    print(f"\nThree-way region overlap analysis:")
    print(f"  Allen regions: {len(allen_regions)} - {sorted(allen_regions)}")
    print(f"  VSV regions: {len(vsv_regions)} - {sorted(vsv_regions)}")
    print(f"  MapSeq regions: {len(mapseq_regions)} - {sorted(mapseq_regions)}")

    # Find common regions
    common_regions = allen_regions & vsv_regions & mapseq_regions
    common_regions = list(common_regions)
    print(f"  Common regions: {len(common_regions)} - {sorted(common_regions)}")

    # Order regions according to target_regions order
    ordered_common_regions = [r for r in target_regions if r in common_regions]
    print(f"Using specified region order: {ordered_common_regions}")

    # Filter to common regions
    allen_final = allen_combined[ordered_common_regions]
    vsv_final = vsv_combined[ordered_common_regions]
    mapseq_final = mapseq_processed[ordered_common_regions]

    # Perform three-way comparison
    three_way_results = perform_three_way_comparison(
        allen_final, vsv_final, mapseq_final, ordered_common_regions
    )

    # Create plots and save results
    create_three_way_plot(three_way_results, output_prefix)
    save_three_way_results(three_way_results, output_prefix)


if __name__ == "__main__":
    main()
