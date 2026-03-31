#!/usr/bin/env python3
"""
Modular Dataset Comparison Pipeline

This script provides a clean, parameterizable pipeline for comparing any two
standardized datasets with consistent methodology.

Usage:
    python compare_datasets_pipeline.py dataset1.csv dataset2.csv [options]

Features:
- Loads any two standardized CSV datasets
- Applies log-softmax normalization consistently
- Performs comprehensive statistical analysis
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


def load_dataset(filepath):
    """Load a standardized dataset CSV file"""
    try:
        df = pd.read_csv(filepath, index_col=0)
        print(f"Loaded {filepath}: {df.shape[0]} subjects x {df.shape[1]} regions")
        print(f"  Regions: {list(df.columns)}")
        print(f"  Value range: {df.values.min():.2e} to {df.values.max():.2e}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def find_common_regions(df1, df2):
    """Find regions common to both datasets, using custom ordering"""
    regions1 = set(df1.columns)
    regions2 = set(df2.columns)

    # Define custom ordering: POR, LI, LM, AL, RL, A, AM, PM, RSPagl, RSPd, RSPv
    custom_order = [
        "VISpor",  # POR
        "VISli",  # LI
        "VISl",  # LM (L is LM)
        "VISal",  # AL
        "VISrl",  # RL
        "VISa",  # A
        "VISam",  # AM
        "VISpm",  # PM
        "RSPagl",  # RSPagl
        "RSPd",  # RSPd
        "RSPv",  # RSPv
    ]

    # Find common regions using custom order as priority
    common_regions = []
    for region in custom_order:
        if region in regions1 and region in regions2:
            common_regions.append(region)

    # Add any remaining common regions not in custom order
    for region in df1.columns:
        if region in regions2 and region not in common_regions:
            common_regions.append(region)

    print(f"Region overlap analysis:")
    print(f"  Dataset 1 regions: {len(regions1)}")
    print(f"  Dataset 2 regions: {len(regions2)}")
    print(f"  Common regions: {len(common_regions)}")
    print(f"  Common: {common_regions}")

    return common_regions


def compare_datasets(
    df1, df2, dataset1_name, dataset2_name, common_regions, exclude_regions=None
):
    """Compare two datasets using log-softmax normalization"""

    # Always exclude VISpl by default, plus any additional exclusions
    default_exclude = ["VISpl"]
    if exclude_regions:
        exclude_regions = list(set(default_exclude + exclude_regions))
    else:
        exclude_regions = default_exclude

    common_regions = [r for r in common_regions if r not in exclude_regions]
    print(
        f"Excluding regions {exclude_regions}, analyzing {len(common_regions)} regions"
    )

    print(f"\n=== COMPARING {dataset1_name} vs {dataset2_name} ===")
    print(f"Analyzing {len(common_regions)} common regions")
    print(f"Using log-softmax normalization for both datasets")

    # Extract common data
    data1_common = df1[common_regions]
    data2_common = df2[common_regions]

    # STEP 1: Compute averages before normalization
    avg1_raw = data1_common.mean(axis=0)
    avg2_raw = data2_common.mean(axis=0)

    print(f"\nRaw averages before normalization:")
    print(f"{dataset1_name} mean: {np.mean(avg1_raw):.2e}, std: {np.std(avg1_raw):.2e}")
    print(f"{dataset2_name} mean: {np.mean(avg2_raw):.2e}, std: {np.std(avg2_raw):.2e}")

    # STEP 2: Apply log-softmax normalization
    avg1_norm = log_transform_and_softmax(avg1_raw)
    avg2_norm = log_transform_and_softmax(avg2_raw)

    # STEP 3: Compute error bars from individually normalized data
    data1_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in data1_common.iterrows()]
    )
    data2_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in data2_common.iterrows()]
    )

    sem1 = stats.sem(data1_individual_norm, axis=0)
    sem2 = stats.sem(data2_individual_norm, axis=0)

    print(f"{dataset1_name} SEM range: {np.min(sem1):.2e} to {np.max(sem1):.2e}")
    print(f"{dataset2_name} SEM range: {np.min(sem2):.2e} to {np.max(sem2):.2e}")

    # STEP 4: Statistical testing
    results = {
        "region": [],
        "dataset1_mean": [],
        "dataset2_mean": [],
        "difference": [],
        "p_value_ttest": [],
        "p_value_welch": [],
        "p_value_ks": [],
        "cohens_d": [],
        "significance_ttest": [],
        "significance_welch": [],
        "significance_ks": [],
    }

    for i, region in enumerate(common_regions):
        vals1 = data1_individual_norm[:, i]
        vals2 = data2_individual_norm[:, i]

        # Statistical tests
        _, p_ttest = stats.ttest_ind(vals1, vals2)
        _, p_welch = stats.ttest_ind(vals1, vals2, equal_var=False)
        _, p_ks = stats.ks_2samp(vals1, vals2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(vals1) - 1) * np.var(vals1) + (len(vals2) - 1) * np.var(vals2))
            / (len(vals1) + len(vals2) - 2)
        )
        cohens_d = (
            (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0
        )

        # Store results
        results["region"].append(region)
        results["dataset1_mean"].append(avg1_norm[i])
        results["dataset2_mean"].append(avg2_norm[i])
        results["difference"].append(avg1_norm[i] - avg2_norm[i])
        results["p_value_ttest"].append(float(p_ttest))
        results["p_value_welch"].append(float(p_welch))
        results["p_value_ks"].append(float(p_ks))
        results["cohens_d"].append(float(cohens_d))

        # Determine significance levels
        for test_name, p_val in [("ttest", p_ttest), ("welch", p_welch), ("ks", p_ks)]:
            if float(p_val) < 0.001:
                sig = "***"
            elif float(p_val) < 0.01:
                sig = "**"
            elif float(p_val) < 0.05:
                sig = "*"
            else:
                sig = "ns"
            results[f"significance_{test_name}"].append(sig)

    # Jensen-Shannon Divergence
    js_divergence = jensenshannon(avg1_norm, avg2_norm)
    print(f"\nJensen-Shannon Divergence: {js_divergence:.6f}")

    return {
        "common_regions": common_regions,
        "dataset1_avg_norm": avg1_norm,
        "dataset2_avg_norm": avg2_norm,
        "dataset1_sem": sem1,
        "dataset2_sem": sem2,
        "results": results,
        "js_divergence": js_divergence,
        "dataset1_name": dataset1_name,
        "dataset2_name": dataset2_name,
    }


def create_comparison_plot(comparison_results, output_prefix):
    """Create publication-ready comparison plot"""

    # Use the already correctly ordered regions from reverse Sheet8 structure
    regions = comparison_results["common_regions"]
    dataset1_avg_norm = comparison_results["dataset1_avg_norm"]
    dataset2_avg_norm = comparison_results["dataset2_avg_norm"]
    dataset1_sem = comparison_results["dataset1_sem"]
    dataset2_sem = comparison_results["dataset2_sem"]

    dataset1_name = comparison_results["dataset1_name"]
    dataset2_name = comparison_results["dataset2_name"]
    n_regions = len(regions)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    x_pos = np.arange(n_regions)
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x_pos - width / 2,
        dataset1_avg_norm,
        width,
        yerr=dataset1_sem,
        label=dataset1_name,
        color="steelblue",
        alpha=0.8,
        capsize=3,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        dataset2_avg_norm,
        width,
        yerr=dataset2_sem,
        label=dataset2_name,
        color="lightcoral",
        alpha=0.8,
        capsize=3,
    )

    ax.set_xlabel("Brain Regions")
    ax.set_ylabel("Normalized Density (Log-Softmax)")
    ax.set_title(
        f'{dataset1_name} vs {dataset2_name}\nJS Divergence: {comparison_results["js_divergence"]:.4f}'
    )
    ax.set_xticks(x_pos)

    # Convert VIS areas to uppercase without VIS prefix, keep RSP areas unchanged
    region_labels = []
    for region in regions:
        if region.startswith("VIS"):
            # Special case: VISl should be labeled as LM
            if region == "VISl":
                region_labels.append("LM")
            else:
                # Remove VIS prefix and make uppercase
                region_labels.append(region[3:].upper())
        else:
            # Keep RSP areas as-is
            region_labels.append(region)

    ax.set_xticklabels(region_labels)
    ax.legend()
    # ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)  # Fixed y-axis from 0 to 1

    plt.tight_layout()

    # Save plot
    plot_file = f"{output_prefix}_comparison.svg"
    plt.savefig(plot_file, format="svg", dpi=300, bbox_inches="tight")
    print(f"Plot saved: {plot_file}")

    return plot_file


def save_detailed_results(comparison_results, output_prefix):
    """Save comprehensive results to text file"""

    results_file = f"{output_prefix}_results.txt"

    with open(results_file, "w") as f:
        f.write("DATASET COMPARISON ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASETS COMPARED:\n")
        f.write(f"Dataset 1: {comparison_results['dataset1_name']}\n")
        f.write(f"Dataset 2: {comparison_results['dataset2_name']}\n")
        f.write(
            f"Common regions analyzed: {len(comparison_results['common_regions'])}\n"
        )
        f.write(f"Normalization method: Log-Softmax\n\n")

        f.write("OVERALL SIMILARITY:\n")
        f.write(
            f"Jensen-Shannon Divergence: {comparison_results['js_divergence']:.6f}\n"
        )
        f.write(
            "(Lower values = more similar, 0 = identical, 1 = maximally different)\n\n"
        )

        f.write("DETAILED REGIONAL ANALYSIS:\n")
        f.write("-" * 120 + "\n")
        f.write(
            "Region     Dataset1     Dataset2     Difference   P-Value(t)   P-Value(W)   P-Value(KS)  Cohen's d    Sig(t/W/KS)\n"
        )
        f.write("-" * 120 + "\n")

        # Use correct Sheet8 reverse ordering for display
        regions_ordered = comparison_results["common_regions"]
        for display_idx, region in enumerate(regions_ordered):
            # Find the original index for this region
            original_idx = comparison_results["common_regions"].index(region)
            res = comparison_results["results"]
            f.write(
                f"{region:<10} {res['dataset1_mean'][original_idx]:<12.6f} {res['dataset2_mean'][original_idx]:<12.6f} "
                f"{res['difference'][original_idx]:<12.6f} {res['p_value_ttest'][original_idx]:<12.6f} "
                f"{res['p_value_welch'][original_idx]:<12.6f} {res['p_value_ks'][original_idx]:<12.6f} "
                f"{res['cohens_d'][original_idx]:<12.2f} {res['significance_ttest'][original_idx]}/{res['significance_welch'][original_idx]}/{res['significance_ks'][original_idx]}\n"
            )

        f.write(
            "\nSIGNIFICANCE CODES: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n"
        )
        f.write(
            "TESTS: t = Student's t-test, W = Welch's t-test, KS = Kolmogorov-Smirnov\n"
        )

    print(f"Detailed results saved: {results_file}")
    return results_file


def save_csv_results(comparison_results, output_prefix):
    """Save results as CSV for further analysis"""

    csv_file = f"{output_prefix}_results.csv"
    results_df = pd.DataFrame(comparison_results["results"])
    results_df.to_csv(csv_file, index=False)
    print(f"CSV results saved: {csv_file}")
    return csv_file


def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description="Compare two standardized datasets with comprehensive analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_datasets_pipeline.py standardized_datasets/abi_projection-density_standardized.csv standardized_datasets/vsv_original_standardized.csv
  python compare_datasets_pipeline.py standardized_datasets/vsv_original_standardized.csv standardized_datasets/vsv_new_standardized.csv --exclude VISpl VISP
  python compare_datasets_pipeline.py dataset1.csv dataset2.csv --output my_comparison
        """,
    )

    parser.add_argument("dataset1", help="Path to first dataset CSV file")
    parser.add_argument("dataset2", help="Path to second dataset CSV file")
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="Regions to exclude from analysis"
    )
    parser.add_argument(
        "--output",
        default="comparison",
        help="Output file prefix (default: comparison)",
    )

    args = parser.parse_args()

    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / "compare_datasets_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / args.output)

    print("MODULAR DATASET COMPARISON PIPELINE")
    print("=" * 60)
    print(f"Dataset 1: {args.dataset1}")
    print(f"Dataset 2: {args.dataset2}")
    if args.exclude:
        print(f"Excluding regions: {args.exclude}")
    print(f"Output prefix: {output_prefix}")

    # Load datasets
    df1 = load_dataset(args.dataset1)
    df2 = load_dataset(args.dataset2)

    # Extract dataset names from file paths
    dataset1_name = Path(args.dataset1).stem.replace("_standardized", "")
    dataset2_name = Path(args.dataset2).stem.replace("_standardized", "")

    # Find common regions
    common_regions = find_common_regions(df1, df2)

    # Perform comparison
    comparison_results = compare_datasets(
        df1, df2, dataset1_name, dataset2_name, common_regions, args.exclude
    )

    # Generate outputs
    plot_file = create_comparison_plot(comparison_results, output_prefix)
    results_file = save_detailed_results(comparison_results, output_prefix)
    csv_file = save_csv_results(comparison_results, output_prefix)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Jensen-Shannon Divergence: {comparison_results['js_divergence']:.6f}")
    print(f"Files generated:")
    print(f"  - Plot: {plot_file}")
    print(f"  - Results: {results_file}")
    print(f"  - CSV: {csv_file}")


if __name__ == "__main__":
    main()
