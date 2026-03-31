#!/usr/bin/env python3
"""
Enhanced Two-Way Dataset Comparison: VSV vs MapSeq
Compares VSV raw pixels and MapSeq datasets with area combining and log-softmax normalization.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
from scipy import stats

# Set font to Helvetica with Arial fallback, and ensure SVG text is editable
plt.rcParams["font.family"] = ['Helvetica', 'Arial']
plt.rcParams["svg.fonttype"] = "none"


def log_transform_and_softmax(values, epsilon=1e-12):
    """Apply log transformation followed by softmax normalization"""
    # Add small epsilon to avoid log(0)
    log_values = np.log(np.array(values) + epsilon)
    # Apply softmax to get probability distribution
    return softmax(log_values)


def load_dataset(file_path, dataset_type):
    """Load and validate dataset"""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    try:
        if dataset_type.lower() == "mapseq":
            # MapSeq files have different structure
            df = pd.read_csv(file_path, index_col=0)
        else:
            # Standard format with first column as index
            df = pd.read_csv(file_path, index_col=0)

        # Fill NaN values with 0
        df = df.fillna(0)

        print(f"Loaded {file_path}: {len(df)} subjects x {len(df.columns)} regions")
        print(f"  Regions: {list(df.columns)}")
        print(f"  Value range: {df.values.min():.2e} to {df.values.max():.2e}")

        return df
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        sys.exit(1)


def preprocess_mapseq_data(df):
    """Preprocess MapSeq data: rename UMI columns to combined region names"""
    print("Preprocessing MapSeq data...")

    # Mapping from MapSeq UMI columns to combined region names
    mapseq_mapping = {
        "UMISum_RSP": "RSPagl+RSPd",
        "UMISum_PM": "PM",
        "UMISum_AM": "AM",
        "UMISum_AL": "AL",
        "UMISum_LM": "LM+Li",
    }

    # Create new dataframe with mapped column names
    df_processed = df.rename(columns=mapseq_mapping)

    # Keep only the renamed columns
    available_regions = [
        region for region in mapseq_mapping.values() if region in df_processed.columns
    ]
    df_final = df_processed[available_regions]

    print(f"MapSeq processed regions: {list(df_final.columns)}")
    return df_final


def create_combined_regions_with_mapping(df, target_regions, dataset_type):
    """Create combined regions from individual regions using mapping"""

    # Define region mapping for allen_vsv datasets
    region_mapping = {
        "LM+Li": ["VISl", "VISli"],  # Lateral medial + Lateral intermediate
        "AL": ["VISal"],  # Anterolateral
        "AM": ["VISam"],  # Anteromedial
        "PM": ["VISpm"],  # Posteromedial
        "RSPagl+RSPd": [
            "RSPagl",
            "RSPd",
        ],  # Retrosplenial agranular + Retrosplenial dorsal
    }

    if dataset_type == "allen_vsv":
        combined_data = {}

        print(f"Creating combined regions for {dataset_type} dataset: {target_regions}")
        for target_region in target_regions:
            if target_region in region_mapping:
                source_regions = region_mapping[target_region]
                available_sources = [r for r in source_regions if r in df.columns]

                if available_sources:
                    print(f"  {target_region} <- {' + '.join(available_sources)}")
                    # Sum the values from source regions
                    combined_data[target_region] = df[available_sources].sum(axis=1)
                else:
                    print(f"  WARNING: No source regions found for {target_region}")
            else:
                print(f"  WARNING: No mapping defined for {target_region}")

        if combined_data:
            return pd.DataFrame(combined_data)
        else:
            print("ERROR: No combined regions could be created")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown dataset type: {dataset_type}")
        sys.exit(1)


def perform_two_way_comparison(vsv_data, mapseq_data, common_regions):
    """Perform two-way statistical comparison"""

    print(f"\n=== TWO-WAY COMPARISON: VSV vs MapSeq ===")
    print(f"Analyzing {len(common_regions)} common regions")
    print(f"Using log-softmax normalization for both datasets")

    # Filter to common regions
    vsv_common = vsv_data[common_regions]
    mapseq_common = mapseq_data[common_regions]

    # Calculate raw averages and ranges
    vsv_avg_raw = vsv_common.mean()
    mapseq_avg_raw = mapseq_common.mean()

    print(f"\nRaw averages before normalization:")
    print(f"VSV mean: {vsv_avg_raw.mean():.2e}, std: {vsv_avg_raw.std():.2e}")
    print(f"MapSeq mean: {mapseq_avg_raw.mean():.2e}, std: {mapseq_avg_raw.std():.2e}")

    # Apply log-softmax normalization to averages
    vsv_avg_norm = log_transform_and_softmax(vsv_avg_raw.values)
    mapseq_avg_norm = log_transform_and_softmax(mapseq_avg_raw.values)

    # Calculate normalized individual samples for error bars
    vsv_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in vsv_common.iterrows()]
    )
    mapseq_individual_norm = np.array(
        [log_transform_and_softmax(row.values) for _, row in mapseq_common.iterrows()]
    )

    # Calculate SEM from normalized data
    vsv_sem = np.std(vsv_individual_norm, axis=0) / np.sqrt(len(vsv_individual_norm))
    mapseq_sem = np.std(mapseq_individual_norm, axis=0) / np.sqrt(
        len(mapseq_individual_norm)
    )

    print(f"VSV SEM range: {vsv_sem.min():.2e} to {vsv_sem.max():.2e}")
    print(f"MapSeq SEM range: {mapseq_sem.min():.2e} to {mapseq_sem.max():.2e}")

    # Calculate Jensen-Shannon divergence
    js_vsv_vs_mapseq = jensenshannon(vsv_avg_norm, mapseq_avg_norm)

    print(f"\nJensen-Shannon Divergence:")
    print(f"  VSV vs MapSeq: {js_vsv_vs_mapseq:.6f}")

    return {
        "common_regions": common_regions,
        "vsv_avg_norm": vsv_avg_norm,
        "mapseq_avg_norm": mapseq_avg_norm,
        "vsv_sem": vsv_sem,
        "mapseq_sem": mapseq_sem,
        "js_vsv_vs_mapseq": js_vsv_vs_mapseq,
    }


def create_two_way_plot(comparison_results, output_prefix):
    """Create publication-ready two-way comparison plot"""

    regions = comparison_results["common_regions"]
    vsv_avg_norm = comparison_results["vsv_avg_norm"]
    mapseq_avg_norm = comparison_results["mapseq_avg_norm"]
    vsv_sem = comparison_results["vsv_sem"]
    mapseq_sem = comparison_results["mapseq_sem"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up bar positions
    x_pos = np.arange(len(regions))
    width = 0.35

    # Reverse data to match region order (consistent with three-way script)
    vsv_avg_norm_reversed = vsv_avg_norm[::-1]
    mapseq_avg_norm_reversed = mapseq_avg_norm[::-1]
    vsv_sem_reversed = vsv_sem[::-1]
    mapseq_sem_reversed = mapseq_sem[::-1]

    # Create bars
    bars1 = ax.bar(
        x_pos - width / 2,
        vsv_avg_norm_reversed,
        width,
        yerr=vsv_sem_reversed,
        capsize=5,
        alpha=0.8,
        label="VSV Raw Pixels",
        color="skyblue",
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        mapseq_avg_norm_reversed,
        width,
        yerr=mapseq_sem_reversed,
        capsize=5,
        alpha=0.8,
        label="MapSeq",
        color="lightcoral",
    )

    # Customize plot
    ax.set_xlabel("Brain Regions")
    ax.set_ylabel("Normalized Density (Log-Softmax)")

    js_value = comparison_results["js_vsv_vs_mapseq"]
    title = f"VSV vs MapSeq Comparison\n" f"Jensen-Shannon Divergence: {js_value:.4f}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xticks(x_pos)

    # Apply region label mapping to match three-way script exactly
    region_label_mapping = {
        "RSPagl+RSPd": "RSP",
        "PM": "PM",
        "AM": "AM",
        "AL": "AL",
        "LM+Li": "LM+LI",
    }

    # Reverse the order of regions for display (consistent with three-way script)
    regions_reversed = regions[::-1]
    region_labels = [
        region_label_mapping.get(region, region) for region in regions_reversed
    ]
    ax.set_xticklabels(region_labels)
    ax.legend()

    plt.tight_layout()

    # Save plot
    plot_file = f"{output_prefix}_vsv_mapseq_comparison.svg"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Two-way plot saved: {plot_file}")
    return plot_file


def save_two_way_results(results, output_prefix):
    """Save detailed two-way comparison results"""
    filename = f"{output_prefix}_vsv_mapseq_results.txt"

    with open(filename, "w") as f:
        f.write("TWO-WAY DATASET COMPARISON RESULTS: VSV vs MapSeq\n")
        f.write("=" * 60 + "\n\n")

        # Jensen-Shannon divergence
        f.write("JENSEN-SHANNON DIVERGENCE:\n")
        f.write(f"VSV vs MapSeq: {results['js_vsv_vs_mapseq']:.6f}\n")
        f.write(
            "(Lower values = more similar, 0 = identical, 1 = maximally different)\n\n"
        )

        # Region-wise comparison
        f.write("DETAILED REGIONAL ANALYSIS:\n")
        f.write("-" * 70 + "\n")
        f.write("Region      VSV (norm)    MapSeq (norm) VSV SEM      MapSeq SEM\n")
        f.write("-" * 70 + "\n")

        regions = results["common_regions"]
        vsv_norm = results["vsv_avg_norm"]
        mapseq_norm = results["mapseq_avg_norm"]
        vsv_sem = results["vsv_sem"]
        mapseq_sem = results["mapseq_sem"]

        for i, region in enumerate(regions):
            f.write(
                f"{region:<11} {vsv_norm[i]:<12.6f} {mapseq_norm[i]:<12.6f} "
                f"{vsv_sem[i]:<12.6f} {mapseq_sem[i]:<12.6f}\n"
            )

        f.write("\nNOTE: Values are normalized using log-softmax transformation\n")
        f.write("SEM = Standard Error of the Mean\n")

    # Save CSV results
    csv_filename = f"{output_prefix}_vsv_mapseq_results.csv"

    csv_data = {
        "region": results["common_regions"],
        "vsv_normalized": results["vsv_avg_norm"],
        "mapseq_normalized": results["mapseq_avg_norm"],
        "vsv_sem": results["vsv_sem"],
        "mapseq_sem": results["mapseq_sem"],
    }

    pd.DataFrame(csv_data).to_csv(csv_filename, index=False)

    print(f"Two-way detailed results saved: {filename}")
    print(f"Two-way CSV results saved: {csv_filename}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare VSV raw pixels and MapSeq datasets with area combining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_vsv_mapseq_two_way.py --vsv standardized_datasets/vsv_raw_pixels_standardized.csv --mapseq "Summary_data_07012025 - mapseq_data.csv"
  python compare_vsv_mapseq_two_way.py --vsv vsv.csv --mapseq mapseq.csv --output my_vsv_mapseq_comparison
        """,
    )

    parser.add_argument("--vsv", required=True, help="Path to VSV dataset CSV file")
    parser.add_argument(
        "--mapseq", required=True, help="Path to MapSeq dataset CSV file"
    )
    parser.add_argument(
        "--output",
        default="vsv_mapseq_comparison",
        help="Output file prefix (default: vsv_mapseq_comparison)",
    )

    return parser.parse_args()


def main():
    """Main comparison function"""
    from pathlib import Path
    args = parse_arguments()

    # Set output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / "compare_vsv_mapseq_two_way"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / args.output)

    # Define target regions (custom order: RSPagl+RSPd, PM, AM, AL, LM+Li)
    # This follows the requested order: POR, LI, LM, AL, RL, A, AM, PM, RSPagl, RSPd, RSPv
    target_regions = ["RSPagl+RSPd", "PM", "AM", "AL", "LM+Li"]

    print("VSV vs MapSeq TWO-WAY COMPARISON PIPELINE")
    print("=" * 60)
    print(f"VSV dataset: {args.vsv}")
    print(f"MapSeq dataset: {args.mapseq}")
    print(f"Output prefix: {output_prefix}")

    # Load datasets
    vsv_data = load_dataset(args.vsv, "VSV")
    mapseq_data = load_dataset(args.mapseq, "MapSeq")

    # Preprocess MapSeq data
    print("\nPreprocessing MapSeq data...")
    mapseq_processed = preprocess_mapseq_data(mapseq_data.copy())
    print(f"MapSeq processed regions: {list(mapseq_processed.columns)}")

    # Create combined regions for VSV dataset
    print(f"\nCreating combined regions for VSV dataset: {target_regions}")
    vsv_combined = create_combined_regions_with_mapping(
        vsv_data, target_regions, "allen_vsv"
    )

    # Find common regions between both datasets
    vsv_regions = set(vsv_combined.columns)
    mapseq_regions = set(mapseq_processed.columns)

    print(f"\nTwo-way region overlap analysis:")
    print(f"  VSV regions: {len(vsv_regions)} - {sorted(vsv_regions)}")
    print(f"  MapSeq regions: {len(mapseq_regions)} - {sorted(mapseq_regions)}")

    # Find common regions
    common_regions = vsv_regions & mapseq_regions
    common_regions = list(common_regions)
    print(f"  Common regions: {len(common_regions)} - {sorted(common_regions)}")

    # Order regions according to target_regions order
    ordered_common_regions = [r for r in target_regions if r in common_regions]
    print(f"Using specified region order: {ordered_common_regions}")

    # Filter to common regions
    vsv_final = vsv_combined[ordered_common_regions]
    mapseq_final = mapseq_processed[ordered_common_regions]

    # Perform two-way comparison
    two_way_results = perform_two_way_comparison(
        vsv_final, mapseq_final, ordered_common_regions
    )

    # Create plots and save results
    create_two_way_plot(two_way_results, output_prefix)
    save_two_way_results(two_way_results, output_prefix)

    print(f"\n{'='*60}")
    print("TWO-WAY ANALYSIS COMPLETE")
    js = two_way_results["js_vsv_vs_mapseq"]
    print(f"Jensen-Shannon Divergence:")
    print(f"  VSV vs MapSeq: {js:.6f}")
    print(f"Files generated:")
    print(f"  - Plot: {output_prefix}_vsv_mapseq_comparison.svg")
    print(f"  - Results: {output_prefix}_vsv_mapseq_results.txt")
    print(f"  - CSV: {output_prefix}_vsv_mapseq_results.csv")


if __name__ == "__main__":
    main()
