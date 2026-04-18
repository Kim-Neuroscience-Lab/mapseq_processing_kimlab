import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent opening windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, kruskal
import ast
from itertools import combinations
import os
import glob
import argparse

# Set font to Helvetica with Arial fallback, and ensure SVG text is editable
plt.rcParams["font.family"] = ['Helvetica', 'Arial']
plt.rcParams["svg.fonttype"] = "none"

# Base directory containing per-animal data
# Default to 02_output directory in the repository root
import sys
from pathlib import Path

# Get the repository root (assuming helpers/ is a subdirectory)
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BASE_DIR = REPO_ROOT / "02_output"

BASE_DIR = str(DEFAULT_BASE_DIR)

# Results directory for saving outputs (relative to helpers directory)
RESULTS_DIR = str(Path(__file__).parent.parent / "outputs" / "01_motif_analysis_per_animal")

# Domain normalization and replicate export are aligned with original motif_analysis_per_animal.py:
# - Domain = number of regions (1 + motif.count("+") for +-joined labels; bracket-list uses len(ast.literal_eval)).
# - Domains 4 and 5 are excluded; only 1,2,3 used. Animal jr0420 excluded from domain data (Option A).
# See comment block before "FIGURE 2: Domain-wise Normalization" and inline comments there for repair context.

# To use a different directory, uncomment and modify the line below:
# set_base_directory("/path/to/your/motif_observed_summary")


def ensure_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    else:
        print(f"Using existing results directory: {RESULTS_DIR}")


def perform_kruskal_wallis_tests(datasets, all_motifs, normalization_type="global"):
    """
    Perform Kruskal-Wallis tests for each motif across timepoints.

    Args:
        datasets: dict of {timepoint: dataframe}
        all_motifs: list of all unique motifs
        normalization_type: 'global' or 'domain'

    Returns:
        pandas.DataFrame with test results
    """
    from scipy.stats import kruskal

    results = []
    datasets_list = list(datasets.keys())

    for motif in all_motifs:
        # Collect data for this motif from all timepoints
        groups = []
        group_labels = []

        for timepoint, df in datasets.items():
            motif_data = df[df["motif label_Clean"] == motif]

            if len(motif_data) > 0:
                if normalization_type == "global":
                    values = motif_data["normalized_freq"].values
                else:
                    # For domain-wise, we'll use domain_normalized_freq if available
                    if "domain_normalized_freq" in motif_data.columns:
                        values = motif_data["domain_normalized_freq"].values
                    else:
                        values = motif_data["normalized_freq"].values

                if len(values) > 0:
                    groups.append(values)
                    group_labels.append(timepoint)

        # Perform Kruskal-Wallis test if we have data from at least 2 groups
        if len(groups) >= 2:
            try:
                # Check if all groups have at least some variation
                valid_groups = [
                    group for group in groups if len(group) > 0 and np.var(group) > 0
                ]

                if len(valid_groups) >= 2:
                    h_stat, p_value = kruskal(*valid_groups)

                    # Get sample sizes for each group
                    sample_sizes = {
                        label: len(group) for label, group in zip(group_labels, groups)
                    }

                    results.append(
                        {
                            "Motif": motif,
                            "H_statistic": h_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "n_groups": len(valid_groups),
                            **sample_sizes,
                            "normalization": normalization_type,
                        }
                    )
                else:
                    results.append(
                        {
                            "Motif": motif,
                            "H_statistic": np.nan,
                            "p_value": np.nan,
                            "significant": False,
                            "n_groups": len(groups),
                            "note": "Insufficient variation in groups",
                            "normalization": normalization_type,
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "Motif": motif,
                        "H_statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_groups": len(groups),
                        "note": f"Error: {str(e)}",
                        "normalization": normalization_type,
                    }
                )
        else:
            results.append(
                {
                    "Motif": motif,
                    "H_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "n_groups": len(groups),
                    "note": "Insufficient groups for testing",
                    "normalization": normalization_type,
                }
            )

    return pd.DataFrame(results)


def set_base_directory(new_base_dir):
    """
    Set a new base directory for loading per-animal data.

    Args:
        new_base_dir (str): Path to directory containing timepoint subdirectories
    """
    global BASE_DIR
    BASE_DIR = new_base_dir
    print(f"Base directory set to: {BASE_DIR}")


def load_per_animal_data(base_dir, parameterization_filter=None, model_type=None):
    """
    Load per-animal motif data from directory structure, preserving individual animal measurements.

    Supports two directory structures:
    1. Old structure (multiple ages):
       base_dir/
       ├── p12/
       │   └── analysis/
       │       ├── uniform/
       │       │   └── *upsetplot_uniform.csv
       │       └── region_specific/
       │           └── *upsetplot_region_specific.csv
       ├── p20/
       │   └── analysis/
       └── p60/
           └── analysis/
    
    2. New structure (single parameterization):
       base_dir/
       └── analysis/
           ├── uniform/
           │   └── *upsetplot_uniform.csv
           └── region_specific/
               └── *upsetplot_region_specific.csv

    Args:
        base_dir: Base directory containing age subdirectories
        parameterization_filter: Optional parameterization name to filter by (e.g., "01.minimal_filter_parameters_i1_r1_t1_u2")
        model_type: Model type to load ('uniform', 'region_specific', or None for backward compatibility)

    Returns:
        dict: {timepoint: dataframe with individual animal data}
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    datasets = {}

    # Check if this is a parameterization directory (has analysis/ subdirectory)
    analysis_dir = os.path.join(base_dir, "analysis")
    if os.path.isdir(analysis_dir):
        # New structure: single parameterization directory
        # Extract age from parent directory path
        base_path = Path(base_dir)
        age_from_path = None
        
        # Try to find age in path (e.g., 02_output/p3/... -> p3)
        for part in base_path.parts:
            if part.lower().startswith('p') and len(part) <= 4 and part[1:].isdigit():
                age_from_path = part.lower()
                break
        
        if not age_from_path:
            # Try to extract from directory name or file names
            # Check both model subdirectories and main directory for backward compatibility
            search_patterns = []
            if model_type:
                search_patterns.append(os.path.join(analysis_dir, model_type, f"*upsetplot_{model_type}.csv"))
            else:
                # Backward compatibility: check main directory and both model subdirectories
                search_patterns.append(os.path.join(analysis_dir, "*upsetplot.csv"))
                search_patterns.append(os.path.join(analysis_dir, "uniform", "*upsetplot_uniform.csv"))
                search_patterns.append(os.path.join(analysis_dir, "region_specific", "*upsetplot_region_specific.csv"))
            
            csv_files = []
            for pattern in search_patterns:
                csv_files.extend(glob.glob(pattern))
            
            if csv_files:
                # Try to extract age from first filename
                first_file = os.path.basename(csv_files[0])
                if first_file.lower().startswith('p'):
                    age_from_path = first_file.split('_')[0].lower()
        
        if not age_from_path:
            # Default to directory name if we can't determine
            age_from_path = base_path.name.split('_')[0].lower() if '_' in base_path.name else 'unknown'
        
        timepoint_name = age_from_path.upper()  # Convert to P3, P12, P20, P60
        model_suffix = f" ({model_type})" if model_type else ""
        print(f"Loading data for {timepoint_name}{model_suffix} from parameterization directory...")
        
        # Look for upsetplot files in analysis directory
        # Support both new dual-model structure and backward compatibility
        csv_files = []
        if model_type:
            # New structure: search in model-specific subdirectory
            csv_files = glob.glob(os.path.join(analysis_dir, model_type, f"*upsetplot_{model_type}.csv"))
        else:
            # Backward compatibility: check main directory first, then model subdirectories
            csv_files = glob.glob(os.path.join(analysis_dir, "*upsetplot.csv"))
            if not csv_files:
                # Fall back to model subdirectories
                csv_files.extend(glob.glob(os.path.join(analysis_dir, "uniform", "*upsetplot_uniform.csv")))
                csv_files.extend(glob.glob(os.path.join(analysis_dir, "region_specific", "*upsetplot_region_specific.csv")))
        
        # Filter out aggregate files (containing "ALL") if individual animal files exist
        individual_files = [f for f in csv_files if "_ALL_" not in os.path.basename(f)]
        if individual_files:
            csv_files = individual_files
        # Otherwise use all files including aggregate
        
        # Validate that we found CSV files
        if not csv_files:
            print(f"  Warning: No CSV files found in {analysis_dir}")
            return datasets
        
        print(f"  Found {len(csv_files)} files")
        
        # Load data from all animals for this timepoint (keeping individual animal data)
        all_animal_data = []
        
        for csv_file in csv_files:
            animal_id = os.path.basename(csv_file).split("_")[0]  # Extract animal ID
            try:
                df = pd.read_csv(csv_file)
                
                # Check if this is an upsetplot.csv file (has Motifs, Observed columns)
                if "Motifs" in df.columns and "Observed" in df.columns:
                    # Convert upsetplot format to expected format
                    import ast
                    df["motif label"] = df["Motifs"].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    ).apply(lambda x: "+".join(sorted(x)) if isinstance(x, list) and x and x[0] else "")
                    df["motif size"] = df.get("Degree", df["Motifs"].apply(
                        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else (len(x) if isinstance(x, list) else 0)
                    ))
                    df["observed"] = df["Observed"].astype(float)
                # Check if this is the expected format
                elif "motif label" in df.columns and "observed" in df.columns:
                    # Already in expected format
                    if "motif size" not in df.columns:
                        # Calculate motif size if not present
                        df["motif size"] = df["motif label"].apply(
                            lambda x: len(str(x).split("+")) if "+" in str(x) else (1 if str(x) else 0)
                        )
                else:
                    print(
                        f"    Warning: {animal_id} has unexpected column format. Expected 'Motifs'/'Observed' or 'motif label'/'observed'"
                    )
                    continue
                
                # Validate required columns
                required_columns = ["motif label", "motif size", "observed"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    print(
                        f"    Warning: {animal_id} missing columns: {missing_columns}"
                    )
                    continue
                
                # Calculate normalized frequency for this animal
                total_observations = df["observed"].sum()
                if total_observations > 0:
                    df["normalized_freq"] = df["observed"] / total_observations
                else:
                    df["normalized_freq"] = 0.0
                df["Animal_ID"] = animal_id  # Add animal identifier
                df["Timepoint"] = timepoint_name  # Add timepoint identifier
                
                all_animal_data.append(df)
                print(
                    f"    Loaded {animal_id}: {len(df)} motifs, {df['observed'].sum():.0f} observations"
                )
            except Exception as e:
                print(f"    Error loading {csv_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Combine all animal data for this timepoint (preserving individual measurements)
        if not all_animal_data:
            print(f"  Warning: No valid data loaded for {timepoint_name}")
            return datasets
        
        combined_df = pd.concat(all_animal_data, ignore_index=True)
        datasets[timepoint_name] = combined_df
        print(
            f"  Combined data from {len(csv_files)} animals with {len(combined_df)} total motif observations"
        )
        
        # Return datasets for NEW structure
        return datasets
        
    else:
        # Old structure: look for timepoint directories (p3, p12, p20, p60)
        # When base_dir is 02_output, we need to search through all age directories
        # and find parameterization subdirectories within each
        timepoint_dirs = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.lower().startswith("p") and len(d) <= 4
        ]

        if not timepoint_dirs:
            raise ValueError(
                f"No timepoint directories (p3, p12, p20, p60) found in {base_dir}, "
                f"and no analysis/ subdirectory found. Expected either:\n"
                f"  - Multiple age directories (p3/, p12/, p20/, p60/) OR\n"
                f"  - Single parameterization directory with analysis/ subdirectory"
            )

        # Process ALL timepoints for cross-age analysis
        timepoint_dirs_to_process = sorted(timepoint_dirs)
        
        # Group CSV files by timepoint and process each separately
        for timepoint_dir in timepoint_dirs_to_process:
            timepoint_path = os.path.join(base_dir, timepoint_dir)
            timepoint_name = timepoint_dir.upper()  # Convert to P3, P12, P20, P60

            print(f"Loading data for {timepoint_name}...")

            # Look for CSV files in parameterization subdirectories
            # Structure: 02_output/p3/01.minimal_filter_parameters_*/analysis/*upsetplot.csv
            timepoint_csvs = []
            
            # Check for parameterization subdirectories with analysis/ subdirectories
            if os.path.isdir(timepoint_path):
                for subdir in os.listdir(timepoint_path):
                    subdir_path = os.path.join(timepoint_path, subdir)
                    if os.path.isdir(subdir_path):
                        # Check if this is a parameterization directory (starts with 01., 02., etc.)
                        if subdir.startswith(('01.', '02.', '03.', '04.', '05.')):
                            # Filter by parameterization if specified
                            if parameterization_filter and subdir != parameterization_filter:
                                continue
                            analysis_path = os.path.join(subdir_path, "analysis")
                            if os.path.isdir(analysis_path):
                                if model_type:
                                    # New structure: search in model-specific subdirectory
                                    timepoint_csvs.extend(glob.glob(os.path.join(analysis_path, model_type, f"*upsetplot_{model_type}.csv")))
                                else:
                                    # Backward compatibility: check main directory first, then model subdirectories
                                    timepoint_csvs.extend(glob.glob(os.path.join(analysis_path, "*upsetplot.csv")))
                                    if not timepoint_csvs:
                                        timepoint_csvs.extend(glob.glob(os.path.join(analysis_path, "uniform", "*upsetplot_uniform.csv")))
                                        timepoint_csvs.extend(glob.glob(os.path.join(analysis_path, "region_specific", "*upsetplot_region_specific.csv")))
            
            # Process this timepoint's files
            if timepoint_csvs:
                # Filter out aggregate files (containing "ALL") if individual animal files exist
                individual_files = [f for f in timepoint_csvs if "_ALL_" not in os.path.basename(f).upper()]
                if individual_files:
                    timepoint_csvs = individual_files
                
                # Load data for this timepoint
                all_animal_data = []
                
                for csv_file in timepoint_csvs:
                    animal_id = os.path.basename(csv_file).split("_")[0]  # Extract animal ID
                    try:
                        df = pd.read_csv(csv_file)

                        # Check if this is an upsetplot.csv file (has Motifs, Observed columns)
                        if "Motifs" in df.columns and "Observed" in df.columns:
                            # Convert upsetplot format to expected format
                            import ast
                            df["motif label"] = df["Motifs"].apply(
                                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                            ).apply(lambda x: "+".join(sorted(x)) if isinstance(x, list) and x and x[0] else "")
                            df["motif size"] = df.get("Degree", df["Motifs"].apply(
                                lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else (len(x) if isinstance(x, list) else 0)
                            ))
                            df["observed"] = df["Observed"].astype(float)
                        # Check if this is the expected format
                        elif "motif label" in df.columns and "observed" in df.columns:
                            # Already in expected format
                            if "motif size" not in df.columns:
                                # Calculate motif size if not present
                                df["motif size"] = df["motif label"].apply(
                                    lambda x: len(str(x).split("+")) if "+" in str(x) else (1 if str(x) else 0)
                                )
                        else:
                            print(
                                f"    Warning: {animal_id} has unexpected column format. Expected 'Motifs'/'Observed' or 'motif label'/'observed'"
                            )
                            continue

                        # Validate required columns
                        required_columns = ["motif label", "motif size", "observed"]
                        missing_columns = [
                            col for col in required_columns if col not in df.columns
                        ]
                        if missing_columns:
                            print(
                                f"    Warning: {animal_id} missing columns: {missing_columns}"
                            )
                            continue

                        # Calculate normalized frequency for this animal
                        total_observations = df["observed"].sum()
                        if total_observations > 0:
                            df["normalized_freq"] = df["observed"] / total_observations
                        else:
                            df["normalized_freq"] = 0.0
                        df["Animal_ID"] = animal_id  # Add animal identifier
                        df["Timepoint"] = timepoint_name  # Add timepoint identifier

                        all_animal_data.append(df)
                        print(
                            f"    Loaded {animal_id}: {len(df)} motifs, {df['observed'].sum():.0f} observations"
                        )
                    except Exception as e:
                        print(f"    Error loading {csv_file}: {e}")
                        import traceback
                        traceback.print_exc()

                # Combine all animal data for this timepoint
                if all_animal_data:
                    combined_df = pd.concat(all_animal_data, ignore_index=True)
                    datasets[timepoint_name] = combined_df
                    print(
                        f"  Combined data from {len(timepoint_csvs)} animals with {len(combined_df)} total motif observations"
                    )
        
        # Check if we loaded any data
        if not datasets:
            raise ValueError(f"No valid data loaded from {base_dir}")
        
        return datasets


def clean_motif_label(label):
    """Clean and standardize motif labels"""
    if isinstance(label, str):
        try:
            # Try to evaluate as a Python literal
            return str(sorted(ast.literal_eval(label)))
        except:
            # If that fails, just return the string
            return label
    return str(label)


def add_bracket_annotation(fig, ax, x_start, x_end, text):
    """Add an upward-opening bracket with text annotation below the x tick labels"""

    # Force a draw to ensure tick labels are rendered
    fig.canvas.draw()

    # Convert x positions to axis coordinates (0-1 range within the axis)
    xlim = ax.get_xlim()
    x_start_ax = (x_start - xlim[0]) / (xlim[1] - xlim[0])
    x_end_ax = (x_end - xlim[0]) / (xlim[1] - xlim[0])

    # Find the bottom of the tick labels by checking their bounding boxes
    tick_bottom = 0  # Start at axis bottom
    for tick in ax.get_xticklabels():
        if tick.get_text():  # Only consider non-empty labels
            bbox = tick.get_window_extent()
            # Convert to axes coordinates
            axes_coords = ax.transAxes.inverted().transform(
                [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
            )
            tick_y_bottom = axes_coords[0][
                1
            ]  # Bottom of tick label in axes coordinates
            tick_bottom = min(tick_bottom, tick_y_bottom)

    # Position brackets below the tick labels
    bracket_spacing = 0.04  # Small gap below tick labels
    bracket_height = 0.015  # Height of bracket
    text_spacing = 0.04  # Gap between bracket and text

    bracket_y = tick_bottom - bracket_spacing
    text_y = bracket_y - bracket_height - text_spacing

    # Draw bracket using plot with axes coordinates and clip_on=False
    # Left vertical line
    ax.plot(
        [x_start_ax, x_start_ax],
        [bracket_y, bracket_y + bracket_height],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    # Right vertical line
    ax.plot(
        [x_end_ax, x_end_ax],
        [bracket_y, bracket_y + bracket_height],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    # Bottom horizontal line
    ax.plot(
        [x_start_ax, x_end_ax],
        [bracket_y, bracket_y],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )

    # Add text below the bracket
    ax.text(
        (x_start_ax + x_end_ax) / 2,
        text_y,
        text,
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        clip_on=False,
    )


def _normalized_freq_arrays(freq_arrays):
    """Epsilon + L1-normalize each frequency vector (shared by JSD helpers)."""
    epsilon = 1e-10
    normalized_freqs = []
    for freq_array in freq_arrays:
        freq = np.array(freq_array, dtype=float) + epsilon
        freq = freq / freq.sum()
        normalized_freqs.append(freq)
    return normalized_freqs


def calculate_distribution_jsd(freq_arrays):
    """
    Legacy triple of JSDs for positions (0,1), (0,2), (1,2) — used for figure brackets
    when there are exactly three timepoints; with two timepoints only (0,1) is filled.
    """
    n_timepoints = len(freq_arrays)

    if n_timepoints < 2:
        return np.nan, np.nan, np.nan

    normalized_freqs = _normalized_freq_arrays(freq_arrays)

    if n_timepoints == 2:
        jsd_12 = jensenshannon(normalized_freqs[0], normalized_freqs[1])
        return jsd_12, np.nan, np.nan
    elif n_timepoints >= 3:
        jsd_12 = jensenshannon(normalized_freqs[0], normalized_freqs[1])
        jsd_13 = jensenshannon(normalized_freqs[0], normalized_freqs[2])
        jsd_23 = jensenshannon(normalized_freqs[1], normalized_freqs[2])
        return jsd_12, jsd_13, jsd_23

    return np.nan, np.nan, np.nan


def pairwise_jsd_dataframe(freq_arrays, labels, normalization, domain="all"):
    """
    Jensen–Shannon distance between all unordered pairs of distributions.

    freq_arrays: list of vectors (same length), aligned with labels.
    domain: 'all' for global; else motif domain index (int) for domain-wise rows.
    """
    cols = ["timepoint_a", "timepoint_b", "jsd", "normalization", "domain"]
    if len(freq_arrays) < 2 or len(labels) != len(freq_arrays):
        return pd.DataFrame(columns=cols)

    normalized_freqs = _normalized_freq_arrays(freq_arrays)
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        rows.append(
            {
                "timepoint_a": str(labels[i]),
                "timepoint_b": str(labels[j]),
                "jsd": float(jensenshannon(normalized_freqs[i], normalized_freqs[j])),
                "normalization": normalization,
                "domain": domain,
            }
        )
    return pd.DataFrame(rows)


def jsd_bracket_annotation_lines(jsd_12, jsd_13, jsd_23, labels):
    """Build bracket text lines using actual timepoint names for legacy (0,1),(0,2),(1,2) triple."""
    parts = []
    n = len(labels)
    if n >= 2 and not np.isnan(jsd_12):
        parts.append(f"{labels[0]}-{labels[1]}: {jsd_12:.3f}")
    if n >= 3 and not np.isnan(jsd_13):
        parts.append(f"{labels[0]}-{labels[2]}: {jsd_13:.3f}")
    if n >= 3 and not np.isnan(jsd_23):
        parts.append(f"{labels[1]}-{labels[2]}: {jsd_23:.3f}")
    return parts


CANONICAL_MIRROR_SUBDIRS = ("cross_age", "p3", "p12", "p20", "p60")


def write_mirrored_model_outputs(model_results_dir, jsd_df, run_metrics_df, summary_text):
    """Write the same analysis artifacts under each canonical model subdirectory."""
    for sub in CANONICAL_MIRROR_SUBDIRS:
        out_dir = os.path.join(model_results_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        jsd_df.to_csv(os.path.join(out_dir, "distribution_jsd_pairwise.csv"), index=False)
        run_metrics_df.to_csv(os.path.join(out_dir, "run_metrics.csv"), index=False)
        summary_path = os.path.join(out_dir, "analysis_run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)


def calculate_summary_stats(datasets, all_motifs, normalization_type="global"):
    """
    Calculate mean and SEM for each motif across animals for each timepoint.

    Args:
        datasets: dict of {timepoint: dataframe}
        all_motifs: list of all unique motifs
        normalization_type: 'global' or 'domain'

    Returns:
        dict: {timepoint: {motif: {'mean': float, 'sem': float, 'n': int, 'values': list}}}
    """
    from scipy import stats

    summary_stats = {}

    for timepoint, df in datasets.items():
        summary_stats[timepoint] = {}

        for motif in all_motifs:
            # Get all animal values for this motif
            motif_data = df[df["motif label_Clean"] == motif]

            if len(motif_data) > 0:
                if normalization_type == "global":
                    values = motif_data["normalized_freq"].values
                else:  # domain-wise normalization will be handled separately
                    values = motif_data["normalized_freq"].values

                if len(values) > 0:
                    mean_val = np.mean(values)
                    sem_val = stats.sem(values) if len(values) > 1 else 0
                    summary_stats[timepoint][motif] = {
                        "mean": mean_val,
                        "sem": sem_val,
                        "n": len(values),
                        "values": values.tolist(),
                    }
                else:
                    summary_stats[timepoint][motif] = {
                        "mean": 0,
                        "sem": 0,
                        "n": 0,
                        "values": [],
                    }
            else:
                summary_stats[timepoint][motif] = {
                    "mean": 0,
                    "sem": 0,
                    "n": 0,
                    "values": [],
                }

    return summary_stats


def plot_bars_with_points_and_errors(
    ax, x_positions, summary_stats, datasets_list, colors, width, alpha=0.8
):
    """
    Plot bars with individual points and error bars.

    Args:
        ax: matplotlib axis
        x_positions: array of x positions
        summary_stats: summary statistics dictionary
        datasets_list: list of dataset names
        colors: list of colors
        width: bar width
        alpha: bar transparency

    Returns:
        list of bar objects
    """
    all_bars = []

    for i, dataset in enumerate(datasets_list):
        means = []
        sems = []

        # Get means and SEMs for all motifs
        for j, x_pos in enumerate(x_positions):
            motif_key = list(summary_stats[dataset].keys())[j]
            stats = summary_stats[dataset][motif_key]
            means.append(stats["mean"])
            sems.append(stats["sem"])

        # Plot bars
        bars = ax.bar(
            x_positions + i * width,
            means,
            width,
            label=dataset,
            color=colors[i],
            alpha=alpha,
            yerr=sems,
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        all_bars.extend(bars)

        # Plot individual points (smaller and fully black)
        for j, x_pos in enumerate(x_positions):
            motif_key = list(summary_stats[dataset].keys())[j]
            values = summary_stats[dataset][motif_key]["values"]

            if values:  # Only plot if there are values
                # Add small random jitter to x position for visibility
                x_jitter = np.random.normal(0, width * 0.1, len(values))
                ax.scatter(
                    x_pos + i * width + x_jitter,
                    values,
                    color="black",
                    alpha=1.0,
                    s=4,
                    zorder=3,
                )

    return all_bars


def generate_per_age_plots(datasets, all_motifs, output_base_dir, normalization_type="global"):
    """
    Generate per-age plots showing individual animals within each age group.
    
    Args:
        datasets: dict of {timepoint: dataframe} (already loaded)
        all_motifs: list of all unique motifs
        output_base_dir: Base output directory (RESULTS_DIR)
        normalization_type: "global" or "domain" (same as existing plots)
    """
    print(f"\nGenerating per-age plots ({normalization_type} normalization)...")
    
    # Process each age group separately
    for timepoint, df in datasets.items():
        age_lower = timepoint.lower()  # P12 -> p12
        age_output_dir = os.path.join(output_base_dir, age_lower)
        os.makedirs(age_output_dir, exist_ok=True)
        
        print(f"  Processing {timepoint}...")
        
        # Create a single-age dataset dict for this timepoint
        single_age_datasets = {timepoint: df}
        
        # Calculate summary statistics for this age only
        if normalization_type == "global":
            age_summary_stats = calculate_summary_stats(single_age_datasets, all_motifs, "global")
        else:
            # For domain normalization, reuse logic aligned with original motif_analysis_per_animal.py:
            # domain count from +-joined labels, domains 4 and 5 removed, jr0420 excluded.
            motif_domains = {}
            for motif in all_motifs:
                if motif.startswith("["):
                    count = len(ast.literal_eval(motif))
                else:
                    count = 1 + motif.count("+") if motif and str(motif).strip() else 1
                if count not in motif_domains:
                    motif_domains[count] = []
                motif_domains[count].append(motif)
            for d in [4, 5]:
                if d in motif_domains:
                    motif_domains.pop(d)

            # Calculate domain-wise normalized frequencies for this age
            domain_data = []
            for _, row in df.iterrows():
                motif = row["motif label_Clean"]
                animal_id = row["Animal_ID"]
                if animal_id == "jr0420":
                    continue
                # Find which domain this motif belongs to
                motif_domain = None
                for domain, domain_motifs in motif_domains.items():
                    if motif in domain_motifs:
                        motif_domain = domain
                        break
                if motif_domain is not None:
                    # Calculate domain total for this animal
                    animal_domain_data = df[(df["Animal_ID"] == animal_id)]
                    domain_motifs_for_animal = animal_domain_data[
                        animal_domain_data["motif label_Clean"].isin(
                            motif_domains[motif_domain]
                        )
                    ]
                    domain_total = domain_motifs_for_animal["observed"].sum()
                    if domain_total > 0:
                        domain_normalized_freq = row["observed"] / domain_total
                    else:
                        domain_normalized_freq = 0
                    domain_data.append({
                        "motif label_Clean": motif,
                        "domain_normalized_freq": domain_normalized_freq,
                        "Animal_ID": animal_id,
                    })
            
            domain_df = pd.DataFrame(domain_data)
            domain_datasets = {timepoint: domain_df}
            
            # Calculate summary stats for domain normalization
            age_summary_stats = {}
            age_summary_stats[timepoint] = {}
            for motif in all_motifs:
                motif_data = domain_df[domain_df["motif label_Clean"] == motif]
                if len(motif_data) > 0:
                    values = motif_data["domain_normalized_freq"].values
                    if len(values) > 0:
                        from scipy import stats
                        mean_val = np.mean(values)
                        sem_val = stats.sem(values) if len(values) > 1 else 0
                        age_summary_stats[timepoint][motif] = {
                            "mean": mean_val,
                            "sem": sem_val,
                            "n": len(values),
                            "values": values.tolist(),
                        }
                    else:
                        age_summary_stats[timepoint][motif] = {
                            "mean": 0,
                            "sem": 0,
                            "n": 0,
                            "values": [],
                        }
                else:
                    age_summary_stats[timepoint][motif] = {
                        "mean": 0,
                        "sem": 0,
                        "n": 0,
                        "values": [],
                    }
        
        # Create plot for this age
        fig, ax = plt.subplots(figsize=(24, 8))
        
        n_motifs = len(all_motifs)
        x = np.arange(n_motifs)
        width = 0.6  # Wider bars for single age group
        
        # Use a single color for this age group
        age_colors = {"P12": "#1f77b4", "P20": "#ff7f0e", "P60": "#2ca02c"}
        color = age_colors.get(timepoint, "#1f77b4")
        
        # Plot bars with individual points
        means = []
        sems = []
        for motif in all_motifs:
            stats = age_summary_stats[timepoint][motif]
            means.append(stats["mean"])
            sems.append(stats["sem"])
        
        # Plot bars
        bars = ax.bar(
            x,
            means,
            width,
            label=timepoint,
            color=color,
            alpha=0.8,
            yerr=sems,
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        
        # Plot individual points
        for j, x_pos in enumerate(x):
            values = age_summary_stats[timepoint][all_motifs[j]]["values"]
            if values:
                x_jitter = np.random.normal(0, width * 0.1, len(values))
                ax.scatter(
                    x_pos + x_jitter,
                    values,
                    color="black",
                    alpha=1.0,
                    s=4,
                    zorder=3,
                )
        
        # Set labels and title
        norm_label = "Global" if normalization_type == "global" else "Domain-wise"
        ax.set_ylabel(f"Normalized Frequency ({norm_label})", fontsize=12)
        ax.set_title(
            f"Motif Frequency Distribution - {norm_label} Normalization ({timepoint})\n(Bars = Mean ± SEM, Points = Individual Animals)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_motifs, rotation=90, ha="center")
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_filename = f"motif_analysis_{normalization_type}_normalization_{timepoint}.svg"
        fig_path = os.path.join(age_output_dir, fig_filename)
        fig.savefig(fig_path, format="svg", dpi=300, bbox_inches="tight")
        print(f"    Saved: {fig_path}")
        plt.close(fig)
        
        # Save summary statistics CSV
        summary_data = []
        for motif in all_motifs:
            stats = age_summary_stats[timepoint][motif]
            summary_data.append({
                "Motif": motif,
                "Mean": stats["mean"],
                "SEM": stats["sem"],
                "N": stats["n"],
            })
        summary_df = pd.DataFrame(summary_data)
        csv_filename = f"motif_summary_{normalization_type}_{timepoint}.csv"
        csv_path = os.path.join(age_output_dir, csv_filename)
        summary_df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")


# Parse command-line arguments for output mode (before data loading)
parser = argparse.ArgumentParser(description="Analyze motif data per animal")
parser.add_argument('--output_mode', default='cross_age', 
                   choices=['cross_age', 'per_age', 'both'],
                   help='Output mode: cross_age (default), per_age, or both')
parser.add_argument('--base_output_dir', type=str, default=None,
                   help='Base output directory for processing results (default: REPO_ROOT/02_output). Helper outputs will be saved in a subdirectory matching the parameterization.')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/01_motif_analysis_per_animal)')
parser.add_argument('--export_wide_for_prism', action='store_true',
                   help='After writing long-format CSVs, also write wide-format TSVs for Prism (per model).')
args = parser.parse_args()
output_mode = args.output_mode

# Update BASE_DIR and RESULTS_DIR if arguments provided
if args.base_output_dir:
    set_base_directory(args.base_output_dir)
if args.helper_output_dir:
    RESULTS_DIR = args.helper_output_dir
    ensure_results_directory()

# Extract parameterization name from helper_output_dir if provided
parameterization_filter = None
if args.helper_output_dir:
    helper_path = Path(args.helper_output_dir)
    # Look for parameterization name in path (e.g., .../01.minimal_filter_parameters_..._helpers/...)
    for part in helper_path.parts:
        if part.startswith(('01.', '02.', '03.', '04.', '05.')) and '_helpers' in part:
            # Extract just the parameterization name (before _helpers)
            parameterization_filter = part.split('_helpers')[0]
            print(f"Filtering by parameterization: {parameterization_filter}")
            break
        elif part.startswith(('01.', '02.', '03.', '04.', '05.')):
            parameterization_filter = part
            print(f"Filtering by parameterization: {parameterization_filter}")
            break

# Process all models separately
models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                     'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                     'bayesian_hierarchical', 'ml_nonparametric']

for model_type in models_to_process:
    print("\n" + "="*80)
    print(f"Processing {model_type.upper()} MODEL")
    print("="*80)
    
    # Read and process data for this model
    print(f"Loading and aggregating per-animal data for {model_type} model...")
    datasets = load_per_animal_data(BASE_DIR, parameterization_filter=parameterization_filter, model_type=model_type)
    
    if not datasets:
        print(f"Warning: No valid datasets found for {model_type} model in {BASE_DIR}")
        continue
    
    print(f"\nSuccessfully loaded {len(datasets)} timepoints for {model_type} model: {list(datasets.keys())}")
    
    # Ensure results directory exists
    ensure_results_directory()
    
    # Create model-specific output directory
    model_results_dir = os.path.join(RESULTS_DIR, model_type)
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Cross-age figures and Kruskal CSVs always use this directory (even if output_mode is per_age-only).
    cross_age_dir = os.path.join(model_results_dir, "cross_age")
    os.makedirs(cross_age_dir, exist_ok=True)
    
    # Store original RESULTS_DIR to restore later
    original_results_dir = RESULTS_DIR
    RESULTS_DIR = model_results_dir

    # Process datasets - clean motif labels and collect all motifs in original order
    # Get motifs in the order they appear in the first dataset
    first_dataset = list(datasets.values())[0]
    all_motifs_original_order = []
    seen_motifs = set()

    for _, row in first_dataset.iterrows():
        clean_motif = clean_motif_label(row["motif label"])
        if clean_motif not in seen_motifs:
            all_motifs_original_order.append(clean_motif)
            seen_motifs.add(clean_motif)

    # Add any motifs from other datasets that weren't in the first dataset
    for name, df in datasets.items():
        df["motif label_Clean"] = df["motif label"].apply(clean_motif_label)
        for motif in df["motif label_Clean"].unique():
            if motif not in seen_motifs:
                all_motifs_original_order.append(motif)
                seen_motifs.add(motif)

    all_motifs = all_motifs_original_order  # Use original order instead of sorted

    for name, df in datasets.items():
        print(
            f"{name}: {df['Animal_ID'].nunique()} animals, {len(df['motif label_Clean'].unique())} unique motifs"
        )

    print(f"\nTotal unique motifs across all timepoints ({model_type} model): {len(all_motifs)}")

    model_summary_lines = []

    def sp(*args):
        """Print to console and capture a single line for analysis_run_summary.txt."""
        line = " ".join(str(a) for a in args) if args else ""
        print(*args)
        model_summary_lines.append(line)

    # Summary statistics required for global figure and JSD (all output modes).
    sp("\nCalculating summary statistics...")
    global_summary_stats = calculate_summary_stats(datasets, all_motifs, "global")

    # Perform Kruskal-Wallis tests for global normalization
    sp("Performing Kruskal-Wallis tests for global normalization...")
    global_kw_results = perform_kruskal_wallis_tests(datasets, all_motifs, "global")

    # =============================================================================
    # FIGURE 1: Global Normalization (each dataset sums to 1)
    # =============================================================================

    # Create Figure 1 with reasonable height
    fig1, ax1 = plt.subplots(figsize=(24, 8))

    n_motifs = len(all_motifs)
    x = np.arange(n_motifs)
    width = 0.25

    datasets_list = list(datasets.keys())
    # Use a color palette that can handle any number of datasets
    import matplotlib.cm as cm
    color_map = cm.get_cmap('tab10')
    colors = [color_map(i) for i in range(len(datasets_list))]

    # Plot bars with individual points and error bars
    all_bars = plot_bars_with_points_and_errors(
        ax1, x, global_summary_stats, datasets_list, colors, width
    )

    # Set labels and title
    ax1.set_ylabel("Normalized Frequency (Global)", fontsize=12)
    ax1.set_title(
        "Motif Frequency Distribution - Global Normalization\n(Bars = Mean ± SEM, Points = Individual Animals)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(all_motifs, rotation=90, ha="center")
    ax1.legend()

    # Calculate global distribution JSD using means
    global_freqs = []
    for dataset in datasets_list:
        dataset_freqs = []
        for motif in all_motifs:
            mean_freq = global_summary_stats[dataset][motif]["mean"]
            dataset_freqs.append(mean_freq)
        global_freqs.append(dataset_freqs)

    jsd_12, jsd_13, jsd_23 = calculate_distribution_jsd(global_freqs)

    # Only add bracket annotation if we have enough timepoints
    if not (np.isnan(jsd_12) and np.isnan(jsd_13) and np.isnan(jsd_23)):
        # Calculate bracket span from the actual x-axis range
        bracket_start = -width / 2  # Start from left edge of first group
        bracket_end = n_motifs - 1 + 2.5 * width  # End at right edge of last group

        jsd_parts = jsd_bracket_annotation_lines(
            jsd_12, jsd_13, jsd_23, datasets_list
        )
        if jsd_parts:
            jsd_text = "\n".join(jsd_parts)
            add_bracket_annotation(fig1, ax1, bracket_start, bracket_end, jsd_text)

    # Remove manual bottom adjustment since bracket function handles this automatically
    plt.tight_layout()

    # Save figure as SVG (to cross_age subdirectory)
    global_fig_path = os.path.join(cross_age_dir, "motif_analysis_global_normalization.svg")
    fig1.savefig(global_fig_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved global normalization figure: {global_fig_path}")
    plt.close(fig1)  # Close figure to free memory

    # Save global Kruskal-Wallis results (to cross_age subdirectory)
    global_kw_path = os.path.join(cross_age_dir, "kruskal_wallis_global_normalization.csv")
    global_kw_results.to_csv(global_kw_path, index=False)
    print(f"Saved global Kruskal-Wallis results: {global_kw_path}")

    # Print summary of significant results
    n_significant_global = global_kw_results["significant"].sum()
    sp(
        f"Global normalization: {n_significant_global}/{len(global_kw_results)} motifs show significant differences (p < 0.05)"
    )

    # =============================================================================
    # FIGURE 2: Domain-wise Normalization (each motif length group sums to 1)
    # =============================================================================
    # ALIGNMENT WITH ORIGINAL motif_analysis_per_animal.py (see plan helper_01_match_original_script_logic):
    # - Domain count: pipeline motifs are +-joined (e.g. "al+am"); use 1 + motif.count("+")
    #   so domain = number of regions. Original used bracket-list format and len(ast.literal_eval).
    # - Domains 4 and 5 are removed so only domains 1,2,3 are used for domain normalization.
    # - Animal jr0420 is excluded from domain normalization (Option A); domain_normalized_freq
    #   is NaN for jr0420 in individual_replicates_per_animal_domain.csv.

    # Group motifs by count (domain)
    motif_domains = {}
    for motif in all_motifs:
        if motif.startswith("["):
            count = len(ast.literal_eval(motif))
        else:
            # +-joined format from pipeline (e.g. "al+am" -> 2 regions)
            count = 1 + motif.count("+") if motif and str(motif).strip() else 1
        if count not in motif_domains:
            motif_domains[count] = []
        motif_domains[count].append(motif)

    # Match original: use only domains 1, 2, 3 for domain normalization (remove 4 and 5)
    domains_to_remove = [4, 5]
    for d in domains_to_remove:
        if d in motif_domains:
            motif_domains.pop(d)

    # Sort domains and motifs within domains (preserve original order within domains)
    sorted_domains = sorted(motif_domains.keys())
    domain_sorted_motifs = []
    domain_boundaries = []
    current_pos = 0

    for domain in sorted_domains:
        # Get motifs in original order for this domain
        domain_motifs_original_order = [
            motif for motif in all_motifs if motif in motif_domains[domain]
        ]
        domain_sorted_motifs.extend(domain_motifs_original_order)
        domain_boundaries.append(
            (current_pos, current_pos + len(domain_motifs_original_order) - 1, domain)
        )
        current_pos += len(domain_motifs_original_order)

    # Calculate domain-wise normalized frequencies for each animal
    print("Calculating domain-wise normalization...")
    domain_datasets = {}

    for timepoint, df in datasets.items():
        domain_data = []

        for _, row in df.iterrows():
            motif = row["motif label_Clean"]
            animal_id = row["Animal_ID"]

            # Match original: exclude jr0420 from domain normalization (Option A)
            if animal_id == "jr0420":
                continue

            # Find which domain this motif belongs to
            motif_domain = None
            for domain, domain_motifs in motif_domains.items():
                if motif in domain_motifs:
                    motif_domain = domain
                    break

            if motif_domain is not None:
                # Calculate domain total for this animal
                animal_domain_data = df[(df["Animal_ID"] == animal_id)]
                domain_motifs_for_animal = animal_domain_data[
                    animal_domain_data["motif label_Clean"].isin(
                        motif_domains[motif_domain]
                    )
                ]
                domain_total = domain_motifs_for_animal["observed"].sum()

                # Calculate domain-wise normalized frequency
                if domain_total > 0:
                    domain_normalized_freq = row["observed"] / domain_total
                else:
                    domain_normalized_freq = 0

                domain_data.append(
                    {
                        "motif label_Clean": motif,
                        "motif size": row["motif size"],
                        "observed": row["observed"],
                        "normalized_freq": row["normalized_freq"],
                        "domain_normalized_freq": domain_normalized_freq,
                        "Animal_ID": animal_id,
                        "Timepoint": timepoint,
                        "Domain": motif_domain,
                    }
                )

        domain_datasets[timepoint] = pd.DataFrame(domain_data)

    # Export individual replicate values: one CSV per normalization type (global vs domain).
    # The cross-age domain plot (motif_analysis_domain_normalization.svg) uses domain_normalized_freq;
    # the global plot uses normalized_freq. Each file is named for the data it contains.
    long_global = pd.concat(
        [
            df[["Timepoint", "Animal_ID", "motif label_Clean", "normalized_freq"]].rename(
                columns={"motif label_Clean": "Motif"}
            )
            for df in datasets.values()
        ],
        ignore_index=True,
    )
    long_domain = pd.concat(
        [
            df[["Timepoint", "Animal_ID", "motif label_Clean", "domain_normalized_freq"]].rename(
                columns={"motif label_Clean": "Motif"}
            )
            for df in domain_datasets.values()
        ],
        ignore_index=True,
    )
    path_global = os.path.join(model_results_dir, "individual_replicates_per_animal_global.csv")
    long_global.to_csv(path_global, index=False)
    print(f"Saved individual replicates (global): {path_global}")
    path_domain = os.path.join(model_results_dir, "individual_replicates_per_animal_domain.csv")
    long_domain.to_csv(path_domain, index=False)
    print(f"Saved individual replicates (domain): {path_domain}")

    if getattr(args, "export_wide_for_prism", False):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from transform_replicates_to_wide import write_wide_for_prism
        try:
            out_global = write_wide_for_prism(path_global, value_column="normalized_freq")
            print(f"Saved wide Prism (global): {out_global}")
            out_domain = write_wide_for_prism(path_domain, value_column="domain_normalized_freq")
            print(f"Saved wide Prism (domain): {out_domain}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: wide-for-Prism export failed: {e}")

    # Calculate summary statistics for domain-wise normalization
    domain_summary_stats = {}
    for timepoint, df in domain_datasets.items():
        domain_summary_stats[timepoint] = {}

        for motif in domain_sorted_motifs:
            motif_data = df[df["motif label_Clean"] == motif]

            if len(motif_data) > 0:
                values = motif_data["domain_normalized_freq"].values

                if len(values) > 0:
                    from scipy import stats

                    mean_val = np.mean(values)
                    sem_val = stats.sem(values) if len(values) > 1 else 0
                    domain_summary_stats[timepoint][motif] = {
                        "mean": mean_val,
                        "sem": sem_val,
                        "n": len(values),
                        "values": values.tolist(),
                    }
                else:
                    domain_summary_stats[timepoint][motif] = {
                        "mean": 0,
                        "sem": 0,
                        "n": 0,
                        "values": [],
                    }
            else:
                domain_summary_stats[timepoint][motif] = {
                    "mean": 0,
                    "sem": 0,
                    "n": 0,
                    "values": [],
                }

    # Perform Kruskal-Wallis tests for domain-wise normalization
    print("Performing Kruskal-Wallis tests for domain-wise normalization...")
    domain_kw_results = perform_kruskal_wallis_tests(
        domain_datasets, domain_sorted_motifs, "domain"
    )

    # Create Figure 2 with reasonable height
    fig2, ax2 = plt.subplots(figsize=(24, 8))

    n_motifs_domain = len(domain_sorted_motifs)
    x_domain = np.arange(n_motifs_domain)

    # Plot bars with individual points and error bars
    all_bars_domain = plot_bars_with_points_and_errors(
        ax2, x_domain, domain_summary_stats, datasets_list, colors, width
    )

    # Set labels and title
    ax2.set_ylabel("Normalized Frequency (Domain-wise)", fontsize=12)
    ax2.set_title(
        "Motif Frequency Distribution - Domain-wise Normalization\n(Bars = Mean ± SEM, Points = Individual Animals)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_domain + width)
    ax2.set_xticklabels(domain_sorted_motifs, rotation=90, ha="center")
    ax2.legend()

    # Add domain separators
    for i, (start, end, domain) in enumerate(domain_boundaries[:-1]):
        separator_x = end + 0.5 + width
        ax2.axvline(x=separator_x, color="k", linestyle="--", alpha=0.7, linewidth=0.5)

    # Calculate domain-wise JSDs and add brackets at the same level (without modifying y-limits)
    for start, end, domain in domain_boundaries:
        # Get frequencies for this domain across all datasets using means
        domain_freqs = []
        domain_motifs = motif_domains[domain]

        for dataset in datasets_list:
            dataset_domain_freqs = []
            for motif in domain_motifs:
                mean_freq = domain_summary_stats[dataset][motif]["mean"]
                dataset_domain_freqs.append(mean_freq)
            domain_freqs.append(dataset_domain_freqs)

        # Calculate JSDs for this domain
        jsd_12, jsd_13, jsd_23 = calculate_distribution_jsd(domain_freqs)

        # Only add bracket annotation if we have enough timepoints
        if not (np.isnan(jsd_12) and np.isnan(jsd_13) and np.isnan(jsd_23)):
            # Calculate bracket span using domain boundaries and actual bar layout
            # Get the actual x positions of bars for this domain
            first_motif_x = start  # x position of first motif in domain
            last_motif_x = end  # x position of last motif in domain

            # Calculate bracket span from leftmost bar edge to rightmost bar edge
            bracket_start = first_motif_x - width / 2  # Left edge of first bar group
            bracket_end = last_motif_x + width + width  # Right edge of last bar group

            jsd_parts = jsd_bracket_annotation_lines(
                jsd_12, jsd_13, jsd_23, datasets_list
            )
            if jsd_parts:
                jsd_text = "\n".join(jsd_parts)
                add_bracket_annotation(fig2, ax2, bracket_start, bracket_end, jsd_text)

    # Remove manual bottom adjustment since bracket function handles this automatically
    plt.tight_layout()

    # Save figure as SVG (to cross_age subdirectory)
    domain_fig_path = os.path.join(cross_age_dir, "motif_analysis_domain_normalization.svg")
    fig2.savefig(domain_fig_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved domain normalization figure: {domain_fig_path}")
    plt.close(fig2)  # Close figure to free memory

    # Save domain-wise Kruskal-Wallis results (to cross_age subdirectory)
    domain_kw_path = os.path.join(cross_age_dir, "kruskal_wallis_domain_normalization.csv")
    domain_kw_results.to_csv(domain_kw_path, index=False)
    print(f"Saved domain Kruskal-Wallis results: {domain_kw_path}")

    # Print summary of significant results
    n_significant_domain = domain_kw_results["significant"].sum()
    sp(
        f"Domain normalization: {n_significant_domain}/{len(domain_kw_results)} motifs show significant differences (p < 0.05)"
    )

    # Create a summary of all statistical results
    sp("\nCreating statistical summary...")
    combined_results = []

    # Add global results
    for _, row in global_kw_results.iterrows():
        combined_results.append(
            {
                "Motif": row["Motif"],
                "Normalization": "Global",
                "H_statistic": row["H_statistic"],
                "p_value": row["p_value"],
                "significant": row["significant"],
                "n_groups": row["n_groups"],
            }
        )

    # Add domain results
    for _, row in domain_kw_results.iterrows():
        combined_results.append(
            {
                "Motif": row["Motif"],
                "Normalization": "Domain",
                "H_statistic": row["H_statistic"],
                "p_value": row["p_value"],
                "significant": row["significant"],
                "n_groups": row["n_groups"],
            }
        )

    combined_df = pd.DataFrame(combined_results)
    summary_path = os.path.join(cross_age_dir, "kruskal_wallis_summary.csv")
    combined_df.to_csv(summary_path, index=False)
    sp(f"Saved combined statistical summary: {summary_path}")

    jsd_frames = [pairwise_jsd_dataframe(global_freqs, datasets_list, "global", "all")]
    for start, end, domain in domain_boundaries:
        domain_motifs = motif_domains[domain]
        domain_freqs = []
        for dataset in datasets_list:
            dataset_domain_freqs = [
                domain_summary_stats[dataset][motif]["mean"] for motif in domain_motifs
            ]
            domain_freqs.append(dataset_domain_freqs)
        jsd_frames.append(
            pairwise_jsd_dataframe(domain_freqs, datasets_list, "domain", domain)
        )
    jsd_export_df = pd.concat(jsd_frames, ignore_index=True)
    jsd_export_df.insert(0, "model", model_type)

    # =============================================================================
    # STATISTICAL SUMMARY
    # =============================================================================

    sp("")
    sp("=" * 100)
    sp("COMPREHENSIVE STATISTICAL ANALYSIS")
    sp("=" * 100)

    sp("\n1. GLOBAL NORMALIZATION - Overall Distribution Comparison:")
    sp("-" * 80)
    global_pair_df = jsd_export_df[jsd_export_df["normalization"] == "global"]
    if len(global_pair_df) == 0:
        sp("Note: Not enough timepoints for JSD comparison (need at least 2)")
    else:
        for _, row in global_pair_df.iterrows():
            sp(f"{row['timepoint_a']} vs {row['timepoint_b']}: JSD = {row['jsd']:.4f}")

    sp("\n2. DOMAIN-WISE NORMALIZATION - Domain-specific Comparisons Only:")
    sp("-" * 80)
    sp("(No global comparison - only within-domain comparisons are meaningful)")
    for start, end, domain in domain_boundaries:
        domain_motifs = motif_domains[domain]
        sp(f"Domain {domain} ({len(domain_motifs)} motifs):")
        sub = jsd_export_df[
            (jsd_export_df["normalization"] == "domain")
            & (jsd_export_df["domain"] == domain)
        ]
        for _, row in sub.iterrows():
            sp(f"  {row['timepoint_a']} vs {row['timepoint_b']}: JSD = {row['jsd']:.4f}")

    sp("\n3. DOMAIN COMPOSITION:")
    sp("-" * 80)
    for domain in sorted_domains:
        motifs_in_domain = motif_domains[domain]
        sp(f"Domain {domain}: {len(motifs_in_domain)} motifs")
        for motif in motifs_in_domain:
            sp(f"  {motif}")

    sp("\n4. SAMPLE SIZES:")
    sp("-" * 80)
    for timepoint in datasets_list:
        n_animals = len(set(datasets[timepoint]["Animal_ID"]))
        sp(f"{timepoint}: {n_animals} animals")

        # Show sample sizes for each domain
        for domain in sorted_domains:
            domain_motifs = motif_domains[domain]
            domain_data = domain_datasets[timepoint]
            animals_with_domain_data = set()
            for motif in domain_motifs:
                motif_animals = domain_data[domain_data["motif label_Clean"] == motif][
                    "Animal_ID"
                ].unique()
                animals_with_domain_data.update(motif_animals)
            sp(f"  Domain {domain}: {len(animals_with_domain_data)} animals with data")

    sp("\n5. KRUSKAL-WALLIS TEST RESULTS:")
    sp("-" * 80)
    sp(
        f"Global normalization: {n_significant_global}/{len(global_kw_results)} motifs show significant differences"
    )
    sp(
        f"Domain normalization: {n_significant_domain}/{len(domain_kw_results)} motifs show significant differences"
    )

    sp("\nMost significant motifs (Global normalization, p < 0.01):")
    if len(global_kw_results) > 0:
        significant_global = global_kw_results[global_kw_results["p_value"] < 0.01].copy()
        if len(significant_global) > 0:
            # Sort by p_value using argsort for better type compatibility
            sort_idx = significant_global["p_value"].argsort()
            significant_global_sorted = significant_global.iloc[sort_idx]
            for _, row in significant_global_sorted.head(10).iterrows():
                sp(
                    f"  {row['Motif']}: H = {row['H_statistic']:.3f}, p = {row['p_value']:.6f}"
                )
        else:
            sp("  No motifs with p < 0.01")
    else:
        sp("  No results available")

    sp("\nMost significant motifs (Domain normalization, p < 0.01):")
    if len(domain_kw_results) > 0:
        significant_domain = domain_kw_results[domain_kw_results["p_value"] < 0.01].copy()
        if len(significant_domain) > 0:
            # Sort by p_value using argsort for better type compatibility
            sort_idx = significant_domain["p_value"].argsort()
            significant_domain_sorted = significant_domain.iloc[sort_idx]
            for _, row in significant_domain_sorted.head(10).iterrows():
                sp(
                    f"  {row['Motif']}: H = {row['H_statistic']:.3f}, p = {row['p_value']:.6f}"
                )
        else:
            sp("  No motifs with p < 0.01")
    else:
        sp("  No results available")

    sp("")
    sp("=" * 100)
    sp("INTERPRETATION")
    sp("=" * 100)
    sp(
        "• Global Normalization: Compares entire frequency distributions across all motifs"
    )
    sp(
        "• Domain-wise Normalization: Compares frequency distributions only within each motif complexity level"
    )
    sp(
        "• JSD values: 0 = identical distributions, 1 = maximally different distributions"
    )
    sp(
        "• Lower JSD = more similar temporal patterns, Higher JSD = more divergent temporal patterns"
    )
    sp(
        "• Each domain represents motifs of the same complexity (number of brain regions involved)"
    )
    sp("• Domain-wise analysis isolates complexity-specific developmental patterns")
    sp("• Error bars represent SEM across individual animals")
    sp("• Individual points show data from each animal")
    sp("• Kruskal-Wallis test: Non-parametric test for differences between timepoints")
    sp(
        "• Significant Kruskal-Wallis results (p < 0.05) indicate developmental changes in motif usage"
    )

    run_metrics_df = pd.DataFrame(
        [
            {
                "model": model_type,
                "timepoints_present": ";".join(datasets_list),
                "n_significant_global": int(n_significant_global),
                "n_significant_domain": int(n_significant_domain),
                "n_motifs_kw_global": len(global_kw_results),
                "n_motifs_kw_domain": len(domain_kw_results),
            }
        ]
    )

    # New per-age plot generation
    if output_mode in ['per_age', 'both']:
        sp("")
        sp("=" * 100)
        sp(f"GENERATING PER-AGE INDIVIDUAL ANIMAL PLOTS ({model_type.upper()} MODEL)")
        sp("=" * 100)
        generate_per_age_plots(datasets, all_motifs, RESULTS_DIR, "global")
        generate_per_age_plots(datasets, all_motifs, RESULTS_DIR, "domain")
        sp(
            "Per-age SVG and motif_summary CSVs saved under each age subdirectory (p3, p12, p20, p60 as applicable)."
        )

    header_lines = [
        "=" * 80,
        f"Model: {model_type}",
        f"Timepoints: {', '.join(datasets_list)}",
        f"Unique motifs (ordered): {len(all_motifs)}",
        "",
    ]
    for name, df in datasets.items():
        header_lines.append(
            f"{name}: {df['Animal_ID'].nunique()} animals, {len(df['motif label_Clean'].unique())} unique motifs"
        )
    header_lines.append("")
    summary_text = "\n".join(header_lines + model_summary_lines)
    write_mirrored_model_outputs(
        model_results_dir, jsd_export_df, run_metrics_df, summary_text
    )

    print(f"\n📁 All {model_type} model results saved to: {RESULTS_DIR}")
    
    # Restore original RESULTS_DIR for next iteration
    RESULTS_DIR = original_results_dir

print(f"\n📁 All results saved to: {RESULTS_DIR}")
print("   - Uniform model: {}/uniform/".format(RESULTS_DIR))
print("   - Region-specific model: {}/region_specific/".format(RESULTS_DIR))
print("📊 SVG figures can be edited in vector graphics software")
print("📈 CSV files contain detailed statistical results for further analysis")
