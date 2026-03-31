import argparse
import os
import re
import sympy
import csv
import shutil
import numpy as np
import pandas as pd
from sympy import symbols, Product, Array, N, latex
from sympy.printing import latex
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import patches
from pathlib import Path
import seaborn as sn
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from scipy.stats import friedmanchisquare, kruskal, binomtest, binom
from sklearn.cluster import KMeans, k_means
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize 
import itertools
from adjustText import adjust_text
import multiprocessing as mp
import itertools
import upsetplot as up
from statsmodels.stats.multitest import fdrcorrection
from collections import Counter

# Argument parser setup
parser = argparse.ArgumentParser(description="Process NBCM data")
parser.add_argument("-o","--out_dir", type=str, required=True, help="Output directory for saving results")
parser.add_argument("-s", "--sample_name", type=str, required=True, help="Sample name")
parser.add_argument("-d","--data_file", type=str, required=True, help="Path to the input nbcm.csv file")
parser.add_argument("-a","--alpha", type=float, default=0.05, help="Significance threshold for Bonferroni correction (default: 0.05)")
parser.add_argument(
    "-i", "--injection_umi_min", 
    type=float, 
    default=1, 
    help="Sets a threshold for minimum 'inj' UMI values. Rows where 'inj' is below this value will be removed. Default: 1."
)
parser.add_argument(
    "-t","--min_target_count", type=float, default=10,
    help="Minimum UMI count required in at least one target area. Rows not meeting this are excluded."
)
parser.add_argument(
    "-r","--min_body_to_target_ratio", type=float, default=10,
    help="Minimum fold-difference between 'inj' value and the highest target count. Rows not meeting this are excluded."
)
parser.add_argument("-u","--target_umi_min", type=float, default=2, help="Sets a threshold filter for target area UMI counts where smaller values will be set to zero. Typically for noise reduction of single UMI values in targets. (default: 2)")
parser.add_argument(
    "-l",
    "--labels",
    type=str,
    help="Comma-separated column labels (e.g., 'target1,target2,target3,target-neg-bio'). These need to match your NBCM columns, and you MUST use the exact label 'neg' in any negative control column and 'inj' in any injection column"
)
parser.add_argument("-A","--special_area_1", type=str, required=False, help="One of your favorite target areas")
parser.add_argument("-B","--special_area_2", type=str, required=False, help="Another of your favorite target areas to compare to the first")
parser.add_argument(
    "-f", "--apply_outlier_filtering", 
    action="store_true", 
    help="Enable outlier filtering (Step 7) using mean + 2*std deviation."
)
parser.add_argument(
    "--force_user_threshold",
    action="store_true",
    help="If set, override all automatic thresholding and use the user-defined target_umi_min."
)
parser.add_argument(
    "--is-anchor-model",
    action="store_true",
    help="Mark this run as the anchor/baseline model. Saves probabilities and correlation matrix for use by other cohorts."
)
parser.add_argument(
    "--anchor-model-file",
    type=str,
    default=None,
    help="Path to anchor model's Region-specific_Probabilities_N0based.csv. When provided, uses anchor's probabilities with local N0."
)
parser.add_argument(
    "--anchor-correlation-file",
    type=str,
    default=None,
    help="Path to anchor model's Conditional_Probability_Matrix.csv for correlated binomial model."
)
parser.add_argument(
    "--model-type",
    type=str,
    choices=["uniform", "region_specific", "correlated", "empirical", "smoothed_empirical", 
             "max_entropy", "hierarchical_correlations", "negative_binomial", "zero_inflated",
             "bayesian_hierarchical", "ml_nonparametric", "all"],
    default="all",
    help="Which probability model(s) to run. Default: all"
)
parser.add_argument(
    "--smoothing-alpha",
    type=float,
    default=1.0,
    help="Smoothing parameter α for smoothed_empirical model (default: 1.0)"
)
parser.add_argument(
    "--skip-sections",
    type=str,
    default=None,
    help="Comma-separated list of sections to skip: visualizations,clustering,heatmaps"
)
parser.add_argument(
    "--illustrator-volcano-dir",
    type=str,
    default=None,
    help="When set, also save an illustrator-ready SVG of the uniform effect significance plot to this directory (fixed axes, no title/axis/tick labels, Helvetica text)."
)
parser.add_argument(
    "--illustrator-report-ranges-only",
    action="store_true",
    help="With --illustrator-volcano-dir: append this run's volcano data range to _data_ranges.csv and do not save the illustrator SVG (used for computing uniform limits)."
)
parser.add_argument(
    "--illustrator-xlim",
    type=float,
    nargs=2,
    default=None,
    metavar=("XMIN", "XMAX"),
    help="X-axis limits for illustrator volcano (e.g. -4 4). Used with --illustrator-volcano-dir."
)
parser.add_argument(
    "--illustrator-ylim",
    type=float,
    nargs=2,
    default=None,
    metavar=("YMIN", "YMAX"),
    help="Y-axis limits for illustrator volcano (e.g. 0 10). Used with --illustrator-volcano-dir."
)

# Parse arguments
args = parser.parse_args()

# Parse skip-sections argument into a set for easy lookup
skip_sections = set()
if getattr(args, 'skip_sections', None):
    skip_sections = set(s.strip().lower() for s in args.skip_sections.split(","))
    print(f"⚠️ Skip sections enabled: {skip_sections}")

# Handle both string and list input for labels
if isinstance(args.labels, str):
    args.labels = args.labels.strip('"').strip("'")  # Strip quotes
    sample_labels = [label.strip().strip('"').strip("'") for label in args.labels.split(",")]
elif isinstance(args.labels, list):
    sample_labels = [label.strip().strip('"').strip("'") for label in args.labels]
else:
    sample_labels = None

# Define variables dynamically from arguments
out_dir = args.out_dir
sample_name = args.sample_name
data_file = args.data_file
alpha = args.alpha
min_target_count = args.min_target_count
min_body_to_target_ratio = args.min_body_to_target_ratio
target_umi_min = args.target_umi_min
special_area_1 = args.special_area_1
special_area_2 = args.special_area_2



# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Safety check: Ensure out_dir is not just a parent directory (e.g., 02_output/p3)
# This prevents accidentally saving files to the wrong location
if out_dir and not any(part.startswith(('01.', '02.', '03.', '04.', '05.')) for part in Path(out_dir).parts):
    import warnings
    warnings.warn(f"Warning: out_dir '{out_dir}' does not appear to contain a parameterization subdirectory. "
                  f"Files should be saved to a parameterization-specific directory like "
                  f"'02_output/p3/01.minimal_filter_parameters_i1_r1_t1_u2'")

#This switch is for excluding some columns. See line 254
full_data = True
motif_join = '+'

def compute_umi_total_counts(matrix: np.ndarray, region_labels: list, out_path: str = None):
    """
    Computes the total summed UMI counts for each brain region (column-wise sum).

    Args:
        matrix (np.ndarray): Filtered matrix (cells × regions).
        region_labels (list): List of region names corresponding to matrix columns.
        out_path (str): Optional path to write CSV of total counts.

    Returns:
        dict: Mapping of region name to summed UMI counts.
    """
    umi_total_counts = {region: float(np.sum(matrix[:, idx])) for idx, region in enumerate(region_labels)}

    print(f"🔍 UMI total counts (summed per region): {umi_total_counts}")

    if out_path:
        df = pd.DataFrame({
            'Region': region_labels,
            'UMI_Sum': [umi_total_counts[r] for r in region_labels]
        })
        df.to_csv(out_path, index=False)
        print(f"💾 UMI total counts saved to: {out_path}")

    return umi_total_counts

def calculate_projections_from_matrix(matrix, sample_labels, out_path=None):
    """
    Calculate projection metrics per region:
    - column_counts: Number of cells projecting to each region (binary presence).
    - total_projections: Sum of column_counts, i.e., total number of projection events.
    
    If out_path is provided, saves both dictionaries as a CSV.
    """
    # Binary projection presence count (how many cells project to each region)
    column_counts = {region: np.count_nonzero(matrix[:, idx]) for idx, region in enumerate(sample_labels)}

    # Total projection events (presence-based)
    total_projections = sum(column_counts.values())

    print(f"🔍 Column counts (neurons per region): {column_counts}")
    print(f"🔍 Total projections (Sums column-wise counts): {total_projections}")

    # Optional CSV output
    if out_path:
        df = pd.DataFrame({
            'Region': sample_labels,
            'Cell_Counts': [column_counts[r] for r in sample_labels],            
        })
        df.to_csv(out_path, index=False)
        print(f"💾 Projection summary saved to: {out_path}")

    return column_counts, total_projections

def calculate_total_projections(projections):
    return sum(projections.values())

def solve_for_roots(projections, observed_cells):
    N0, k = symbols('N_0 k')
    m = len(projections) - 1
    s = Array(list(projections.values()))
    pi = (1 - Product((1 - (s[k]/N0)), (k, 0, m)).doit())
    soln = sympy.solve(pi * N0 - observed_cells)
    roots = [N(x).as_real_imag()[0] for x in soln]
    return roots, pi

def save_latex_expression(expression, title, filename):
    """
    Properly renders and saves a LaTeX equation image.
    """
    latex_output = r"$" + latex(expression) + r"$"  # Use single-dollar format
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, latex_output, fontsize=16, va='center', ha='center', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)  # Remove borders

    plt.title(title, fontsize=16)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_probabilities(projections, total_projections):
    return {region: (count / total_projections) for region, count in projections.items()}

def load_anchor_model(anchor_file, anchor_corr_file=None):
    """
    Load anchor model probabilities and optional correlation matrix.
    
    Args:
        anchor_file: Path to anchor model's Region-specific_Probabilities_N0based.csv
        anchor_corr_file: Optional path to Conditional_Probability_Matrix.csv
        
    Returns:
        anchor_probs_dict: Dictionary mapping region names to probabilities
        anchor_n0: The N0 value from the anchor model
        anchor_corr: Conditional probability matrix (DataFrame) or None
    """
    anchor_probs = pd.read_csv(anchor_file)
    anchor_probs_dict = dict(zip(anchor_probs['Region'], anchor_probs['Probability']))
    anchor_n0 = anchor_probs['N0'].iloc[0]
    
    anchor_corr = None
    if anchor_corr_file and os.path.exists(anchor_corr_file):
        anchor_corr = pd.read_csv(anchor_corr_file, index_col=0)
    
    return anchor_probs_dict, anchor_n0, anchor_corr

def binomial_test(value, total, probability):
    return binomtest(value, n=total, p=probability).pvalue

def normalize_rows(matrix):
    """
    Normalize each row by its maximum value.
    - If the max value is 0, the row remains unchanged.
    - Prevents division errors.

    Args:
        matrix (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Normalized matrix.
    """
    if matrix.shape[0] == 0:  # 🚨 If empty, return immediately
        print("⚠ WARNING: Normalized matrix is empty. Skipping normalization.")
        return matrix

    return np.apply_along_axis(lambda x: x / np.amax(x) if np.amax(x) > 0 else x, axis=1, arr=matrix)
def clean_and_filter(matrix, sample_labels, target_umi_min, injection_umi_min,
                     apply_outlier_filtering=False, force_user_threshold=False):
    """
    Clean and filter the matrix:
    - Remove header row and barcode column
    - Remove zero-projection rows
    - Remove rows where any 'neg' column has a nonzero value
    - Remove rows where any value >= the corresponding 'inj' column value
    - Apply UMI threshold and optionally remove high UMI outliers
    """
    
    # 🚨 Step 1: Remove headers & barcode column
    matrix = matrix[1:, 1:]
    print(f"🔍 Step 1: Removed headers & barcode. Shape: {matrix.shape}")

    # 🚨 Step 2: Remove rows with all zeros
    matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
    print(f"🔍 Step 2: Removed zero-projection rows. Shape: {matrix.shape}")

    # 🚨 Step 2b: Remove rows where no target regions are > min_target_count
    non_neg_inj_cols = [i for i, label in enumerate(sample_labels) if label not in ["neg", "inj"]]
    if non_neg_inj_cols:
        target_max = np.nanmax(matrix[:, non_neg_inj_cols], axis=1)
        matrix = matrix[target_max >= min_target_count]
        print(f"🔍 Step 2b: Removed rows with no targets > {min_target_count}. Shape: {matrix.shape}")
    else:
        print("⚠ WARNING: No valid target columns found (excluding 'inj' and 'neg'). Skipping Step 2b.")

    # 🚨 Step 3: Remove rows where any value >= the corresponding 'inj' column value
    if "inj" in sample_labels:
        inj_col_idx = sample_labels.index("inj")
        inj_values = matrix[:, inj_col_idx]
        print(f"🔍 Step 3: 'inj' column detected at index {inj_col_idx}")
        print(f"🔍 Step 3: Injection Site values = min: {np.min(inj_values)}, max: {np.max(inj_values)}, mean: {np.mean(inj_values)}")
        mask = (
            np.all(matrix[:, :inj_col_idx] < inj_values[:, None], axis=1) &
            np.all(matrix[:, inj_col_idx + 1:] < inj_values[:, None], axis=1)
        )
        matrix = matrix[mask]
        print(f"🔍 Step 3: Removed rows with values >= 'inj'. Shape: {matrix.shape}")
    else:
        print("⚠ WARNING: 'inj' column not found in sample labels. Skipping this step.")

    # 🚨 Step 3b: Remove rows where 'inj' value is below injection_umi_min
    if "inj" in sample_labels:
        inj_col_idx = sample_labels.index("inj")
        matrix = matrix[matrix[:, inj_col_idx] >= injection_umi_min]
        print(f"✅ CHECK THIS Step 3b: Removed rows with 'inj' < {injection_umi_min}. Shape: {matrix.shape}")
    else:
        print("⚠ WARNING: 'inj' column not found. Skipping Step 3b.")

    # 🚨 Step 3c: Remove rows where inj < (max target value * ratio threshold)
    if "inj" in sample_labels:
        inj_col_idx = sample_labels.index("inj")
        non_neg_inj_cols = [i for i, label in enumerate(sample_labels) if label not in ["neg", "inj"]]
        inj_values = matrix[:, inj_col_idx]
        if non_neg_inj_cols:
            max_target_values = np.nanmax(matrix[:, non_neg_inj_cols], axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                valid_mask = inj_values >= (max_target_values * min_body_to_target_ratio)
                valid_mask = np.nan_to_num(valid_mask, nan=False)
            matrix = matrix[valid_mask.astype(bool)]
            print(f"🔍 Step 3c: Removed rows where 'inj' < max(targets) * {min_body_to_target_ratio}. Shape: {matrix.shape}")
        else:
            print("⚠ WARNING: No valid target columns found (excluding 'inj' and 'neg'). Skipping Step 3c.")
    else:
        print("⚠ WARNING: 'inj' column not found. Skipping Step 3c.")

    # 🚨 Step 4: Extract max value from 'neg' column **before removing rows with neg > 0**
    neg_columns = [i for i, label in enumerate(sample_labels) if "neg" in label.lower()]
    if neg_columns:
        neg_values = matrix[:, neg_columns]
        neg_values = neg_values[~np.isnan(neg_values)]
        if neg_values.size > 0:
            max_neg_value = np.nanmax(neg_values)
        else:
            max_neg_value = target_umi_min
            print("⚠ WARNING: 'neg' column contains only NaN values. Using argparse default.")
        print(f"🚨 Stored max value from 'neg' column: {max_neg_value}")
    else:
        max_neg_value = target_umi_min
        print("⚠ WARNING: 'neg' column not found. Using argparse default UMI threshold.")

    # 🚨 Step 5: Remove rows where any 'neg' column has a nonzero value
    neg_columns = [i for i, label in enumerate(sample_labels) if "neg" in label.lower()]
    if neg_columns:
        matrix = matrix[np.all(matrix[:, neg_columns] == 0, axis=1)]
    print(f"🔍 Step 5: Removed rows with 'neg' > 0. Shape: {matrix.shape}")

    # 🚨 Step 6a: Dynamically calculate the noise threshold value using histogram elbow
    non_neg_inj_cols = [i for i, label in enumerate(sample_labels) if label not in ["neg", "inj"]]
    if non_neg_inj_cols:
        all_target_vals = matrix[:, non_neg_inj_cols].flatten()
        non_zero_target_vals = all_target_vals[all_target_vals > 0]
        if non_zero_target_vals.size > 0:
            from scipy.stats import gaussian_kde
            log_vals = np.log10(non_zero_target_vals + 1e-5)
            # KDE requires at least 2 data points
            if len(log_vals) >= 2:
                density = gaussian_kde(log_vals)
                xs = np.linspace(log_vals.min(), log_vals.max(), 1000)
                ys = density(xs)
                d2 = np.gradient(np.gradient(ys))
                elbow_idx = np.argmin(d2)
                elbow_log_value = xs[elbow_idx]
                dynamic_threshold = 10 ** elbow_log_value
                print(f"🔍 Step 6a: Dynamic noise threshold estimated via elbow method: {dynamic_threshold:.4f}")
            else:
                # Use default threshold if insufficient data for KDE
                dynamic_threshold = target_umi_min
                print(f"⚠ WARNING: Insufficient data points ({len(log_vals)}) for KDE threshold estimation. Using user defined or argparse default value.")
        else:
            dynamic_threshold = target_umi_min
            print("⚠ WARNING: No nonzero target values found for threshold estimation. Using user defined or argparse default value.")
    else:
        dynamic_threshold = target_umi_min
        print("⚠ WARNING: No target columns found for dynamic thresholding. Using user defined or argparse default value.")

    print(f"Calculated Threshold: {dynamic_threshold}")
    print(f"User defined or argparse minimum default(2): {target_umi_min}")
    print(f"Max Negative Control Value: {max_neg_value}")

    # 🚨 Step 6b: Choose UMI threshold
    if force_user_threshold:
        final_umi_threshold = target_umi_min
        print(f"⚠️ Step 6b: Forcing user-defined UMI threshold: {final_umi_threshold}")
    else:
        final_umi_threshold = max(target_umi_min, max_neg_value, dynamic_threshold)
        print(f"✅ Step 6b: Looking for the maximum value: user= ({target_umi_min}), MAXneg= ({max_neg_value:.4f}), "
              f"and dynamic= ({dynamic_threshold:.4f}) ➜ Final threshold choice: {final_umi_threshold:.4f}")

    # 🚨 Step 6c: Remove rows that became all zeros after thresholding
    matrix[matrix < final_umi_threshold] = 0
    num_zero_after_threshold = np.sum(np.sum(matrix > 0, axis=1) == 0)
    print(f"✅ CHECK THIS Step 6c: Applied threshold ({final_umi_threshold}). New zero rows: {num_zero_after_threshold}")
    
    # 🚨 Step 6d: Remove rows that became all zeros after thresholding
    matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
    print(f"🔍 Step 6d: Removed new zero rows. Shape: {matrix.shape}")

    # 🚨 Step 7: Apply optional high-UMI outlier filtering
    if apply_outlier_filtering:
        non_neg_inj_cols = [i for i, label in enumerate(sample_labels) if label not in ["neg", "inj"]]
        if non_neg_inj_cols:
            mean_values = np.mean(matrix[:, non_neg_inj_cols], axis=0)
            std_values = np.std(matrix[:, non_neg_inj_cols], axis=0)
            upper_threshold = mean_values + 2 * std_values
            filtered_matrix = []
            for row in matrix:
                if all(row[i] <= upper_threshold[idx] for idx, i in enumerate(non_neg_inj_cols)):
                    filtered_matrix.append(row)
            matrix = np.array(filtered_matrix)
        print(f"✅ CHECK THIS Step 7: Removed high-UMI outliers where value was > (mean+2*StdDev). Shape: {matrix.shape}")

    return matrix, max_neg_value, final_umi_threshold

def compute_motif_probabilities(pe_num, total_regions):
    """
    Compute probabilities for each possible motif type.
    
    Args:
    - pe_num (float): Probability of an edge (p_e).
    - total_regions (int): Number of brain regions.

    Returns:
    - motif_probs (dict): Dictionary with motif type as key and probability as value.
    """
    # Ensure pe_num is a native float
    pe_num = float(pe_num)

    # Compute motif probabilities using safe probability mass function (PMF)
    motif_probs = {
        n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n))
        for n in range(1, total_regions + 1)
    }
    
    return motif_probs


### Main Calculations

# Load barcodes
    """
    Note that you can change this delimiter to ',' if you are using a custom CSV file rather than the core provided TSV.
    """
barcodematrix = np.genfromtxt(data_file, delimiter='\t')
barcodematrix = np.array(barcodematrix, dtype=np.float64)
print("Barcode Matrix Shape:", barcodematrix.shape)

#check zeros before filtering
num_zero_before = np.sum(np.sum(barcodematrix > 0, axis=1) == 0)
print(f"🔍 BEFORE ANY FILTERING: Neurons with Zero Projections: {num_zero_before}")

# Perform cleaning and filtering
apply_outlier_filtering = args.apply_outlier_filtering  # Get argument value

filtered_matrix, max_neg_value, final_umi_threshold = clean_and_filter(
    barcodematrix,
    sample_labels,
    target_umi_min,
    args.injection_umi_min,
    args.apply_outlier_filtering,
    force_user_threshold=args.force_user_threshold
)

if args.force_user_threshold:
    print(f"⚠️ User-forced threshold in effect: {final_umi_threshold}")
else:
    print(f"✅ CHECK THIS: Final UMI Threshold Used (max of user, neg, dynamic): {final_umi_threshold:.4f}")

print("🔍 Filtered Matrix Shape:", filtered_matrix.shape)

# Calculate mean injection value after all filtering is complete
if "inj" in sample_labels:
    inj_col_idx = sample_labels.index("inj")
    mean_inj_value_filtered_cells_only = np.mean(filtered_matrix[:, inj_col_idx])
    print(f"🔍 Mean injection value (filtered cells only): {mean_inj_value_filtered_cells_only:.4f}")
else:
    mean_inj_value_filtered_cells_only = np.nan
    print("⚠️ WARNING: 'inj' column not found. Cannot calculate mean injection value.")

# Drop "neg" and "inj" columns from the filtered matrix
neg_inj_columns = [i for i, label in enumerate(sample_labels) if "neg" in label.lower() or label == "inj"]
if neg_inj_columns:
    filtered_matrix = np.delete(filtered_matrix, neg_inj_columns, axis=1)
    print(f"Dropped 'neg' and 'inj' columns at indices: {neg_inj_columns}.")
else:
    print("🚨 No 'neg' or 'inj' columns found. Skipping column removal.")

# Update the columns list to match the remaining matrix columns
columns = [label for i, label in enumerate(sample_labels) if i not in neg_inj_columns]
print(f"✅ CHECK THIS: Columns remaining for analyis: {columns}")

# Normalize rows of the filtered matrix (AFTER dropping columns)
normalized_matrix = normalize_rows(filtered_matrix)
print(f"🔍 Normalized Matrix Shape: {normalized_matrix.shape}")

# 🚨 Final Step: Remove rows with all zeros after normalization
normalized_matrix = normalized_matrix[np.sum(normalized_matrix > 0, axis=1) > 0]
print(f"🚨 Final Step: Removed all-zero rows post-normalization. Shape: {normalized_matrix.shape}")

# Recalculate Observed Cells after all filtering steps
observed_cells = normalized_matrix.shape[0]  # Update Observed Cells count
print(f"🔍 Updated Observed Cells: {observed_cells}")

# Verify alignment before saving
assert normalized_matrix.shape[1] == len(columns), (
    f"Mismatch: Normalized matrix columns {normalized_matrix.shape[1]}, headers {len(columns)}"
)

# Save the filtered matrix to CSV for future analysis in the script
filtered_matrix_file = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Filtered_Matrix.csv"))
pd.DataFrame(filtered_matrix, columns=columns).to_csv(filtered_matrix_file, index=False, float_format="%.8f")
print(f"💾 Filtered matrix saved to: 📂 {filtered_matrix_file}")

# Safely compute and save UMI total counts
umi_counts_outfile = os.path.join(out_dir, f"{sample_name}_UMI_Total_Counts.csv")

try:
    umi_total_counts = compute_umi_total_counts(filtered_matrix, columns, out_path=umi_counts_outfile)

    if not isinstance(umi_total_counts, dict):
        raise TypeError("compute_umi_total_counts did not return a dictionary.")

    missing_keys = [col for col in columns if col not in umi_total_counts]
    if missing_keys:
        print(f"⚠️ Warning: Missing UMI counts for regions: {missing_keys}")
    else:
        print(f"✅ All expected UMI counts present: {list(umi_total_counts.keys())}")

    print(f"🔍 UMI total counts (summed per region): {umi_total_counts}")
    print(f"💾 UMI total counts saved to: {umi_counts_outfile}")

except Exception as e:
    print(f"❌ Error during UMI total counts computation or saving: {e}")
    umi_total_counts = {col: 0.0 for col in columns}  # fallback to avoid pipeline crash


# Save the normalized matrix to CSV for future analysis in the script
normalized_matrix_file = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Normalized_Matrix.csv"))
pd.DataFrame(normalized_matrix, columns=columns).to_csv(normalized_matrix_file, index=False, float_format="%.8f")
print(f"💾 Normalized matrix saved to: 📂 {normalized_matrix_file}")

# Calculate projections dynamically from the filtered matrix
"""
    See the associated function
"""
projections, total_projections = calculate_projections_from_matrix(normalized_matrix, columns)


# Solve for N0
roots, pi = solve_for_roots(projections, observed_cells)
print("🔍 All Roots for N0:", roots)  # Print all calculated roots

# Filter for real, positive roots that are also greater than observed_cells
valid_N0 = [root for root in roots if root.is_real and root > observed_cells]

if valid_N0:
    # Choose the largest valid N0 (assuming overestimation is safer)
    N0_value = max(valid_N0)  
    print(f"🚨 Selected N0: {N0_value}, which is greater than observed_cells ({observed_cells}).")
else:
    raise ValueError(f"🚨 No valid positive real root found for N0 that is greater than observed_cells ({observed_cells}).")


simplified_pi = sympy.simplify(pi)
print("Simplified Pi:", simplified_pi)

# Save LaTeX representation of simplified Pi
save_latex_expression(simplified_pi, "Simplified Pi Visualization", os.path.normpath(os.path.join(out_dir, f"{sample_name}_Simplified_Pi.png")))

# Calculate probabilities
psdict = calculate_probabilities(projections, total_projections)
print("Region-specific Probabilities:", psdict)

# Define symbolic variable for p_e
pe = symbols('p_e')

# Solve for symbolic p_e using N0 (estimated population) instead of total_projections
# This is the mathematically correct equation: (1 - (1 - pₑ)^n) × N₀ = N_obs
pe_solutions = sympy.solve((1 - (1 - pe)**len(projections)) * N0_value - observed_cells, pe, force=True)

# Extract only real solutions within (0,1)
#valid_symbolic_solutions = [sol.evalf() for sol in pe_solutions if sol.is_real and 0 < sol < 1]

#Debug to print solutions before the possible failures in the next step
print(f"🔍 Raw pe_solutions: {pe_solutions}")

#Extract only real solutions within (0,1) updated to handle complex solution evaluations
valid_symbolic_solutions = []
for sol in pe_solutions:
    try:
        if sol.is_real:
            val = float(sol.evalf())
            if 0 < val < 1:
                valid_symbolic_solutions.append(val)
    except Exception as e:
        print(f"⚠️ Skipping symbolic solution {sol} due to error: {e}")


# Compute empirical p_e
pe_empirical = np.mean(list(psdict.values()))

# Pick the best estimate: FIRST valid symbolic solution or fallback to empirical
#pe_num = valid_symbolic_solutions[0] if valid_symbolic_solutions else pe_empirical

# Pick the best estimate: AVERAGE valid symbolic solution or fallback to empirical
pe_num = np.mean(valid_symbolic_solutions) if valid_symbolic_solutions else pe_empirical


# Ensure pe_num is within (0,1), otherwise warn the user
if not (0 < pe_num < 1):
    print(f"⚠ WARNING: Selected p_e = {pe_num}, but it is outside (0,1). Check your computations.")

# Print debug information
print(f"🔍 Symbolic solutions: {pe_solutions}")
print(f"🔍 Valid p_e symbolic solution(s): {valid_symbolic_solutions}")
print(f"🔍 Empirical p_e solution: {pe_empirical}")
print(f"🚨 Numeric p_e being used: {pe_num}, computed as mean valid symbolic solution if one exists else uses emperical solution.")

# ============================================================================
# METHOD 1: UNIFORM EDGE PROBABILITY MODEL (pₑ-based)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: UNIFORM EDGE PROBABILITY MODEL (pₑ-based)")
print("="*80)

# Define total_regions BEFORE computing motif probabilities
total_regions = len(columns)  # Number of regions after filtering

# Compute motif probabilities using uniform pₑ (Ensuring n starts from 1, since n=0 isn't meaningful here)
motif_probs_uniform = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n))
    for n in range(1, total_regions + 1)  # Start at 1
}

# Normalize probabilities to ensure they sum to exactly 1
total_motif_prob_uniform = sum(motif_probs_uniform.values())

if total_motif_prob_uniform > 0:
    motif_probs_uniform = {k: v / total_motif_prob_uniform for k, v in motif_probs_uniform.items()}

# Debugging print statements
print(f"🔍 [UNIFORM MODEL] Total Motif Probability Before Normalization: {total_motif_prob_uniform}")
print(f"🔍 [UNIFORM MODEL] Total Motif Probability After Normalization (must sum to 1): {sum(motif_probs_uniform.values())}")

# Final check: Ensure sum is 1
if not np.isclose(float(sum(motif_probs_uniform.values())), 1, atol=0.01):
    print(f"🚨 WARNING: [UNIFORM MODEL] Motif probabilities sum to {sum(motif_probs_uniform.values())}, not 1.")

# Keep for backward compatibility
motif_probs = motif_probs_uniform

# ============================================================================
# METHOD 2: REGION-SPECIFIC PROBABILITY MODEL (Old Pipeline Method)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: REGION-SPECIFIC PROBABILITY MODEL (Old Pipeline Method)")
print("="*80)

# Calculate region-specific probabilities using N0 (like old pipeline)
# Old pipeline: p[i] = N_i / N_total
# Where N_i = number of neurons projecting to region i
#       N_total = estimated total population (N0)
print(f"🔍 [REGION-SPECIFIC MODEL] Calculating probabilities using N0 = {N0_value}")
psdict_region_specific = {region: (count / float(N0_value)) for region, count in projections.items()}
print(f"🔍 [REGION-SPECIFIC MODEL] Region-specific probabilities (p_i = N_i / N0): {psdict_region_specific}")

# Verify probabilities sum to reasonable value (should be < 1 since neurons can project to multiple regions)
total_region_prob = sum(psdict_region_specific.values())
print(f"🔍 [REGION-SPECIFIC MODEL] Sum of region probabilities: {total_region_prob} (expected > 1 since neurons can project to multiple regions)")

# ============================================================================
# ANCHOR MODEL LOGIC: Load external probabilities if provided
# ============================================================================
anchor_probs_loaded = None
anchor_n0_loaded = None
anchor_corr_loaded = None

if getattr(args, 'anchor_model_file', None):
    print("\n" + "="*80)
    print("ANCHOR MODEL MODE: Loading external probabilities from anchor dataset")
    print("="*80)
    anchor_probs_loaded, anchor_n0_loaded, anchor_corr_loaded = load_anchor_model(
        args.anchor_model_file,
        getattr(args, 'anchor_correlation_file', None)
    )
    print(f"🔍 [ANCHOR MODEL] Loaded anchor probabilities from: {args.anchor_model_file}")
    print(f"🔍 [ANCHOR MODEL] Anchor N0: {anchor_n0_loaded}, Local N0: {N0_value}")
    print(f"🔍 [ANCHOR MODEL] Anchor probabilities: {anchor_probs_loaded}")
    if anchor_corr_loaded is not None:
        print(f"🔍 [ANCHOR MODEL] Loaded correlation matrix from: {args.anchor_correlation_file}")
    print(f"🔍 [ANCHOR MODEL] Expected counts will use: Anchor_prob × Local_N0 ({N0_value})")
    print("="*80 + "\n")

# Determine which probabilities to use for expected count calculations
# If anchor model is provided, use anchor probabilities; otherwise use local
psdict_for_expected = anchor_probs_loaded if anchor_probs_loaded else psdict_region_specific
corr_matrix_for_expected = anchor_corr_loaded  # Will be None if not using anchor or not provided

def compute_motif_probabilities_region_specific(motif_labels, region_probs_dict, region_names):
    """
    Compute motif probabilities using region-specific probabilities (old pipeline method).
    
    Formula: P(motif) = ∏_{i in motif} p_i × ∏_{j not in motif} (1 - p_j)
    
    Args:
        motif_labels: List of motif labels (each is a list of region names)
        region_probs_dict: Dictionary mapping region names to probabilities
        region_names: List of all region names (in order)
    
    Returns:
        Dictionary mapping motif index to probability
    """
    motif_probs_rs = {}
    
    for i, motif_regions in enumerate(motif_labels):
        if not motif_regions or (len(motif_regions) == 1 and not motif_regions[0]):
            # Empty motif (no projections)
            motif_probs_rs[i] = 0.0
            continue
        
        # Calculate probability for this specific motif
        # P(motif) = ∏_{regions in motif} p_region × ∏_{regions not in motif} (1 - p_region)
        prob = 1.0
        for region in region_names:
            if region in motif_regions:
                # Region is in motif: multiply by p_region
                prob *= region_probs_dict.get(region, 0.0)
            else:
                # Region is not in motif: multiply by (1 - p_region)
                prob *= (1.0 - region_probs_dict.get(region, 0.0))
        
        motif_probs_rs[i] = prob
    
    return motif_probs_rs

def compute_motif_probabilities_correlated(motif_labels, region_probs_dict, cond_prob_matrix, region_names):
    """
    Compute motif probabilities accounting for pairwise correlations.
    Uses P(A,B) = P(A) * P(B|A) for pairs, extended to higher-order motifs using chain rule.
    
    Args:
        motif_labels: List of motif labels (each is a list of region names)
        region_probs_dict: Dictionary mapping region names to marginal probabilities
        cond_prob_matrix: DataFrame with conditional probabilities P(B|A) where rows=A, cols=B
        region_names: List of all region names (in order)
    
    Returns:
        Dictionary mapping motif index to probability
    """
    motif_probs_corr = {}
    
    for i, motif_regions in enumerate(motif_labels):
        if not motif_regions or (len(motif_regions) == 1 and not motif_regions[0]):
            # Empty motif (no projections)
            motif_probs_corr[i] = 0.0
            continue
        
        if len(motif_regions) == 1:
            # Single region motif: just use marginal probability
            region = motif_regions[0]
            motif_probs_corr[i] = region_probs_dict.get(region, 0.0)
        else:
            # Multi-region motif: use chain rule with conditional probabilities
            # P(A,B,C,...) = P(A) * P(B|A) * P(C|B) * ...
            # Sort regions for consistent ordering
            sorted_regions = sorted(motif_regions)
            
            # Start with first region's marginal probability
            prob = region_probs_dict.get(sorted_regions[0], 0.0)
            
            # Multiply by conditional probabilities for subsequent regions
            for j in range(1, len(sorted_regions)):
                prev_region = sorted_regions[j-1]
                curr_region = sorted_regions[j]
                
                # Get conditional probability P(curr|prev) from matrix
                try:
                    cond_prob = cond_prob_matrix.loc[prev_region, curr_region]
                except (KeyError, TypeError):
                    # Fallback if regions not found: use marginal probability
                    cond_prob = region_probs_dict.get(curr_region, 0.0)
                
                prob *= cond_prob
            
            motif_probs_corr[i] = prob
    
    return motif_probs_corr

def compute_motif_probabilities_empirical(motif_labels, observed_counts, n0, normalize=False):
    """
    Compute empirical motif probabilities from observed frequencies.
    For anchor run: P(motif) = observed_count(motif) / N0
    
    Args:
        motif_labels: List of motif labels
        observed_counts: Observed counts for each motif
        n0: Total population size (N0)
        normalize: If True, normalize probabilities to sum to 1. 
                   If False (default), use raw frequencies (observed/N0).
                   For anchor models, should be False to ensure perfect fit.
    
    Returns:
        Dictionary mapping motif index to probability
    """
    motif_probs = {}
    total_observed = sum(observed_counts)
    
    for i, motif_regions in enumerate(motif_labels):
        if total_observed > 0:
            motif_probs[i] = observed_counts[i] / float(n0)
        else:
            motif_probs[i] = 0.0
    
    # Only normalize if explicitly requested
    # For anchor models, we want P = observed/N0 so that expected = N0 * P = observed
    if normalize:
        total_prob = sum(motif_probs.values())
        if total_prob > 0:
            motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_smoothed_empirical(motif_labels, observed_counts, n0, alpha=1.0):
    """
    Compute smoothed empirical motif probabilities with additive smoothing (Laplace).
    P(motif) = (observed_count + α) / (N0 + α × total_motifs)
    """
    motif_probs = {}
    total_motifs = len(motif_labels)
    denominator = float(n0) + alpha * total_motifs
    
    for i, motif_regions in enumerate(motif_labels):
        motif_probs[i] = (observed_counts[i] + alpha) / denominator
    
    # Normalize to ensure sum = 1
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_max_entropy(motif_labels, region_probs_dict, cond_prob_matrix, region_names):
    """
    Compute motif probabilities using maximum entropy approach.
    Uses Iterative Proportional Fitting (IPF) to match constraints:
    - Region marginals: P(region_i) = p_i
    - Pairwise correlations: P(region_i, region_j) = p_ij
    """
    from scipy.optimize import minimize
    import numpy as np
    
    # For now, use a simplified approach: match marginals and pairwise correlations
    # Full IPF implementation would be more complex
    motif_probs = {}
    
    # Start with independent model as initial guess
    for i, motif_regions in enumerate(motif_labels):
        if not motif_regions:
            motif_probs[i] = 0.0
            continue
        
        # Use region-specific probabilities as base
        prob = 1.0
        for region in region_names:
            if region in motif_regions:
                prob *= region_probs_dict.get(region, 0.0)
            else:
                prob *= (1.0 - region_probs_dict.get(region, 0.0))
        
        # Adjust for pairwise correlations if available
        if cond_prob_matrix is not None and len(motif_regions) > 1:
            sorted_regions = sorted(motif_regions)
            # Apply correlation adjustments
            for j in range(1, len(sorted_regions)):
                prev_region = sorted_regions[j-1]
                curr_region = sorted_regions[j]
                try:
                    cond_prob = cond_prob_matrix.loc[prev_region, curr_region]
                    marginal_prob = region_probs_dict.get(curr_region, 0.0)
                    # Adjust if conditional differs from marginal
                    if marginal_prob > 0:
                        adjustment = cond_prob / marginal_prob
                        prob *= adjustment
                except (KeyError, TypeError):
                    pass
        
        motif_probs[i] = max(0.0, prob)
    
    # Normalize
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_negative_binomial(motif_labels, region_probs_dict, region_names, dispersion=1.0):
    """
    Compute motif probabilities using negative binomial (overdispersed) model.
    For now, uses region-specific probabilities with dispersion adjustment.
    """
    # Use region-specific as base, dispersion affects variance not mean
    motif_probs = {}
    
    for i, motif_regions in enumerate(motif_labels):
        prob = 1.0
        for region in region_names:
            if region in motif_regions:
                prob *= region_probs_dict.get(region, 0.0)
            else:
                prob *= (1.0 - region_probs_dict.get(region, 0.0))
        motif_probs[i] = prob
    
    # Normalize
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_zero_inflated(motif_labels, observed_counts, n0, region_probs_dict, region_names, zero_inflation=0.0):
    """
    Compute motif probabilities using zero-inflated model.
    Two-component: structural zeros + count distribution
    """
    motif_probs = {}
    total_observed = sum(observed_counts)
    
    for i, motif_regions in enumerate(motif_labels):
        if observed_counts[i] == 0:
            # Structural zero with probability zero_inflation
            motif_probs[i] = zero_inflation * (1.0 / len(motif_labels))  # Uniform over zeros
        else:
            # Count distribution with probability (1 - zero_inflation)
            # Use region-specific probabilities
            prob = 1.0
            for region in region_names:
                if region in motif_regions:
                    prob *= region_probs_dict.get(region, 0.0)
                else:
                    prob *= (1.0 - region_probs_dict.get(region, 0.0))
            motif_probs[i] = (1.0 - zero_inflation) * prob
    
    # Normalize
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_bayesian_hierarchical(motif_labels, observed_counts, n0, alpha_prior=1.0):
    """
    Compute motif probabilities using Bayesian hierarchical model.
    Dirichlet-Multinomial: P ~ Dirichlet(α), counts ~ Multinomial(P, N0)
    Posterior: P ~ Dirichlet(α + observed_counts)
    """
    motif_probs = {}
    total_motifs = len(motif_labels)
    
    # Posterior parameters: alpha + observed
    posterior_alphas = [alpha_prior + count for count in observed_counts]
    total_alpha = sum(posterior_alphas)
    
    for i, motif_regions in enumerate(motif_labels):
        # Posterior mean of Dirichlet
        motif_probs[i] = posterior_alphas[i] / total_alpha if total_alpha > 0 else 0.0
    
    return motif_probs

def compute_motif_probabilities_ml_nonparametric(motif_labels, observed_counts, n0, region_names):
    """
    Compute motif probabilities using ML/non-parametric approach.
    For now, uses a simple random forest-like feature-based approach.
    """
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Prepare features: motif size, region composition
    X = []
    y = []
    
    for i, motif_regions in enumerate(motif_labels):
        # Features: motif size, one-hot for each region
        features = [len(motif_regions)]
        for region in region_names:
            features.append(1 if region in motif_regions else 0)
        X.append(features)
        y.append(observed_counts[i] / float(n0) if n0 > 0 else 0.0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train simple model (if we have enough data)
    if len(X) > 5 and np.sum(y) > 0:
        try:
            model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X)
            # Use predictions as probabilities
            motif_probs = {i: max(0.0, float(pred)) for i, pred in enumerate(predictions)}
        except:
            # Fallback to empirical
            motif_probs = compute_motif_probabilities_empirical(motif_labels, observed_counts, n0)
    else:
        # Fallback to empirical
        motif_probs = compute_motif_probabilities_empirical(motif_labels, observed_counts, n0)
    
    # Normalize
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def compute_motif_probabilities_hierarchical_correlations(motif_labels, region_probs_dict, cond_prob_matrix, region_names):
    """
    Compute motif probabilities using hierarchical/higher-order correlations.
    Models 3-way, 4-way correlations explicitly.
    """
    # For now, extend correlated model to higher orders
    # Full implementation would estimate 3-way, 4-way correlations
    motif_probs = {}
    
    for i, motif_regions in enumerate(motif_labels):
        if not motif_regions:
            motif_probs[i] = 0.0
            continue
        
        if len(motif_regions) == 1:
            motif_probs[i] = region_probs_dict.get(motif_regions[0], 0.0)
        elif len(motif_regions) == 2:
            # Use pairwise correlation
            sorted_regions = sorted(motif_regions)
            prob = region_probs_dict.get(sorted_regions[0], 0.0)
            try:
                cond_prob = cond_prob_matrix.loc[sorted_regions[0], sorted_regions[1]]
                prob *= cond_prob
            except (KeyError, TypeError):
                prob *= region_probs_dict.get(sorted_regions[1], 0.0)
            motif_probs[i] = prob
        else:
            # For 3+ regions, use chain rule with adjustments
            sorted_regions = sorted(motif_regions)
            prob = region_probs_dict.get(sorted_regions[0], 0.0)
            
            for j in range(1, len(sorted_regions)):
                prev_region = sorted_regions[j-1]
                curr_region = sorted_regions[j]
                try:
                    cond_prob = cond_prob_matrix.loc[prev_region, curr_region]
                    prob *= cond_prob
                except (KeyError, TypeError):
                    prob *= region_probs_dict.get(curr_region, 0.0)
            
            # Apply higher-order adjustment (simplified)
            # In full implementation, would use 3-way, 4-way correlation estimates
            motif_probs[i] = prob
    
    # Normalize
    total_prob = sum(motif_probs.values())
    if total_prob > 0:
        motif_probs = {k: v / total_prob for k, v in motif_probs.items()}
    
    return motif_probs

def determine_models_to_process(model_type_arg):
    """Determine which models to process based on CLI argument."""
    all_models = [
        "uniform", "region_specific", "correlated", 
        "empirical", "smoothed_empirical", "max_entropy",
        "hierarchical_correlations", "negative_binomial", "zero_inflated",
        "bayesian_hierarchical", "ml_nonparametric"
    ]
    
    if model_type_arg == "all":
        return all_models
    elif model_type_arg in all_models:
        return [model_type_arg]
    else:
        # Fallback: return original three models
        return ["uniform", "region_specific", "correlated"]

# Note: Region-specific motif probabilities will be calculated later when motif_labels are available
# (after line 1114 where motif_labels = gen_motifs(...))

normalized_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Normalized_Matrix.csv"))

# Dynamically match labels for important areas
special_area_1_label = next((label for label in columns if re.match(f"{args.special_area_1}\\d*", label, re.IGNORECASE)), None)
special_area_2_label = next((label for label in columns if re.match(f"{args.special_area_2}\\d*", label, re.IGNORECASE)), None)

column_counts_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Column_Counts.csv"))

# if special_area_1_label and special_area_2_label:
#     print(f"Matched labels: {special_area_1_label}, #{special_area_2_label}")
#     # Replace hardcoded logic with dynamic labels

root_save_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Roots.csv"))
pi_save_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Simplified_Pi.csv"))
region_probs_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Region-specific_Probabilities.csv"))
region_probs_rs_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Region-specific_Probabilities_N0based.csv"))
calculated_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Calculated_Value.csv"))
std_dev_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Standard_Deviation.csv"))
motif_test_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Motif_Binomial_Results.csv"))

# Safe log-sum-exp calculation to avoid log(0)
safe_psdict = {label: max(psdict.get(label, 0), 1e-10) for label in columns} # assign a very small floor value to avoid log(0)
log_scaled_value = sum(np.log(safe_psdict[label]) for label in columns)
scaled_value = np.exp(log_scaled_value) #convert back from log scaled
zero_labels = [label for label in columns if psdict.get(label, 0) <= 0]
if zero_labels:
    print(f"⚠️ Warning: Zero or missing probabilities for labels: {zero_labels}")

# Dynamic calculation using N0 (estimated population) - this verifies the pₑ solution
calculated_value = (1 - (1 - pe_num)**len(columns)) * N0_value
print(f"🔍 Expected Observed Projections [(1-(1-p_e)*#areas)*N0]: {calculated_value}")
print(f"🔍 This should approximately equal observed_cells ({observed_cells}) if pₑ is correct")

# Save LaTeX representation of calculated value
save_latex_expression(calculated_value, "Calculated Value Visualization", os.path.normpath(os.path.join(out_dir, f"{sample_name}_Calculated_Value.png")))

# Perform statistical tests
if not (0 <= scaled_value <= 1):
    raise ValueError("Scaled value must be in range [0,1] for valid probability interpretation.")

std_dev = np.sqrt(scaled_value * total_projections * (1 - scaled_value))

print("🔍 Standard Deviation [valid range 0-1]:", std_dev)

# Identify observed motif sizes and counts
observed_motif_sizes = np.unique(np.sum(normalized_matrix > 0, axis=1))  # Unique motif sizes
motif_counts = [np.sum(np.sum(normalized_matrix > 0, axis=1) == size) for size in observed_motif_sizes]

# Debugging printout: Show motif counts (Observed vs Expected)
# FIXED: Use N0 for expected counts to match alternative pipeline
print("\n==== Motif Observed vs Expected Counts (using N0) ============")
for i, motif_size in enumerate(observed_motif_sizes):
    observed = motif_counts[i]  # Observed count
    expected = int(motif_probs.get(motif_size, 0) * N0_value)  # Expected count based on probabilities and N0

    print(f"Motif Size: {motif_size:5} | Observed: {observed:5} | Expected: {expected:5} (N0={N0_value:.1f})")

print("\n===================================================")

# Compute probabilities for observed motif sizes only
motif_probs = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n)) for n in observed_motif_sizes
}

# Normalize probabilities
total_motif_prob = sum(motif_probs.values())
motif_probs = {k: v / total_motif_prob for k, v in motif_probs.items()}

# Perform binomial test for each observed motif size
# FIXED: Use two-tailed test (binomtest) with N0 instead of one-tailed (binom.sf) with observed_cells
# This matches the alternative pipeline's more rigorous approach
print(f"🔍 [UNIFORM MODEL] Performing two-tailed binomial tests using N0 = {N0_value} (not observed_cells = {observed_cells})")
binomial_test_results = []
for n_proj in observed_motif_sizes:
    obs_count = int(motif_counts[observed_motif_sizes.tolist().index(n_proj)])  # Ensure integer
    prob = float(motif_probs.get(n_proj, 0))  # Ensure float
    # Use binomtest (two-tailed) with N0_value instead of binom.sf (one-tailed) with observed_cells
    p_value = binomtest(obs_count, n=int(N0_value), p=prob).pvalue  # Two-tailed test with N0
    binomial_test_results.append((n_proj, prob, p_value))  # Append tuple with 3 elements

# Debugging
print("Checking structure of binomial_test_results")
for entry in binomial_test_results:
    print(f"Entry: {entry}, Type: {type(entry)}, Length: {len(entry)}")

# Flatten results into a CSV-friendly structure
flat_results = [
    {"Motif Size": n_proj, "Expected Probability": prob, "P-Value": p_value}
    for n_proj, prob, p_value in binomial_test_results
]

# Save to CSV 
binomial_results_file = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Motif_Binomial_Results.csv"))
pd.DataFrame(flat_results).to_csv(binomial_results_file, index=False)
print(f"Motif binomial test results saved to: {binomial_results_file}")

# Output results
for n_proj, prob, p_value in binomial_test_results:
    print(f"Motif Size {n_proj}: Observed = {motif_counts[observed_motif_sizes.tolist().index(n_proj)]}, "
          f"Expected Probability = {prob:.5f}, P-Value = {p_value:.50f}")

print("\nBinomial Test Results for All Detected Motif Sizes:")
for n_proj, prob, p_value in binomial_test_results:
    print(f"  Motif with {n_proj} projections: P-Value = {p_value:.50f}")  # Increase decimal precision
    print(f"Motif Size {n_proj}: Expected Probability = {motif_probs[n_proj]:.50f}")
    print(f"Motif Size {n_proj}: Observed Count = {motif_counts[observed_motif_sizes.tolist().index(n_proj)]}")

# Save other results
results = {
    "Roots": roots,
    "Simplified Pi": [simplified_pi],
    "Region-specific Probabilities": list(psdict.values()),
    "Calculated Value": [calculated_value],
    "Standard Deviation": [std_dev],
    "Binomial Test Results": [binomial_test_results]  # Save correctly formatted results
}

# Create output directory if it doesn’t exist
os.makedirs(out_dir, exist_ok=True)

print("\n💾 Saving computed results to CSV files...\n")

# Save each result to a separate CSV file
for key, value in results.items():
    file_path = os.path.join(out_dir, f"{sample_name}_{key.replace(' ', '_')}.csv")

    if key == "Binomial Test Results":
        print(f"📂 Saving {key} to {file_path} (formatted as a DataFrame)")
        pd.DataFrame(flat_results).to_csv(file_path, index=False)
    else:
        print(f"📂 Saving {key} to {file_path} (single-column DataFrame)")
        pd.DataFrame({key: value}).to_csv(file_path, index=False)

print("\n✅ All result files have been successfully saved!\n")


### Visualizations
# Region-specific probabilities
plt.figure(figsize=(8, 5))
plt.bar(psdict.keys(), psdict.values())
plt.title("Region-specific Probabilities")
plt.ylabel("Probability")
plt.xlabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.normpath(os.path.join(out_dir, f"{sample_name}_Region_Probabilities.png")))
plt.close()

# Roots scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(range(len(roots)), roots)
plt.title("Roots")
plt.ylabel("Root Value")
plt.xlabel("Index")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.normpath(os.path.join(out_dir, f"{sample_name}_Roots.png")))
plt.close()

### Analysis and Plotting Integration

# Where is the normalized_matrix.csv
data_dir = out_dir
file_name = os.path.normpath(os.path.join(data_dir, f"{sample_name}_Normalized_Matrix.csv"))


# Load normalized matrix as input for analysis
normalized_matrix_file = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Normalized_Matrix.csv"))
normalized_matrix = pd.read_csv(normalized_matrix_file)


# Ensure 'analysis' subdirectory exists within 'out_dir'
analysis_dir = os.path.normpath(os.path.join(out_dir, 'analysis'))
os.makedirs(analysis_dir, exist_ok=True)

#Where do you want the analysis output to go?
plot_dir = analysis_dir
csv_output_dir = os.path.join(plot_dir, "motif_raw_data")

# Determine which models to process
models_to_process = determine_models_to_process(args.model_type)
print(f"🔍 Models to process: {models_to_process}")

# Create model-specific subdirectories for multi-model plots
model_plot_dirs = {}
for model_name in models_to_process:
    model_plot_dir = os.path.normpath(os.path.join(plot_dir, model_name))
    os.makedirs(model_plot_dir, exist_ok=True)
    model_plot_dirs[model_name] = model_plot_dir
    print(f"📁 Created directory for {model_name} model: {model_plot_dir}")

# Keep backward compatibility with existing code
uniform_plot_dir = model_plot_dirs.get('uniform', os.path.normpath(os.path.join(plot_dir, 'uniform')))
region_specific_plot_dir = model_plot_dirs.get('region_specific', os.path.normpath(os.path.join(plot_dir, 'region_specific')))
correlated_plot_dir = model_plot_dirs.get('correlated', os.path.normpath(os.path.join(plot_dir, 'correlated')))

n0 = N0_value  # Use estimated population (N0) instead of observed_cells for motif expected counts

np.set_printoptions(suppress=True)

def load_df(file, remove_cols=None, subset=None):
    """
    Loads CSV file specified in `file` (a full path string).
    remove_cols: list of column names to drop.
    subset: list of column names to retain (drops all others).
    """
    try:
        df = pd.read_csv(file)  # Supports both Windows/Linux paths via normpath above
    except Exception as e:
        raise IOError(f"Failed to load file {file}: {e}")

    print(f"✅ Columns in loaded file: {df.columns.tolist()}")

    if remove_cols is not None:
        try:
            df = df.drop(columns=remove_cols)
        except Exception as e:
            print(f"⚠️ Warning: Could not remove columns {remove_cols} — {e}")

    if subset is not None:
        try:
            df = df[subset]
        except Exception as e:
            print(f"⚠️ Warning: Could not subset to columns {subset} — {e}")

    return df


# Prepare file path
file_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Normalized_Matrix.csv"))

# Load DataFrame based on `full_data` flag
if full_data:
    df = load_df(file_path)
else:
    df = load_df(file_path, remove_cols=['RSP'], subset=['PM', 'AM', 'A', 'RL', 'AL', 'LM'])

print(f"df shape: {df.shape}")
print("DF Head:")
print(df.head())
print("Number of NAs:")
print(df.isnull().sum())


X = df.to_numpy()
K = list(range(2, 15))  # skip k=1 for silhouette and BIC

# 1. Elbow Method (Inertia)
inertias = []
for k_val in K:
    k = min(k_val, X.shape[0])  # Prevents ValueError when too few samples
    km = KMeans(n_clusters=k, n_init="auto").fit(X)
    inertias.append(km.inertia_)

# Compute elbow using max second derivative
inertia_deltas = np.diff(inertias)
inertia_deltas2 = np.diff(inertia_deltas)
elbow_k = K[np.argmax(inertia_deltas2) + 2] if len(inertia_deltas2) > 0 else K[0]

# 2. Silhouette Score
sil_scores = []
for k_val in K:
    if k_val >= X.shape[0]:
        sil_scores.append(-1)
        continue
    km = KMeans(n_clusters=k_val, n_init="auto").fit(X)
    sil_scores.append(silhouette_score(X, km.labels_))
silhouette_k = K[np.argmax(sil_scores)]

# 3. Gap Statistic
def compute_gap_statistic(X, refs=10):
    gaps = []
    for k_val in K:
        if k_val >= X.shape[0]:
            gaps.append(-np.inf)
            continue
        km = KMeans(n_clusters=k_val, n_init="auto").fit(X)
        disp = np.mean(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1))

        ref_disps = []
        for _ in range(refs):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            km_ref = KMeans(n_clusters=k_val, n_init="auto").fit(X_ref)
            ref_disp = np.mean(np.min(cdist(X_ref, km_ref.cluster_centers_, 'euclidean'), axis=1))
            ref_disps.append(ref_disp)

        gap = np.log(np.mean(ref_disps)) - np.log(disp)
        gaps.append(gap)
    return gaps

gaps = compute_gap_statistic(X)
gap_k = K[np.argmax(gaps)]

# 4. BIC using GMM (guarded)
bics = []
bic_valid_k = []
for k_val in K:
    if k_val >= X.shape[0]:
        continue
    try:
        gmm = GaussianMixture(n_components=k_val, n_init=1).fit(X)
        bics.append(gmm.bic(X))
        bic_valid_k.append(k_val)
    except:
        continue

bic_k = bic_valid_k[np.argmin(bics)] if bics else None

# Consensus vote
votes = [v for v in [elbow_k, silhouette_k, gap_k, bic_k] if v is not None]
vote_counts = Counter(votes)
consensus_k = vote_counts.most_common(1)[0][0]

# Diagnostic info
print(f"Elbow k = {elbow_k}, Silhouette k = {silhouette_k}, Gap k = {gap_k}, BIC k = {bic_k}")
print(f"Consensus k = {consensus_k}")

# Plot
plt.figure(figsize=(10, 7))
plt.plot(K[:len(inertias)], inertias, 'x-', color='blue', label='Inertia')
plt.axvline(x=elbow_k, color='gray', linestyle='--', label=f'Elbow: k={elbow_k}')
plt.axvline(x=silhouette_k, color='green', linestyle='--', label=f'Silhouette: k={silhouette_k}')
plt.axvline(x=gap_k, color='orange', linestyle='--', label=f'Gap: k={gap_k}')
if bic_k is not None:
    plt.axvline(x=bic_k, color='purple', linestyle='--', label=f'BIC: k={bic_k}')
plt.axvline(x=consensus_k, color='red', linestyle=':', linewidth=2.0, label=f'Consensus: k={consensus_k}')
plt.xlabel('k', fontsize=20)
plt.ylabel('Inertia', fontsize=20)
plt.title('Cluster Evaluation Methods', fontsize=20)
plt.legend()
plt.tight_layout()

for ext in ["pdf", "svg", "png"]:
    elbow_plt = plt.gcf()
    elbow_plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_cluster_diagnostics.{ext}")))

# 🚀 Final clustering
safe_k = min(consensus_k, X.shape[0])
km = KMeans(n_clusters=safe_k, n_init="auto").fit(X)

from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import os

# Clustering
X = df.to_numpy()
k = min(k, X.shape[0])  # Prevents ValueError when too few samples
km = KMeans(n_clusters=consensus_k, n_init="auto").fit(X)
clusters, regions = km.cluster_centers_.shape

# Save Data
df.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_obs_exp.csv")))

# Plotting setup
scolors = ['black', 'red', 'orange', 'yellow']
scm = LinearSegmentedColormap.from_list('white_to_red', scolors, N=100)
fig, ax = plt.subplots(figsize=(10, 10))

# Axes labels and style
ax.set_title("K-means Clustering")
ax.set_xlabel("Regions")
ax.set_ylabel("Cluster")
ax.set_xticks(range(regions))
ax.set_xticklabels(df.columns.to_list(), rotation=45, ha='right')
ax.set_yticks(range(clusters))
ax.set_yticklabels(range(1, clusters + 1))
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)

# Data plot
X_vals = range(regions)
for i in range(km.n_clusters):
    y_vals = np.full(regions, i)
    size = km.cluster_centers_[i]
    size_norm = (size - size.min()) / (size.max() - size.min()) if size.max() > size.min() else size
    ax_ = ax.scatter(x=X_vals, y=y_vals, s=1000, c=size_norm, cmap=scm)

# Add colorbar
fig.colorbar(ax_, label='Normalized Projection Strength')

# Save plot
#fig.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_kmeans.pdf")))
for ext in ["pdf", "svg", "png"]:
    fig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_kmeans.{ext}")))


def concatenate_list_data(slist,join=motif_join):
    result = []
    for i in slist:
        sub = ''
        for j in i:
            if sub:
                sub = sub + join + str(j)
            else:
                sub += str(j)
        result.append(sub)
    return result

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def gen_motifs(r,labels): #r is number of regions, so number of motifis is 2^r
    num_motifs = 2**r
    motifs = np.zeros((num_motifs,r)).astype(bool)
    motif_ids = list(powerset(np.arange(r)))
    motif_labels = [] #list of labels e.g. PFC-LH or PFC-LS-BNST
    for i in range(num_motifs):
        idx = motif_ids[i]
        motifs[i,idx] = True
        label = labels[np.array(idx)].to_list() if idx else ['']
        motif_labels.append(label)
    return motifs, motif_labels

def count_motifs(df,motifs,return_ids=False):
    """
    Returns a vector with counts that indicated number of motifs present for each possible motif
    A motif is a combination of target areas represented as a binary array, 
    e.g. [1,1,0] represents a motif where a cell targets the first two regions but not the 3rd
    A motif can be obtained by simply thresholding each cell's projection strength vector such that
    non-zero elements are 1.
    Also returns the labels for each motif
    """
    cells, regions = df.shape
    data = df.to_numpy().astype(bool)
    counts = np.zeros(motifs.shape[0])
    cell_ids = []
    for i in range(motifs.shape[0]): #loop through motifs (128x7)
        cell_ids_ = []
        for j in range(data.shape[0]): #loop through observed data cells X regions
            if np.array_equal(motifs[i],data[j]):
                counts[i] = counts[i] + 1
                cell_ids_.append(j)
        cell_ids.append(cell_ids_)
        
    if return_ids:
        return counts, cell_ids#, motifs
    else:
        return counts

def zip_with_scalar(l, o):
    return zip(l, itertools.repeat(o))

motifs, motif_labels = gen_motifs(df.shape[1],df.columns)

dcounts, cell_ids = count_motifs(df,motifs, return_ids=True) #observed data

def convert_counts_to_df(columns,counts,labels):
    """ 
    Returns dataframe showing cell counts for each motif
    """
    motifdf = pd.DataFrame(columns=columns)
    for i in range(len(counts)):
        cols = labels[i]
        if len(cols) == 1 and not cols[0]:
            continue
        motifdf.loc[i,cols] = 1
        motifdf.loc[i,"Count"] = counts[i]
    return motifdf.fillna(0).infer_objects(copy=False)

motif_df = convert_counts_to_df(df.columns,dcounts,motif_labels)

from sympy import N

def get_expected_counts(motifs, num_regions = 7, prob_edge=pe_num,n=n0):
    # Ensure variables are numeric
    prob_edge = float(prob_edge.evalf()) if hasattr(prob_edge, "evalf") else float(prob_edge)
    n_motifs = len(motifs)
    res = np.zeros(n_motifs)
    probs = np.zeros(n_motifs)
    for i,motif in enumerate(motifs):
        e1 = int(len(motif))
        e2 = num_regions - e1
        p = (prob_edge ** e1) * (1 - prob_edge) ** e2
        exp = float(N(p)) * n
        res[i] = exp
        probs[i] = p
    res[0] = 0
    return res, probs

# ============================================================================
# UNIFORM MODEL: Expected counts and results
# ============================================================================
print("\n" + "="*80)
print("[UNIFORM MODEL] Calculating expected counts for all motifs...")
print("="*80)

exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels),\
                                dcounts,\
                                exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif','Observed','Expected']
df_obs_exp.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_obs_exp_uniform.csv")))
print(f"💾 [UNIFORM MODEL] Saved motif obs/exp to: {sample_name}_motif_obs_exp_uniform.csv")

##suggested addition to give another csv without the null combination from the powerset at the top row.
exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels), dcounts, exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif', 'Observed', 'Expected']

# Exclude empty motifs
df_obs_exp = df_obs_exp[df_obs_exp['Motif'] != ""]  # Adjust condition as needed for your data format - chatGPT

# Save filtered data to CSV
df_obs_exp.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_obs_exp_uniform_filtered.csv")),
    index=False
)
print(f"💾 [UNIFORM MODEL] Saved filtered motif obs/exp to: {sample_name}_motif_obs_exp_uniform_filtered.csv")

# ============================================================================
# REGION-SPECIFIC MODEL: Expected counts and results (Old Pipeline Method)
# ============================================================================
print("\n" + "="*80)
print("[REGION-SPECIFIC MODEL] Calculating probabilities and expected counts for all motifs...")
print("="*80)

# Use anchor probabilities if provided, otherwise use local region-specific probabilities
# This ensures P60-anchored comparison when --anchor-model-file is provided
probs_for_region_specific = psdict_for_expected if anchor_probs_loaded else psdict_region_specific
if anchor_probs_loaded:
    print(f"🔗 [REGION-SPECIFIC MODEL] Using P60 ANCHOR probabilities for expected counts")
    print(f"🔗 [REGION-SPECIFIC MODEL] Anchor probabilities: {probs_for_region_specific}")
else:
    print(f"🔍 [REGION-SPECIFIC MODEL] Using LOCAL probabilities (no anchor provided)")

# Compute motif probabilities using region-specific model
print(f"🔍 [REGION-SPECIFIC MODEL] Computing probabilities for {len(motif_labels)} motifs...")
motif_probs_region_specific = compute_motif_probabilities_region_specific(
    motif_labels, probs_for_region_specific, columns
)

# Calculate total probability (should be close to 1, but may not be exactly 1 due to numerical precision)
total_motif_prob_rs = sum(motif_probs_region_specific.values())
print(f"🔍 [REGION-SPECIFIC MODEL] Total motif probability sum: {total_motif_prob_rs}")

# Normalize if needed (though theoretically should already sum to 1)
if total_motif_prob_rs > 0:
    motif_probs_region_specific = {k: v / total_motif_prob_rs for k, v in motif_probs_region_specific.items()}
    print(f"🔍 [REGION-SPECIFIC MODEL] Normalized probabilities (sum = {sum(motif_probs_region_specific.values())})")
else:
    print(f"🚨 WARNING: [REGION-SPECIFIC MODEL] Total motif probability is 0 or negative!")

print(f"✅ [REGION-SPECIFIC MODEL] Computed probabilities for {len(motif_probs_region_specific)} motifs")

# Save region-specific probabilities to CSV
df_region_probs_rs = pd.DataFrame({
    'Region': list(psdict_region_specific.keys()),
    'Probability': list(psdict_region_specific.values()),
    'Projection_Count': [projections.get(r, 0) for r in psdict_region_specific.keys()],
    'N0': [N0_value] * len(psdict_region_specific)
})
df_region_probs_rs.to_csv(region_probs_rs_path, index=False)
print(f"💾 [REGION-SPECIFIC MODEL] Saved region-specific probabilities (N0-based) to: {region_probs_rs_path}")

# Calculate expected counts using region-specific probabilities
def get_expected_counts_region_specific(motif_labels, motif_probs_dict, n=n0):
    """
    Calculate expected counts for each motif using region-specific probabilities.
    
    Args:
        motif_labels: List of motif labels (each is a list of region names)
        motif_probs_dict: Dictionary mapping motif index to probability
        n: Total population size (N0)
    
    Returns:
        Array of expected counts for each motif
    """
    n_motifs = len(motif_labels)
    res = np.zeros(n_motifs)
    probs = np.zeros(n_motifs)
    
    for i in range(n_motifs):
        prob = motif_probs_dict.get(i, 0.0)
        probs[i] = prob
        exp = float(prob) * float(n)
        res[i] = exp
    
    res[0] = 0  # Empty motif (no projections)
    return res, probs

exp_counts_rs, motif_probs_rs_array = get_expected_counts_region_specific(
    motif_labels, motif_probs_region_specific, n=n0
)

print(f"🔍 [REGION-SPECIFIC MODEL] Calculated expected counts for {len(exp_counts_rs)} motifs")
print(f"🔍 [REGION-SPECIFIC MODEL] Total expected count: {np.sum(exp_counts_rs):.2f} (should be close to N0 = {n0})")

# Create DataFrame for region-specific model
df_obs_exp_rs = pd.DataFrame(data=[
    concatenate_list_data(motif_labels),
    dcounts,
    exp_counts_rs.astype(int)
]).T
df_obs_exp_rs.columns = ['Motif', 'Observed', 'Expected']
df_obs_exp_rs.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_obs_exp_region_specific.csv")))
print(f"💾 [REGION-SPECIFIC MODEL] Saved motif obs/exp to: {sample_name}_motif_obs_exp_region_specific.csv")

# Exclude empty motifs
df_obs_exp_rs_filtered = df_obs_exp_rs[df_obs_exp_rs['Motif'] != ""].copy()
df_obs_exp_rs_filtered.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_obs_exp_region_specific_filtered.csv")),
    index=False
)
print(f"💾 [REGION-SPECIFIC MODEL] Saved filtered motif obs/exp to: {sample_name}_motif_obs_exp_region_specific_filtered.csv")

# ============================================================================
# PROPORTIONAL EFFECT SIZE MODEL: Observed/Expected CSV
# ============================================================================
# Create DataFrame for proportional effect size model (same data as region_specific, different effect size calculation)
df_obs_exp_prop = pd.DataFrame(data=[
    concatenate_list_data(motif_labels),
    dcounts,
    exp_counts_rs.astype(int)
]).T
df_obs_exp_prop.columns = ['Motif', 'Observed', 'Expected']
df_obs_exp_prop.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_obs_exp_proportional_effectsize.csv")))
print(f"💾 [PROPORTIONAL EFFECT SIZE MODEL] Saved motif obs/exp to: {sample_name}_motif_obs_exp_proportional_effectsize.csv")

# Exclude empty motifs
df_obs_exp_prop_filtered = df_obs_exp_prop[df_obs_exp_prop['Motif'] != ""].copy()
df_obs_exp_prop_filtered.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_obs_exp_proportional_effectsize_filtered.csv")),
    index=False
)
print(f"💾 [PROPORTIONAL EFFECT SIZE MODEL] Saved filtered motif obs/exp to: {sample_name}_motif_obs_exp_proportional_effectsize_filtered.csv")


def standardize_pos(x):
    return (x + 1) / (x.std())
def standardize(x):
    return (x + 1e-13) / (x.max() - x.min())
def subset_list(lis, ids):
    return [lis[i] for i in ids]

#dcounts are motif counts from observed data
def get_motif_sig_pts(dcounts,labels,\
                            prob_edge=pe_num, n0 = n0, \
                      exclude_zeros=True, \
                      p_transform=lambda x: -1 * np.log10(x)):
    num_motifs = dcounts.shape[0]
    expected, probs = get_expected_counts(labels, prob_edge=pe_num,n=n0)
    assert dcounts.shape[0] == expected.shape[0]
    if exclude_zeros:
        nonzid = np.nonzero(dcounts)[0]
    else:
        nonzid = np.arange(dcounts.shape[0])
    num_nonzid_motifs = nonzid.shape[0]
    dcounts_ = dcounts[nonzid]
    expected_ = expected[nonzid]
    probs_ = probs[nonzid]
    #Effect size is log2(observed/expected)
    effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
    matches = np.zeros(num_nonzid_motifs)
    assert dcounts_.shape[0] == expected_.shape[0]
    dcounts_ = dcounts_.astype(int)
    for i in range(num_nonzid_motifs):
        pi = max(probs_[i], 1e-10) #avoid zero or very small probs
        # Uses binomtest (two-tailed) with n0 (which is N0_value) - matches alternative pipeline
        # Convert n0 to int as binomtest requires integer n
        matches[i] = binomtest(int(dcounts_[i]),n=int(n0),p=pi).pvalue
        matches[i] = max(matches[i], 1e-10)
    matches = p_transform(matches)
    #matches is the significance level
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

# ============================================================================
# UNIFORM MODEL: Statistical testing
# ============================================================================
print("\n" + "="*80)
print("[UNIFORM MODEL] Performing statistical tests...")
print("="*80)

#SET TO TRUE IF YOU WANT TO EXCLUDE ZERO MOTIFS
sigs, slabels = get_motif_sig_pts(dcounts,motif_labels,exclude_zeros=False)
print(f"✅ [UNIFORM MODEL] Statistical tests completed for {len(slabels)} motifs")

# ============================================================================
# REGION-SPECIFIC MODEL: Statistical testing (Old Pipeline Method)
# ============================================================================
print("\n" + "="*80)
print("[REGION-SPECIFIC MODEL] Performing statistical tests...")
print("="*80)

def get_motif_sig_pts_region_specific(dcounts, labels, motif_probs_dict, n0=n0,
                                      exclude_zeros=True,
                                      p_transform=lambda x: -1 * np.log10(x)):
    """
    Calculate significance for motifs using region-specific probabilities (old pipeline method).
    
    Args:
        dcounts: Observed counts for each motif
        labels: Motif labels
        motif_probs_dict: Dictionary mapping motif index to probability
        n0: Total population size (N0)
        exclude_zeros: Whether to exclude zero-count motifs
        p_transform: Transform function for p-values
    
    Returns:
        List of (effect_size, significance) tuples, and list of motif labels
    """
    num_motifs = dcounts.shape[0]
    expected, probs = get_expected_counts_region_specific(labels, motif_probs_dict, n=n0)
    assert dcounts.shape[0] == expected.shape[0]
    
    if exclude_zeros:
        nonzid = np.nonzero(dcounts)[0]
    else:
        nonzid = np.arange(dcounts.shape[0])
    
    num_nonzid_motifs = nonzid.shape[0]
    dcounts_ = dcounts[nonzid]
    expected_ = expected[nonzid]
    probs_ = probs[nonzid]
    
    # Effect size is log2(observed/expected)
    effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
    matches = np.zeros(num_nonzid_motifs)
    assert dcounts_.shape[0] == expected_.shape[0]
    dcounts_ = dcounts_.astype(int)
    
    for i in range(num_nonzid_motifs):
        pi = max(probs_[i], 1e-10)  # avoid zero or very small probs
        # Uses binomtest (two-tailed) with n0 (which is N0_value) - matches alternative pipeline
        # Convert n0 to int as binomtest requires integer n
        matches[i] = binomtest(int(dcounts_[i]), n=int(n0), p=pi).pvalue
        matches[i] = max(matches[i], 1e-10)
    
    matches = p_transform(matches)
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

sigs_rs, slabels_rs = get_motif_sig_pts_region_specific(
    dcounts, motif_labels, motif_probs_region_specific, n0=n0, exclude_zeros=False
)
print(f"✅ [REGION-SPECIFIC MODEL] Statistical tests completed for {len(slabels_rs)} motifs")

# ============================================================================
# PROPORTIONAL EFFECT SIZE MODEL: Statistical testing
# ============================================================================
print("\n" + "="*80)
print("[PROPORTIONAL EFFECT SIZE MODEL] Performing statistical tests...")
print("="*80)

def get_motif_sig_pts_proportional_effectsize(dcounts, labels, motif_probs_dict, n0=n0,
                                              exclude_zeros=True,
                                              p_transform=lambda x: -1 * np.log10(x)):
    """
    Calculate significance for motifs using region-specific probabilities with proportion-based effect size.
    
    This uses the same probabilities and p-values as region_specific model, but calculates
    effect size using proportions: log2((k/n0 + 1) / (π + 1)) instead of log2((k + 1) / (π×n0 + 1)).
    
    Args:
        dcounts: Observed counts for each motif
        labels: Motif labels
        motif_probs_dict: Dictionary mapping motif index to probability
        n0: Total population size (N0)
        exclude_zeros: Whether to exclude zero-count motifs
        p_transform: Transform function for p-values
    
    Returns:
        List of (effect_size, significance) tuples, and list of motif labels
    """
    num_motifs = dcounts.shape[0]
    expected, probs = get_expected_counts_region_specific(labels, motif_probs_dict, n=n0)
    assert dcounts.shape[0] == expected.shape[0]
    
    if exclude_zeros:
        nonzid = np.nonzero(dcounts)[0]
    else:
        nonzid = np.arange(dcounts.shape[0])
    
    num_nonzid_motifs = nonzid.shape[0]
    dcounts_ = dcounts[nonzid]
    expected_ = expected[nonzid]
    probs_ = probs[nonzid]
    
    # Effect size using proportion-based formula: log2((k/n0 + 1) / (π + 1))
    # This is equivalent to: log2((k + n0) / (π×n0 + n0))
    n0_float = float(n0)
    effect_size = np.log2((dcounts_ / n0_float + 1) / (probs_ + 1))
    matches = np.zeros(num_nonzid_motifs)
    assert dcounts_.shape[0] == expected_.shape[0]
    dcounts_ = dcounts_.astype(int)
    
    for i in range(num_nonzid_motifs):
        pi = max(probs_[i], 1e-10)  # avoid zero or very small probs
        # Uses binomtest (two-tailed) with n0 (which is N0_value) - same as region_specific
        # Convert n0 to int as binomtest requires integer n
        matches[i] = binomtest(int(dcounts_[i]), n=int(n0), p=pi).pvalue
        matches[i] = max(matches[i], 1e-10)
    
    matches = p_transform(matches)
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

def get_motif_sig_pts_proportional_effectsize_raw(dcounts, labels, motif_probs_dict, n0=n0,
                                                  exclude_zeros=True,
                                                  p_transform=lambda x: -1 * np.log10(x)):
    """
    Same as get_motif_sig_pts_proportional_effectsize but effect size is (k/n0+1)/(pi+1) - 1 (no log2).
    Returns effect size centered at 0: 0 = no effect, positive = over-represented, negative = under-represented.
    """
    num_motifs = dcounts.shape[0]
    expected, probs = get_expected_counts_region_specific(labels, motif_probs_dict, n=n0)
    assert dcounts.shape[0] == expected.shape[0]
    if exclude_zeros:
        nonzid = np.nonzero(dcounts)[0]
    else:
        nonzid = np.arange(dcounts.shape[0])
    num_nonzid_motifs = nonzid.shape[0]
    dcounts_ = dcounts[nonzid]
    expected_ = expected[nonzid]
    probs_ = probs[nonzid]
    n0_float = float(n0)
    # Deviation from 1 so 0 = no effect, positive = over, negative = under (same as log2 version)
    effect_size = (dcounts_ / n0_float + 1) / (probs_ + 1) - 1.0
    matches = np.zeros(num_nonzid_motifs)
    assert dcounts_.shape[0] == expected_.shape[0]
    dcounts_ = dcounts_.astype(int)
    for i in range(num_nonzid_motifs):
        pi = max(probs_[i], 1e-10)
        matches[i] = binomtest(int(dcounts_[i]), n=int(n0), p=pi).pvalue
        matches[i] = max(matches[i], 1e-10)
    matches = p_transform(matches)
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

sigs_prop, slabels_prop = get_motif_sig_pts_proportional_effectsize(
    dcounts, motif_labels, motif_probs_region_specific, n0=n0, exclude_zeros=False
)
print(f"✅ [PROPORTIONAL EFFECT SIZE MODEL] Statistical tests completed for {len(slabels_prop)} motifs")

sigs_prop_raw, slabels_prop_raw = get_motif_sig_pts_proportional_effectsize_raw(
    dcounts, motif_labels, motif_probs_region_specific, n0=n0, exclude_zeros=False
)

# Create minimal CSV file early with effect sizes for unified range calculation
# This ensures the CSV exists when calculate_unified_xaxis_range() is called
# The full CSV will be created and overwritten later with complete data
effect_sizes_prop = [e for e, _ in sigs_prop]
p_values_prop = [p for _, p in sigs_prop]
dfraw_prop_minimal = pd.DataFrame({
    'Motifs': slabels_prop,
    'Effect Size': effect_sizes_prop,
    'P-value': p_values_prop
})
dfraw_prop_minimal.to_csv(
    os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_proportional_effectsize.csv")),
    index=False
)
print(f"💾 [PROPORTIONAL EFFECT SIZE MODEL] Saved minimal upsetplot data (for unified range calculation) to: {region_specific_plot_dir}/{sample_name}_upsetplot_proportional_effectsize.csv")

effect_sizes_prop_raw = [e for e, _ in sigs_prop_raw]
dfraw_prop_raw_minimal = pd.DataFrame({
    'Motifs': slabels_prop_raw,
    'Effect Size': effect_sizes_prop_raw,
    'P-value': p_values_prop
})
dfraw_prop_raw_minimal.to_csv(
    os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_proportional_effectsize_raw.csv")),
    index=False
)
print(f"💾 [PROPORTIONAL EFFECT SIZE MODEL] Saved minimal upsetplot data (raw) for unified range to: {region_specific_plot_dir}/{sample_name}_upsetplot_proportional_effectsize_raw.csv")

# ============================================================================
# Function to calculate unified x-axis range for volcano plots
# ============================================================================
def calculate_unified_xaxis_range(output_dir, model_suffix, current_effect_sizes=None):
    """
    Calculate unified x-axis range for volcano plots across ALL ages for the same parameterization.
    
    This function scans existing upsetplot CSV files across all age directories (p3, p12, p20, p60)
    for the same parameterization to find all effect sizes for the given model type, then calculates
    a symmetric range that encompasses all samples. This ensures consistent x-axis scales for 
    cross-cohort comparisons.
    
    Args:
        output_dir: Directory containing upsetplot CSV files (e.g., region_specific_plot_dir)
                   Format: 02_output/{age}/{parameterization}/analysis/{model}/
        model_suffix: Model type (e.g., 'region_specific', 'proportional_effectsize', 'uniform')
        current_effect_sizes: Optional list of effect sizes from current sample to include
    
    Returns:
        unified_xlim: Tuple (x_min, x_max) for unified x-axis limits, or None if no data found
    """
    all_effect_sizes = []
    files_scanned = 0
    ages_found = []
    
    # Include current sample's effect sizes if provided
    if current_effect_sizes is not None:
        all_effect_sizes.extend(current_effect_sizes)
    
    # Extract base directory and parameterization from output_dir
    # output_dir format: 02_output/{age}/{parameterization}/analysis/{model}/
    path_parts = Path(output_dir).parts
    
    # Find parameterization (starts with 01., 02., 03., 04., or 05.)
    param_idx = None
    for i, part in enumerate(path_parts):
        if any(part.startswith(f"{j:02d}.") for j in range(1, 6)):
            param_idx = i
            break
    
    if param_idx is not None:
        # Successfully extracted parameterization
        # NOTE: path_parts[:param_idx] includes the age folder (e.g., ".../02_output/p12"),
        # but we need the directory that CONTAINS all age folders (e.g., ".../02_output").
        base_dir = os.path.join(*path_parts[:param_idx - 1])  # parent of age folder, e.g., 02_output
        param_name = path_parts[param_idx]  # e.g., 05.HAN_filter_parameters_i300_r10_t10_u5
        model_dir = path_parts[-1]  # e.g., region_specific
        
        # Scan all age directories
        ages = ['p3', 'p12', 'p20', 'p60']
        # Also try capitalized versions for robustness
        ages.extend(['P3', 'P12', 'P20', 'P60'])
        
        for age in ages:
            age_model_dir = os.path.join(base_dir, age, param_name, 'analysis', model_dir)
            if os.path.exists(age_model_dir):
                ages_found.append(age)
                # Scan this age directory for CSV files
                try:
                    for fname in os.listdir(age_model_dir):
                        # Match pattern: *_ALL_*_upsetplot_{model_suffix}.csv (aggregate samples only)
                        if "_ALL_" in fname and fname.endswith(f"_upsetplot_{model_suffix}.csv"):
                            fpath = os.path.join(age_model_dir, fname)
                            try:
                                df = pd.read_csv(fpath)
                                if "Effect Size" in df.columns:
                                    # Extract effect sizes, filtering out any non-numeric values
                                    effect_sizes = df["Effect Size"].dropna()
                                    # Convert to numeric, coercing errors to NaN
                                    effect_sizes = pd.to_numeric(effect_sizes, errors='coerce').dropna()
                                    all_effect_sizes.extend(effect_sizes.tolist())
                                    files_scanned += 1
                            except Exception as e:
                                # Skip files that can't be read
                                continue
                except Exception as e:
                    # Skip directories that can't be accessed
                    continue
    else:
        # Fall back to scanning only current directory if parameterization cannot be extracted
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                # Match pattern: *_ALL_*_upsetplot_{model_suffix}.csv (aggregate samples only)
                if "_ALL_" in fname and fname.endswith(f"_upsetplot_{model_suffix}.csv"):
                    fpath = os.path.join(output_dir, fname)
                    try:
                        df = pd.read_csv(fpath)
                        if "Effect Size" in df.columns:
                            # Extract effect sizes, filtering out any non-numeric values
                            effect_sizes = df["Effect Size"].dropna()
                            # Convert to numeric, coercing errors to NaN
                            effect_sizes = pd.to_numeric(effect_sizes, errors='coerce').dropna()
                            all_effect_sizes.extend(effect_sizes.tolist())
                            files_scanned += 1
                    except Exception as e:
                        # Skip files that can't be read
                        continue
    
    # If no effect sizes found, return None (will use per-plot range)
    if not all_effect_sizes:
        return None, []
    
    # Calculate global min and max
    global_min = min(all_effect_sizes)
    global_max = max(all_effect_sizes)
    
    # Symmetric range for all effect sizes (centered at 0)
    # Raw proportional uses (ratio - 1), so it's also centered at 0 like log2 version
    max_abs = max(abs(global_min), abs(global_max))
    unified_range = max_abs * 1.1
    return (-unified_range, unified_range), all_effect_sizes

def validate_unified_range(xlim, model_suffix, current_effect_sizes, all_effect_sizes):
    """
    Validate that unified x-axis range makes sense.
    
    Args:
        xlim: Tuple (x_min, x_max) for unified x-axis limits
        model_suffix: Model type (e.g., 'region_specific', 'proportional_effectsize', 'uniform')
        current_effect_sizes: List of effect sizes from current sample
        all_effect_sizes: List of all effect sizes collected from all samples
    
    Returns:
        (is_valid, warnings_list): Tuple of boolean and list of warning messages
    """
    warnings = []
    
    if xlim is None:
        return True, []  # No range to validate
    
    # Check range is not too extreme
    range_size = xlim[1] - xlim[0]
    if model_suffix == 'proportional_effectsize' and range_size > 2.0:
        warnings.append(f"⚠️ Proportional effect size range is unusually large: {range_size:.4f}")
    elif model_suffix in ['region_specific', 'uniform'] and range_size > 20.0:
        warnings.append(f"⚠️ {model_suffix} range is unusually large: {range_size:.4f}")
    
    # Check symmetry
    if abs(xlim[0] + xlim[1]) > 0.01:
        warnings.append(f"⚠️ Range is not symmetric: [{xlim[0]:.4f}, {xlim[1]:.4f}]")
    
    # Check range is not zero or near-zero
    if range_size < 0.001:
        warnings.append(f"⚠️ Range is near-zero: {range_size:.4f} (all data may be at zero)")
    
    # Check data coverage
    if current_effect_sizes:
        outside_range = [es for es in current_effect_sizes if es < xlim[0] or es > xlim[1]]
        if outside_range:
            warnings.append(f"⚠️ {len(outside_range)} data points outside calculated range")
    
    # Check range encompasses all data
    if all_effect_sizes:
        data_min = min(all_effect_sizes)
        data_max = max(all_effect_sizes)
        if data_min < xlim[0] or data_max > xlim[1]:
            warnings.append(f"⚠️ Calculated range does not fully encompass data: data=[{data_min:.4f}, {data_max:.4f}], range={xlim}")
    
    return len(warnings) == 0, warnings

# ============================================================================
# Function to generate effect significance (volcano) plot
# ============================================================================
def plot_effect_significance(sigs_data, slabels_data, output_dir, model_suffix, sample_name, alpha_val=alpha, xlabel=None, xlim=None, ylim=None, illustrator_output_dir=None, illustrator_report_ranges_only=False, illustrator_xlim=None, illustrator_ylim=None):
    """
    Generate effect significance (volcano) plot for a given model.
    
    Args:
        sigs_data: List of (effect_size, significance) tuples
        slabels_data: List of motif labels
        output_dir: Directory to save plots
        model_suffix: Suffix for filename (e.g., 'uniform' or 'region_specific')
        sample_name: Sample name for plot title and filename
        alpha_val: Significance threshold for Bonferroni correction
        xlabel: Optional custom x-axis label. If None, uses default label.
        xlim: Optional tuple (x_min, x_max) for unified x-axis limits. If None, calculates from current data.
        ylim: Optional tuple (y_min, y_max) for y-axis limits. If None, uses data-based range.
        illustrator_output_dir: If set, also save an illustrator-ready SVG (fixed axes, no title/axis/tick labels, Helvetica text).
        illustrator_report_ranges_only: If True with illustrator_output_dir, append data range to _data_ranges.csv and do not save SVG.
        illustrator_xlim: Optional (xmin, xmax) for illustrator export. If None, uses (-4, 4).
        illustrator_ylim: Optional (ymin, ymax) for illustrator export. If None, uses (0, 10).
    """
    from adjustText import adjust_text
    
    # Bonferroni correction: p-threshold / Num comparisons
    pcutoff = -1 * np.log10(alpha_val / len(slabels_data))
    
    list_sig = [i for (i, (e, s)) in enumerate(sigs_data) if s > pcutoff]
    color_labels = ['gray' for i in range(len(sigs_data))]
    for i in list_sig:
        e, s = sigs_data[i]
        if e > 0:  # overrepresented
            color_labels[i] = 'red'
        else:
            color_labels[i] = 'blue'
    
    hide_singlets = True
    if hide_singlets:
        mask = [i for (i, l) in enumerate(slabels_data) if len(l) > 1]
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 20)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    ax.set_title(sample_name.replace('_', ''), fontsize=16)
    
    # Use custom xlabel if provided, otherwise use default
    if xlabel is None:
        xlabel = "Effect Size \n$log_2($observed/expected$)$"
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Significance\n $-log_{10}(P)$", fontsize=16)
    ax.axhline(y=pcutoff, linestyle='--')
    ax.axvline(x=0, linestyle='--')
    pvalue_text = ax.text(x=-.5, y=pcutoff + 0.05, s='P-value cutoff', fontsize=16)
    
    # Scatter plot
    ax.scatter(*zip(*subset_list(sigs_data, mask)), c=subset_list(color_labels, mask))
    
    # Prepare text labels
    pretty_slabels = concatenate_list_data(subset_list(slabels_data, mask))
    coordinates = subset_list(sigs_data, mask)
    texts = []
    
    for n, (z, y) in enumerate(coordinates):
        txt = pretty_slabels[n]
        texts.append(ax.text(z, y, txt, fontsize=12))
    
    # Adjust y-axis limits
    y_vals = [y for _, y in subset_list(sigs_data, mask)]
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        padding = 0.1 * (max(y_vals) - min(y_vals))  # Add 10% padding
        ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)
    
    # Adjust x-axis limits
    if xlim is not None:
        # Use provided unified limits for cross-cohort comparisons
        ax.set_xlim(xlim[0], xlim[1])
    else:
        # Calculate from current data (original behavior)
        x_vals = [x for x, _ in subset_list(sigs_data, mask)]
        x_padding = 0.1 * (max(x_vals) - min(x_vals))  # 10% padding
        
        x_min = min(x_vals) - x_padding
        x_max = max(x_vals) + x_padding
        
        # Ensure symmetric padding if range is around 0
        x_abs_max = max(abs(x_min), abs(x_max))
        ax.set_xlim(-x_abs_max, x_abs_max)
    
    # Adjust text positions to avoid overlap
    adjust_text(
        texts,
        expand_points=(1.5, 2.5),  # Add padding around points
        force_text=1,  # Increase separation force for text
        force_points=1  # Increase separation force for points
    )
    
    # Save plots
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(os.path.normpath(os.path.join(output_dir, f"{sample_name}_effect_significance_{model_suffix}.{ext}")))
    
    # Optionally report data range only (for computing uniform limits) or save illustrator-ready version
    if illustrator_output_dir is not None:
        os.makedirs(illustrator_output_dir, exist_ok=True)
        masked_sigs = subset_list(sigs_data, mask)
        x_vals = [x for x, _ in masked_sigs]
        y_vals = [y for _, y in masked_sigs]
        data_x_min, data_x_max = min(x_vals), max(x_vals)
        data_y_min, data_y_max = min(y_vals), max(y_vals)

        if illustrator_report_ranges_only:
            ranges_file = os.path.normpath(os.path.join(illustrator_output_dir, "_data_ranges.csv"))
            file_exists = os.path.isfile(ranges_file)
            with open(ranges_file, "a") as f:
                if not file_exists:
                    f.write("sample_name,x_min,x_max,y_min,y_max\n")
                f.write(f"{sample_name},{data_x_min},{data_x_max},{data_y_min},{data_y_max}\n")
            plt.close(fig)
            return

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelbottom=False, labelleft=False, length=0)
        pvalue_text.remove()
        xlim_use = illustrator_xlim if illustrator_xlim is not None else (-4.0, 4.0)
        ylim_use = illustrator_ylim if illustrator_ylim is not None else (0.0, 10.0)
        ax.set_xlim(xlim_use[0], xlim_use[1])
        ax.set_ylim(ylim_use[0], ylim_use[1])
        for t in texts:
            t.set_fontfamily('sans-serif')
            t.set_fontname('Helvetica')
        out_path = os.path.normpath(os.path.join(illustrator_output_dir, f"{sample_name}_effect_significance_{model_suffix}.svg"))
        # Keep text as editable <text> elements in SVG (not converted to paths)
        with mpl.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(out_path, format='svg')
        print(f"Illustrator volcano scales for {sample_name}: x=[{xlim_use[0]}, {xlim_use[1]}], y=[{ylim_use[0]}, {ylim_use[1]}]")
    plt.close(fig)

# Calculate unified x-axis ranges for each model type
print("\n" + "="*80)
print("Calculating unified x-axis ranges for cross-cohort comparisons...")
print("="*80)

# Extract current effect sizes from each model
current_effect_sizes_uniform = [e for e, _ in sigs]
current_effect_sizes_rs = [e for e, _ in sigs_rs]
current_effect_sizes_prop = [e for e, _ in sigs_prop]
current_effect_sizes_prop_raw = [e for e, _ in sigs_prop_raw]

# Calculate unified ranges for each model type
xlim_uniform, all_effect_sizes_uniform = calculate_unified_xaxis_range(uniform_plot_dir, 'uniform', current_effect_sizes_uniform)
xlim_rs, all_effect_sizes_rs = calculate_unified_xaxis_range(region_specific_plot_dir, 'region_specific', current_effect_sizes_rs)
xlim_prop, all_effect_sizes_prop = calculate_unified_xaxis_range(region_specific_plot_dir, 'proportional_effectsize', current_effect_sizes_prop)
xlim_prop_raw, all_effect_sizes_prop_raw = calculate_unified_xaxis_range(region_specific_plot_dir, 'proportional_effectsize_raw', current_effect_sizes_prop_raw)

# Validate ranges
print("\n" + "-"*80)
print("Validating unified x-axis ranges...")
print("-"*80)

is_valid_uniform, warnings_uniform = validate_unified_range(xlim_uniform, 'uniform', current_effect_sizes_uniform, all_effect_sizes_uniform)
is_valid_rs, warnings_rs = validate_unified_range(xlim_rs, 'region_specific', current_effect_sizes_rs, all_effect_sizes_rs)
is_valid_prop, warnings_prop = validate_unified_range(xlim_prop, 'proportional_effectsize', current_effect_sizes_prop, all_effect_sizes_prop)

# Log validation results
if warnings_uniform:
    for warning in warnings_uniform:
        print(f"[UNIFORM] {warning}")
if warnings_rs:
    for warning in warnings_rs:
        print(f"[REGION-SPECIFIC] {warning}")
if warnings_prop:
    for warning in warnings_prop:
        print(f"[PROPORTIONAL EFFECT SIZE] {warning}")

# Cross-model type validation
print("\n" + "-"*80)
print("Cross-model type range comparison...")
print("-"*80)

if xlim_prop and xlim_rs:
    prop_range = xlim_prop[1] - xlim_prop[0]
    rs_range = xlim_rs[1] - xlim_rs[0]
    if prop_range > rs_range * 0.5:  # Proportional should be much smaller
        print(f"⚠️ WARNING: Proportional effect size range ({prop_range:.4f}) is unexpectedly large compared to region_specific ({rs_range:.4f})")
    elif prop_range < rs_range * 0.1:  # But shouldn't be too small either
        print(f"ℹ️ INFO: Proportional effect size range ({prop_range:.4f}) is {rs_range/prop_range:.1f}x smaller than region_specific ({rs_range:.4f}) - expected behavior")

# Verify ranges are consistent across similar model types
if xlim_uniform and xlim_rs:
    uniform_range = xlim_uniform[1] - xlim_uniform[0]
    rs_range = xlim_rs[1] - xlim_rs[0]
    range_ratio = max(uniform_range, rs_range) / min(uniform_range, rs_range)
    if range_ratio > 2.0:  # Ranges should be similar
        print(f"⚠️ WARNING: Uniform range ({uniform_range:.4f}) and region_specific range ({rs_range:.4f}) differ significantly (ratio: {range_ratio:.2f})")
    else:
        print(f"ℹ️ INFO: Uniform range ({uniform_range:.4f}) and region_specific range ({rs_range:.4f}) are consistent (ratio: {range_ratio:.2f})")

# Log calculated ranges
print("\n" + "-"*80)
print("Calculated unified x-axis ranges:")
print("-"*80)
if xlim_uniform:
    print(f"✅ [UNIFORM] Unified x-axis range: [{xlim_uniform[0]:.4f}, {xlim_uniform[1]:.4f}]")
else:
    print(f"⚠️ [UNIFORM] No existing data found, using per-plot range")
if xlim_rs:
    print(f"✅ [REGION-SPECIFIC] Unified x-axis range: [{xlim_rs[0]:.4f}, {xlim_rs[1]:.4f}]")
else:
    print(f"⚠️ [REGION-SPECIFIC] No existing data found, using per-plot range")
if xlim_prop:
    print(f"✅ [PROPORTIONAL EFFECT SIZE] Unified x-axis range: [{xlim_prop[0]:.4f}, {xlim_prop[1]:.4f}]")
else:
    print(f"⚠️ [PROPORTIONAL EFFECT SIZE] No existing data found, using per-plot range")
if xlim_prop_raw:
    print(f"✅ [PROPORTIONAL EFFECT SIZE (raw)] Unified x-axis range: [{xlim_prop_raw[0]:.4f}, {xlim_prop_raw[1]:.4f}]")
else:
    print(f"⚠️ [PROPORTIONAL EFFECT SIZE (raw)] No existing data found, using per-plot range")

# Generate effect significance plots for both models
print("\n" + "="*80)
print("Generating effect significance plots for both models...")
print("="*80)

# Uniform model plot
plot_effect_significance(
    sigs, slabels, uniform_plot_dir, 'uniform', sample_name, alpha, xlim=xlim_uniform,
    illustrator_output_dir=getattr(args, 'illustrator_volcano_dir', None),
    illustrator_report_ranges_only=getattr(args, 'illustrator_report_ranges_only', False),
    illustrator_xlim=tuple(args.illustrator_xlim) if getattr(args, 'illustrator_xlim', None) else None,
    illustrator_ylim=tuple(args.illustrator_ylim) if getattr(args, 'illustrator_ylim', None) else None,
)
print(f"✅ [UNIFORM MODEL] Effect significance plot saved to {uniform_plot_dir}")

# Region-specific model plot
plot_effect_significance(sigs_rs, slabels_rs, region_specific_plot_dir, 'region_specific', sample_name, alpha, xlim=xlim_rs)
print(f"✅ [REGION-SPECIFIC MODEL] Effect significance plot saved to {region_specific_plot_dir}")

# Proportional effect size model plot (log2 x-axis)
plot_effect_significance(
    sigs_prop, slabels_prop, region_specific_plot_dir, 
    'proportional_effectsize', sample_name, alpha,
    xlabel="Effect Size (Proportional)\n$log_2(\\frac{k/n_0 + 1}{\\pi + 1})$",
    xlim=xlim_prop
)
print(f"✅ [PROPORTIONAL EFFECT SIZE MODEL] Effect significance plot saved to {region_specific_plot_dir}")

# Proportional effect size model plot (raw x-axis, no log2)
plot_effect_significance(
    sigs_prop_raw, slabels_prop_raw, region_specific_plot_dir,
    'proportional_effectsize_raw', sample_name, alpha,
    xlabel="Effect Size (Proportional, no log2)\n$\\frac{k/n_0 + 1}{\\pi + 1} - 1$",
    xlim=xlim_prop_raw
)
print(f"✅ [PROPORTIONAL EFFECT SIZE MODEL (raw)] Effect significance plot saved to {region_specific_plot_dir}")

#per cell projection strength
def gen_per_cell_plot(df, cell_ids, motif_labels, dcounts, expected,
                      savepath=plot_dir, hide_singlets=True, figsize=(16, 35),
                      sample_name=None, export_csvs=False, csv_dir=None):
    """
    This plots each cell of a given motif on the same plot as an individual line.
    Each line's points are the corresponding projection strengths at that region.
    Now also optionally saves raw data per motif as CSVs using the sample_name.
    
    Parameters:
        ...
        export_csvs : bool
            Whether to export raw data as CSVs.
        csv_dir : str or None
            Optional directory to save per-motif CSVs. Defaults to savepath's directory.
    """
    import pandas as pd
    import os
    import re

    raw_data_output = {}

    if hide_singlets:  # Only show motifs with two or more regions
        mask = [i for (i, l) in enumerate(motif_labels) if len(l) > 1]
        cell_ids = subset_list(cell_ids, mask)
        dcounts = subset_list(dcounts, mask)
        expected = subset_list(expected, mask)
        motif_labels = subset_list(motif_labels, mask)

    non0cell_ids = [(i, x) for (i, x) in enumerate(cell_ids) if len(x) > 0]
    obs_ex = [(dcounts[i], expected[i]) for i, _ in non0cell_ids]
    plot_titles = concatenate_list_data(motif_labels)

    ncols = 2
    nrows = int(np.ceil(len(non0cell_ids) / ncols))
    fig = plt.figure(figsize=figsize)

    # Determine CSV output directory
    if export_csvs:
        csv_outdir = csv_dir if csv_dir else os.path.dirname(savepath)
        os.makedirs(csv_outdir, exist_ok=True)

    n = 1
    for i, cellids in non0cell_ids:
        ax = fig.add_subplot(nrows, ncols, n)
        title = plot_titles[i]
        ax.set_title(title)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.to_list(), rotation=90)
        ax.set_ylabel("Projection Strength")

        x = df.iloc[cellids, :].to_numpy()

        # ⬇️ OPTIONAL CSV EXPORT
        if export_csvs and sample_name:
            df_raw = pd.DataFrame(
                x,
                index=[f'cell_{cid}' for cid in cellids],
                columns=df.columns
            )
            safe_title = re.sub(r'\W+', '_', title)[:100]
            fname = f"{sample_name}_{safe_title}_raw_data.csv"
            full_path = os.path.join(csv_outdir, fname)
            df_raw.to_csv(full_path)
            print(f"[CSV EXPORT] Wrote: {full_path}")

        # Add observed/expected legend
        obs, ex = obs_ex[n - 1]
        textstr = f'Observed: {int(obs)} \n Expected: {int(ex)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        yerr = x.std(axis=0) / np.sqrt(x.shape[0])
        for j in range(x.shape[0]):
            ax.plot(np.arange(df.shape[1]), x[j], markerfacecolor='none', alpha=0.2, c='gray')
        ax.errorbar(x=np.arange(df.shape[1]), y=x.mean(axis=0), yerr=yerr,
                    ecolor='gray', c='black', linewidth=3)
        n += 1

    if savepath:
        # Always save in multiple formats (PNG, PDF, SVG) for compatibility
        # Pattern matches other analysis files: {sample_name}_{descriptive_name}.{ext}
        # savepath should already be constructed as: plot_dir/{sample_name}_per_cell_proj_strength
        
        if os.path.isdir(savepath):
            # If savepath is a directory, construct filename using sample_name
            if sample_name:
                base_path = os.path.normpath(os.path.join(savepath, f"{sample_name}_per_cell_proj_strength"))
            else:
                raise ValueError("Cannot save plot: savepath is a directory but sample_name is not provided")
        else:
            # Remove extension if present to get base path
            base_path = savepath.rsplit('.', 1)[0] if '.' in savepath else savepath
            
            # Always ensure we're saving to plot_dir (analysis directory) with proper sample_name
            # This prevents issues where savepath might be malformed or truncated
            if sample_name and plot_dir:
                # Normalize paths for comparison
                abs_base = os.path.abspath(base_path)
                abs_plot_dir = os.path.abspath(plot_dir)
                
                # If base_path doesn't start with plot_dir, or doesn't contain sample_name, reconstruct it
                if not abs_base.startswith(abs_plot_dir) or sample_name not in os.path.basename(base_path):
                    base_path = os.path.normpath(os.path.join(plot_dir, f"{sample_name}_per_cell_proj_strength"))
        
        # Ensure the directory exists (should be plot_dir/analysis)
        output_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else plot_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in all formats (matching pattern of other analysis outputs)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(f"{base_path}.{ext}", format=ext)

    return ax

# Generate per_cell_proj_strength plots for both models
print("\n" + "="*80)
print("Generating per_cell_proj_strength plots for both models...")
print("="*80)

# Uniform model per_cell_proj_strength plot
gprcpplot_uniform = gen_per_cell_plot(
    df, cell_ids, motif_labels, dcounts, exp_counts,
    figsize=(20, 5 * len([m for m in motif_labels if len(m) > 1])),
    savepath=os.path.normpath(os.path.join(uniform_plot_dir, sample_name + "_per_cell_proj_strength_uniform")),
    sample_name=sample_name,
    export_csvs=True,
    csv_dir=csv_output_dir
)
print(f"✅ [UNIFORM MODEL] Per-cell projection strength plot saved to {uniform_plot_dir}")

# Region-specific model per_cell_proj_strength plot
gprcpplot_rs = gen_per_cell_plot(
    df, cell_ids, motif_labels, dcounts, exp_counts_rs,
    figsize=(20, 5 * len([m for m in motif_labels if len(m) > 1])),
    savepath=os.path.normpath(os.path.join(region_specific_plot_dir, sample_name + "_per_cell_proj_strength_region_specific")),
    sample_name=sample_name,
    export_csvs=False,  # Don't duplicate CSV exports
    csv_dir=csv_output_dir
)
print(f"✅ [REGION-SPECIFIC MODEL] Per-cell projection strength plot saved to {region_specific_plot_dir}")


from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import seaborn as sns
import re
import sys

# Workaround for Windows recursion bug and large datasets
# Increase recursion limit for large datasets (p60 has ~20k cells)
n_cells = df.shape[0]
if n_cells > 10000:
    sys.setrecursionlimit(10000)
    print(f"⚠️ Large dataset detected ({n_cells} cells). Increased recursion limit to 10000.")
else:
    sys.setrecursionlimit(5000)

### === Green-White Cluster Heatmap === ###
print("🔍 Generating Green-White cluster heatmap...")

# Dynamically build full order list
order_full = [col for pattern in ['LM', 'AL', 'RL', 'A', 'AM', 'PM', 'RSP']
              for col in df.columns if re.match(f"{pattern}\\d*", col, re.IGNORECASE)]
order_full = list(dict.fromkeys(order_full))

if not order_full:
    raise ValueError("❌ No matching columns found for green-white cluster heatmap.")

order_partial = ['LM', 'AL', 'RL', 'AM', 'PM']
order_partial = [col for col in order_partial if col in df.columns]

print(f"Adjusted order_full: {order_full}")
print(f"Adjusted order_partial: {order_partial}")

df_ = df[order_full] if full_data else df[order_partial]
print(f"Adjusted df_ columns: {df_.columns.tolist()}")

# Normalize
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_.astype(float)),
    columns=df_.columns
)

# Ensure clean native float matrix
df_scaled_np = df_scaled.to_numpy(copy=True).astype(float)

# Colormap
grn_white_cm = LinearSegmentedColormap.from_list('white_to_green', ['white', 'green'], N=100)

# Drop constant or all-zero rows
df_scaled = df_scaled.loc[df_scaled.var(axis=1) > 0]

# Final check before clustering
if df_scaled.shape[0] < 2:
    raise ValueError("❌ Too few rows remaining after variance filtering to perform clustering.")

# For very large datasets, subsample or disable row clustering to avoid recursion errors
n_rows = df_scaled.shape[0]
if n_rows > 15000:
    print(f"⚠️ Very large dataset ({n_rows} rows). Subsampling to 10000 rows for clustering to avoid recursion errors.")
    # Randomly subsample to 10000 rows
    np.random.seed(42)
    sample_indices = np.random.choice(n_rows, size=10000, replace=False)
    df_scaled = df_scaled.iloc[sample_indices].reset_index(drop=True)
    print(f"   Subsampled to {df_scaled.shape[0]} rows for clustering.")

# Draw clustermap with error handling
clusterfig = None
use_simple_heatmap = False

try:
    clusterfig = sns.clustermap(
        df_scaled,
        col_cluster=False,
        metric='cosine',
        method='average',
        cbar_kws=dict(label='Projection Strength'),
        cmap=grn_white_cm,
        vmin=0.0,
        vmax=1.0
    )
    clusterfig.ax_heatmap.set_title(sample_name.replace('_', ' '))
    clusterfig.ax_heatmap.axes.get_yaxis().set_visible(False)
except RecursionError as e:
    print(f"⚠️ RecursionError during clustering for large dataset ({n_rows} rows).")
    print(f"   Attempting with row_cluster=False to avoid dendrogram calculation...")
    try:
        # Fallback: disable row clustering to avoid dendrogram recursion
        clusterfig = sns.clustermap(
            df_scaled,
            row_cluster=False,
            col_cluster=False,
            cbar_kws=dict(label='Projection Strength'),
            cmap=grn_white_cm,
            vmin=0.0,
            vmax=1.0
        )
        clusterfig.ax_heatmap.set_title(sample_name.replace('_', ' '))
        clusterfig.ax_heatmap.axes.get_yaxis().set_visible(False)
    except Exception as e2:
        print(f"⚠️ Second attempt also failed: {e2}")
        print(f"   Using simple heatmap without clustering...")
        use_simple_heatmap = True
except Exception as e:
    print(f"⚠️ Error during clustermap generation: {e}")
    print(f"   Using simple heatmap without clustering...")
    use_simple_heatmap = True

if use_simple_heatmap:
    # Final fallback: simple heatmap without clustering
    fig, ax = plt.subplots(figsize=(12, max(8, min(n_rows / 200, 50))))
    sns.heatmap(
        df_scaled,
        cmap=grn_white_cm,
        vmin=0.0,
        vmax=1.0,
        cbar_kws=dict(label='Projection Strength'),
        ax=ax,
        yticklabels=False
    )
    ax.set_title(sample_name.replace('_', ' '))
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_green_white_cluster_heatmap.{ext}")))
    plt.close(fig)
else:
    for ext in ['pdf', 'svg', 'png']:
        clusterfig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_green_white_cluster_heatmap.{ext}")))
    plt.close(clusterfig.fig)

print("✅ Green-White cluster heatmap saved.")

### === Han-Style Heatmap === ###
print("🔍 Generating Han-style heatmap...")

# Han colormap
han_cm = LinearSegmentedColormap.from_list('white_to_green', ['white', 'green'], N=100)

# Define Han-style targets
han_targets = ['LM', 'AL', 'PM', 'AM', 'RL']
han_order_full = [col for pattern in han_targets for col in df.columns if re.match(f"{pattern}\\d*", col, re.IGNORECASE)]
han_order_full = list(dict.fromkeys(han_order_full))

if not han_order_full:
    raise ValueError("❌ No matching columns found for Han-style target area pattern.")

df_han = df[han_order_full] if full_data else df[[col for col in han_targets if col in df.columns]]
print(f"🧬 Han target columns: {df_han.columns.tolist()}")
print("Han df shape:", df_han.shape)

# Log-transform and normalize
df_han = np.log1p(df_han + 1e-3)
df_han = df_han.div(df_han.max(axis=1), axis=0)

if df_han.isnull().values.any():
    raise ValueError("❌ NaNs found in df_han after normalization.")

# Filter out zero-variance rows
df_han = df_han.loc[df_han.var(axis=1) > 0].reset_index(drop=True)
if df_han.shape[0] < 2:
    raise ValueError("❌ Not enough valid rows in df_han after filtering.")

# Sort rows by max projection column index
df_han['max_proj_col'] = df_han.values.argmax(axis=1)
df_han = df_han.sort_values('max_proj_col').drop(columns='max_proj_col').reset_index(drop=True)

# Linkage use if you want dendrogram sorting
#row_linkage = linkage(pdist(df_han, metric='euclidean'), method='ward')

# Ensure clean float type in DataFrame
df_han = df_han.astype(float)

# Draw Han-style heatmap
clusterfig_han = sns.clustermap(
    df_han,
    row_cluster=False,
    col_cluster=False,
    cmap=han_cm,
    vmin=0.0,
    vmax=1.0,
    cbar_kws=dict(label='Projection Strength')
)


clusterfig_han.ax_heatmap.set_title(sample_name.replace('_', ' ') + ' (Han-style)')
clusterfig_han.ax_heatmap.axes.get_yaxis().set_visible(False)

for ext in ['pdf', 'svg', 'png']:
    clusterfig_han.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_Hanstyle_cluster_heatmap.{ext}")))

print("✅ Han-style heatmap generated and saved.")


def gen_prob_matrix(df : pd.DataFrame):
    data = df.to_numpy(copy=True)
    cells,regions = data.shape
    mat = np.zeros((regions,regions)) #area B x area A
    #loop over columns (region )
    for col in range(regions):
        #find all cells (rows in data) that project to 'col'
        ids_col = np.where(data[:,col] != 0)[0]
        sub_col = data[ids_col]
        #of these, how many project to region B
        for row in range(regions):
            ids_row = np.where(sub_col[:,row] != 0)[0]
            if ids_col.shape[0] == 0:
                prob = 0
            else:
                prob = ids_row.shape[0] / ids_col.shape[0]
            #print("P({} | {}) = {}".format(df.columns[row],df.columns[col],prob))
            mat[col,row] = prob
    mat = pd.DataFrame(mat, columns=df.columns)
    mat.index = df.columns
    return mat

probmat = gen_prob_matrix(df)

# Save conditional probability matrix for correlated model
probmat_path = os.path.join(out_dir, f"{sample_name}_Conditional_Probability_Matrix.csv")
probmat.to_csv(probmat_path)
print(f"💾 Saved conditional probability matrix to: {probmat_path}")

fig, ax = plt.subplots(figsize=(10,10))
ax.set_title(sample_name.replace('_',''),fontsize=20)
colors2 = ['darkblue','#1f9ed1','#26ffc5','#ffc526','yellow']
cm2 = LinearSegmentedColormap.from_list(
        'white_to_red', colors2, N=100)
ax.set_facecolor('#a8a8a8')
ax = sn.heatmap(probmat.T,mask=probmat.T == 1,ax=ax,cbar_kws=dict(label='$P(B | A)$'),cmap=cm2) #can add vmax=number for scale
ax.set_xlabel("Area A",fontsize=16)
ax.set_ylabel("Area B",fontsize=16)
#plt.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_blueyellow_probability_heatmap.pdf")))
for ext in ["pdf", "svg", "png"]:
    plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_blueyellow_probability_heatmap.{ext}")))


def remove_zero_rows(df):
    df_ = df.fillna(0)
    df = df.loc[~(df_==0).all(axis=1)].astype('float32')
    return df

def get_overlaps(df):
    """
    Returns the number of cells that target both regions in a pair
    """
    cells,regions = df.shape
    pairs = list(itertools.combinations(df.columns,2)) #remove null id
    pairs_unzip = list(zip(*pairs))
    from_r = list(pairs_unzip[0])
    to_r = list(pairs_unzip[1])
    counts=[]
    df = df.copy()
    for i in pairs:
        sub = df.T.loc[list(i)].T
        sub = remove_zero_rows(sub)
        counts.append(sub.shape[0])
    res = pd.DataFrame(columns=['from','to','value'])
    res['from'] = from_r
    res['to'] = to_r
    res['value'] = counts
    return res #counts, pairs

oo = get_overlaps(df)
oo.head()

def get_motif_count(motif,counts,labels):
    """
    Get the number of cells that project to this specific motif
    where motif is a list of column names e.g. ['LH','PFC']
    """
    for i in range(len(labels)):
        if set(motif) == set(labels[i]):
            return counts[i]

get_motif_count(['PM','AL'],dcounts,motif_labels)

import re
pattern = re.compile(r'([^\s\w]|_)+')

def strip_nonchars(string):
    strip = pattern.sub('', string)
    return strip

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def get_pair_reg_props(df,counts,labels):
    """
    Given each pair in region_list, find number of cells that target either in the pair
    and then find proportion of cells that target both in pair exclusively
    
    counts: motif counts
    labels: motif labels
    """
    region_list = df.columns.to_list()
    R = len(region_list)
    tot = df.shape[0] / 100
    pairs = findsubsets(region_list,2)
    results = []
    for i,pair in enumerate(pairs):
        p1,p2 = pair
        num_cells_p1 = df[df[p1] > 0.].shape[0]
        num_cells_p2 = df[df[p2] > 0.].shape[0]
        tot_cells = num_cells_p1 + num_cells_p2
        num_doublets = get_motif_count([p1,p2],counts,labels)
        perc = np.around((100.0 * num_doublets) / tot_cells,3)
        results.append((p1,p2,tot_cells,num_doublets,perc))
    return results

get_pair_reg_props(df,dcounts,motif_labels)

def get_all_counts(df,motifs,counts,labels):
    """
    Returns an array where each row is a motif and the counts of 
    number of cells targeting each member of the motif (non-exclusive), total number of cells targeting any of 
    the members of the motif, number of cells targeting all members of motif, and percentage exclusively targeting full
    motif (relative to any member of the motif), e.g.
    columns: PFC BNST LS CeA Total Motif Perc
    row 1  : 10   20  30  NA  60    6     10%
    where NA means that region is not part of the motif
    
    Input: df; dataframe of normalized data, Num cells (N) x Num regions (R)
    motifs M (num motifs) x R binary matrix indicating which regions present in each motif (row)
    counts vector containg counts of cells that exclusively project to each matching motif/row in motifs
    labels string labels for regions that make each matching motif in motifs
    """
    ret = pd.DataFrame(columns=df.columns.to_list() + ['Total', 'Motif Num', 'Motif Perc'])
    num_cols = len(ret.columns.to_list())
    for i,motif in enumerate(motifs): #loop through motifs
        m = [index for (index,x) in enumerate(motif) if x]
        if len(m) < 1:
            continue
        sums = df.iloc[:,m].astype(bool).astype(int).sum().to_numpy()
        ap = np.zeros(num_cols)
        ap[:] = np.nan
        ap[m] = sums
        ap = ap.reshape(1,ap.shape[0])
        ap = pd.DataFrame(ap,columns=ret.columns)
        tot = ap.iloc[:,0:-3].dropna(axis=1).to_numpy().sum()
        ap.iloc[:,-3] = tot
        ap.iloc[:,-2] = counts[i]
        # Safeguard against division by zero
        if tot == 0:
            ap.iloc[:, -1] = 0.0
        else:
            ap.iloc[:, -1] = 100.0 * (counts[i] / tot)
        ret = pd.concat([ret, ap], ignore_index=True)
    return ret

def get_all_counts_nondf(df,motifs,counts,labels):
    """
    Returns an array where each row is a motif and the columns are the counts of 
    number of cells targeting each member of the motif (non-exclusive), total number of cells targeting any of 
    the members of the motif, number of cells targeting all members of motif, and percentage exclusively targeting full
    motif (relative to any member of the motif), e.g.
    columns: PFC BNST LS CeA Total Motif Perc
    row 1  : 10   20  30  NA  60    6     10%
    where NA means that region is not part of the motif
    
    Input: df; dataframe of normalized data, Num cells (N) x Num regions (R)
    motifs M (num motifs) x R binary matrix indicating which regions present in each motif (row)
    counts vector containg counts of cells that exclusively project to each matching motif/row in motifs
    labels string labels for regions that make each matching motif in motifs
    """
    retdf = [] #return list
    #each element is a list [Labels, R1 count, R2 count ... Rn count, Total Count, Motif Count, Motif Perc]
    for i,motif in enumerate(motifs): #loop through motifs
        m = [index for (index,x) in enumerate(motif) if x]
        row = list(np.zeros(1+len(m)+3)) #1 (labels) + num-regions-in-motifs + 3 (total,motif count,motif perc)
        if len(m) < 1:
            continue
        sums = df.iloc[:,m].astype(bool).astype(int).sum().to_numpy()
        row[0] = labels[i]
        row[1:len(m)+1] = sums
        tot = sums.sum()
        
        # Prevent division by zero
        if tot == 0:
            row[len(m) + 1] = np.nan  # or 0, depending on how you want to handle this case
            row[len(m) + 2] = np.nan  # Handle the motif count
            row[len(m) + 3] = np.nan  # Handle the motif percentage
        else:
            row[len(m) + 1] = tot
            row[len(m) + 2] = counts[i]
            row[len(m) + 3] = 100.0 * (counts[i] / tot)
        
        retdf.append(row)
    return retdf

unstruct_counts = get_all_counts_nondf(df,motifs,dcounts,motif_labels)

def write_motif_counts(path,counts):
    with open(path, 'w') as f:
        for item in counts:
            f.write("%s\n" % item)
write_motif_counts(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_counts.txt")),
    unstruct_counts
)

mdf = get_all_counts(df,motifs,dcounts,motif_labels)

mdf.head()

mdf.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_counts.csv")))

def get_target_pie(df : pd.DataFrame):
    """
    For each cell (row), determine how many projections it makes
    """
    data = df.to_numpy(copy=True)
    cells,regions = data.shape
    res = []#np.zeros(regions)
    for cell in range(cells):
        num_targets = int(np.nonzero(data[cell])[0].shape[0])
        res.append(num_targets)
    ret = np.array(res)
    #ret = pd.DataFrame(ret)
    return ret

df_pie = get_target_pie(df)

g,c = np.unique(df_pie,return_counts=True)

c_row_names = ['1 target']
c_row_names += ["{} targets".format(i+2) for i in range(c.shape[0]-1)]
c = pd.DataFrame(c,columns=['# Cells'], index=c_row_names)
c_np = c.to_numpy(copy=True).flatten()
c.head()

c.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_pie_chart_data.csv")))

c_tot = c_np.sum()
c_tot

plt.figure(figsize=(10,10))
plt.title(sample_name.replace('_',''))
glabels = ["1 target \n {:0.3}%%".format(100*c_np[0] / c_tot)]
glabels += ["{} targets \n {:0.3}%%".format(i+2,100*j/c_tot) for (i,j) in zip(range(c_np.shape[0]-1),c_np[1:])]
patches, texts = plt.pie(c.to_numpy().flatten(),labels=glabels)
[txt.set_fontsize(8) for txt in texts]
#plt.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_num_targets_pie.pdf")))
for ext in ["pdf", "svg", "png"]:
    plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_num_targets_pie.{ext}")))

# Calculate appropriate perplexity based on sample size
n_samples = len(df)
perplexity = min(30, max(1, n_samples - 1))

# Only run t-SNE if we have enough samples
if n_samples > perplexity:
    maxproj = TSNE(n_components=2, metric='cosine', perplexity=perplexity).fit_transform(df.to_numpy(copy=True))
    
    #maxprojclusters = kmeans(X=maxproj,n_clusters=6)
    
    tlabels = df.to_numpy(copy=True).argmax(axis=1)
    #tlabels = km[1]
    
    plt.figure(figsize=(12,9))
    plt.title(sample_name.replace('_',''),fontsize=20)
    plt.xlabel("tSNE Component 1",fontsize=20)
    plt.ylabel("tSNE Component 2",fontsize=20)
    sc = plt.scatter(maxproj[:,0],maxproj[:,1],c=tlabels) #c=maxprojclusters[1]
    cb = plt.colorbar(sc)
    cb.set_label("Maximum Projection Target",fontsize=20)
    for ext in ["pdf", "svg", "png"]:
        plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_tsne.{ext}")))
    plt.close()
else:
    # Skip t-SNE for datasets that are too small
    print(f"⚠️ Skipping t-SNE visualization: insufficient samples ({n_samples} samples, need > {perplexity})")


def prepare_upset_data(df):
    #mask1 = [i for (i,x) in enumerate(motif_labels) if len(x) > 1]
    mask1 = [i for (i,x) in enumerate(df['Degree'].to_list()) if x > 1]
    a = subset_list(df['Motifs'].to_list(), mask1)
    b = df['Observed'][mask1]
    c = df['Expected'][mask1]
    d = df['Expected SD'][mask1]
    e = df['Effect Size'][mask1]
    f = df['P-value'][mask1]
    g = df['Group'][mask1]
    mask2 = [i for i in range(b.shape[0]) if b.iloc[i] > 0]
    a = subset_list(a, mask2)
    b = b.iloc[mask2]
    b = b.to_numpy().astype(int)
    #
    c = c.iloc[mask2]
    c = c.to_numpy().astype(int)
    #
    d = d.iloc[mask2]
    #
    e = e.iloc[mask2]
    #
    f = f.iloc[mask2]
    #
    g = g.iloc[mask2]
    dfdata = pd.DataFrame(data=[a,b,c,d,e,f,g]).T
    dfdata.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Group']
    #dfdata = dfdata.sort_values(by="Observed",ascending=False)
    return dfdata

sigsraw, slabelsraw = get_motif_sig_pts(dcounts,motif_labels,exclude_zeros=False, p_transform=lambda x:x)

effectsigsraw = np.array(sigsraw)
# Convert motif_probs values to float (handles sympy Float objects)
# motif_probs is a numpy array from get_expected_counts, indexed by motif_labels position
# slabelsraw contains motif labels (lists of regions), need to find their indices in motif_labels
expected_sd_raw = []
for i, motif_label in enumerate(slabelsraw):
    # Find the index in motif_labels that corresponds to this motif label
    try:
        motif_idx = motif_labels.index(motif_label)
        prob_val = float(motif_probs[motif_idx])
    except (ValueError, IndexError, TypeError):
        # Fallback: if index lookup fails, try direct array access (slabelsraw order should match)
        prob_val = float(motif_probs[i]) if i < len(motif_probs) else 0.0
    n0_val = float(n0)
    expected_sd_raw.append(np.sqrt(prob_val * n0_val * (1.0 - prob_val)))
expected_sd_raw = np.array(expected_sd_raw)

degree = [len(x) for x in motif_labels]
degree[0] = 0

group = []
bonferroni_correction = len(slabels)
for i in range(len(degree)):
    """
    Group 1 = significant + over-represented
    Group 2 = significant + under-represented
    Group 3 = non-significant + over-represented
    Group 4 = non-significant + under-represented
    """
    grp = 0
    thr = 0.05 / bonferroni_correction
    if effectsigsraw[i,0] > 0:  # over-represented
        if effectsigsraw[i,1] < thr:  # statistically significant
            grp = 1  # significant + over-represented
        else:
            grp = 3  # non-significant + over-represented
    else:  # under-represented
        if effectsigsraw[i,1] > thr:  # not significant
            grp = 4  # non-significant + under-represented
        else:  # statistically significant
            grp = 2  # significant + under-represented
    group.append(grp)

# ============================================================================
# UNIFORM MODEL: Upsetplot data preparation
# ============================================================================
print("\n" + "="*80)
print("[UNIFORM MODEL] Preparing upsetplot data...")
print("="*80)

dfraw = pd.DataFrame(data=[
                           motif_labels,\
                           dcounts,exp_counts.astype(int), \
                          expected_sd_raw,effectsigsraw[:,0], effectsigsraw[:,1], degree, group]).T
dfraw.columns=['Motifs','Observed','Expected', 'Expected SD','Effect Size', 'P-value', 'Degree', 'Group']
dfraw.to_csv(
    os.path.normpath(os.path.join(uniform_plot_dir, f"{sample_name}_upsetplot_uniform.csv")),
    index=False
)
print(f"💾 [UNIFORM MODEL] Saved upsetplot data to: {uniform_plot_dir}/{sample_name}_upsetplot_uniform.csv")

dfraw.iloc[40:70]

dfdata = prepare_upset_data(dfraw)
dfdata = dfdata.sort_values(by=['Group','Observed'], ascending=[True,False])

# ============================================================================
# REGION-SPECIFIC MODEL: Upsetplot data preparation (Old Pipeline Method)
# ============================================================================
print("\n" + "="*80)
print("[REGION-SPECIFIC MODEL] Preparing upsetplot data...")
print("="*80)

# Calculate raw significance values for region-specific model
sigsraw_rs, slabelsraw_rs = get_motif_sig_pts_region_specific(
    dcounts, motif_labels, motif_probs_region_specific, n0=n0, exclude_zeros=False, p_transform=lambda x: x
)

effectsigsraw_rs = np.array(sigsraw_rs)
# Expected SD using region-specific probabilities
# Convert to float to handle sympy objects
expected_sd_raw_rs = []
for i in range(len(slabelsraw_rs)):
    prob_val = float(motif_probs_rs_array[i])
    n0_val = float(n0)
    expected_sd_raw_rs.append(np.sqrt(prob_val * n0_val * (1.0 - prob_val)))
expected_sd_raw_rs = np.array(expected_sd_raw_rs)

# Degree is the same for both models
degree_rs = [len(x) for x in motif_labels]
degree_rs[0] = 0

# Group classification for region-specific model
group_rs = []
bonferroni_correction_rs = len(slabels_rs)
for i in range(len(degree_rs)):
    """
    Group 1 = significant + over-represented
    Group 2 = significant + under-represented
    Group 3 = non-significant + over-represented
    Group 4 = non-significant + under-represented
    """
    grp = 0
    thr = 0.05 / bonferroni_correction_rs
    if effectsigsraw_rs[i, 0] > 0:  # over-represented
        if effectsigsraw_rs[i, 1] < thr:  # statistically significant
            grp = 1  # significant + over-represented
        else:
            grp = 3  # non-significant + over-represented
    else:  # under-represented
        if effectsigsraw_rs[i, 1] > thr:  # not significant
            grp = 4  # non-significant + under-represented
        else:  # statistically significant
            grp = 2  # significant + under-represented
    group_rs.append(grp)

dfraw_rs = pd.DataFrame(data=[
    motif_labels,
    dcounts,
    exp_counts_rs.astype(int),
    expected_sd_raw_rs,
    effectsigsraw_rs[:, 0],
    effectsigsraw_rs[:, 1],
    degree_rs,
    group_rs
]).T
dfraw_rs.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Degree', 'Group']
dfraw_rs.to_csv(
    os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_region_specific.csv")),
    index=False
)
print(f"💾 [REGION-SPECIFIC MODEL] Saved upsetplot data to: {region_specific_plot_dir}/{sample_name}_upsetplot_region_specific.csv")

dfdata_rs = prepare_upset_data(dfraw_rs)
dfdata_rs = dfdata_rs.sort_values(by=['Group', 'Observed'], ascending=[True, False])
print(f"✅ [REGION-SPECIFIC MODEL] Upsetplot data prepared for {len(dfdata_rs)} motifs")

# ============================================================================
# PROPORTIONAL EFFECT SIZE MODEL: Upsetplot data preparation
# ============================================================================
print("\n" + "="*80)
print("[PROPORTIONAL EFFECT SIZE MODEL] Preparing upsetplot data...")
print("="*80)

# Calculate raw significance values for proportional effect size model
sigsraw_prop, slabelsraw_prop = get_motif_sig_pts_proportional_effectsize(
    dcounts, motif_labels, motif_probs_region_specific, n0=n0, exclude_zeros=False, p_transform=lambda x: x
)

effectsigsraw_prop = np.array(sigsraw_prop)
# Expected SD using region-specific probabilities (same as region_specific model)
# Convert to float to handle sympy objects
expected_sd_raw_prop = []
for i in range(len(slabelsraw_prop)):
    prob_val = float(motif_probs_rs_array[i])
    n0_val = float(n0)
    expected_sd_raw_prop.append(np.sqrt(prob_val * n0_val * (1.0 - prob_val)))
expected_sd_raw_prop = np.array(expected_sd_raw_prop)

# Degree is the same for all models
degree_prop = [len(x) for x in motif_labels]
degree_prop[0] = 0

# Group classification for proportional effect size model
group_prop = []
bonferroni_correction_prop = len(slabels_prop)
for i in range(len(degree_prop)):
    """
    Group 1 = significant + over-represented
    Group 2 = significant + under-represented
    Group 3 = non-significant + over-represented
    Group 4 = non-significant + under-represented
    """
    grp = 0
    thr = 0.05 / bonferroni_correction_prop
    if effectsigsraw_prop[i, 0] > 0:  # over-represented
        if effectsigsraw_prop[i, 1] < thr:  # statistically significant
            grp = 1  # significant + over-represented
        else:
            grp = 3  # non-significant + over-represented
    else:  # under-represented
        if effectsigsraw_prop[i, 1] > thr:  # not significant
            grp = 4  # non-significant + under-represented
        else:  # statistically significant
            grp = 2  # significant + under-represented
    group_prop.append(grp)

dfraw_prop = pd.DataFrame(data=[
    motif_labels,
    dcounts,
    exp_counts_rs.astype(int),
    expected_sd_raw_prop,
    effectsigsraw_prop[:, 0],
    effectsigsraw_prop[:, 1],
    degree_prop,
    group_prop
]).T
dfraw_prop.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Degree', 'Group']
dfraw_prop.to_csv(
    os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_proportional_effectsize.csv")),
    index=False
)
print(f"💾 [PROPORTIONAL EFFECT SIZE MODEL] Saved upsetplot data to: {region_specific_plot_dir}/{sample_name}_upsetplot_proportional_effectsize.csv")

dfdata_prop = prepare_upset_data(dfraw_prop)
dfdata_prop = dfdata_prop.sort_values(by=['Group', 'Observed'], ascending=[True, False])
print(f"✅ [PROPORTIONAL EFFECT SIZE MODEL] Upsetplot data prepared for {len(dfdata_prop)} motifs")


#upsetplot fxn
def kplot(df, size=(30,12)):
    """
    data : pd.DataFrame
    data is a dataframe with columns "Motifs" and "Counts"
    where "Motifs" is a list of lists e.g. [['PFC','LS'],['LS']]
    and "Counts" is a simple array of integers
    """
    motiflabels = df['Motifs'].to_list()
    data = up.from_memberships(motiflabels,data=df['Observed'].to_numpy())
    xlen = df.shape[0]
    xticks = np.arange(xlen)
    uplot = up.UpSet(data, sort_by=None) #sort_by='cardinality'
    fig,ax=plt.subplots(2,2,gridspec_kw={'width_ratios': [1, 3], 'height_ratios':[3,1]})
    fig.set_size_inches(size)
    ax[1,0].set_ylabel("Set Totals")
    uplot.plot_matrix(ax[1,1])
    uplot.plot_totals(ax[1,0])
    ax[0,0].axis('off')
    ax[0,1].spines['bottom'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    width=0.35
    dodge=width/2
    x = np.arange(8)
    ax[1,0].set_title("Totals")
    ax[0,1].set_ylabel("Counts")
    ax[0,1].set_xlim(ax[1,1].get_xlim())
    ox = xticks-dodge
    ex = xticks+dodge
    #colorlist = ['cyan','darkgray','darkgray','red']
    colorlist = ['red','darkblue','black','black']
    cs = [colorlist[i-1] for i in df['Group']]
    ax[0,1].bar(ox,df['Observed'].to_numpy(),width=width,label="Observed", align="center",color=cs, edgecolor='lightgray')
    ax[0,1].bar(ex,df['Expected'].to_numpy(),yerr=df['Expected SD'].to_numpy(),width=width/2,label="Expected", align="center",color='gray',alpha=0.5,ecolor='lightgray')
    grp_ = dfdata['Group'].to_numpy()
    idsig = np.concatenate([np.where(grp_ == 1)[0],np.where(grp_ == 2)[0]])
    [ax[0,1].text(ox[idsig][i]-0.5*dodge,df['Observed'].to_numpy()[idsig][i]+1,s="*") for i in range(idsig.shape[0])]
    # 
    ax[0,1].xaxis.grid(False)
    ax[0,1].xaxis.set_visible(False)
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].xaxis.grid(False)
    #ax[0,1].legend()
    fig.tight_layout()
    return fig,ax

# Generate upsetplot visual plots for both models
print("\n" + "="*80)
print("Generating upsetplot visual plots for both models...")
print("="*80)

# Uniform model upsetplot
fig_uniform, _ = kplot(dfdata)
for ext in ["pdf", "svg", "png"]:
    fig_uniform.savefig(os.path.normpath(os.path.join(uniform_plot_dir, f"{sample_name}_upsetplot_uniform.{ext}")))
plt.close(fig_uniform)
print(f"✅ [UNIFORM MODEL] Upsetplot saved to {uniform_plot_dir}")

# Region-specific model upsetplot
fig_rs, _ = kplot(dfdata_rs)
for ext in ["pdf", "svg", "png"]:
    fig_rs.savefig(os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_region_specific.{ext}")))
plt.close(fig_rs)
print(f"✅ [REGION-SPECIFIC MODEL] Upsetplot saved to {region_specific_plot_dir}")

# ============================================================================
# CORRELATED MODEL: Compute and save outputs using conditional probabilities
# ============================================================================
# Only compute if we have a conditional probability matrix (either from anchor or local)
if 'probmat' in dir() and probmat is not None:
    print("\n" + "="*80)
    print("METHOD 3: CORRELATED BINOMIAL PROBABILITY MODEL")
    print("="*80)
    
    # Use anchor correlation matrix if provided, otherwise use local probmat
    cond_prob_matrix = corr_matrix_for_expected if corr_matrix_for_expected is not None else probmat
    
    # Compute correlated model motif probabilities
    motif_probs_correlated = compute_motif_probabilities_correlated(
        motif_labels, psdict_for_expected, cond_prob_matrix, columns
    )
    print(f"🔍 [CORRELATED MODEL] Computed probabilities for {len(motif_probs_correlated)} motifs")
    
    # Convert to array for expected count calculation
    motif_probs_corr_array = np.array([motif_probs_correlated.get(i, 0.0) for i in range(len(motif_labels))])
    
    # Compute expected counts
    exp_counts_corr = n0 * motif_probs_corr_array
    exp_counts_corr[0] = 0  # Empty motif
    
    print(f"🔍 [CORRELATED MODEL] Total expected count: {np.sum(exp_counts_corr):.2f}")
    
    # Compute effect sizes and significance for correlated model
    expected_sd_raw_corr = []
    effectsigsraw_corr = []
    degree_corr = []
    group_corr = []
    
    # Convert n0 to Python float (may be sympy Float)
    n0_float = float(n0)
    
    for i, label in enumerate(motif_labels):
        d = len(label)
        degree_corr.append(d)
        
        p_corr = float(motif_probs_corr_array[i])  # Convert to Python float
        expected_corr = float(exp_counts_corr[i])  # Convert to Python float
        expected_sd = np.sqrt(n0_float * p_corr * (1 - p_corr)) if p_corr > 0 and p_corr < 1 else 0
        expected_sd_raw_corr.append(expected_sd)
        
        observed = dcounts[i]
        
        # Effect size: log2((observed + 1) / (expected + 1))
        # This matches the formula used in uniform and region_specific models
        if expected_corr > 0 or observed > 0:
            effect_size = np.log2((observed + 1) / (expected_corr + 1))
        else:
            effect_size = 0
        
        # P-value using binomial test
        if expected_corr > 0 and p_corr > 0:
            p_value = binomtest(int(observed), n=int(n0_float), p=max(p_corr, 1e-10)).pvalue
        else:
            p_value = 1.0
        
        effectsigsraw_corr.append([effect_size, p_value])
        
        # Group assignment (numeric like other models)
        # Group 1 = significant + over-represented
        # Group 2 = significant + under-represented
        # Group 3 = non-significant + over-represented
        # Group 4 = non-significant + under-represented
        bonferroni_thr = alpha / len(motif_labels)
        if effect_size > 0:  # over-represented
            if p_value < bonferroni_thr:
                grp = 1  # significant + over-represented
            else:
                grp = 3  # non-significant + over-represented
        else:  # under-represented
            if p_value < bonferroni_thr:
                grp = 2  # significant + under-represented
            else:
                grp = 4  # non-significant + under-represented
        group_corr.append(grp)
    
    effectsigsraw_corr = np.array(effectsigsraw_corr)
    
    # Create DataFrame for correlated model
    dfraw_corr = pd.DataFrame(data=[
        motif_labels,
        dcounts,
        exp_counts_corr.astype(int),
        expected_sd_raw_corr,
        effectsigsraw_corr[:, 0],
        effectsigsraw_corr[:, 1],
        degree_corr,
        group_corr
    ]).T
    dfraw_corr.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Degree', 'Group']
    dfraw_corr.to_csv(
        os.path.normpath(os.path.join(correlated_plot_dir, f"{sample_name}_upsetplot_correlated.csv")),
        index=False
    )
    print(f"💾 [CORRELATED MODEL] Saved upsetplot data to: {correlated_plot_dir}/{sample_name}_upsetplot_correlated.csv")
    
    # Prepare upset data
    dfdata_corr = prepare_upset_data(dfraw_corr)
    dfdata_corr = dfdata_corr.sort_values(by=['Group', 'Observed'], ascending=[True, False])
    print(f"✅ [CORRELATED MODEL] Upsetplot data prepared for {len(dfdata_corr)} motifs")
    
    # Generate upsetplot for correlated model
    fig_corr, _ = kplot(dfdata_corr)
    for ext in ["pdf", "svg", "png"]:
        fig_corr.savefig(os.path.normpath(os.path.join(correlated_plot_dir, f"{sample_name}_upsetplot_correlated.{ext}")))
    plt.close(fig_corr)
    print(f"✅ [CORRELATED MODEL] Upsetplot saved to {correlated_plot_dir}")
    
    # Generate effect significance plot for correlated model
    # Transform p-values to -log10 scale and create sigs list
    sigs_corr = []
    slabels_corr = motif_labels
    for i in range(len(effectsigsraw_corr)):
        effect_size = effectsigsraw_corr[i, 0]
        p_val = max(effectsigsraw_corr[i, 1], 1e-10)  # Avoid log(0)
        sig_val = -1 * np.log10(p_val)
        sigs_corr.append((effect_size, sig_val))
    
    # Calculate unified x-axis range for correlated model
    current_effect_sizes_corr = [e for e, _ in sigs_corr]
    xlim_corr, all_effect_sizes_corr = calculate_unified_xaxis_range(correlated_plot_dir, 'correlated', current_effect_sizes_corr)
    
    # Validate range
    is_valid_corr, warnings_corr = validate_unified_range(xlim_corr, 'correlated', current_effect_sizes_corr, all_effect_sizes_corr)
    if warnings_corr:
        for warning in warnings_corr:
            print(f"[CORRELATED] {warning}")
    
    if xlim_corr:
        print(f"✅ [CORRELATED] Unified x-axis range: [{xlim_corr[0]:.4f}, {xlim_corr[1]:.4f}]")
    
    plot_effect_significance(sigs_corr, slabels_corr, correlated_plot_dir, 'correlated', sample_name, alpha, xlim=xlim_corr)
    print(f"✅ [CORRELATED MODEL] Effect significance plot saved to {correlated_plot_dir}")
else:
    print("⚠️ [CORRELATED MODEL] Skipped - no conditional probability matrix available")

# ============================================================================
# GENERIC MODEL PROCESSING FUNCTION
# ============================================================================
def process_model_generic(model_name, motif_labels, dcounts, n0, motif_probs_dict, 
                         plot_dir_model, sample_name, alpha, 
                         anchor_probs_dict=None, anchor_corr_matrix=None,
                         smoothing_alpha=1.0, zero_inflation=0.0):
    """
    Generic function to process any model type.
    
    Args:
        model_name: Name of the model (e.g., 'empirical', 'smoothed_empirical')
        motif_labels: List of motif labels
        dcounts: Observed counts for each motif
        n0: Estimated population size
        motif_probs_dict: Dictionary of motif probabilities (index -> prob)
        plot_dir_model: Output directory for this model
        sample_name: Sample name for file naming
        alpha: Significance threshold
        anchor_probs_dict: Anchor probabilities (for P60-anchored models)
        anchor_corr_matrix: Anchor correlation matrix (for models that use it)
        smoothing_alpha: Smoothing parameter for smoothed_empirical
        zero_inflation: Zero-inflation parameter for zero_inflated model
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*80}")
        print(f"[{model_name.upper()} MODEL] Processing...")
        print(f"{'='*80}")
        
        # Convert motif_probs_dict to array
        motif_probs_array = np.array([motif_probs_dict.get(i, 0.0) for i in range(len(motif_labels))])
        
        # Numerical stability: clamp probabilities to [0, 1]
        motif_probs_array = np.clip(motif_probs_array, 0.0, 1.0)
        
        # Check for numerical issues
        prob_sum = np.sum(motif_probs_array)
        if prob_sum > 1.01 or prob_sum < 0.99:
            print(f"⚠️ [{model_name.upper()} MODEL] Warning: Probability sum = {prob_sum:.6f} (expected ~1.0)")
        
        # Compute expected counts
        exp_counts = n0 * motif_probs_array
        exp_counts[0] = 0  # Empty motif
        
        print(f"🔍 [{model_name.upper()} MODEL] Total expected count: {np.sum(exp_counts):.2f}")
        
        # Compute effect sizes, p-values, and groups
        n0_float = float(n0)
        expected_sd_raw = []
        effectsigsraw = []
        degree = []
        group = []
        
        for i, label in enumerate(motif_labels):
            d = len(label)
            degree.append(d)
            
            p_val = float(motif_probs_array[i])
            expected = float(exp_counts[i])
            expected_sd = np.sqrt(n0_float * p_val * (1 - p_val)) if p_val > 0 and p_val < 1 else 0
            expected_sd_raw.append(expected_sd)
            
            observed = dcounts[i]
            
            # Effect size: log2((observed + 1) / (expected + 1))
            # This matches the formula used in uniform and region_specific models
            # The +1 pseudocount prevents division by zero and log of zero
            if expected > 0 or observed > 0:
                effect_size = np.log2((observed + 1) / (expected + 1))
                # Clamp extreme effect sizes to reasonable range (±50) for numerical stability
                if abs(effect_size) > 50:
                    print(f"⚠️ [{model_name.upper()} MODEL] Warning: Extreme effect size {effect_size:.2f} for motif {i}, clamping to ±50")
                    effect_size = np.clip(effect_size, -50, 50)
            else:
                effect_size = 0
            
            # P-value using binomial test
            if expected > 0 and p_val > 0:
                p_value = binomtest(int(observed), n=int(n0_float), p=max(p_val, 1e-10)).pvalue
            else:
                p_value = 1.0
            
            effectsigsraw.append([effect_size, p_value])
            
            # Group assignment
            bonferroni_thr = alpha / len(motif_labels)
            if effect_size > 0:  # over-represented
                if p_value < bonferroni_thr:
                    grp = 1  # significant + over-represented
                else:
                    grp = 3  # non-significant + over-represented
            else:  # under-represented
                if p_value < bonferroni_thr:
                    grp = 2  # significant + under-represented
                else:
                    grp = 4  # non-significant + under-represented
            group.append(grp)
        
        effectsigsraw = np.array(effectsigsraw)
        
        # Create DataFrame
        dfraw = pd.DataFrame(data=[
            motif_labels,
            dcounts,
            exp_counts.astype(int),
            expected_sd_raw,
            effectsigsraw[:, 0],
            effectsigsraw[:, 1],
            degree,
            group
        ]).T
        dfraw.columns = ['Motifs', 'Observed', 'Expected', 'Expected SD', 'Effect Size', 'P-value', 'Degree', 'Group']
        
        # Save CSV
        csv_path = os.path.normpath(os.path.join(plot_dir_model, f"{sample_name}_upsetplot_{model_name}.csv"))
        dfraw.to_csv(csv_path, index=False)
        print(f"💾 [{model_name.upper()} MODEL] Saved upsetplot data to: {csv_path}")
        
        # Save obs/exp CSV
        df_obs_exp = pd.DataFrame({
            'Motif': [concatenate_list_data([label])[0] if label else '' for label in motif_labels],
            'Observed': dcounts,
            'Expected': exp_counts.astype(int)
        })
        obs_exp_path = os.path.normpath(os.path.join(plot_dir_model, f"{sample_name}_motif_obs_exp_{model_name}.csv"))
        df_obs_exp.to_csv(obs_exp_path, index=False)
        print(f"💾 [{model_name.upper()} MODEL] Saved obs/exp to: {obs_exp_path}")
        
        # Prepare upset data
        dfdata = prepare_upset_data(dfraw)
        dfdata = dfdata.sort_values(by=['Group', 'Observed'], ascending=[True, False])
        print(f"✅ [{model_name.upper()} MODEL] Upsetplot data prepared for {len(dfdata)} motifs")
        
        # Generate upsetplot
        if len(dfdata) > 0:
            fig, _ = kplot(dfdata)
            for ext in ["pdf", "svg", "png"]:
                fig.savefig(os.path.normpath(os.path.join(plot_dir_model, f"{sample_name}_upsetplot_{model_name}.{ext}")))
            plt.close(fig)
            print(f"✅ [{model_name.upper()} MODEL] Upsetplot saved to {plot_dir_model}")
        
        # Generate effect significance plot
        sigs = []
        slabels = motif_labels
        for i in range(len(effectsigsraw)):
            effect_size = effectsigsraw[i, 0]
            p_val = max(effectsigsraw[i, 1], 1e-10)  # Avoid log(0)
            sig_val = -1 * np.log10(p_val)
            sigs.append((effect_size, sig_val))
        
        # Calculate unified x-axis range for this model type
        current_effect_sizes = [e for e, _ in sigs]
        xlim_model, all_effect_sizes_model = calculate_unified_xaxis_range(plot_dir_model, model_name, current_effect_sizes)
        
        # Validate range
        is_valid_model, warnings_model = validate_unified_range(xlim_model, model_name, current_effect_sizes, all_effect_sizes_model)
        if warnings_model:
            for warning in warnings_model:
                print(f"[{model_name.upper()}] {warning}")
        
        if xlim_model:
            print(f"✅ [{model_name.upper()}] Unified x-axis range: [{xlim_model[0]:.4f}, {xlim_model[1]:.4f}]")
        
        plot_effect_significance(sigs, slabels, plot_dir_model, model_name, sample_name, alpha, xlim=xlim_model)
        print(f"✅ [{model_name.upper()} MODEL] Effect significance plot saved to {plot_dir_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ [{model_name.upper()} MODEL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# PROCESS NEW MODELS
# ============================================================================

# Process empirical model
if 'empirical' in models_to_process:
    if args.is_anchor_model:
        # For anchor run, use observed frequencies WITHOUT normalization
        # This ensures expected = N0 * P = N0 * (observed/N0) = observed (perfect fit)
        motif_probs_empirical = compute_motif_probabilities_empirical(motif_labels, dcounts, n0, normalize=False)
        # Save anchor probabilities for empirical model
        empirical_anchor_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Empirical_Probabilities.csv"))
        df_empirical_anchor = pd.DataFrame({
            'Motif': [concatenate_list_data([label])[0] if label else '' for label in motif_labels],
            'Probability': [motif_probs_empirical.get(i, 0.0) for i in range(len(motif_labels))],
            'Observed_Count': dcounts,
            'N0': [n0] * len(motif_labels)
        })
        df_empirical_anchor.to_csv(empirical_anchor_path, index=False)
        print(f"💾 [EMPIICAL MODEL] Saved anchor probabilities to: {empirical_anchor_path}")
    else:
        # For non-anchor runs, load anchor probabilities if available
        empirical_anchor_file = None
        if anchor_probs_loaded:
            # Try to find empirical anchor file (would need to be generated separately)
            empirical_anchor_file = args.anchor_model_file.replace('Region-specific_Probabilities', 'Empirical_Probabilities')
            if os.path.exists(empirical_anchor_file):
                df_empirical_anchor = pd.read_csv(empirical_anchor_file)
                empirical_probs_dict = dict(zip(range(len(motif_labels)), df_empirical_anchor['Probability'].values))
                motif_probs_empirical = empirical_probs_dict
            else:
                # Fallback: use current observed frequencies (normalize for consistency with other models)
                motif_probs_empirical = compute_motif_probabilities_empirical(motif_labels, dcounts, n0, normalize=True)
        else:
            motif_probs_empirical = compute_motif_probabilities_empirical(motif_labels, dcounts, n0, normalize=True)
    
    process_model_generic('empirical', motif_labels, dcounts, n0, motif_probs_empirical,
                         model_plot_dirs['empirical'], sample_name, alpha)

# Process smoothed empirical model
if 'smoothed_empirical' in models_to_process:
    if args.is_anchor_model:
        motif_probs_smoothed = compute_motif_probabilities_smoothed_empirical(
            motif_labels, dcounts, n0, alpha=args.smoothing_alpha
        )
        # Save anchor probabilities
        smoothed_anchor_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Smoothed_Empirical_Probabilities.csv"))
        df_smoothed_anchor = pd.DataFrame({
            'Motif': [concatenate_list_data([label])[0] if label else '' for label in motif_labels],
            'Probability': [motif_probs_smoothed.get(i, 0.0) for i in range(len(motif_labels))],
            'Observed_Count': dcounts,
            'N0': [n0] * len(motif_labels),
            'Smoothing_Alpha': [args.smoothing_alpha] * len(motif_labels)
        })
        df_smoothed_anchor.to_csv(smoothed_anchor_path, index=False)
        print(f"💾 [SMOOTHED EMPIRICAL MODEL] Saved anchor probabilities to: {smoothed_anchor_path}")
    else:
        # Load anchor if available, otherwise use current data
        if anchor_probs_loaded:
            smoothed_anchor_file = args.anchor_model_file.replace('Region-specific_Probabilities', 'Smoothed_Empirical_Probabilities')
            if os.path.exists(smoothed_anchor_file):
                df_smoothed_anchor = pd.read_csv(smoothed_anchor_file)
                smoothed_probs_dict = dict(zip(range(len(motif_labels)), df_smoothed_anchor['Probability'].values))
                motif_probs_smoothed = smoothed_probs_dict
            else:
                motif_probs_smoothed = compute_motif_probabilities_smoothed_empirical(
                    motif_labels, dcounts, n0, alpha=args.smoothing_alpha
                )
        else:
            motif_probs_smoothed = compute_motif_probabilities_smoothed_empirical(
                motif_labels, dcounts, n0, alpha=args.smoothing_alpha
            )
    
    process_model_generic('smoothed_empirical', motif_labels, dcounts, n0, motif_probs_smoothed,
                         model_plot_dirs['smoothed_empirical'], sample_name, alpha)

# Process maximum entropy model
if 'max_entropy' in models_to_process:
    probs_for_max_entropy = psdict_for_expected if anchor_probs_loaded else psdict_region_specific
    cond_matrix_for_max_entropy = corr_matrix_for_expected if corr_matrix_for_expected is not None else (probmat if 'probmat' in dir() else None)
    
    motif_probs_max_entropy = compute_motif_probabilities_max_entropy(
        motif_labels, probs_for_max_entropy, cond_matrix_for_max_entropy, columns
    )
    
    process_model_generic('max_entropy', motif_labels, dcounts, n0, motif_probs_max_entropy,
                         model_plot_dirs['max_entropy'], sample_name, alpha,
                         anchor_probs_dict=probs_for_max_entropy, anchor_corr_matrix=cond_matrix_for_max_entropy)

# Process negative binomial model
if 'negative_binomial' in models_to_process:
    probs_for_nb = psdict_for_expected if anchor_probs_loaded else psdict_region_specific
    # Estimate dispersion from data (simplified)
    dispersion = 1.0  # Could be estimated from P60 data
    motif_probs_nb = compute_motif_probabilities_negative_binomial(
        motif_labels, probs_for_nb, columns, dispersion=dispersion
    )
    
    process_model_generic('negative_binomial', motif_labels, dcounts, n0, motif_probs_nb,
                         model_plot_dirs['negative_binomial'], sample_name, alpha)

# Process zero-inflated model
if 'zero_inflated' in models_to_process:
    probs_for_zi = psdict_for_expected if anchor_probs_loaded else psdict_region_specific
    # Estimate zero-inflation from data
    zero_inflation = sum(1 for c in dcounts if c == 0) / len(dcounts) if len(dcounts) > 0 else 0.0
    motif_probs_zi = compute_motif_probabilities_zero_inflated(
        motif_labels, dcounts, n0, probs_for_zi, columns, zero_inflation=zero_inflation
    )
    
    process_model_generic('zero_inflated', motif_labels, dcounts, n0, motif_probs_zi,
                         model_plot_dirs['zero_inflated'], sample_name, alpha,
                         zero_inflation=zero_inflation)

# Process hierarchical correlations model
if 'hierarchical_correlations' in models_to_process:
    probs_for_hier = psdict_for_expected if anchor_probs_loaded else psdict_region_specific
    cond_matrix_for_hier = corr_matrix_for_expected if corr_matrix_for_expected is not None else (probmat if 'probmat' in dir() else None)
    
    if cond_matrix_for_hier is not None:
        motif_probs_hier = compute_motif_probabilities_hierarchical_correlations(
            motif_labels, probs_for_hier, cond_matrix_for_hier, columns
        )
        process_model_generic('hierarchical_correlations', motif_labels, dcounts, n0, motif_probs_hier,
                             model_plot_dirs['hierarchical_correlations'], sample_name, alpha,
                             anchor_probs_dict=probs_for_hier, anchor_corr_matrix=cond_matrix_for_hier)
    else:
        print("⚠️ [HIERARCHICAL CORRELATIONS MODEL] Skipped - no correlation matrix available")

# Process Bayesian hierarchical model
if 'bayesian_hierarchical' in models_to_process:
    alpha_prior = 1.0  # Dirichlet prior parameter
    if args.is_anchor_model:
        motif_probs_bayesian = compute_motif_probabilities_bayesian_hierarchical(
            motif_labels, dcounts, n0, alpha_prior=alpha_prior
        )
        # Save anchor probabilities
        bayesian_anchor_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Bayesian_Hierarchical_Probabilities.csv"))
        df_bayesian_anchor = pd.DataFrame({
            'Motif': [concatenate_list_data([label])[0] if label else '' for label in motif_labels],
            'Probability': [motif_probs_bayesian.get(i, 0.0) for i in range(len(motif_labels))],
            'Observed_Count': dcounts,
            'N0': [n0] * len(motif_labels),
            'Alpha_Prior': [alpha_prior] * len(motif_labels)
        })
        df_bayesian_anchor.to_csv(bayesian_anchor_path, index=False)
        print(f"💾 [BAYESIAN HIERARCHICAL MODEL] Saved anchor probabilities to: {bayesian_anchor_path}")
    else:
        # Load anchor if available, otherwise use current data
        if anchor_probs_loaded:
            bayesian_anchor_file = args.anchor_model_file.replace('Region-specific_Probabilities', 'Bayesian_Hierarchical_Probabilities')
            if os.path.exists(bayesian_anchor_file):
                df_bayesian_anchor = pd.read_csv(bayesian_anchor_file)
                bayesian_probs_dict = dict(zip(range(len(motif_labels)), df_bayesian_anchor['Probability'].values))
                motif_probs_bayesian = bayesian_probs_dict
            else:
                motif_probs_bayesian = compute_motif_probabilities_bayesian_hierarchical(
                    motif_labels, dcounts, n0, alpha_prior=alpha_prior
                )
        else:
            motif_probs_bayesian = compute_motif_probabilities_bayesian_hierarchical(
                motif_labels, dcounts, n0, alpha_prior=alpha_prior
            )
    
    process_model_generic('bayesian_hierarchical', motif_labels, dcounts, n0, motif_probs_bayesian,
                         model_plot_dirs['bayesian_hierarchical'], sample_name, alpha)

# Process ML non-parametric model
if 'ml_nonparametric' in models_to_process:
    motif_probs_ml = compute_motif_probabilities_ml_nonparametric(
        motif_labels, dcounts, n0, columns
    )
    
    process_model_generic('ml_nonparametric', motif_labels, dcounts, n0, motif_probs_ml,
                         model_plot_dirs['ml_nonparametric'], sample_name, alpha)

###CHATGPT OPTIMIZED VERSION DOESNT WORK
def kplot(df, size=(30, 12)):
    """
    df : pd.DataFrame
        DataFrame with columns: "Motifs", "Observed", "Expected", "Expected SD", "Group"
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import upsetplot as up

    # Ensure zero-observed motifs are still included
    motiflabels = df['Motifs'].to_list()
    obs_counts = df['Observed'].fillna(0).astype(float).to_numpy()
    data = up.from_memberships(motiflabels, data=obs_counts)

    xlen = df.shape[0]
    xticks = np.arange(xlen)

    uplot = up.UpSet(data, sort_by=None)
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 3], 'height_ratios': [3, 1]})
    fig.set_size_inches(size)

    ax[1, 0].set_ylabel("Set Totals")
    uplot.plot_matrix(ax[1, 1])
    uplot.plot_totals(ax[1, 0])
    ax[0, 0].axis('off')

    # Clean axis formatting
    ax[0, 1].spines['bottom'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)

    width = 0.35
    dodge = width / 2
    ox = xticks - dodge
    ex = xticks + dodge

    ax[1, 0].set_title("Totals")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].set_xlim(ax[1, 1].get_xlim())

    # Assign colors by group
    colorlist = ['red', 'darkblue', 'black', 'black']
    cs = [colorlist[i - 1] for i in df['Group']]
    
    # Bar plots
    ax[0, 1].bar(ox, obs_counts, width=width, label="Observed", align="center",
                 color=cs, edgecolor='lightgray')
    ax[0, 1].bar(ex, df['Expected'].to_numpy(), yerr=df['Expected SD'].to_numpy(),
                 width=width / 2, label="Expected", align="center",
                 color='gray', alpha=0.5, ecolor='lightgray')

    # Annotate Group 1 and Group 2 motifs even if Observed == 0
    grp_ = df['Group'].to_numpy()
    idsig = np.where((grp_ == 1) | (grp_ == 2))[0]
    for i in idsig:
        ax[0, 1].text(ox[i] - 0.5 * dodge, obs_counts[i] + 1, s="*")

    # Final axis settings
    ax[0, 1].xaxis.grid(False)
    ax[0, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.grid(False)

    fig.tight_layout()
    return fig, ax

# Generate GPT-optimized upsetplot visual plots for both models
fig_uniform_gpt, _ = kplot(dfdata)
for ext in ["pdf", "svg", "png"]:
    fig_uniform_gpt.savefig(os.path.normpath(os.path.join(uniform_plot_dir, f"{sample_name}_upsetplot_gpt_uniform.{ext}")))
plt.close(fig_uniform_gpt)

fig_rs_gpt, _ = kplot(dfdata_rs)
for ext in ["pdf", "svg", "png"]:
    fig_rs_gpt.savefig(os.path.normpath(os.path.join(region_specific_plot_dir, f"{sample_name}_upsetplot_gpt_region_specific.{ext}")))
plt.close(fig_rs_gpt)

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import ast

print("\n--- Generating 2-region motif graph from canonical CSV ---")

# Process both models
for model_type, plot_dir_model in [("uniform", uniform_plot_dir), ("region_specific", region_specific_plot_dir)]:
    print(f"\n[{model_type.upper()} MODEL] Generating 2-region motif graph...")
    
    # === Load canonical motif results CSV ===
    motif_csv_path = os.path.normpath(os.path.join(plot_dir_model, f"{sample_name}_upsetplot_{model_type}.csv"))
    
    if not os.path.exists(motif_csv_path):
        print(f"⚠️ Warning: Expected motif CSV not found: {motif_csv_path}")
        print(f"⚠️ Skipping 2-region motif graph generation for {model_type} model")
        continue
    
    try:
        df_motifs = pd.read_csv(motif_csv_path)

        # === Parse 'Motifs' column into real Python lists ===
        if isinstance(df_motifs['Motifs'].iloc[0], str):
            df_motifs['Motifs'] = df_motifs['Motifs'].apply(ast.literal_eval)

        # Drop malformed rows just in case
        df_motifs = df_motifs[df_motifs['Motifs'].apply(lambda x: isinstance(x, list))]

        # Filter for 2-region motifs only
        df_motifs['motif_size'] = df_motifs['Motifs'].apply(len)
        df_2region = df_motifs[df_motifs['motif_size'] == 2].copy()

        # Ensure numeric values
        for col in ['Observed', 'Expected', 'P-value']:
            df_2region[col] = pd.to_numeric(df_2region[col], errors='coerce')

        # Classify significance label
        def get_sig_label_from_group(row):
            if row['Group'] == 1:
                return 'over'
            elif row['Group'] == 2:
                return 'under'
            else:
                return 'ns'
        df_2region['sig_label'] = df_2region.apply(get_sig_label_from_group, axis=1)

        # === Build networkx graph ===
        G = nx.Graph()
        max_obs = df_2region['Observed'].max()
        if pd.isna(max_obs) or max_obs <= 0:
            max_obs = 1  # prevent division by zero

        for row in df_2region.itertuples():
            regions = row.Motifs
            if len(regions) != 2:
                continue
            r1, r2 = regions
            # Convert to uppercase to match manual_region_order
            r1_upper = r1.upper()
            r2_upper = r2.upper()
            color = {'over': 'red', 'under': 'blue', 'ns': 'black'}.get(row.sig_label, 'black')
            width = 1 + 9 * (row.Observed / max_obs)
            G.add_edge(r1_upper, r2_upper, weight=width, color=color)

        # === Layout and draw ===
        manual_region_order = ["RSP", "PM", "AM", "AL", "LM"]
        sorted_nodes = [r for r in manual_region_order if r in G.nodes]
        if not sorted_nodes:
            print(f"[{model_type.upper()} MODEL] No valid 2-region motifs found. Skipping plot.")
        else:
            angle_step = 2 * np.pi / len(sorted_nodes)
            pos = {
                region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
                for i, region in enumerate(sorted_nodes)
            }

            edges = G.edges(data=True)
            colors = [e[2]['color'] for e in edges]
            widths = [e[2]['weight'] for e in edges]

            plt.figure(figsize=(8, 8))
            nx.draw_networkx(G, pos, with_labels=True, node_size=1000, edge_color=colors, width=widths)
            plt.title(f"Fig 10g: 2-Region Broadcasting Motifs ({model_type.replace('_', ' ').title()} Model)")

            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Overrepresented'),
                Line2D([0], [0], color='blue', lw=2, label='Underrepresented'),
                Line2D([0], [0], color='black', lw=2, label='Not Significant')
            ]
            plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
            plt.tight_layout()

            save_path = os.path.join(plot_dir_model, f"{sample_name}_panel_g_broadcasting_from_canonical_{model_type}.svg")
            plt.savefig(save_path, format='svg')
            plt.close()
            print(f"✅ [{model_type.upper()} MODEL] Saved 2-region motif plot to {save_path}")
    
    except Exception as e:
        print(f"❌ [{model_type.upper()} MODEL] Error generating 2-region motif graph: {e}")
        import traceback
        traceback.print_exc()
        continue


### === K-Means Cluster Centroid Heatmap === ###
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

scolors = ['white', 'red']
scm = LinearSegmentedColormap.from_list('white_to_red', scolors, N=256)

# Optionally reorder columns (if you want Han-style ordering)
# region_order = ["RSP", "PM", "AM", "A", "RL", "AL", "LM"]
# df = df[[col for col in region_order if col in df.columns]]

k_clusters = consensus_k if 'consensus_k' in locals() else 8 #this can be commented out to be set to a specific number if you want to have comparable plots across ages and datasets.
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
kmeans.fit(df)
centroids = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(centroids, aspect='auto', cmap=scm, vmin=0, vmax=1)
ax.set_title("Projection Motif Clusters (Extended Data Fig. 10 Style)", fontsize=14)
ax.set_xlabel("Target Regions", fontsize=12)
ax.set_ylabel("Cluster ID", fontsize=12)
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha='right')
ax.set_yticks(range(k_clusters))
ax.set_yticklabels([f"Cluster {i+1}" for i in range(k_clusters)])

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(top=False, bottom=True, left=True, right=False)

cbar = fig.colorbar(im, ax=ax, orientation='vertical')
cbar.set_label('Normalized Projection Strength', rotation=270, labelpad=15)

fig.tight_layout()
# Save in multiple formats (PNG, PDF, SVG) to match other plots
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(
        os.path.normpath(os.path.join(plot_dir, f"{sample_name}_ExtendedDataFig10_Recreation.{ext}")),
        format=ext
    )
plt.close()


df.astype(bool).sum()

def append_summary_wide_format_extended(
    args, projections, umi_total_counts, total_projections, observed_cells,
    N0_value, pe_num, consensus_k, normalized_matrix, output_dir,
    motif_over, motif_under, mean_inj_value_filtered_cells_only,
    model_type
):
    import os
    import pandas as pd
    import numpy as np
    import csv
    import json
    from scipy.stats import entropy

    # #region agent log
    with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"process-nbcm-tsv.py:2539","message":"Function entry","data":{"model_type":model_type,"output_dir":str(output_dir),"sample_name":args.sample_name},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion

    if not isinstance(normalized_matrix, pd.DataFrame):
        raise ValueError("normalized_matrix must be a pandas DataFrame")

    columns = normalized_matrix.columns.tolist()
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.normpath(os.path.join(output_dir, "projection_summary.csv"))
    
    # #region agent log
    with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"process-nbcm-tsv.py:2556","message":"Directory check","data":{"output_dir":str(output_dir),"dir_exists":os.path.exists(output_dir),"summary_path":str(summary_path)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    write_header = not os.path.exists(summary_path)

    mean_umis = dict(zip(columns, normalized_matrix.mean(axis=0)))
    proj_counts = projections

    counts = np.array(list(proj_counts.values()), dtype=float)
    probs = counts / counts.sum() if counts.sum() > 0 else np.ones_like(counts) / len(counts)
    norm_entropy = entropy(probs) / np.log(len(probs)) if len(probs) > 1 else 0.0

    row = {
        "Sample": args.sample_name,
        "Model": model_type,
        "injection min": args.injection_umi_min,
        "target:inj ratio": args.min_body_to_target_ratio,
        "at least 1 target minimum": args.min_target_count,
        "user umi min": args.target_umi_min,
        "force_user_threshold": args.force_user_threshold,
        "threshold used": final_umi_threshold,
        "mean_inj_value_filtered_cells_only": float(mean_inj_value_filtered_cells_only) if not np.isnan(mean_inj_value_filtered_cells_only) else np.nan,
        "Labels": ",".join(columns),
        "TotalProjections": total_projections,
        "ObservedCells": observed_cells,
        "N0": float(N0_value) if N0_value else np.nan,
        "p_e": pe_num,
        "k_consensus": consensus_k,
        "Entropy": norm_entropy,
        "MotifOverrepresented": " ".join(motif_over) if motif_over else "",
        "MotifUnderrepresented": " ".join(motif_under) if motif_under else "",
    }

    # Reordered block: group by metric type (ProjCount, UMISum, MeanUMI)
    for region in columns:
        row[f"ProjCount_{region}"] = int(proj_counts.get(region, 0))

    for region in columns:
        val = umi_total_counts.get(region)
        if val is None:
            print(f"⚠️ Warning: Region '{region}' not in umi_total_counts. Defaulting to 0.")
            val = 0.0
        row[f"UMISum_{region}"] = float(val)

    for region in columns:
        row[f"MeanUMI_{region}"] = float(mean_umis.get(region, 0.0))

    df_row = pd.DataFrame([row])
    
    # #region agent log
    with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"process-nbcm-tsv.py:2600","message":"Before CSV write","data":{"summary_path":str(summary_path),"file_exists_before":os.path.exists(summary_path),"write_header":write_header,"row_count":len(df_row)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    try:
        df_row.to_csv(summary_path, mode='a', header=write_header, index=False, quoting=csv.QUOTE_ALL)
        
        # #region agent log
        with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"process-nbcm-tsv.py:2601","message":"After CSV write","data":{"summary_path":str(summary_path),"file_exists_after":os.path.exists(summary_path),"file_size":os.path.getsize(summary_path) if os.path.exists(summary_path) else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        print(f"📈 Summary extended metrics appended to {summary_path}")
        print(f"🧪 MotifOver: {motif_over}")
        print(f"🧪 MotifUnder: {motif_under}")
    except Exception as e:
        # #region agent log
        with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"process-nbcm-tsv.py:2601","message":"CSV write exception","data":{"summary_path":str(summary_path),"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        raise


# ----------------------
# SAFE motif extraction using the final upsetplot CSV (trusted final logic)
# ----------------------

def extract_motifs_from_upsetplot(plot_dir, sample_name, model_type):
    """
    Extract overrepresented and underrepresented motifs from model-specific upsetplot CSV.
    
    Args:
        plot_dir: Directory containing the upsetplot CSV file
        sample_name: Sample name
        model_type: "uniform" or "region_specific"
        
    Returns:
        tuple: (motif_over, motif_under) - lists of motif names
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Construct filename based on model type
    if model_type == "uniform":
        filename = f"{sample_name}_upsetplot_uniform.csv"
    elif model_type == "region_specific":
        filename = f"{sample_name}_upsetplot_region_specific.csv"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    upset_file = os.path.normpath(os.path.join(plot_dir, filename))
    motif_over, motif_under = [], []

    try:
        if os.path.exists(upset_file):
            print(f"✅ [{model_type.upper()} MODEL] Found motif file: {upset_file}")
            df_upset = pd.read_csv(upset_file)

            # Sanitize column names
            df_upset.columns = [col.strip().lower() for col in df_upset.columns]

            # Rename columns to standardized lowercase names
            df_upset.rename(columns={
                "motifs": "motif",
                "p-value": "pval",
                "observed": "observed",
                "expected": "expected"
            }, inplace=True)

            # Validate required columns exist
            required_cols = {"observed", "expected", "pval", "motif"}
            if not required_cols.issubset(set(df_upset.columns)):
                raise ValueError(f"Missing expected columns in upsetplot CSV: {required_cols - set(df_upset.columns)}")

            # Ensure types are correct
            df_upset["observed"] = pd.to_numeric(df_upset["observed"], errors="coerce")
            df_upset["expected"] = pd.to_numeric(df_upset["expected"], errors="coerce")
            df_upset["pval"] = pd.to_numeric(df_upset["pval"], errors="coerce")

            corrected_threshold = 0.05

            motif_over = (
                df_upset.loc[
                    (df_upset["observed"] > df_upset["expected"]) & (df_upset["pval"] < corrected_threshold),
                    "motif"
                ]
                .dropna().astype(str).tolist()
            )

            motif_under = (
                df_upset.loc[
                    (df_upset["observed"] < df_upset["expected"]) & (df_upset["pval"] < corrected_threshold),
                    "motif"
                ]
                .dropna().astype(str).tolist()
            )

            print(f"🧪 [{model_type.upper()} MODEL] MotifOver: {motif_over}")
            print(f"🧪 [{model_type.upper()} MODEL] MotifUnder: {motif_under}")

        else:
            print(f"⚠️ [{model_type.upper()} MODEL] Motif file not found at expected path: {upset_file}")

    except Exception as e:
        print(f"❌ [{model_type.upper()} MODEL] Failed to load upsetplot file {upset_file}: {e}")
        motif_over, motif_under = [], []
    
    return motif_over, motif_under

# Extract motifs for both models
# #region agent log
import json as _debug_json
with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
    f.write(_debug_json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"process-nbcm-tsv.py:2694","message":"Before motif extraction","data":{"uniform_plot_dir":str(uniform_plot_dir) if 'uniform_plot_dir' in globals() else "UNDEFINED","region_specific_plot_dir":str(region_specific_plot_dir) if 'region_specific_plot_dir' in globals() else "UNDEFINED","sample_name":args.sample_name},"timestamp":int(__import__('time').time()*1000)}) + '\n')
# #endregion

motif_over_uniform, motif_under_uniform = extract_motifs_from_upsetplot(
    uniform_plot_dir, args.sample_name, "uniform"
)

motif_over_rs, motif_under_rs = extract_motifs_from_upsetplot(
    region_specific_plot_dir, args.sample_name, "region_specific"
)

# #region agent log
with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
    f.write(_debug_json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"process-nbcm-tsv.py:2701","message":"After motif extraction","data":{"motif_over_uniform_count":len(motif_over_uniform),"motif_over_rs_count":len(motif_over_rs)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
# #endregion

# ----------------------
# Call summary writer
# ----------------------
# ============================================================================
# Final Summary: Both Models Comparison
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: UNIFORM vs REGION-SPECIFIC MODELS")
print("="*80)
print(f"✅ [UNIFORM MODEL] pₑ = {pe_num:.6f}")
print(f"✅ [REGION-SPECIFIC MODEL] Region probabilities: {psdict_region_specific}")
print(f"✅ Both models calculated expected counts and performed statistical tests")
print(f"✅ Results saved with '_uniform' and '_region_specific' suffixes")
print("="*80)

# #region agent log
import json as _debug_json2
with open('/Users/matt/git/mapseq_processing_Jacobs/.cursor/debug.log', 'a') as f:
    f.write(_debug_json2.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"process-nbcm-tsv.py:2717","message":"Before function calls","data":{"uniform_plot_dir":str(uniform_plot_dir),"region_specific_plot_dir":str(region_specific_plot_dir),"pe_num":float(pe_num) if isinstance(pe_num, (int, float)) else str(pe_num)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
# #endregion

# Call for uniform model
append_summary_wide_format_extended(
    args,
    projections,
    umi_total_counts,
    total_projections,
    observed_cells,
    N0_value,
    pe_num,
    consensus_k,
    normalized_matrix,
    uniform_plot_dir,
    motif_over_uniform,
    motif_under_uniform,
    mean_inj_value_filtered_cells_only,
    "uniform"
)

# Call for region-specific model
# Convert psdict_region_specific to string representation for p_e field
pe_rs_str = ",".join([f"{k}:{v:.6f}" for k, v in psdict_region_specific.items()])
append_summary_wide_format_extended(
    args,
    projections,
    umi_total_counts,
    total_projections,
    observed_cells,
    N0_value,
    pe_rs_str,
    consensus_k,
    normalized_matrix,
    region_specific_plot_dir,
    motif_over_rs,
    motif_under_rs,
    mean_inj_value_filtered_cells_only,
    "region_specific"
)
