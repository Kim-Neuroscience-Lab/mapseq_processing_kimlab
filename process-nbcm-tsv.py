import argparse
import os
import re
import sympy
import csv
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
import seaborn as sn
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
from scipy.stats import friedmanchisquare, kruskal, binomtest, binom
from sklearn.cluster import k_means
from sklearn.preprocessing import normalize 
import itertools
from adjustText import adjust_text
import multiprocessing as mp
import itertools
import upsetplot as up
from statsmodels.stats.multitest import fdrcorrection

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




# Parse arguments
args = parser.parse_args()

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

#This switch is for excluding some columns. See line 254
full_data = True
motif_join = '+'

### Helper Functions

def calculate_projections_from_matrix(matrix, sample_labels):
    """Calculate TOTAL projections per region (including multiple per neuron).

    1. column_counts: Iterates through each column (indexed by idx). Counts the number of nonzero values in that column using np.count_nonzero(). This counts how many neurons project to each brain region. Stores results in a dictionary.
    2. **commented out*** total_projections: 'matrix > 0' creates a Boolean mask where True (1) means a projection exists. False (0) means no projection. 'np.sum(matrix > 0)' counts all nonzero elements across the entire matrix.Counts how many cells (neurons) project to any region (ignoring magnitude). Does not account for projection strength—it treats all values > 0 the same.
This gives an idea of how many neurons are connected, but not their relative strengths.
    3. total_projections: Sums column-wise counts instead. Uses actual projection strength values instead of just binary presence. This is the approach used by Kheirbeck Lab's codebook.
    
    """
    
    column_counts = {region: np.count_nonzero(matrix[:, idx]) for idx, region in enumerate(sample_labels)}
    #total_projections = int(np.sum(matrix > 0)) #counts all projections as a 1 regardless of strength.
    total_projections = sum(column_counts.values())  # sums all the column counts like Kheirbeck lab does
    
    print(f"🔍 Column counts (neurons per region): {column_counts}")
    print(f"🔍 Total projections (Sums column-wise counts): {total_projections}")
    
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
        target_max = np.max(matrix[:, non_neg_inj_cols], axis=1)
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
            max_target_values = np.max(matrix[:, non_neg_inj_cols], axis=1)
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
            max_neg_value = np.max(neg_values)
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
            density = gaussian_kde(log_vals)
            xs = np.linspace(log_vals.min(), log_vals.max(), 1000)
            ys = density(xs)
            d2 = np.gradient(np.gradient(ys))
            elbow_idx = np.argmin(d2)
            elbow_log_value = xs[elbow_idx]
            dynamic_threshold = 10 ** elbow_log_value
            print(f"🔍 Step 6a: Dynamic noise threshold estimated via elbow method: {dynamic_threshold:.4f}")
        else:
            dynamic_threshold = target_umi_min
            print("⚠ WARNING: No nonzero target values found for threshold estimation. Using user defined or argparse default value.")
    else:
        dynamic_threshold = target_umi_min
        print("⚠ WARNING: No target columns found for dynamic thresholding. Using user defined or argparse default value.")

    print(f"Calculated Threshold: {dynamic_threshold}")
    print(f"User defined or argparse minimum default(2): {target_umi_min}")
    print(f"Max Negative Control Value: {max_neg_value}")

    # 🚨 Step 6b: Apply UMI threshold
    if force_user_threshold:
        final_umi_threshold = target_umi_min
        print(f"⚠️ Step 6b: Forcing user-defined UMI threshold: {final_umi_threshold}")
    else:
        final_umi_threshold = max(target_umi_min, max_neg_value, dynamic_threshold)
        print(f"✅ Step 6b: Looking for the maximum value: user= ({target_umi_min}), MAXneg= ({max_neg_value:.4f}), "
              f"and dynamic= ({dynamic_threshold:.4f}) ➜ Final threshold choice: {final_umi_threshold:.4f}")

    # 🚨 Step 6c: Remove rows that became all zeros after thresholding
    matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
    print(f"🔍 Step 6c: Removed new zero rows. Shape: {matrix.shape}")

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

# Solve for symbolic p_e
pe_solutions = sympy.solve((1 - (1 - pe)**len(projections)) * total_projections - observed_cells, pe, force=True)

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

# Define total_regions BEFORE computing motif probabilities
total_regions = len(columns)  # Number of regions after filtering

# Compute motif probabilities (Ensuring n starts from 1, since n=0 isn't meaningful here)
motif_probs = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n))
    for n in range(1, total_regions + 1)  # Start at 1
}

# Normalize probabilities to ensure they sum to exactly 1
total_motif_prob = sum(motif_probs.values())

if total_motif_prob > 0:
    motif_probs = {k: v / total_motif_prob for k, v in motif_probs.items()}

# Debugging print statements
print(f"🔍 Total Motif Probability Before Normalization: {total_motif_prob}")
print(f"🔍 Total Motif Probability After Normalization (must sum to 1): {sum(motif_probs.values())}")

# Final check: Ensure sum is 1
if not np.isclose(float(sum(motif_probs.values())), 1, atol=0.01):
    print(f"🚨 WARNING: Motif probabilities sum to {sum(motif_probs.values())}, not 1.")

normalized_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Normalized_Matrix.csv"))

# Dynamically match labels for important areas
special_area_1_label = next((label for label in columns if re.match(f"{args.special_area_1}\\d*", label)), None)
special_area_2_label = next((label for label in columns if re.match(f"{args.special_area_2}\\d*", label)), None)

column_counts_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Column_Counts.csv"))

# if special_area_1_label and special_area_2_label:
#     print(f"Matched labels: {special_area_1_label}, #{special_area_2_label}")
#     # Replace hardcoded logic with dynamic labels

root_save_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Roots.csv"))
pi_save_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Simplified_Pi.csv"))
region_probs_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Region-specific_Probabilities.csv"))
calculated_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Calculated_Value.csv"))
std_dev_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Standard_Deviation.csv"))
motif_test_path = os.path.normpath(os.path.join(out_dir, f"{sample_name}_Motif_Binomial_Results.csv"))

log_scaled_value = sum(np.log(psdict[label]) for label in columns)
scaled_value = np.exp(log_scaled_value)  # Convert back from log scale



# Dynamic calculation using sample_labels and total_projections
calculated_value = (1 - (1 - pe_num)**len(columns)) * observed_cells #total_projections
print(f"🔍 Expected Observed Projections [(1-(1-p_e)*#areas)*observed cells]: {calculated_value}")

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
print("\n==== Motif Observed vs Expected Counts ============")
for i, motif_size in enumerate(observed_motif_sizes):
    observed = motif_counts[i]  # Observed count
    expected = int(motif_probs.get(motif_size, 0) * observed_cells)  # Expected count based on probabilities

    print(f"Motif Size: {motif_size:5} | Observed: {observed:5} | Expected: {expected:5}")

print("\n===================================================")

# Compute probabilities for observed motif sizes only
motif_probs = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n)) for n in observed_motif_sizes
}

# Normalize probabilities
total_motif_prob = sum(motif_probs.values())
motif_probs = {k: v / total_motif_prob for k, v in motif_probs.items()}

# Perform binomial test for each observed motif size
binomial_test_results = []
for n_proj in observed_motif_sizes:
    obs_count = int(motif_counts[observed_motif_sizes.tolist().index(n_proj)])  # Ensure integer
    prob = float(motif_probs.get(n_proj, 0))  # Ensure float
    p_value = binom.sf(obs_count - 1, int(observed_cells), prob)  # Use proper types
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

n0 = observed_cells #import from stats at beginning

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
from scipy.spatial.distance import cdist
import os
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

X = df.to_numpy()
K = range(2, 15)  # skip k=1 for silhouette and BIC

# 1. Elbow Method (Inertia)
inertias = []
for k in K:
    km = KMeans(n_clusters=k, n_init="auto").fit(X)
    inertias.append(km.inertia_)

# Compute elbow using max second derivative
inertia_deltas = np.diff(inertias)
inertia_deltas2 = np.diff(inertia_deltas)
elbow_k = K[np.argmax(inertia_deltas2) + 2]  # +2 due to double diff

# 2. Silhouette Score
sil_scores = []
for k in K:
    km = KMeans(n_clusters=k, n_init="auto").fit(X)
    sil_scores.append(silhouette_score(X, km.labels_))
silhouette_k = K[np.argmax(sil_scores)]

# 3. Gap Statistic
def compute_gap_statistic(X, refs=10):
    gaps = []
    for k in K:
        km = KMeans(n_clusters=k, n_init="auto").fit(X)
        disp = np.mean(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1))

        ref_disps = []
        for _ in range(refs):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            km_ref = KMeans(n_clusters=k, n_init="auto").fit(X_ref)
            ref_disp = np.mean(np.min(cdist(X_ref, km_ref.cluster_centers_, 'euclidean'), axis=1))
            ref_disps.append(ref_disp)

        gap = np.log(np.mean(ref_disps)) - np.log(disp)
        gaps.append(gap)
    return gaps

gaps = compute_gap_statistic(X)
gap_k = K[np.argmax(gaps)]

# 4. BIC using GMM
bics = []
for k in K:
    gmm = GaussianMixture(n_components=k, n_init=1).fit(X)
    bics.append(gmm.bic(X))
bic_k = K[np.argmin(bics)]

# Consensus vote
votes = [elbow_k, silhouette_k, gap_k, bic_k]
vote_counts = Counter(votes)
consensus_k = vote_counts.most_common(1)[0][0]

# Optional: Print diagnostic info
print(f"Elbow k = {elbow_k}, Silhouette k = {silhouette_k}, Gap k = {gap_k}, BIC k = {bic_k}")
print(f"Consensus k = {consensus_k}")

# Plot Inertia + Delta
plt.figure(figsize=(10, 7))
plt.plot(K, inertias, 'x-', color='blue', label='Inertia')
plt.axvline(x=elbow_k, color='gray', linestyle='--', label=f'Elbow: k={elbow_k}')
plt.axvline(x=silhouette_k, color='green', linestyle='--', label=f'Silhouette: k={silhouette_k}')
plt.axvline(x=gap_k, color='orange', linestyle='--', label=f'Gap: k={gap_k}')
plt.axvline(x=bic_k, color='purple', linestyle='--', label=f'BIC: k={bic_k}')
plt.axvline(x=consensus_k, color='red', linestyle=':', linewidth=2.0, label=f'Consensus: k={consensus_k}')
plt.xlabel('k', fontsize=20)
plt.ylabel('Inertia', fontsize=20)
plt.title('Cluster Evaluation Methods', fontsize=20)
plt.legend()
plt.tight_layout()

# Save plots
elbow_plt = plt.gcf()
#elbow_plt.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_cluster_diagnostics.pdf")))
for ext in ["pdf", "svg", "png"]:
    elbow_plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_cluster_diagnostics.{ext}")))


# 🚀 Use consensus_k for final clustering
km = KMeans(n_clusters=consensus_k, n_init="auto").fit(X)

from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
import os

# Clustering
X = df.to_numpy()
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

exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels),\
                                dcounts,\
                                exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif','Observed','Expected']
df_obs_exp.to_csv(os.path.normpath(os.path.join(plot_dir, sample_name + "_motif_obs_exp.csv")))
df_obs_exp

##suggested addition to give another csv without the null combination from the powerset at the top row.
exp_counts, motif_probs = get_expected_counts(motif_labels)
df_obs_exp = pd.DataFrame(data=[concatenate_list_data(motif_labels), dcounts, exp_counts.astype(int)]).T
df_obs_exp.columns = ['Motif', 'Observed', 'Expected']

# Exclude empty motifs
df_obs_exp = df_obs_exp[df_obs_exp['Motif'] != ""]  # Adjust condition as needed for your data format - chatGPT

# Save filtered data to CSV
df_obs_exp.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_obs_exp_filtered.csv")),
    index=False
)


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
        matches[i] = binomtest(int(dcounts_[i]),n=n0,p=pi).pvalue
        matches[i] = max(matches[i], 1e-10)
    matches = p_transform(matches)
    #matches is the significance level
    res = zip(effect_size, matches)
    mlabels = [labels[h] for h in nonzid]
    return list(res), mlabels

sigs, slabels = get_motif_sig_pts(dcounts,motif_labels,exclude_zeros=True)

#Bonferroni correction: p-threshold / Num comparisons
pcutoff = -1*np.log10(alpha / len(slabels)) #adjust alpha with argument

list_sig = [i for (i,(e,s)) in enumerate(sigs) if s > pcutoff ]
color_labels = ['gray' for i in range(len(sigs))]
for i in list_sig:
    e,s = sigs[i]
    if e > 0: #overrepresented
        color_labels[i] = 'red'
    else:
        color_labels[i] = 'blue'
#color_labels

hide_singlets = True
if hide_singlets:
    mask = [i for (i,l) in enumerate(slabels) if len(l) > 1]
#subset_list(slabels,[1,5,7])
fig,ax = plt.subplots(1,1)#plt.figure(figsize=(13,11))
fig.set_size_inches(20,20)
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
ax.set_title(sample_name.replace('_',''),fontsize=16)
ax.set_xlabel("Effect Size \n$log_2($observed/expected$)$",fontsize=16)
ax.set_ylabel("Significance\n $-log_{10}(P)$",fontsize=16)
ax.axhline(y=pcutoff, linestyle='--')
ax.axvline(x=0, linestyle='--')
ax.text(x=-.5,y=pcutoff+0.05,s='P-value cutoff',fontsize=16)
##CHATGPT UPDATES FOR FORMATTING - Not sure these all work
from adjustText import adjust_text

# Scatter plot
ax.scatter(*zip(*subset_list(sigs, mask)), c=subset_list(color_labels, mask))

# Prepare text labels
pretty_slabels = concatenate_list_data(subset_list(slabels, mask))
coordinates = subset_list(sigs, mask)
texts = []

for n, (z, y) in enumerate(coordinates):
    txt = pretty_slabels[n]
    texts.append(ax.text(z, y, txt, fontsize=12))

# Adjust y-axis limits
y_vals = [y for _, y in subset_list(sigs, mask)]
padding = 0.1 * (max(y_vals) - min(y_vals))  # Add 10% padding
ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)

# Adjust text positions to avoid overlap
adjust_text(
    texts,
    #only_move={'points': 'y', 'text': 'y'},  # Allow vertical movement
    arrowprops=dict(
        arrowstyle="->",
        color='gray',
        lw=1,
        shrinkA=1,  # These values won't matter much for adjust_text
        shrinkB=1
    ),
    expand_points=(1.5, 2.5),  # Add padding around points
    force_text=1,  # Increase separation force for text
    force_points=1  # Increase separation force for points
)

#fig.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_effect_significance.pdf")))
for ext in ["pdf", "svg", "png"]:
    fig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_effect_significance.{ext}")))


def gen_per_cell_plot(df,cell_ids,motif_labels,dcounts,expected,savepath=plot_dir, hide_singlets=True,figsize=(16,35)):
    """
    This plots each cell of a given motif on the same plot as an individual line
    Each line's points are the corresponding projection strengths at that region
    So this plot shows the projection strengths of all the cells for each motif
    """
    if hide_singlets: #Only show motifs with two or more regions
        mask = [i for (i,l) in enumerate(motif_labels) if len(l) > 1]
        cell_ids = subset_list(cell_ids,mask)
        
    non0cell_ids = [(i,x) for (i,x) in enumerate(cell_ids) if len(x) > 0]
    dcounts = subset_list(dcounts,mask)
    exp_counts_ = subset_list(expected,mask)
    obs_ex = []
    for i,x in non0cell_ids:
        oe = (dcounts[i],exp_counts_[i])
        obs_ex.append(oe)
    num_plots = len(non0cell_ids)
    plot_titles = concatenate_list_data(subset_list(motif_labels,mask))
    #print(num_plots)
    ncols = 2
    nrows = int(np.ceil(num_plots / ncols))
    fig = plt.figure(figsize=figsize)
    n = 1
    for cellids_ in non0cell_ids:
        """if n > 2:
            break"""
        i,cellids = cellids_
        ax = fig.add_subplot(nrows,ncols,n)
        title = plot_titles[i]
        ax.set_title(title)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.to_list())
        ax.set_ylabel("Projection Strength")
        x = df.iloc[cellids,:].to_numpy()
        ## add observed/expected legend
        obs,ex = obs_ex[n-1]
        textstr = 'Observed: {} \n Expected: {}'.format(int(obs),int(ex))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ##
        yerr = x.std(axis=0) / np.sqrt(x.shape[0])
        #ax.errorbar(x=np.arange(df.shape[1]), y=x.mean(axis=0),yerr=yerr,ecolor='black') # mean with error bars
        for j in range(x.shape[0]):
            ax.plot(np.arange(df.shape[1]), x[j], markerfacecolor='none', alpha=0.2, c='gray') 
        ax.errorbar(x=np.arange(df.shape[1]), y=x.mean(axis=0),yerr=yerr,ecolor='gray', c='black',linewidth=3) # mean with error bars
        n+=1
    if savepath:
        fig.savefig(savepath)
    return ax

#Run the function
if full_data:
    fig_size2 = (20,140) #(20,140)
else:
    fig_size2 = (20,10)
gprcpplot = gen_per_cell_plot(df,cell_ids,motif_labels,dcounts,exp_counts,figsize=fig_size2, savepath = os.path.normpath(os.path.join(plot_dir, sample_name + "_per_cell_proj_strength.svg")))
#
#


from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import seaborn as sns
import re
import sys

# Workaround for Windows recursion bug
sys.setrecursionlimit(5000)

### === Red-White Cluster Heatmap === ###
print("🔍 Generating Red-White cluster heatmap...")

# Dynamically build full order list
order_full = [col for pattern in ['RSP', 'PM', 'AM', 'A', 'RL', 'AL', 'LM']
              for col in df.columns if re.match(f"{pattern}\\d*", col)]
order_full = list(dict.fromkeys(order_full))

if not order_full:
    raise ValueError("❌ No matching columns found for red-white cluster heatmap.")

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
red_white_cm = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'], N=100)

# Drop constant or all-zero rows
df_scaled = df_scaled.loc[df_scaled.var(axis=1) > 0]

# Final check before clustering
if df_scaled.shape[0] < 2:
    raise ValueError("❌ Too few rows remaining after variance filtering to perform clustering.")

# Draw clustermap
clusterfig = sns.clustermap(
    df_scaled_np,
    col_cluster=False,
    metric='cosine',
    method='average',
    cbar_kws=dict(label='Projection Strength'),
    cmap=red_white_cm,
    vmin=0.0,
    vmax=1.0
)

clusterfig.ax_heatmap.set_title(sample_name.replace('_', ' '))
clusterfig.ax_heatmap.axes.get_yaxis().set_visible(False)

for ext in ['pdf', 'svg', 'png']:
    clusterfig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_red_white_cluster_heatmap.{ext}")))

print("✅ Red-White cluster heatmap saved.")

### === Han-Style Heatmap === ###
print("🔍 Generating Han-style heatmap...")

# Han colormap
han_cm = LinearSegmentedColormap.from_list('white_to_orange', ['white', 'orange'], N=100)

# Define Han-style targets
han_targets = ['LM', 'AL', 'PM', 'AM', 'RL']
han_order_full = [col for pattern in han_targets for col in df.columns if re.match(f"{pattern}\\d*", col)]
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

# Linkage
row_linkage = linkage(pdist(df_han, metric='euclidean'), method='ward')

# Ensure clean native float matrix
df_han_np = df_han.to_numpy(copy=True).astype(float)

# Draw Han-style heatmap
clusterfig_han = sns.clustermap(
    df_han_np,
    row_linkage=row_linkage,
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
pattern = re.compile('([^\s\w]|_)+')

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
glabels = ["1 target \n {:0.3}\%".format(100*c_np[0] / c_tot)]
glabels += ["{} targets \n {:0.3}\%".format(i+2,100*j/c_tot) for (i,j) in zip(range(c_np.shape[0]-1),c_np[1:])]
patches, texts = plt.pie(c.to_numpy().flatten(),labels=glabels)
[txt.set_fontsize(8) for txt in texts]
#plt.savefig(os.path.normpath(os.path.join(plot_dir, sample_name + "_num_targets_pie.pdf")))
for ext in ["pdf", "svg", "png"]:
    plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_num_targets_pie.{ext}")))

maxproj = TSNE(n_components=2,metric='cosine').fit_transform(df.to_numpy(copy=True))

#maxprojclusters = k_means(X=maxproj,n_clusters=6)

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
expected_sd_raw = np.array([np.sqrt(motif_probs[i] * n0 * (1-motif_probs[i])) for i in range(len(slabelsraw))])

degree = [len(x) for x in motif_labels]
degree[0] = 0

group = []
bonferroni_correction = len(slabels)
for i in range(len(degree)):
    """
    Group 1: motifs significantly over represented
    Group 2: motifs non-sig over-represented
    Group 3: motifs non-sig under-represented
    Group 4: motifs significantly under represented
    """
    grp = 0
    thr = 0.05 / bonferroni_correction
    if effectsigsraw[i,0] > 0: #over-represented
        if effectsigsraw[i,1] < thr: #statistically significant
            grp = 1
        else:
            grp = 3 #2
    else: #under-represented
        if effectsigsraw[i,1] > thr:
            grp = 4
        else: #statistically significant
            grp = 2
    group.append(grp)

dfraw = pd.DataFrame(data=[
                           motif_labels,\
                           dcounts,exp_counts.astype(int), \
                          expected_sd_raw,effectsigsraw[:,0], effectsigsraw[:,1], degree, group]).T
dfraw.columns=['Motifs','Observed','Expected', 'Expected SD','Effect Size', 'P-value', 'Degree', 'Group']
dfraw.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_upsetplot.csv")),
    index=False
)

dfraw.iloc[40:70]

dfdata = prepare_upset_data(dfraw)
dfdata = dfdata.sort_values(by=['Group','Observed'], ascending=[True,False])

###CHATGPT OPTIMIZED VERSION FOR ANY NUMBER OF GROUPS
def kplot(df, size=(30,12)):
    """
    data : pd.DataFrame
    data is a dataframe with columns "Motifs" and "Counts"
    where "Motifs" is a list of lists e.g. [['PFC','LS'],['LS']]
    and "Counts" is a simple array of integers
    """
    motiflabels = df['Motifs'].to_list()
    data = up.from_memberships(motiflabels, data=df['Observed'].to_numpy())
    xlen = df.shape[0]
    xticks = np.arange(xlen)
    uplot = up.UpSet(data, sort_by=None)  # sort_by='cardinality'
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 3], 'height_ratios': [3, 1]})
    fig.set_size_inches(size)
    ax[1, 0].set_ylabel("Set Totals")
    uplot.plot_matrix(ax[1, 1])
    uplot.plot_totals(ax[1, 0])
    ax[0, 0].axis('off')
    ax[0, 1].spines['bottom'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    
    width = 0.35
    dodge = width / 2
    x = np.arange(8)
    ax[1, 0].set_title("Totals")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].set_xlim(ax[1, 1].get_xlim())
    
    # Get unique group labels and create color map for each group dynamically
    unique_groups = df['Group'].unique()
    colorlist = ['red', 'darkblue', 'black', 'green', 'purple']  # Extend as needed
    color_map = {group: colorlist[i % len(colorlist)] for i, group in enumerate(unique_groups)}
    
    # Map the colors based on the group
    cs = [color_map[group] for group in df['Group']]
    
    # Plot the bars with colors
    ax[0, 1].bar(xticks - dodge, df['Observed'].to_numpy(), width=width, label="Observed", align="center", color=cs, edgecolor='lightgray')
    ax[0, 1].bar(xticks + dodge, df['Expected'].to_numpy(), yerr=df['Expected SD'].to_numpy(), width=width / 2, label="Expected", align="center", color='gray', alpha=0.5, ecolor='lightgray')
    
    # Draw significance asterisks
    for group in unique_groups:
        group_ids = np.where(df['Group'].to_numpy() == group)[0]
        for i in group_ids:
            ax[0, 1].text(xticks[i] - 0.5 * dodge, df['Observed'].to_numpy()[i] + 1, "*", fontsize=12, color=color_map[group])

    # Hide axis ticks and grids
    ax[0, 1].xaxis.grid(False)
    ax[0, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.set_visible(False)
    ax[1, 1].xaxis.grid(False)
    
    fig.tight_layout()
    return fig, ax

fig, _ = kplot(dfdata)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_upsetplot_gpt.{ext}")))

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

fig,_ = kplot(dfdata)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_upsetplot.{ext}")))

### === Arrow Diagram Panels f and g === ###
import networkx as nx
from itertools import combinations
from scipy.stats import binom
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.lines import Line2D

manual_region_order = ["RSP", "PM", "AM", "A", "RL", "AL", "LM"]
regions = [r for r in manual_region_order if r in df.columns]
matrix = df[regions].apply(pd.to_numeric, errors='coerce').to_numpy()

# Binary projection matrix
binary_matrix = (matrix > 0).astype(int)
n_cells = binary_matrix.shape[0]

# === Panel f: Dedicated Projections ===
proj_counts = binary_matrix.sum(axis=1)
only_one_proj = (proj_counts == 1)
dedicated_cells = binary_matrix[only_one_proj]
dedicated_counts = dedicated_cells.sum(axis=0)

plt.figure(figsize=(8, 5))
plt.bar(regions, dedicated_counts, color='gray')
plt.ylabel("Number of Dedicated Neurons")
plt.title("Fig 10f: Dedicated Projection Neurons")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_panel_f_dedicated.svg")), format='svg')
plt.close()

# === Panel g: Motif Significance Analysis ===
region_probs = binary_matrix.mean(axis=0)
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

# FDR correction
pvals = [m['pval'] for m in motif_stats]
rejected, corrected_pvals = fdrcorrection(pvals, alpha=0.05)
for i, val in enumerate(motif_stats):
    val['FDR_pval'] = corrected_pvals[i]
    val['significant'] = rejected[i]

motif_df = pd.DataFrame(motif_stats)
motif_df.to_csv(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_significance.csv")),
    index=False
)


print("\n--- Generating 2-region motif graphs ---")
# === NetworkX for 2-region motifs ===
pair_counts = {k: v for k, v in observed_counts.items() if len(k) == 2}
G = nx.Graph()
for label in regions:
    G.add_node(label)

max_count = max(pair_counts.values())
for (i, j), count in pair_counts.items():
    region_i, region_j = regions[i], regions[j]
    motif_label = '+'.join(sorted([region_i, region_j]))
    row = motif_df[(motif_df['motif'] == motif_label) & (motif_df['size'] == 2)]
    if not row.empty:
        observed = row['observed'].values[0]
        expected = row['expected'].values[0]
        diff = observed - expected
        color = 'red' if row['significant'].values[0] and diff > 0 else ('blue' if row['significant'].values[0] and diff < 0 else 'black')
        width = 1 + 9 * (observed / max_count)
        G.add_edge(region_i, region_j, weight=width, color=color)

# Draw
angle_step = 2 * np.pi / len(regions)
pos = {
    region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
    for i, region in enumerate(manual_region_order)
}
edges = G.edges(data=True)
colors = [e[2]['color'] for e in edges]
widths = [e[2]['weight'] for e in edges]

plt.figure(figsize=(8, 8))
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, edge_color=colors, width=widths)
plt.title("Fig 10g: Broadcasting Neurons - Significant 2-Region Motifs")
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Overrepresented'),
    Line2D([0], [0], color='blue', lw=2, label='Underrepresented'),
    Line2D([0], [0], color='black', lw=2, label='Not Significant')
]
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.normpath(os.path.join(plot_dir, f"{sample_name}_panel_g_broadcasting.svg")))
plt.close()

print("\n--- Generating 3- and 4-region motif graphs ---")
for r in [3, 4]:
    print(f"Checking {r}-way motifs...")
    filtered = motif_df[(motif_df['size'] == r)]
    if not filtered.empty:
        print(f"→ Found {len(filtered)} {r}-way motifs. Building graph...")
        max_count_r = filtered['observed'].max()
        G_multi = nx.Graph()

        for _, row in filtered.iterrows():
            motif_nodes = row['motif'].split('+')
            observed = row['observed']
            expected = row['expected']
            diff = observed - expected
            color = 'red' if row['significant'] and diff > 0 else ('blue' if row['significant'] and diff < 0 else 'black')
            width = 1 + 4 * (observed / max(1, max_count_r))

            for i, j in combinations(motif_nodes, 2):
                if G_multi.has_edge(i, j):
                    if G_multi[i][j]['weight'] < width:
                        G_multi[i][j]['color'] = color
                        G_multi[i][j]['weight'] = width
                else:
                    G_multi.add_edge(i, j, color=color, weight=width)

        angle_step = 2 * np.pi / len(manual_region_order)
        pos = {
            region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
            for i, region in enumerate(manual_region_order)
            if region in G_multi.nodes
        }

        edge_colors = [e[2]['color'] for e in G_multi.edges(data=True)]
        edge_weights = [e[2]['weight'] for e in G_multi.edges(data=True)]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx(G_multi, pos, with_labels=True, node_size=1000,
                         edge_color=edge_colors, width=edge_weights)
        plt.title(f"Fig 10g Extension: {r}-Region Motifs")

        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Overrepresented'),
            Line2D([0], [0], color='blue', lw=2, label='Underrepresented'),
            Line2D([0], [0], color='black', lw=2, label='Not Significant')
        ]
        plt.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

        plt.tight_layout()
        fig_path = os.path.normpath(os.path.join(plot_dir, f"{sample_name}_panel_g_{r}way_broadcasting.svg"))
        plt.savefig(fig_path, format='svg')
        plt.close()
        print(f"✅ Saved {r}-way broadcasting motif network to {fig_path}")

# Optional: Save significant motif lists to text files
for r in [3, 4]:
    filtered = motif_df[(motif_df['size'] == r) & (motif_df['significant'])]
    if not filtered.empty:
        txt_path = os.path.normpath(os.path.join(plot_dir, f"{sample_name}_motif_{r}way_significant.txt"))
        with open(txt_path, 'w') as f:
            for _, row in filtered.iterrows():
                f.write(f"{row['motif']}: Observed={row['observed']}, "
                        f"Expected={row['expected']:.2f}, "
                        f"p={row['FDR_pval']:.3g}\n")
        print(f"✅ Saved significant {r}-way motifs to {txt_path}")


### === K-Means Cluster Centroid Heatmap === ###
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

scolors = ['white', 'red']
scm = LinearSegmentedColormap.from_list('white_to_red', scolors, N=256)

# Optionally reorder columns (if you want Han-style ordering)
# region_order = ["RSP", "PM", "AM", "A", "RL", "AL", "LM"]
# df = df[[col for col in region_order if col in df.columns]]

k_clusters = consensus_k if 'consensus_k' in locals() else 8
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
fig.savefig(
    os.path.normpath(os.path.join(plot_dir, f"{sample_name}_ExtendedDataFig10_Recreation.svg")),
    format='svg'
)
plt.close()


df.astype(bool).sum()
