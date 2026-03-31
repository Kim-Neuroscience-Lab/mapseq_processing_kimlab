# motif_analysis_pipeline.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent opening windows
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind, ks_2samp, entropy
import numpy as np
from scipy.spatial.distance import jensenshannon

# === Load Data ===
import os
import glob
import ast
from pathlib import Path

# Get repository root and output directory
import argparse
parser = argparse.ArgumentParser(description="Motif analysis pipeline")
parser.add_argument('--base_output_dir', type=str, default=None,
                   help='Base output directory for processing results (default: REPO_ROOT/02_output)')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/05_motif_analysis)')
args = parser.parse_args()

script_dir = Path(__file__).parent
REPO_ROOT = script_dir.parent.parent
if args.base_output_dir:
    OUTPUT_DIR = Path(args.base_output_dir)
else:
    OUTPUT_DIR = REPO_ROOT / "02_output"
# Output directory for this script
if args.helper_output_dir:
    OUTPUT_SCRIPT_DIR = Path(args.helper_output_dir)
else:
    OUTPUT_SCRIPT_DIR = script_dir.parent / "outputs" / "05_motif_analysis"
OUTPUT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Extract parameterization name and filter type from helper_output_dir if provided
parameterization_filter = None
filter_type = None
if args.helper_output_dir:
    helper_path = Path(args.helper_output_dir)
    # Look for parameterization name in path (e.g., .../01.minimal_filter_parameters_..._helpers/...)
    for part in helper_path.parts:
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
            print(f"Filtering by parameterization: {parameterization_filter}, filter type: {filter_type}")
            break

# P3 data was removed in manuscript before submission - exclude from analysis
age_groups = ['p12', 'p20', 'p60']
# age_groups = ['p3', 'p12', 'p20', 'p60']  # Original with P3

print("Generating motif_percent_matrix_by_age.csv from 02_output aggregate upsetplot files...")

# Dictionary to store data for each age
age_motif_data = {}

# Process all three models separately
models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                     'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                     'bayesian_hierarchical', 'ml_nonparametric']

for model_type in models_to_process:
    print("\n" + "="*80)
    print(f"Processing {model_type.upper()} MODEL")
    print("="*80)
    
    age_motif_data = {}
    
    for age in age_groups:
        # Look for aggregate upsetplot files in model-specific subdirectories
        # Filter by parameterization and filter type if specified
        if parameterization_filter and filter_type:
            # Search only in the specific parameterization directory
            param_path = OUTPUT_DIR / age / parameterization_filter
            if param_path.exists():
                patterns = [
                    str(param_path / "analysis" / model_type / f"*ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"*alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"{age.upper()}_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"{age.lower()}_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"{age.upper()}_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"{age.lower()}_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    # Handle p60 uppercase variation
                    str(param_path / "analysis" / model_type / f"P60_ALL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    str(param_path / "analysis" / model_type / f"P60_alL_{filter_type}_filters_upsetplot_{model_type}.csv"),
                    # Backward compatibility: also check main analysis directory
                    str(param_path / "analysis" / f"*ALL_{filter_type}_filters_upsetplot.csv"),
                    str(param_path / "analysis" / f"*alL_{filter_type}_filters_upsetplot.csv"),
                ]
            else:
                patterns = []
        else:
            # Original behavior: search for HAN filters (backward compatibility)
            patterns = [
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"*ALL_HAN_filters_upsetplot_{model_type}.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"*alL_HAN_filters_upsetplot_{model_type}.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"{age.upper()}_ALL_HAN_filters_upsetplot_{model_type}.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"{age.lower()}_ALL_HAN_filters_upsetplot_{model_type}.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"{age.upper()}_alL_HAN_filters_upsetplot_{model_type}.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / model_type / f"{age.lower()}_alL_HAN_filters_upsetplot_{model_type}.csv"),
                # Backward compatibility: also check main analysis directory
                str(OUTPUT_DIR / age / "**" / "analysis" / f"*ALL_HAN_filters_upsetplot.csv"),
                str(OUTPUT_DIR / age / "**" / "analysis" / f"*alL_HAN_filters_upsetplot.csv"),
            ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
        files = list(set(files))  # Remove duplicates
        
        if not files:
            print(f"Warning: No aggregate upsetplot file found for {age} ({model_type} model)")
            age_motif_data[age] = {}
            continue
        
        # Use the first matching file (should only be one aggregate file per age)
        file_path = files[0]
        print(f"Loading {age} data ({model_type} model) from: {file_path}")
        
        # Read the upsetplot CSV file
        df_temp = pd.read_csv(file_path)
        
        # Parse Motifs column and calculate percentages
        # Filter out rows with empty motif lists ([] or [''] with zero observed) before calculating total
        rows_to_drop = []
        for idx, row in df_temp.iterrows():
            motifs_str = row['Motifs']
            observed_count = float(row['Observed'])
            
            # Skip rows with zero observed counts (these are the empty/placeholder rows)
            if observed_count == 0.0:
                rows_to_drop.append(idx)
                continue
            
            try:
                if isinstance(motifs_str, str):
                    motifs_list = ast.literal_eval(motifs_str)
                else:
                    motifs_list = motifs_str
                
                # Skip rows with empty motif lists or lists containing only empty strings
                if not isinstance(motifs_list, list):
                    rows_to_drop.append(idx)
                elif len(motifs_list) == 0:
                    rows_to_drop.append(idx)
                elif all(not m or m.strip() == '' for m in motifs_list):
                    # List contains only empty strings
                    rows_to_drop.append(idx)
            except:
                # If we can't parse and observed is zero, drop it
                if observed_count == 0.0:
                    rows_to_drop.append(idx)
        
        # Drop the filtered rows
        df_filtered = df_temp.drop(rows_to_drop)
        
        # Calculate total from filtered data (excluding empty motif rows)
        total_observed = df_filtered['Observed'].sum()
        
        if total_observed == 0:
            print(f"  Warning: No valid data found for {age} after filtering empty rows")
            age_motif_data[age] = {}
            continue
        
        motif_percentages = {}
        for _, row in df_filtered.iterrows():
            # Parse the Motifs column (it's a string representation of a list)
            motifs_str = row['Motifs']
            try:
                if isinstance(motifs_str, str):
                    motifs_list = ast.literal_eval(motifs_str)
                else:
                    motifs_list = motifs_str
                
                # Skip empty motif lists (should already be filtered, but double-check)
                if not isinstance(motifs_list, list) or len(motifs_list) == 0:
                    continue
                if all(not m or m.strip() == '' for m in motifs_list):
                    continue
                
                # Create a sorted, normalized motif label
                # Sort and join with + for consistency
                motif_label = sorted(motifs_list)
                
                # Convert to string representation for consistency
                motif_key = str(motif_label)
                
                # Calculate percentage
                observed_count = float(row['Observed'])
                if total_observed > 0:
                    percentage = (observed_count / total_observed) * 100
                else:
                    percentage = 0.0
                
                motif_percentages[motif_key] = percentage
                
            except Exception as e:
                print(f"  Warning: Could not parse motif {motifs_str}: {e}")
                continue
        
        age_motif_data[age] = motif_percentages
        print(f"  Loaded {len(motif_percentages)} motifs for {age} ({model_type} model)")

    # Get all unique motifs across all ages
    all_motifs = set()
    for age_data in age_motif_data.values():
        all_motifs.update(age_data.keys())
    all_motifs = sorted(all_motifs)

    # Build the matrix
    matrix_data = {'Motif_Label': []}
    for age in age_groups:
        matrix_data[age.upper()] = []

    for motif in all_motifs:
        matrix_data['Motif_Label'].append(motif)
        for age in age_groups:
            percentage = age_motif_data[age].get(motif, 0.0)
            matrix_data[age.upper()].append(percentage)

    # Create DataFrame
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.set_index("Motif_Label", inplace=True)

    # Save the generated matrix for reference with model suffix
    output_csv = OUTPUT_SCRIPT_DIR / f"motif_percent_matrix_by_age_{model_type}.csv"
    matrix_df.to_csv(output_csv)
    print(f"\nGenerated and saved motif_percent_matrix_by_age_{model_type}.csv to: {output_csv}")
    print(f"Matrix shape: {matrix_df.shape}")
    print(f"Columns: {list(matrix_df.columns)}")

    # === Melt into Long Format ===
    long_df = matrix_df.reset_index().melt(id_vars="Motif_Label", var_name="Age", value_name="Percent")
    long_df["Motif_List"] = long_df["Motif_Label"].apply(eval)
    long_df["Motif_Count"] = long_df["Motif_List"].apply(len)

    # P3 data was removed in manuscript before submission - exclude from analysis
    ordered_ages = ["P12", "P20", "P60"]
    # ordered_ages = ["P3", "P12", "P20", "P60"]  # Original with P3
    unique_degrees = sorted(long_df["Motif_Count"].unique())

    # === Prepare PDF Outputs ===
    pdf_path_sorted = OUTPUT_SCRIPT_DIR / f"motif_barplots_sorted_{model_type}.pdf"
    pdf_path_by_rank = OUTPUT_SCRIPT_DIR / f"motif_barplots_ranked_{model_type}.pdf"

    # === Plot PDFs ===
    for by_rank, out_path in zip([False, True], [pdf_path_sorted, pdf_path_by_rank]):
        with PdfPages(out_path) as pdf:
            for age in ordered_ages:
                age_data = long_df[long_df['Age'] == age]
                # Get degrees that actually have data for this age
                degrees_with_data = sorted([d for d in unique_degrees 
                                          if not age_data[age_data["Motif_Count"] == d].empty])
                
                if not degrees_with_data:
                    print(f"Warning: No data found for {age}, skipping plot")
                    continue
                
                # Determine number of subplots needed (max 5, or number of degrees with data)
                n_plots = min(5, len(degrees_with_data))
                if n_plots == 0:
                    continue
                fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
                # Ensure axs is always a list/array for consistent indexing
                if n_plots == 1:
                    axs = [axs]
                fig.suptitle(f"Motif Observed % per Degree - {age}" + (" (Ranked)" if by_rank else ""), fontsize=16)

                for i, degree in enumerate(degrees_with_data[:n_plots]):
                    subset = age_data[age_data["Motif_Count"] == degree]
                    if subset.empty:
                        axs[i].axis("off")
                        continue

                    subset = subset.copy()
                    if by_rank:
                        subset = subset.sort_values("Percent", ascending=False)
                    else:
                        subset = subset.sort_values("Motif_Label")

                    sns.barplot(data=subset, x="Motif_Label", y="Percent", ax=axs[i], hue="Motif_Label", palette='viridis', legend=False)
                    axs[i].set_title(f"{degree}-Target Motifs")
                    axs[i].set_ylabel('% of Observed')
                    axs[i].set_xlabel('')
                    axs[i].tick_params(axis='x', rotation=90)
                    axs[i].set_ylim(0, 100)

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)

    # === Histogram Similarity Metrics ===
    js_data = []
    similarity_lines = ["Histogram Comparison Results:\n"]
    histogram_file = OUTPUT_SCRIPT_DIR / f"histogram_similarity_summary_{model_type}.txt"
    with open(histogram_file, "w") as f:
        for degree in unique_degrees:
            # Filter to only include ages in ordered_ages (exclude P3)
            valid_ages = [age for age in ordered_ages if age in long_df["Age"].values]
            subsets = {
                age: long_df[(long_df["Age"] == age) & (long_df["Motif_Count"] == degree)]["Percent"].values
                for age in valid_ages
            }
            age_transitions = [(valid_ages[i], valid_ages[i + 1]) for i in range(len(valid_ages) - 1)] + [("P12", "P60")]
            for age1, age2 in age_transitions:
                vec1, vec2 = subsets[age1], subsets[age2]

                p = np.asarray(vec1) / np.sum(vec1) if np.sum(vec1) > 0 else np.zeros_like(vec1)
                q = np.asarray(vec2) / np.sum(vec2) if np.sum(vec2) > 0 else np.zeros_like(vec2)
                min_len = min(len(p), len(q))
                js_div = jensenshannon(p[:min_len], q[:min_len])**2

                t_stat, t_p = ttest_ind(vec1, vec2, equal_var=False)
                ks_stat, ks_p = ks_2samp(vec1, vec2)

                line = (f"Degree {degree}: {age1} vs {age2} -> "
                        f"Welch's p = {t_p:.4e}, KS p = {ks_p:.4e}, JS Divergence = {js_div:.4f}, "
                        f"Significant = {t_p < 0.05}, {ks_p < 0.05}, {js_div > 0.05}")
                print(line)
                f.write(line + "\n")
                js_data.append({"Degree": degree, "Comparison": f"{age1} vs {age2}", "JS_Divergence": js_div})

    # === Plot JS Divergence from Histogram Similarity ===
    if js_data:
        js_df = pd.DataFrame(js_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=js_df, x="Comparison", y="JS_Divergence", hue="Degree", palette="magma")
        plt.title(f"JS Divergence Across Age Comparisons by Motif Degree ({model_type} model)")
        plt.ylabel("JS Divergence")
        plt.xlabel("Age Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_SCRIPT_DIR / f"js_divergence_histogram_comparisons_{model_type}.png")
        plt.close()

    # === Per-Motif Percent Change Analysis Across Ages ===
    transition_file = OUTPUT_SCRIPT_DIR / f"motif_transition_significance_summary_{model_type}.txt"
    with open(transition_file, "w") as f:
        f.write("Per-Motif Transition Comparison\n")
        # Build list of transitions including P12 vs P60 (exclude P3)
        # Filter to only include ages that exist in the data
        valid_ages_for_transitions = [age for age in ordered_ages if age in matrix_df.columns]
        age_transitions = [(valid_ages_for_transitions[i], valid_ages_for_transitions[i + 1]) 
                           for i in range(len(valid_ages_for_transitions) - 1)] + [("P12", "P60")]
        for age1, age2 in age_transitions:
            # Skip if either age is not in the data
            if age1 not in matrix_df.columns or age2 not in matrix_df.columns:
                continue
            f.write(f"\n{age1} vs {age2}\n")
            for motif in matrix_df.index:
                v1 = matrix_df.loc[motif, age1]
                v2 = matrix_df.loc[motif, age2]
                p_vec = np.array([v1, 100 - v1]) / 100 if v1 + v2 > 0 else np.zeros(2)
                q_vec = np.array([v2, 100 - v2]) / 100 if v1 + v2 > 0 else np.zeros(2)
                js_motif = jensenshannon(p_vec, q_vec)**2
                significance_flag = js_motif > 0.05
                line = f"{motif}: {age1} = {v1:.2f}%, {age2} = {v2:.2f}% -> JS Divergence = {js_motif:.4f}, Significant = {significance_flag}"
                f.write(line + "\n")

    # === Hierarchical Clustering Heatmap ===
    # Filter to only include ages in ordered_ages (exclude P3 if present in data)
    available_ages = [age for age in ordered_ages if age in matrix_df.columns]
    if len(available_ages) != len(ordered_ages):
        missing = set(ordered_ages) - set(available_ages)
        print(f"Warning: Missing age columns in data: {missing}")
        print(f"Available columns: {list(matrix_df.columns)}")
        print(f"Using available ages: {available_ages}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix_df[available_ages].values)
    sns.clustermap(X_scaled, method='ward', cmap='viridis', row_cluster=True, col_cluster=False,
                   figsize=(12, 16), yticklabels=matrix_df.index, xticklabels=available_ages)
    plt.title(f"Hierarchical Clustering (Standardized % by Age) - {model_type} model")
    plt.savefig(OUTPUT_SCRIPT_DIR / f"hclust_heatmap_{model_type}.png")
    plt.close()

    # === PCA + KMeans ===
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(pca_result)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='tab10')
    plt.title(f"PCA + KMeans Clustering - {model_type} model")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(OUTPUT_SCRIPT_DIR / f"pca_kmeans_plot_{model_type}.png")
    plt.close()

    # === Annotate PCA axes with contributing features ===
    pca_loadings = pd.DataFrame(pca.components_, columns=available_ages, index=["PC1", "PC2"])
    pca_loadings.T.to_csv(OUTPUT_SCRIPT_DIR / f"pca_feature_loadings_{model_type}.csv")

    print(f"\nTop contributing features to PC1 and PC2 ({model_type} model):")
    print(pca_loadings.T.abs().sort_values("PC1", ascending=False).head(5))
    print(pca_loadings.T.abs().sort_values("PC2", ascending=False).head(5))

    # === Export PCA results with cluster assignments ===
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=matrix_df.index)
    pca_df["KMeans_Cluster"] = labels
    pca_df.to_csv(OUTPUT_SCRIPT_DIR / f"motif_pca_clusters_{model_type}.csv")
    
    print(f"\n✅ Completed processing for {model_type} model")

print(f"\n📁 All results saved to: {OUTPUT_SCRIPT_DIR}")
print("   - Uniform model outputs: *_uniform.csv, *_uniform.pdf, etc.")
print("   - Region-specific model outputs: *_region_specific.csv, *_region_specific.pdf, etc.")
