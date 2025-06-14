import os
import argparse
import pandas as pd
import numpy as np

def get_column_mapping(original_columns, filename):
    print(f"\n📂 Processing file: {filename}")
    print("Detected columns:")

    mapping = {}
    for current_col in original_columns:
        # Build line with bold for current column
        line = []
        for col in original_columns:
            if col == current_col:
                line.append(f"\033[1m{col}\033[0m")  # ANSI bold
            else:
                line.append(col)
        print("   " + " | ".join(line))

        # Prompt for user input
        new_col = input(f"🔤 Enter standardized name for column '{current_col}': ").strip()
        mapping[current_col] = new_col
    return mapping

def identify_neg_column(columns, default_neg=None, max_attempts=3):
    print(f"\nStandardized columns: {columns}")
    attempts = 0
    while attempts < max_attempts:
        neg_col = input("❓ Enter the standardized name of the 'neg' column (press Enter to skip): ").strip()
        if not neg_col:
            if default_neg and default_neg in columns:
                print(f"✅ Defaulting to 'neg' column: {default_neg}")
                return default_neg
            else:
                print("⚠ No 'neg' column provided and no default available.")
                return None
        if neg_col in columns:
            return neg_col
        print("⚠ Invalid column name. Try again.")
        attempts += 1
    print("❌ Max attempts reached. Skipping 'neg' thresholding.")
    return None

def preprocess_file(filepath, outdir, fallback_threshold=2):
    df = pd.read_csv(filepath, sep='\t', header=0)
    base = os.path.basename(filepath).replace('.tsv', '')

    original_file_columns = df.columns.tolist()
    column_mapping = get_column_mapping(original_file_columns, os.path.basename(filepath))
    df = df.rename(columns=column_mapping)
    standardized_cols = df.columns.tolist()

    neg_col = identify_neg_column(standardized_cols, default_neg='neg')

    if "barcodes" not in df.columns:
        raise ValueError(f"❌ You must map a column to 'barcodes' in {filepath}")

    # Determine threshold
    if neg_col is None or neg_col not in df.columns:
        threshold = fallback_threshold
        print(f"⚠ No valid 'neg' column. Using fallback threshold = {threshold}")
    else:
        neg_values = df[neg_col].dropna().to_numpy()
        if len(neg_values) == 0:
            threshold = fallback_threshold
            print(f"⚠ No values in 'neg' column. Using fallback threshold = {threshold}")
        else:
            threshold = np.mean(neg_values) + np.std(neg_values)
            print(f"✅ Using threshold = mean + std = {threshold:.4f}")

    non_vbc = [col for col in df.columns if col.lower() not in ["barcodes"]]
    df[non_vbc] = df[non_vbc].apply(pd.to_numeric, errors="coerce")

    print(f"📊 Applying threshold to columns: {non_vbc}")
    pre_thresh_nonzero = (df[non_vbc] > 0).sum().sum()
    df[non_vbc] = df[non_vbc].apply(lambda col: col.where(col >= threshold, 0))
    post_thresh_nonzero = (df[non_vbc] > 0).sum().sum()
    print(f"🧹 Thresholding complete: {int(pre_thresh_nonzero - post_thresh_nonzero)} values set to zero.")

    df = df.loc[(df[non_vbc] > 0).any(axis=1)]

    if neg_col in df.columns:
        df = df[df[neg_col] == 0]

    cleaned_path = os.path.join(outdir, f"{base}_cleaned.tsv")
    df.to_csv(cleaned_path, sep='\t', index=False)
    print(f"💾 Saved cleaned file to {cleaned_path}\n")

    return df



def main(input_dir, output_dir, fallback_threshold):
    os.makedirs(output_dir, exist_ok=True)
    cleaned_dfs = []

    # Track column order in order of first appearance
    column_order = []
    seen_columns = set()

    for file in os.listdir(input_dir):
        if file.endswith(".tsv"):
            full_path = os.path.join(input_dir, file)
            cleaned_df = preprocess_file(full_path, output_dir, fallback_threshold)
            cleaned_dfs.append(cleaned_df)
            for col in cleaned_df.columns:
                if col not in seen_columns:
                    column_order.append(col)
                    seen_columns.add(col)

    # Align all columns across datasets
    aligned_dfs = []
    for df in cleaned_dfs:
        df_aligned = df.copy()
        for col in column_order:
            if col not in df_aligned.columns:
                df_aligned[col] = 0
        df_aligned = df_aligned[column_order]  # preserve original order
        aligned_dfs.append(df_aligned)

    # Aggregate all aligned cleaned data
    if aligned_dfs:
        final_df = pd.concat(aligned_dfs, axis=0)
        aggregate_path = os.path.join(output_dir, "aggregated_cleaned_matrix.tsv")
        final_df.to_csv(aggregate_path, sep='\t', index=False)
        print(f"\n✅ Aggregated matrix saved to:\n📂 {aggregate_path}")
    else:
        print("⚠ No cleaned datasets found to aggregate.")
        return

    # 🧾 Post-run summary
    total_files = len(cleaned_dfs)
    total_zeroed = sum(((df == 0).sum().sum() for df in aligned_dfs))
    print(f"\n🧾 Summary: Processed {total_files} file(s), output written to:")
    print(f"📄 {aggregate_path}")
    print(f"🧹 Total zeroed matrix entries across all files: {total_zeroed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and align replicate TSVs for aggregation.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory with replicate .tsv files")
    parser.add_argument("-o", "--output_dir", required=True, help="Where to save cleaned and aggregated files")
    parser.add_argument("-t", "--fallback_threshold", type=float, default=2.0, help="Used if neg column has no data")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.fallback_threshold)
