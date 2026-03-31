#Takes the motif_raw_data directory as input to create projection strength plots. 
# For individual files: processes files from one age group and saves to age-specific directory (p12/, p20/, p60/)
# For aggregate files: collects aggregate files from all ages and compares them in ALL_ages/ directory

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend for headless environments
matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep text editable in SVGs
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arial']  # List of fonts to try

import matplotlib.pyplot as plt
from pathlib import Path

def detect_age_from_path(path):
    """Extract age group (p12, p20, p60) from file path"""
    path_str = str(path).lower()
    # Look for p12, p20, p60 in the path
    age_match = re.search(r'/(p(12|20|60))/', path_str)
    if age_match:
        return age_match.group(1)
    return None

def main(data_dir, output_dir=None, file_type='auto', all_ages_base_dir=None):
    """
    Plot normalized projection strength data.
    
    Args:
        data_dir: Directory containing *_raw_data.csv files (for individual) or base directory (for aggregate all_ages)
        output_dir: Base directory to save plots (default: helpers/outputs/09_plot_normalized_projection_strength_data)
        file_type: 'individual', 'aggregate', or 'auto' (default: 'auto')
        all_ages_base_dir: Base directory containing p12/, p20/, p60/ subdirectories (for aggregate mode)
    """
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = str(script_dir.parent / "outputs" / "09_plot_normalized_projection_strength_data")
    
    all_csv_files = glob.glob(os.path.join(data_dir, "*_raw_data.csv"))
    
    # Detect file types: individual animal files vs aggregate files
    # Individual: starts with lowercase letters + digits (e.g., jr0674, m777)
    # Aggregate: contains "ALL" or starts with p12/p20/p60/P12/P20/P60
    individual_pattern = re.compile(r'^[a-z]{1,3}\d+', re.IGNORECASE)
    aggregate_pattern = re.compile(r'(ALL|^p(12|20|60))', re.IGNORECASE)
    
    individual_files = []
    aggregate_files = []
    
    for file in all_csv_files:
        filename = Path(file).stem
        if aggregate_pattern.search(filename):
            aggregate_files.append(file)
        elif individual_pattern.match(filename):
            individual_files.append(file)
    
    # Select files based on file_type argument
    file_type = file_type.lower()
    if file_type == 'individual':
        csv_files = individual_files
        if not csv_files:
            print(f"Error: No individual animal files found in {data_dir}")
            return
        
        # Detect age from input directory path
        age = detect_age_from_path(data_dir)
        if not age:
            print(f"Warning: Could not detect age from path {data_dir}. Using 'unknown'")
            age = 'unknown'
        
        # Create age-specific output directory
        age_output_dir = os.path.join(output_dir, age)
        os.makedirs(age_output_dir, exist_ok=True)
        print(f"Using {len(csv_files)} individual animal files from {age}")
        print(f"Saving plots to: {age_output_dir}")
        
    elif file_type == 'aggregate':
        # For aggregate mode, we need to collect files from all ages
        if all_ages_base_dir is None:
            # Try to infer from data_dir (go up to find p12/p20/p60 parent)
            all_ages_base_dir = data_dir
            # Check if we're already in an age-specific directory
            age = detect_age_from_path(data_dir)
            if age:
                # Go up to find the parent directory containing all ages
                path_parts = Path(data_dir).parts
                for i, part in enumerate(path_parts):
                    if part.lower() == age:
                        all_ages_base_dir = str(Path(*path_parts[:i]))
                        break
        
        # Extract parameterization name from output_dir if provided
        parameterization_filter = None
        if output_dir:
            output_path = Path(output_dir)
            # Look for parameterization name in path (e.g., .../01.minimal_filter_parameters_..._helpers/...)
            for part in output_path.parts:
                if part.startswith(('01.', '02.', '03.', '04.', '05.')) and '_helpers' in part:
                    # Extract just the parameterization name (before _helpers)
                    parameterization_filter = part.split('_helpers')[0]
                    print(f"Filtering by parameterization: {parameterization_filter}")
                    break
                elif part.startswith(('01.', '02.', '03.', '04.', '05.')):
                    parameterization_filter = part
                    print(f"Filtering by parameterization: {parameterization_filter}")
                    break
        
        # Find aggregate files from all age groups
        age_groups = ['p12', 'p20', 'p60']
        all_age_aggregate_files = {}
        
        for age in age_groups:
            # Search for aggregate files in this age's motif_raw_data directory
            # Filter by parameterization if specified
            if parameterization_filter:
                # Search only in the specific parameterization directory
                param_path = Path(all_ages_base_dir) / age / parameterization_filter
                if param_path.exists():
                    age_pattern = str(param_path / "**" / "*ALL*_raw_data.csv")
                    age_files = glob.glob(age_pattern, recursive=True)
                    # Also try case variations
                    if not age_files:
                        age_pattern = str(param_path / "**" / f"*{age.upper()}*ALL*_raw_data.csv")
                        age_files = glob.glob(age_pattern, recursive=True)
                else:
                    age_files = []
            else:
                # Original behavior: search all parameterizations
                age_pattern = os.path.join(all_ages_base_dir, age, "**", "*ALL*_raw_data.csv")
                age_files = glob.glob(age_pattern, recursive=True)
                # Also try case variations
                if not age_files:
                    age_pattern = os.path.join(all_ages_base_dir, age, "**", f"*{age.upper()}*ALL*_raw_data.csv")
                    age_files = glob.glob(age_pattern, recursive=True)
            
            if age_files:
                all_age_aggregate_files[age] = age_files
                print(f"Found {len(age_files)} aggregate files for {age}")
            else:
                print(f"Warning: No aggregate files found for {age}")
        
        if not all_age_aggregate_files:
            print(f"Error: No aggregate files found in {all_ages_base_dir}")
            return
        
        # Create ALL_ages output directory
        all_ages_output_dir = os.path.join(output_dir, "ALL_ages")
        os.makedirs(all_ages_output_dir, exist_ok=True)
        print(f"Saving aggregate comparison plots to: {all_ages_output_dir}")
        
        # Process aggregate files across all ages
        process_aggregate_all_ages(all_age_aggregate_files, all_ages_output_dir)
        return
        
    elif file_type == 'auto':
        # Auto-detect: prioritize individual files if both exist
        if individual_files and aggregate_files:
            print(f"Warning: Both individual ({len(individual_files)}) and aggregate ({len(aggregate_files)}) files found.")
            print(f"Using individual files only. Use --file_type aggregate to use aggregate files instead.")
            csv_files = individual_files
            file_type = 'individual'
        elif individual_files:
            csv_files = individual_files
            file_type = 'individual'
        elif aggregate_files:
            csv_files = aggregate_files
            file_type = 'aggregate'
        else:
            print(f"Error: No valid *_raw_data.csv files found in {data_dir}")
            return
        
        # Recursively call with determined file_type
        return main(data_dir, output_dir, file_type, all_ages_base_dir)
    else:
        print(f"Error: Invalid file_type '{file_type}'. Must be 'individual', 'aggregate', or 'auto'")
        return
    
    # Process individual files for a single age group
    grouped_files = {}

    for file in csv_files:
        filename = Path(file).stem
        parts = filename.split('_')
        
        # For individual files, title is the motif (regions) - everything between sample ID and "raw"
        title = '_'.join(parts[1:-2]) if parts[-2] == "raw" else '_'.join(parts[1:-1])
        
        if title not in grouped_files:
            grouped_files[title] = []
        grouped_files[title].append(file)

    region_order = ['LM', 'AL', 'AM', 'PM', 'RSP']  # Desired x-axis order

    for title, files in grouped_files.items():
        plt.figure(figsize=(12, 6))

        # Determine unique sample IDs and assign colors
        sample_ids = [Path(f).stem.split('_')[0] for f in files]
        unique_samples = sorted(set(sample_ids))
        color_map = plt.colormaps.get_cmap('tab10')
        color_lookup = {
            sample: color_map(i / max(1, len(unique_samples) - 1))
            for i, sample in enumerate(unique_samples)
        }

        all_data = []
        plotted_samples = set()
        region_labels = None  # Will be set from first file and used consistently

        for file in files:
            df = pd.read_csv(file)
            sample_id = Path(file).stem.split('_')[0]

            # Always skip the first column, and reorder based on region_order
            # Match columns case-insensitively (CSV has lowercase, region_order has uppercase)
            df_cols_lower = {col.lower(): col for col in df.columns[1:]}
            region_cols = []
            region_col_indices = [0]  # Always include first column (row labels)
            
            for region in region_order:
                region_lower = region.lower()
                if region_lower in df_cols_lower:
                    actual_col = df_cols_lower[region_lower]
                    region_cols.append(actual_col)
                    region_col_indices.append(df.columns.get_loc(actual_col))
            
            if not region_cols:
                print(f"Warning: No matching region columns found in {file}. Available columns: {list(df.columns[1:])}")
                continue
            
            # Set region_labels from first file (use uppercase for display)
            if region_labels is None:
                region_labels = [r.upper() for r in region_cols]
                
            df = df.iloc[:, region_col_indices]

            normalized_data = []
            for _, row in df.iterrows():
                values = row.values[1:].astype(float)  # Skip row label column
                normalized_data.append(values)
                color = color_lookup[sample_id]
                label = sample_id if sample_id not in plotted_samples else None
                plt.plot(region_labels, values, color=color, alpha=0.9, label=label)

            plotted_samples.add(sample_id)
            norm_array = np.array(normalized_data)
            all_data.append(norm_array)

        if not all_data:
            print(f"Warning: No data to plot for {title}")
            plt.close()
            continue
            
        combined_data = np.vstack(all_data)
        mean_vals = combined_data.mean(axis=0)
        sem_vals = combined_data.std(axis=0) / np.sqrt(combined_data.shape[0])
        std_vals = combined_data.std(axis=0)

        plt.errorbar(region_labels, mean_vals, fmt='-o', color='black',
                     linewidth=2, capsize=5, label='Mean')

        plt.title(f"Normalized Regional Data: {title} ({age.upper()})")
        plt.xlabel("Region")
        plt.ylabel("Normalized Value (0-1)")
        plt.grid(False)

        # Deduplicate and sort legend entries, with 'Mean' last
        handles, labels = plt.gca().get_legend_handles_labels()
        label_handle_pairs = dict(zip(labels, handles))

        mean_handle = label_handle_pairs.pop('Mean', None)
        sorted_pairs = sorted(label_handle_pairs.items(), key=lambda x: x[0])
        if mean_handle:
            sorted_pairs.append(('Mean', mean_handle))

        sorted_labels, sorted_handles = zip(*sorted_pairs)
        plt.legend(sorted_handles, sorted_labels, title="Sample ID")

        plt.tight_layout()
        output_path = os.path.join(age_output_dir, f"{title}_normalized_plot.svg")
        plt.savefig(output_path, format='svg')
        plt.close()

def process_aggregate_all_ages(all_age_aggregate_files, output_dir):
    """Process aggregate files from all ages and create comparison plots"""
    region_order = ['LM', 'AL', 'AM', 'PM', 'RSP']
    
    # Group files by motif (region combination) across all ages
    # First, extract motifs from all files
    motif_to_ages = {}  # {motif: {age: [files]}}
    
    for age, files in all_age_aggregate_files.items():
        for file in files:
            filename = Path(file).stem
            parts = filename.split('_')
            
            # Extract motif from aggregate filename (e.g., "p12_ALL_HAN_filters_pm_am_lm_raw_data.csv" -> "pm_am_lm")
            try:
                filters_idx = next(i for i, p in enumerate(parts) if p.upper() == 'FILTERS')
                raw_idx = next(i for i, p in enumerate(parts) if p.lower() == 'raw')
                motif = '_'.join(parts[filters_idx + 1:raw_idx])
            except StopIteration:
                # Fallback
                motif = '_'.join(parts[1:-2]) if parts[-2] == "raw" else '_'.join(parts[1:-1])
            
            if motif not in motif_to_ages:
                motif_to_ages[motif] = {}
            if age not in motif_to_ages[motif]:
                motif_to_ages[motif][age] = []
            motif_to_ages[motif][age].append(file)
    
    # Create plots for each motif, comparing all ages
    age_colors = {'p12': '#1f77b4', 'p20': '#ff7f0e', 'p60': '#2ca02c'}  # Blue, Orange, Green
    
    for motif, age_files_dict in motif_to_ages.items():
        plt.figure(figsize=(12, 6))
        
        all_age_data = {}  # {age: [arrays]}
        region_labels = None
        
        for age in ['p12', 'p20', 'p60']:
            if age not in age_files_dict:
                continue
            
            # Use the first file for this age (should only be one aggregate file per motif per age)
            file = age_files_dict[age][0]
            df = pd.read_csv(file)
            
            # Match columns case-insensitively
            df_cols_lower = {col.lower(): col for col in df.columns[1:]}
            region_cols = []
            region_col_indices = [0]
            
            for region in region_order:
                region_lower = region.lower()
                if region_lower in df_cols_lower:
                    actual_col = df_cols_lower[region_lower]
                    region_cols.append(actual_col)
                    region_col_indices.append(df.columns.get_loc(actual_col))
            
            if not region_cols:
                continue
            
            if region_labels is None:
                region_labels = [r.upper() for r in region_cols]
            
            df = df.iloc[:, region_col_indices]
            
            # Collect all rows for this age
            age_data = []
            for _, row in df.iterrows():
                values = row.values[1:].astype(float)
                age_data.append(values)
            
            if age_data:
                all_age_data[age] = np.array(age_data)
                # Plot mean for this age
                mean_vals = np.array(age_data).mean(axis=0)
                plt.plot(region_labels, mean_vals, 'o-', color=age_colors[age], 
                        linewidth=2, markersize=8, label=f'{age.upper()} Mean', alpha=0.8)
        
        # Plot overall mean across all ages
        if all_age_data:
            # Combine all ages
            combined_all = np.vstack(list(all_age_data.values()))
            overall_mean = combined_all.mean(axis=0)
            plt.plot(region_labels, overall_mean, 'o-', color='black', 
                    linewidth=3, markersize=10, label='Overall Mean', alpha=0.9)
        
        plt.title(f"Normalized Regional Data: {motif} (All Ages Comparison)")
        plt.xlabel("Region")
        plt.ylabel("Normalized Value (0-1)")
        plt.grid(False)
        plt.legend(title="Age Group")
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{motif}_all_ages_comparison.svg")
        plt.savefig(output_path, format='svg')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot normalized data with mean and SEM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process individual animal files from a specific age (saves to p12/, p20/, or p60/)
  python plot_normalized_projection_strength_data.py /path/to/p12/.../motif_raw_data --file_type individual
  
  # Process aggregate files from all ages (saves to ALL_ages/)
  python plot_normalized_projection_strength_data.py /path/to/02_output --file_type aggregate --all_ages_base_dir /path/to/02_output
  
  # Auto-detect (default: prioritizes individual if both exist)
  python plot_normalized_projection_strength_data.py /path/to/motif_raw_data --file_type auto
        """
    )
    parser.add_argument('data_dir', help="Directory containing *_raw_data.csv files (for individual) or base directory (for aggregate)")
    parser.add_argument('--output_dir', default=None, help="Base directory to save plots (default: helpers/outputs/09_plot_normalized_projection_strength_data)")
    parser.add_argument('--file_type', default='auto', choices=['individual', 'aggregate', 'auto'],
                       help="Type of files to process: 'individual' (animal-specific, saves to age/), 'aggregate' (ALL_HAN_filters, saves to ALL_ages/), or 'auto' (default)")
    parser.add_argument('--all_ages_base_dir', default=None,
                       help="Base directory containing p12/, p20/, p60/ subdirectories (required for aggregate mode)")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.file_type, args.all_ages_base_dir)
