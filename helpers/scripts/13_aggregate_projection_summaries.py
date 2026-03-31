#!/usr/bin/env python3
"""
Aggregate all projection_summary.csv files across all parameterizations.

This script:
1. Finds all projection_summary.csv files in 02_output/
2. Filters for aggregate samples (containing "_ALL_" in Sample name)
3. Extracts metadata (age, parameterization) from file paths
4. Combines all into a single summary CSV

Usage:
    python aggregate_projection_summaries.py [--output_dir OUTPUT_DIR] [--base_dir BASE_DIR]
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import glob

def extract_metadata_from_path(file_path):
    """
    Extract age and parameterization from file path.
    
    Args:
        file_path: Path to projection_summary.csv file
        
    Returns:
        dict with 'age' and 'parameterization' keys
    """
    path_parts = Path(file_path).parts
    age = None
    parameterization = None
    
    # Find age (p3, p12, p20, p60)
    for part in path_parts:
        if part.lower().startswith('p') and len(part) <= 4 and part[1:].isdigit():
            age = part.lower()
            break
    
    # Find parameterization (starts with 01., 02., 03., 04., or 05.)
    for part in path_parts:
        if part.startswith(('01.', '02.', '03.', '04.', '05.')):
            parameterization = part
            break
    
    return {
        'age': age,
        'parameterization': parameterization,
        'file_path': str(file_path)
    }

def load_and_filter_aggregate_rows(file_path):
    """
    Load projection_summary.csv and return only aggregate rows.
    
    Args:
        file_path: Path to projection_summary.csv
        
    Returns:
        DataFrame with only aggregate rows (Sample contains "_ALL_"), or None if no aggregate rows
    """
    try:
        df = pd.read_csv(file_path)
        
        # Filter for aggregate samples (containing "_ALL_" in Sample column)
        if 'Sample' not in df.columns:
            print(f"Warning: No 'Sample' column in {file_path}")
            return None
        
        aggregate_rows = df[df['Sample'].str.contains('_ALL_', case=False, na=False)]
        
        if len(aggregate_rows) == 0:
            return None
        
        return aggregate_rows
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all projection_summary.csv files across parameterizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (02_output/ and ~/helpers/scripts/13_summary_generation/)
  python aggregate_projection_summaries.py
  
  # Custom base directory
  python aggregate_projection_summaries.py --base_dir /path/to/output
  
  # Custom output directory
  python aggregate_projection_summaries.py --output_dir /path/to/output
        """
    )
    
    parser.add_argument('--base_output_dir', type=str, default=None,
                       help='Base output directory for processing results (default: REPO_ROOT/02_output)')
    parser.add_argument('--helper_output_dir', type=str, default=None,
                       help='Directory for helper script outputs (default: helpers/outputs/13_aggregate_projection_summaries)')
    
    args = parser.parse_args()
    
    # Get repository root (assuming helpers/ is a subdirectory)
    REPO_ROOT = Path(__file__).parent.parent.parent
    
    # Determine base directory
    if args.base_output_dir:
        OUTPUT_BASE = Path(args.base_output_dir)
    else:
        OUTPUT_BASE = REPO_ROOT / "02_output"
    
    if not OUTPUT_BASE.exists():
        print(f"Error: Output directory not found: {OUTPUT_BASE}")
        print(f"Please specify --base_output_dir or ensure 02_output/ exists relative to script location")
        return 1
    
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
    
    # Determine output directory - follow same pattern as other helper scripts:
    # 02_output/{parameterization}_helpers/13_aggregate_projection_summaries/
    SCRIPT_DIR = Path(__file__).parent
    if args.helper_output_dir:
        output_dir = Path(args.helper_output_dir)
    else:
        if parameterization_filter:
            output_dir = OUTPUT_BASE / f"{parameterization_filter}_helpers" / "13_aggregate_projection_summaries"
        else:
            # Default fallback
            output_dir = SCRIPT_DIR.parent / "outputs" / "13_aggregate_projection_summaries"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for projection_summary.csv files in: {OUTPUT_BASE}")
    print(f"Output will be saved to: {output_dir}")
    print("=" * 80)
    
    # Find all projection_summary.csv files
    pattern = str(OUTPUT_BASE / "**" / "projection_summary.csv")
    summary_files = glob.glob(pattern, recursive=True)
    
    if not summary_files:
        print(f"Error: No projection_summary.csv files found in {OUTPUT_BASE}")
        return 1
    
    print(f"Found {len(summary_files)} projection_summary.csv files")
    
    # Process each file
    all_aggregate_data = []
    
    for file_path in sorted(summary_files):
        print(f"\nProcessing: {file_path}")
        
        # Extract metadata
        metadata = extract_metadata_from_path(file_path)
        
        if not metadata['age'] or not metadata['parameterization']:
            print(f"  Warning: Could not extract age or parameterization from path. Skipping.")
            continue
        
        # Load and filter for aggregate rows
        df_aggregate = load_and_filter_aggregate_rows(file_path)
        
        if df_aggregate is None or len(df_aggregate) == 0:
            print(f"  No aggregate rows found (no samples with '_ALL_' in name)")
            continue
        
        print(f"  Found {len(df_aggregate)} aggregate row(s)")
        
        # Add metadata columns
        df_aggregate = df_aggregate.copy()
        df_aggregate['Age'] = metadata['age']
        df_aggregate['Parameterization'] = metadata['parameterization']
        df_aggregate['SourceFile'] = metadata['file_path']
        
        all_aggregate_data.append(df_aggregate)
    
    if not all_aggregate_data:
        print("\nError: No aggregate data found in any projection_summary.csv files")
        return 1
    
    # Combine all data
    print("\n" + "=" * 80)
    print("Combining all aggregate data...")
    combined_df = pd.concat(all_aggregate_data, ignore_index=True)
    
    # Reorder columns to put metadata first
    metadata_cols = ['Age', 'Parameterization', 'SourceFile', 'Sample']
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    column_order = metadata_cols + other_cols
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[column_order]
    
    # Save summary CSV
    output_file = output_dir / "aggregate_projection_summary_all_parameterizations.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Summary saved to: {output_file}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Ages: {sorted(combined_df['Age'].unique())}")
    print(f"   Parameterizations: {sorted(combined_df['Parameterization'].unique())}")
    print(f"   Columns: {len(combined_df.columns)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    
    if 'TotalProjections' in combined_df.columns:
        print(f"\nTotal Projections by Age:")
        print(combined_df.groupby('Age')['TotalProjections'].sum())
    
    if 'ObservedCells' in combined_df.columns:
        print(f"\nObserved Cells by Age:")
        print(combined_df.groupby('Age')['ObservedCells'].sum())
    
    if 'Entropy' in combined_df.columns:
        print(f"\nMean Entropy by Parameterization:")
        print(combined_df.groupby('Parameterization')['Entropy'].mean())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

