#!/usr/bin/env python3
"""
Model Group Comparison Script

This script aggregates group information from upsetplot_{model}.csv files across
multiple models (uniform, region_specific) and ages (p3, p12, p20, p60).

The output is a wide-format CSV where each row is a motif and columns show the
group value for each age+model combination.

Usage:
    python 14_model_group_comparison.py [--base_output_dir BASE_DIR] [--helper_output_dir OUTPUT_DIR]
"""

import pandas as pd
import numpy as np
import ast
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Try to import PIL/Pillow for TIFF support
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def extract_parameterization_from_path(path):
    """
    Extract parameterization name from helper_output_dir path.
    
    Args:
        path: Path object or string
        
    Returns:
        str or None: Parameterization name if found, None otherwise
    """
    helper_path = Path(path)
    for part in helper_path.parts:
        if part.startswith(('01.', '02.', '03.', '04.', '05.')):
            # Extract parameterization name (before _helpers if present)
            if '_helpers' in part:
                return part.split('_helpers')[0]
            else:
                return part
    return None


def normalize_motif(motif_str):
    """
    Convert motif string representation to consistent format.
    
    Args:
        motif_str: String representation of motif list (e.g., "['rsp', 'pm']")
        
    Returns:
        str: Normalized motif string representation (sorted, consistent format)
    """
    try:
        if isinstance(motif_str, str):
            motifs_list = ast.literal_eval(motif_str)
        else:
            motifs_list = motif_str
        
        # Handle empty or invalid motifs
        if not isinstance(motifs_list, list):
            return None
        
        # Filter out empty strings and normalize
        motifs_list = [m.strip() for m in motifs_list if m and m.strip()]
        
        if len(motifs_list) == 0:
            return None
        
        # Sort for consistency and return as string representation
        sorted_motifs = sorted(motifs_list)
        return str(sorted_motifs)
    
    except Exception as e:
        # If parsing fails, return None
        return None


def find_upsetplot_files(base_dir, age, model, parameterization_filter=None):
    """
    Find upsetplot CSV files for given age and model.
    
    Args:
        base_dir: Base output directory (e.g., 02_output)
        age: Age string (e.g., 'p3', 'p12', 'p20', 'p60')
        model: Model type ('uniform' or 'region_specific')
        parameterization_filter: Optional parameterization name to filter by
        
    Returns:
        list: List of matching file paths
    """
    base_path = Path(base_dir)
    
    if parameterization_filter:
        # Search only in the specific parameterization directory
        param_path = base_path / age / parameterization_filter
        if not param_path.exists():
            return []
        
        # Try multiple patterns for aggregate files
        patterns = [
            str(param_path / "analysis" / model / f"*ALL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"*alL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"*All_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"*aLl_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"{age.upper()}_ALL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"{age.lower()}_ALL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"{age.upper()}_alL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"{age.lower()}_alL_*_filters_upsetplot_{model}.csv"),
            # Handle p60 uppercase variation
            str(param_path / "analysis" / model / f"P60_ALL_*_filters_upsetplot_{model}.csv"),
            str(param_path / "analysis" / model / f"P60_alL_*_filters_upsetplot_{model}.csv"),
        ]
    else:
        # Search across all parameterizations
        patterns = [
            str(base_path / age / "**" / "analysis" / model / f"*ALL_*_filters_upsetplot_{model}.csv"),
            str(base_path / age / "**" / "analysis" / model / f"*alL_*_filters_upsetplot_{model}.csv"),
            str(base_path / age / "**" / "analysis" / model / f"*All_*_filters_upsetplot_{model}.csv"),
            str(base_path / age / "**" / "analysis" / model / f"*aLl_*_filters_upsetplot_{model}.csv"),
            str(base_path / age / "**" / "analysis" / model / f"{age.upper()}_ALL_*_filters_upsetplot_{model}.csv"),
            str(base_path / age / "**" / "analysis" / model / f"{age.lower()}_ALL_*_filters_upsetplot_{model}.csv"),
        ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and filter for aggregate files (must contain _ALL_ pattern)
    files = list(set(files))
    # Explicitly check for _ALL_ pattern (case-insensitive) to ensure we only get aggregate files
    # This ensures we don't accidentally process individual animal files
    aggregate_files = []
    for f in files:
        filename_upper = Path(f).name.upper()
        # Check for _ALL_ pattern (with underscores on both sides) to match aggregate file naming
        # This matches patterns like: p3_ALL_HAN_filters..., P60_ALL_HAN_filters..., etc.
        if '_ALL_' in filename_upper:
            aggregate_files.append(f)
    
    return aggregate_files


def load_and_extract_groups(file_path):
    """
    Load CSV and extract motif->group mapping.
    
    Args:
        file_path: Path to upsetplot CSV file
        
    Returns:
        dict: Dictionary mapping normalized motif strings to group values
    """
    # Safety check: verify this is an aggregate _ALL_ file
    filename = Path(file_path).name.upper()
    if '_ALL_' not in filename:
        print(f"  Warning: File does not appear to be an aggregate _ALL_ file: {file_path}")
        print(f"  Skipping to ensure we only process aggregate data.")
        return {}
    
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'Motifs' not in df.columns or 'Group' not in df.columns:
            print(f"  Warning: Missing required columns in {file_path}")
            return {}
        
        motif_groups = {}
        
        for _, row in df.iterrows():
            motif_str = row['Motifs']
            group_value = row['Group']
            
            # Normalize motif first to check if it's valid
            normalized_motif = normalize_motif(motif_str)
            
            if normalized_motif is None:
                continue  # Skip empty/placeholder motifs
            
            # Check if Observed is 0.0 - if so, set group to 0
            if 'Observed' in df.columns and float(row['Observed']) == 0.0:
                motif_groups[normalized_motif] = 0
            else:
                # Store group value (convert to int if possible)
                try:
                    group_value = int(float(group_value))
                except (ValueError, TypeError):
                    # If conversion fails, skip this row
                    continue
                motif_groups[normalized_motif] = group_value
        
        return motif_groups
    
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return {}


def get_domain_sorted_motifs(motifs_list):
    """
    Sort motifs by domain (number of regions) matching script 01's domain ordering.
    
    Groups motifs by domain (number of regions in motif), sorts domains numerically,
    and preserves original order within each domain.
    
    Args:
        motifs_list: List of motif strings (e.g., ["['al', 'am']", "['al']", ...])
        
    Returns:
        list: Motifs sorted by domain, matching script 01's domain_sorted_motifs order
    """
    # Group motifs by domain (number of regions)
    motif_domains = {}
    for motif in motifs_list:
        try:
            if isinstance(motif, str):
                motifs_parsed = ast.literal_eval(motif)
            else:
                motifs_parsed = motif
            
            if isinstance(motifs_parsed, list):
                count = len(motifs_parsed)
            else:
                count = 1
            
            if count not in motif_domains:
                motif_domains[count] = []
            motif_domains[count].append(motif)
        except Exception:
            # If parsing fails, treat as single region
            if 1 not in motif_domains:
                motif_domains[1] = []
            motif_domains[1].append(motif)
    
    # Sort domains numerically
    sorted_domains = sorted(motif_domains.keys())
    
    # Build ordered list preserving original order within each domain
    domain_sorted_motifs = []
    for domain in sorted_domains:
        # Get motifs in original order for this domain by iterating through original list
        domain_motifs_original_order = []
        seen_in_domain = set()
        for motif in motifs_list:
            if motif in motif_domains[domain] and motif not in seen_in_domain:
                domain_motifs_original_order.append(motif)
                seen_in_domain.add(motif)
        domain_sorted_motifs.extend(domain_motifs_original_order)
    
    return domain_sorted_motifs


def plot_model_group_comparison(csv_path, output_dir):
    """
    Plot model group comparison data with motifs on X-axis and group values on Y-axis.
    
    Creates separate subplots for each age+model combination, with motifs ordered
    by domain (matching script 01's domain normalization plot order).
    
    Args:
        csv_path: Path to model_group_comparison.csv file
        output_dir: Directory to save the plot
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("Warning: CSV file is empty, cannot create plot")
        return
    
    # Get all motifs and sort by domain
    all_motifs = df['Motif'].tolist()
    sorted_motifs = get_domain_sorted_motifs(all_motifs)
    
    # Create a mapping from motif to index in sorted order
    motif_to_index = {motif: idx for idx, motif in enumerate(sorted_motifs)}
    
    # Get all data columns (excluding 'Motif')
    data_columns = [col for col in df.columns if col != 'Motif']
    
    if not data_columns:
        print("Warning: No data columns found, cannot create plot")
        return
    
    # Calculate subplot layout (2 columns for uniform/region_specific pairs)
    n_plots = len(data_columns)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    
    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Y-axis mapping: labels 0, 4, 3, 2, 1 from bottom to top (0 is origin, 1 is highest)
    # Group values map to y-positions: 0->0, 1->4, 2->3, 3->2, 4->1
    group_to_ypos = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}
    y_ticks = [0, 1, 2, 3, 4]  # Y-axis positions
    y_labels = ['0', '4', '3', '2', '1']  # Labels at each position
    
    # Plot each column
    for idx, col_name in enumerate(data_columns):
        ax = axes[idx]
        
        # Get data for this column
        x_positions = []
        y_positions = []
        colors = []
        
        for _, row in df.iterrows():
            motif = row['Motif']
            group_value = row[col_name]
            
            # Skip NaN values
            if pd.isna(group_value):
                continue
            
            # Get x position based on sorted motif order
            if motif in motif_to_index:
                x_pos = motif_to_index[motif]
                
                # Convert group value to y position using mapping
                try:
                    group_val = int(float(group_value))
                    if group_val in group_to_ypos:
                        y_pos = group_to_ypos[group_val]
                    else:
                        # If group value not in expected range, skip
                        continue
                except (ValueError, TypeError):
                    continue
                
                x_positions.append(x_pos)
                y_positions.append(y_pos)
                
                # Determine color: red for 1, blue for 2, black for others
                if group_val == 1:
                    colors.append('red')
                elif group_val == 2:
                    colors.append('blue')
                else:
                    colors.append('black')
        
        # Add small horizontal jitter to prevent overlap
        if x_positions:
            x_jitter = np.random.normal(0, 0.1, len(x_positions))
            x_positions_jittered = np.array(x_positions) + x_jitter
            
            # Plot points grouped by color for efficiency
            # Group by color
            red_mask = np.array(colors) == 'red'
            blue_mask = np.array(colors) == 'blue'
            black_mask = np.array(colors) == 'black'
            
            if red_mask.any():
                ax.scatter(x_positions_jittered[red_mask], np.array(y_positions)[red_mask], 
                          c='red', s=30, alpha=0.7, edgecolors='none', marker='o')
            if blue_mask.any():
                ax.scatter(x_positions_jittered[blue_mask], np.array(y_positions)[blue_mask], 
                          c='blue', s=30, alpha=0.7, edgecolors='none', marker='o')
            if black_mask.any():
                ax.scatter(x_positions_jittered[black_mask], np.array(y_positions)[black_mask], 
                          c='black', s=30, alpha=0.7, edgecolors='none', marker='o')
        
        # Set axis properties
        ax.set_xlim(-0.5, len(sorted_motifs) - 0.5)
        ax.set_ylim(-0.3, 4.3)  # Add padding above and below
        ax.set_xticks(range(len(sorted_motifs)))
        ax.set_xticklabels(sorted_motifs, rotation=90, ha='center', fontsize=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Group Value', fontsize=10)
        ax.set_title(col_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Create legend for the entire figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, label='1 = over-represented (significant)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='2 = under-represented (significant)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=8, label='3/4 = not significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markersize=8, label='0 = not detected')
    ]
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space for legend
    
    # Save plot
    output_path = Path(output_dir) / "model_group_comparison.svg"
    fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Also save as PNG
    output_path_png = Path(output_dir) / "model_group_comparison.png"
    fig.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path_png}")
    
    plt.close(fig)


def find_effect_significance_files(base_dir, parameterization_filter=None):
    """
    Find effect_significance PNG files for each age+model combination.
    
    Args:
        base_dir: Base output directory (e.g., 02_output)
        parameterization_filter: Optional parameterization name to filter by
        
    Returns:
        dict: Dictionary mapping "{age}_{model}" to file path (or None if not found)
    """
    base_path = Path(base_dir)
    age_groups = ['p3', 'p12', 'p20', 'p60']
    models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                         'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                         'bayesian_hierarchical', 'ml_nonparametric']
    
    file_map = {}
    
    for age in age_groups:
        for model in models_to_process:
            key = f"{age}_{model}"
            
            if parameterization_filter:
                # Search in specific parameterization directory
                param_path = base_path / age / parameterization_filter / "analysis" / model
            else:
                # Search across all parameterizations (find first match)
                param_path = None
                for param_dir in base_path.glob(f"{age}/*"):
                    if param_dir.is_dir() and any(part.startswith(('01.', '02.', '03.', '04.', '05.')) for part in param_dir.parts):
                        test_path = param_dir / "analysis" / model
                        if test_path.exists():
                            param_path = test_path
                            break
                
                if param_path is None:
                    file_map[key] = None
                    continue
            
            if param_path is None or not param_path.exists():
                file_map[key] = None
                continue
            
            # Try multiple filename patterns (case variations)
            patterns = [
                f"{age}_ALL_HAN_filters_effect_significance_{model}.png",
                f"{age.upper()}_ALL_HAN_filters_effect_significance_{model}.png",
                f"*ALL_*_filters_effect_significance_{model}.png",
            ]
            
            found_file = None
            for pattern in patterns:
                matches = list(param_path.glob(pattern))
                if matches:
                    found_file = matches[0]
                    break
            
            file_map[key] = str(found_file) if found_file else None
    
    return file_map


def plot_effect_significance_grid(base_dir, output_dir, parameterization_filter=None):
    """
    Load effect_significance PNG files and arrange them in a subplot grid.
    
    Creates a figure with the same subplot layout as model_group_comparison plot,
    displaying the effect_significance images for each age+model combination.
    
    Args:
        base_dir: Base output directory
        output_dir: Directory to save the plot
        parameterization_filter: Optional parameterization name to filter by
    """
    # Find all effect_significance files
    file_map = find_effect_significance_files(base_dir, parameterization_filter)
    
    # Define order matching data_columns order from model_group_comparison
    age_groups = ['p3', 'p12', 'p20', 'p60']
    models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                         'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                         'bayesian_hierarchical', 'ml_nonparametric']
    
    # Create subplot layout: 2 columns, 4 rows (same as model_group_comparison)
    n_cols = 2
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Plot each age+model combination
    plot_idx = 0
    for age in age_groups:
        for model in models_to_process:
            key = f"{age}_{model}"
            ax = axes[plot_idx]
            
            file_path = file_map.get(key)
            
            if file_path and Path(file_path).exists():
                try:
                    # Load image
                    img = mpimg.imread(file_path)
                    ax.imshow(img, aspect='auto')
                    ax.set_title(key, fontsize=12, fontweight='bold')
                    ax.axis('off')  # Remove axes
                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")
                    ax.text(0.5, 0.5, f'Image not found\n{key}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                # Show placeholder for missing file
                ax.text(0.5, 0.5, f'File not found\n{key}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            plot_idx += 1
    
    plt.tight_layout()
    
    # Save as high-resolution TIFF
    output_path = Path(output_dir) / "effect_significance_grid.tiff"
    
    # Try to save as TIFF (requires PIL/Pillow for best quality)
    if HAS_PIL:
        # Save using PIL for better TIFF support
        fig.savefig(output_path, format='tiff', dpi=300, bbox_inches='tight', 
                   pil_kwargs={'compression': 'tiff_lzw'})
    else:
        # Fallback: try matplotlib's TIFF support (may not work on all systems)
        try:
            fig.savefig(output_path, format='tiff', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"  Warning: Could not save as TIFF: {e}")
            print(f"  Saving as PNG instead...")
            output_path = Path(output_dir) / "effect_significance_grid.png"
            fig.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Effect significance grid saved to: {output_path}")
    
    plt.close(fig)


def aggregate_group_data(base_dir, parameterization_filter=None):
    """
    Main aggregation function to collect group data across all ages and models.
    
    Args:
        base_dir: Base output directory
        parameterization_filter: Optional parameterization name to filter by
        
    Returns:
        pd.DataFrame: Wide-format DataFrame with motifs as rows and age+model combinations as columns
    """
    age_groups = ['p3', 'p12', 'p20', 'p60']
    models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                         'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                         'bayesian_hierarchical', 'ml_nonparametric']
    
    # Dictionary to store data: {age: {model: {motif: group}}}
    all_data = defaultdict(lambda: defaultdict(dict))
    
    # Track which combinations were found
    found_combinations = []
    missing_combinations = []
    
    print("=" * 80)
    print("Searching for upsetplot files and extracting group data...")
    print("=" * 80)
    
    for age in age_groups:
        for model in models_to_process:
            print(f"\nProcessing {age.upper()} - {model} model...")
            
            # Find files
            files = find_upsetplot_files(base_dir, age, model, parameterization_filter)
            
            if not files:
                print(f"  No files found for {age} - {model}")
                missing_combinations.append(f"{age}_{model}")
                continue
            
            # Use the first matching file (should only be one aggregate file per age/model)
            file_path = files[0]
            if len(files) > 1:
                print(f"  Warning: Multiple files found, using: {file_path}")
            
            print(f"  Loading: {file_path}")
            
            # Extract group data
            motif_groups = load_and_extract_groups(file_path)
            
            if not motif_groups:
                print(f"  Warning: No valid group data extracted from {file_path}")
                missing_combinations.append(f"{age}_{model}")
                continue
            
            # Store data
            all_data[age][model] = motif_groups
            found_combinations.append(f"{age}_{model}")
            print(f"  Extracted {len(motif_groups)} motifs with group values")
    
    # Collect all unique motifs across all ages and models
    all_motifs = set()
    for age_data in all_data.values():
        for model_data in age_data.values():
            all_motifs.update(model_data.keys())
    
    all_motifs = sorted(all_motifs)
    
    if not all_motifs:
        print("\nError: No motifs found in any files!")
        return pd.DataFrame()
    
    print(f"\n{'=' * 80}")
    print(f"Found {len(all_motifs)} unique motifs across all ages and models")
    print(f"Found {len(found_combinations)} age+model combinations")
    if missing_combinations:
        print(f"Missing {len(missing_combinations)} combinations: {missing_combinations}")
    print(f"{'=' * 80}\n")
    
    # Build wide-format DataFrame
    # Columns: Motif, p3_uniform, p3_region_specific, p12_uniform, ...
    column_order = ['Motif']
    for age in age_groups:
        for model in models_to_process:
            column_order.append(f"{age}_{model}")
    
    # Build data rows
    rows = []
    for motif in all_motifs:
        row = {'Motif': motif}
        for age in age_groups:
            for model in models_to_process:
                col_name = f"{age}_{model}"
                # Get group value if available, otherwise NaN
                group_value = all_data[age][model].get(motif, np.nan)
                row[col_name] = group_value
        rows.append(row)
    
    # Create DataFrame
    df_result = pd.DataFrame(rows, columns=column_order)
    
    return df_result


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Aggregate group information from upsetplot files across models and ages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python 14_model_group_comparison.py
  
  # Custom base directory
  python 14_model_group_comparison.py --base_output_dir /path/to/02_output
  
  # Custom output directory (will extract parameterization from path)
  python 14_model_group_comparison.py --helper_output_dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--base_output_dir',
        type=str,
        default=None,
        help='Base output directory for processing results (default: REPO_ROOT/02_output)'
    )
    parser.add_argument(
        '--helper_output_dir',
        type=str,
        default=None,
        help='Directory for helper script outputs (default: helpers/outputs/14_model_group_comparison)'
    )
    
    args = parser.parse_args()
    
    # Get repository root
    script_dir = Path(__file__).parent
    REPO_ROOT = script_dir.parent.parent
    
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
    # Follow same pattern as script 13
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
    
    # If parameterization not found from helper_output_dir, try to detect from OUTPUT_BASE
    if not parameterization_filter:
        # Look for *_helpers directories in OUTPUT_BASE
        helpers_dirs = list(OUTPUT_BASE.glob("*_helpers"))
        if helpers_dirs:
            # Extract parameterization from first matching directory
            first_helpers_dir = helpers_dirs[0]
            param_name = first_helpers_dir.name.replace("_helpers", "")
            if param_name.startswith(('01.', '02.', '03.', '04.', '05.')):
                parameterization_filter = param_name
                print(f"Detected parameterization from OUTPUT_BASE: {parameterization_filter}")
    
    # Determine output directory - follow same pattern as other helper scripts:
    # 02_output/{parameterization}_helpers/14_model_group_comparison/
    SCRIPT_DIR = Path(__file__).parent
    if args.helper_output_dir:
        output_dir = Path(args.helper_output_dir)
    else:
        if parameterization_filter:
            output_dir = OUTPUT_BASE / f"{parameterization_filter}_helpers" / "14_model_group_comparison"
        else:
            # Default fallback
            output_dir = SCRIPT_DIR.parent / "outputs" / "14_model_group_comparison"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("MODEL GROUP COMPARISON SCRIPT")
    print("=" * 80)
    print(f"Base output directory: {OUTPUT_BASE}")
    print(f"Output directory: {output_dir}")
    if parameterization_filter:
        print(f"Parameterization filter: {parameterization_filter}")
    print("=" * 80)
    
    # Aggregate group data
    result_df = aggregate_group_data(OUTPUT_BASE, parameterization_filter)
    
    if result_df.empty:
        print("\nError: No data aggregated. Please check that upsetplot files exist.")
        return 1
    
    # Save output
    output_file = output_dir / "model_group_comparison.csv"
    result_df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"   Total motifs: {len(result_df)}")
    print(f"   Columns: {len(result_df.columns)}")
    print(f"   Age+Model combinations: {len(result_df.columns) - 1}")  # Exclude 'Motif' column
    print(f"{'=' * 80}\n")
    
    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    for col in result_df.columns:
        if col != 'Motif':
            non_null_count = result_df[col].notna().sum()
            print(f"  {col}: {non_null_count} motifs with group values")
    
    # Generate plot
    print(f"\n{'=' * 80}")
    print("Generating plot...")
    print(f"{'=' * 80}")
    try:
        plot_model_group_comparison(output_file, output_dir)
    except Exception as e:
        print(f"Warning: Error generating plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate effect significance grid
    print(f"\n{'=' * 80}")
    print("Generating effect significance grid...")
    print(f"{'=' * 80}")
    try:
        plot_effect_significance_grid(OUTPUT_BASE, output_dir, parameterization_filter)
    except Exception as e:
        print(f"Warning: Error generating effect significance grid: {e}")
        import traceback
        traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
