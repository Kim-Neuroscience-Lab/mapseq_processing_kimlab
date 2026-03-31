# Per-cell projection strength across ages: one plot per motif with all cell lines
# colored by age (p12/p20/p60) and a black mean line with round points at the five regions.
# Reads existing *ALL*_raw_data.csv from each age's motif_raw_data directory.

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arial']

import matplotlib.pyplot as plt
from pathlib import Path

REGION_ORDER = ['LM', 'AL', 'AM', 'PM', 'RSP']
AGE_GROUPS = ['p12', 'p20', 'p60']
AGE_COLORS = {'p12': '#bf2000', 'p20': '#45df00', 'p60': '#17becf'}

# Figure size in px at border; use 72 dpi so SVG user units = same as px for 1:1 export
FIG_WIDTH_PX = 85.5
FIG_HEIGHT_PX = 110
FIG_SIZE = (FIG_WIDTH_PX / 72.0, FIG_HEIGHT_PX / 72.0)

CELL_LINEWIDTH = 0.25
MEAN_LINEWIDTH = 2.0 * CELL_LINEWIDTH  # 0.5 pt, twice as thick as cell lines
MEAN_MARKER_SIZE_PT = 1.0


def _parameterization_from_output_dir(output_dir):
    """Extract parameterization name from output_dir (e.g. 05.HAN_filter_parameters_...)."""
    if not output_dir:
        return None
    for part in Path(output_dir).parts:
        if part.startswith(('01.', '02.', '03.', '04.', '05.')) and '_helpers' in part:
            return part.split('_helpers')[0]
        if part.startswith(('01.', '02.', '03.', '04.', '05.')):
            return part
    return None


def _collect_aggregate_files(all_ages_base_dir, parameterization_filter):
    """Return {age: [file paths]} for *ALL*_raw_data.csv under each age/parameterization."""
    all_age_aggregate_files = {}
    for age in AGE_GROUPS:
        param_path = Path(all_ages_base_dir) / age / parameterization_filter
        if not param_path.exists():
            continue
        age_pattern = str(param_path / "**" / "*ALL*_raw_data.csv")
        age_files = glob.glob(age_pattern, recursive=True)
        if not age_files:
            age_pattern = str(param_path / "**" / f"*{age.upper()}*ALL*_raw_data.csv")
            age_files = glob.glob(age_pattern, recursive=True)
        if age_files:
            all_age_aggregate_files[age] = age_files
    return all_age_aggregate_files


def _motif_from_filename(filepath):
    """Extract motif from aggregate raw_data filename (e.g. p12_ALL_HAN_filters_pm_am_lm_raw_data -> pm_am_lm)."""
    filename = Path(filepath).stem
    parts = filename.split('_')
    try:
        filters_idx = next(i for i, p in enumerate(parts) if p.upper() == 'FILTERS')
        raw_idx = next(i for i, p in enumerate(parts) if p.lower() == 'raw')
        return '_'.join(parts[filters_idx + 1:raw_idx])
    except StopIteration:
        return '_'.join(parts[1:-2]) if len(parts) >= 3 and parts[-2].lower() == 'raw' else '_'.join(parts[1:-1])


def _build_motif_to_ages(all_age_aggregate_files):
    """Build {motif: {age: path}} (one file per age per motif)."""
    motif_to_ages = {}
    for age, files in all_age_aggregate_files.items():
        for file in files:
            motif = _motif_from_filename(file)
            if motif not in motif_to_ages:
                motif_to_ages[motif] = {}
            # One file per motif per age
            motif_to_ages[motif][age] = file
    return motif_to_ages


def _read_region_ordered_values(csv_path):
    """
    Read CSV and return (N, 5) array of values in REGION_ORDER, and success.
    First column is row label; remaining columns matched case-insensitively to regions.
    """
    df = pd.read_csv(csv_path)
    df_cols_lower = {col.lower(): col for col in df.columns[1:]}
    region_col_indices = [0]
    for region in REGION_ORDER:
        rl = region.lower()
        if rl in df_cols_lower:
            region_col_indices.append(df.columns.get_loc(df_cols_lower[rl]))
    if len(region_col_indices) <= 1:
        return None, False
    df = df.iloc[:, region_col_indices]
    rows = []
    for _, row in df.iterrows():
        vals = row.values[1:].astype(float)
        if len(vals) == 5:
            rows.append(vals)
    if not rows:
        return None, False
    return np.array(rows), True


def main(data_dir, output_dir=None, all_ages_base_dir=None):
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "outputs" / "10_plot_per_cell_projection_strength_across_ages")
    if all_ages_base_dir is None:
        all_ages_base_dir = data_dir

    parameterization_filter = _parameterization_from_output_dir(output_dir)
    if not parameterization_filter:
        print("Error: Could not infer parameterization from output_dir. Use output_dir containing e.g. 05.HAN_filter_parameters_..._helpers/...")
        return

    all_age_aggregate_files = _collect_aggregate_files(all_ages_base_dir, parameterization_filter)
    if not all_age_aggregate_files:
        print("Error: No aggregate *ALL*_raw_data.csv files found under ages.")
        return

    motif_to_ages = _build_motif_to_ages(all_age_aggregate_files)
    # Only multi-region motifs (match pipeline hide_singlets)
    multi_region = [m for m in motif_to_ages if len(m.split('_')) >= 2]
    # Sort by domain (number of regions) ascending, then alphabetically by motif
    sorted_motifs = sorted(multi_region, key=lambda m: (len(m.split('_')), m))

    os.makedirs(output_dir, exist_ok=True)
    x_indices = np.arange(5)

    for motif in sorted_motifs:
        age_files = motif_to_ages[motif]
        all_cells = []   # list of (age, row_array)
        for age in AGE_GROUPS:
            if age not in age_files:
                print(f"Warning: No data for motif {motif} at age {age}")
                continue
            arr, ok = _read_region_ordered_values(age_files[age])
            if not ok:
                print(f"Warning: Could not read or empty: {age_files[age]}")
                continue
            for i in range(arr.shape[0]):
                all_cells.append((age, arr[i]))

        if not all_cells:
            print(f"Warning: No cells for motif {motif}, skipping.")
            continue

        # Overall mean
        stack = np.array([c[1] for c in all_cells])
        mean_vals = stack.mean(axis=0)

        # Draw order: p60 first (bottom), then p20, then p12 (top) so smaller age groups stay visible
        draw_order = ['p60', 'p20', 'p12']
        ordered_cells = [c for a in draw_order for c in all_cells if c[0] == a]

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        # Axis box = edge of plot: no margins, no background layers
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        ax.set_position([0, 0, 1, 1])
        # Per-cell lines: 0.25 pt, no markers; slight alpha so overlapping lines remain visible
        for age, row in ordered_cells:
            ax.plot(x_indices, row, color=AGE_COLORS[age], linewidth=CELL_LINEWIDTH, alpha=0.85, solid_capstyle='round')

        # Mean: 0.5 pt line + round points at 5 positions
        ax.plot(x_indices, mean_vals, color='black', linewidth=MEAN_LINEWIDTH, marker='o',
                markersize=MEAN_MARKER_SIZE_PT, markeredgewidth=0, markerfacecolor='black', solid_capstyle='round')

        ax.set_xlim(-0.25, 4.25)
        ymin, ymax = stack.min(), stack.max()
        margin = (ymax - ymin) * 0.05 if ymax > ymin else 0.1
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)

        safe_motif = re.sub(r'\W+', '_', motif)[:80]
        out_path = os.path.join(output_dir, f"{safe_motif}_per_cell_across_ages.svg")
        plt.savefig(out_path, format='svg', bbox_inches=None, pad_inches=0)
        plt.close()
        # Force SVG dimensions to 85.5px x 110px for AI (px) import
        with open(out_path, 'r') as f:
            content = f.read()
        content = re.sub(r'\bwidth="[^"]*"', f'width="{FIG_WIDTH_PX}px"', content, count=1)
        content = re.sub(r'\bheight="[^"]*"', f'height="{FIG_HEIGHT_PX}px"', content, count=1)
        with open(out_path, 'w') as f:
            f.write(content)

    print(f"Saved {len(sorted_motifs)} motif plots to {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-cell projection strength across ages (one plot per motif).")
    p.add_argument('data_dir', help="Base directory (e.g. 02_output).")
    p.add_argument('--output_dir', default=None, help="Output directory (default: helpers/outputs/10_plot_per_cell_projection_strength_across_ages)")
    p.add_argument('--all_ages_base_dir', default=None, help="Base dir containing p12/, p20/, p60/ (default: data_dir)")
    args = p.parse_args()
    main(args.data_dir, args.output_dir, args.all_ages_base_dir)
