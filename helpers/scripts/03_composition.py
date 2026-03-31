import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent opening windows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import os
import glob
from pathlib import Path
from scipy import stats

# Set font preferences for editable SVG text
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']

# Try to find projection_summary.csv files in 02_output subdirectories
import argparse
parser = argparse.ArgumentParser(description="Generate composition plots")
parser.add_argument('--base_output_dir', type=str, default=None,
                   help='Base output directory for processing results (default: REPO_ROOT/02_output)')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/03_composition)')
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent.parent
if args.base_output_dir:
    OUTPUT_DIR = Path(args.base_output_dir)
else:
    OUTPUT_DIR = REPO_ROOT / "02_output"

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

# Look for projection_summary.csv files in subdirectories
# Filter by parameterization if specified
if parameterization_filter:
    summary_files = []
    for age_dir in ['p3', 'p12', 'p20', 'p60']:
        param_path = OUTPUT_DIR / age_dir / parameterization_filter
        if param_path.exists():
            summary_files.extend(glob.glob(str(param_path / "**" / "projection_summary.csv"), recursive=True))
else:
    summary_files = glob.glob(str(OUTPUT_DIR / "**" / "projection_summary.csv"), recursive=True)

if summary_files:
    # Combine all projection_summary.csv files
    dfs = []
    for f in summary_files:
        df_temp = pd.read_csv(f)
        # Extract age from file path (e.g., .../p20/... -> p20)
        path_parts = Path(f).parts
        age_from_path = None
        for part in path_parts:
            if part.lower().startswith('p') and len(part) <= 4 and part[1:].isdigit():
                age_from_path = part.lower()
                break
        
        # If no age in path, try to extract from Sample column
        if age_from_path is None and 'Sample' in df_temp.columns:
            # Try to extract from sample name (e.g., "p20_alL_HAN_filters" -> "p20")
            sample_name = str(df_temp['Sample'].iloc[0]) if len(df_temp) > 0 else ""
            if sample_name.lower().startswith('p') and len(sample_name) >= 2:
                age_from_path = sample_name[:3].lower() if sample_name[1:3].isdigit() else sample_name[:4].lower()
        
        if age_from_path:
            df_temp['Age'] = age_from_path
        dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    
    # If Age column doesn't exist, create it from Sample column
    if 'Age' not in df.columns and 'Sample' in df.columns:
        df['Age'] = df['Sample'].str.extract(r'(p\d+)', expand=False).str.lower()
    
    print(f"Loaded {len(df)} rows from {len(summary_files)} projection_summary.csv files")
else:
    raise FileNotFoundError(
        f"Could not find projection_summary.csv files in {OUTPUT_DIR}. "
        f"Searched in: {OUTPUT_DIR}/**/projection_summary.csv"
    )

# Define the brain areas of interest (case-insensitive matching)
areas = ['RSP', 'PM', 'AM', 'AL', 'LM']
# Try to find columns with case-insensitive matching
df_columns_lower = {col.lower(): col for col in df.columns}

umisum_cols = []
projcount_cols = []
for area in areas:
    # Try exact match first, then case-insensitive
    umi_exact = f'UMISum_{area}'
    umi_lower = f'umisum_{area.lower()}'
    proj_exact = f'ProjCount_{area}'
    proj_lower = f'projcount_{area.lower()}'
    
    if umi_exact in df.columns:
        umisum_cols.append(umi_exact)
    elif umi_lower in df_columns_lower:
        umisum_cols.append(df_columns_lower[umi_lower])
    
    if proj_exact in df.columns:
        projcount_cols.append(proj_exact)
    elif proj_lower in df_columns_lower:
        projcount_cols.append(df_columns_lower[proj_lower])

# Check if we found the required columns
if not umisum_cols:
    print(f"Warning: Could not find UMISum columns. Available columns: {list(df.columns)[:10]}...")
    # Try to find any UMISum columns
    umisum_cols = [col for col in df.columns if 'umisum' in col.lower()]
    print(f"Found UMISum columns: {umisum_cols}")

if not projcount_cols:
    print(f"Warning: Could not find ProjCount columns. Available columns: {list(df.columns)[:10]}...")
    # Try to find any ProjCount columns
    projcount_cols = [col for col in df.columns if 'projcount' in col.lower()]
    print(f"Found ProjCount columns: {projcount_cols}")

# Extract and sort UMI data
if not umisum_cols:
    raise ValueError("No UMISum columns found. Cannot create UMI composition plot.")

df_comp = df[['Age'] + umisum_cols].copy()
# Dynamically determine available age categories from the data
available_ages = sorted(df_comp['Age'].dropna().unique())
if not available_ages:
    raise ValueError("No age data found. Cannot create UMI composition plot.")

# Normalize UMI counts per row to percentage first (before aggregation)
df_comp_norm = df_comp.copy()
df_comp_norm[umisum_cols] = df_comp_norm[umisum_cols].div(df_comp_norm[umisum_cols].sum(axis=1), axis=0) * 100

# Calculate statistics for error bars and individual points
from scipy import stats
age_stats = {}
for age in available_ages:
    age_data = df_comp_norm[df_comp_norm['Age'] == age][umisum_cols]
    age_stats[age] = {
        'mean': age_data.mean(),
        'sem': age_data.sem(),
        'individual': age_data.values  # All individual samples for this age
    }

# Aggregate by age (take mean of normalized percentages)
df_percent = df_comp_norm.groupby('Age')[umisum_cols].mean()

# Set categorical ordering for the index
df_percent.index = pd.Categorical(df_percent.index, categories=available_ages, ordered=True)
df_percent = df_percent.sort_index()

# Plot UMI composition
fig, ax = plt.subplots(figsize=(8, 3))
colors = plt.cm.tab10.colors[:len(umisum_cols)]

bottom = [0] * len(df_percent)
# Map column names back to area labels for legend
area_labels = []
for col in umisum_cols:
    # Extract area name from column (e.g., "UMISum_rsp" -> "RSP")
    area_name = col.split('_')[-1].upper()
    area_labels.append(area_name)

# Get y positions for each age (numeric for plotting)
y_positions = {age: i for i, age in enumerate(df_percent.index)}

for i, col in enumerate(umisum_cols):
    values = df_percent[col]
    ax.barh(df_percent.index, values, left=bottom, color=colors[i], label=area_labels[i], alpha=0.8, zorder=1)
    
    # Add SEM error bars for the segment width (centered on segment)
    for j, age in enumerate(df_percent.index):
        y_pos = y_positions[age]
        segment_center = bottom[j] + values.iloc[j] / 2  # Center of this segment
        sem_val = age_stats[age]['sem'][col]
        # Horizontal error bar showing uncertainty in this segment's width
        ax.errorbar(segment_center, y_pos, xerr=sem_val, fmt='none', 
                   color='black', capsize=2, capthick=1, linewidth=1, zorder=3, alpha=0.7)
    
    # Add individual points within each segment
    # Each point represents one sample's value for this region, positioned at segment center
    for j, age in enumerate(df_percent.index):
        y_pos = y_positions[age]
        individual_samples = age_stats[age]['individual']
        segment_center = bottom[j] + values.iloc[j] / 2
        
        # Get each sample's value for this specific region
        region_values = [sample_row[i] for sample_row in individual_samples]
        
        if len(region_values) > 0:
            # Position points at segment center with slight horizontal jitter
            # The jitter helps visualize that these are individual samples
            x_jitter = np.random.normal(0, values.iloc[j] * 0.1, len(region_values))  # 10% of segment width
            y_jitter = np.random.normal(0, 0.08, len(region_values))
            
            ax.scatter([segment_center + xj for xj in x_jitter], 
                      [y_pos + yj for yj in y_jitter],
                      color='black', s=4, alpha=0.6, zorder=4, edgecolors='white', linewidths=0.5)
    
    bottom = [sum(x) for x in zip(bottom, values)]

ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_yticks(range(len(df_percent)))
ax.set_yticklabels(df_percent.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

legend_elements = [Patch(facecolor=colors[i], label=area_labels[i]) for i in range(len(umisum_cols))]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
if args.helper_output_dir:
    HELPER_OUTPUT_DIR = Path(args.helper_output_dir)
else:
    HELPER_OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "03_composition"
HELPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Save in multiple formats for compatibility
for ext in ['png', 'pdf', 'svg']:
    output_file = HELPER_OUTPUT_DIR / f"UMI_composition_by_age.{ext}"
    plt.savefig(output_file, format=ext, dpi=300)
plt.close()

# Extract and sort ProjCount data
if not projcount_cols:
    raise ValueError("No ProjCount columns found. Cannot create ProjCount composition plot.")

df_proj = df[['Age'] + projcount_cols].copy()
# Dynamically determine available age categories from the data
available_ages_proj = sorted(df_proj['Age'].dropna().unique())
if not available_ages_proj:
    raise ValueError("No age data found. Cannot create ProjCount composition plot.")

# Normalize projection counts per row to percentage first (before aggregation)
df_proj_norm = df_proj.copy()
df_proj_norm[projcount_cols] = df_proj_norm[projcount_cols].div(df_proj_norm[projcount_cols].sum(axis=1), axis=0) * 100

# Calculate statistics for error bars and individual points
age_stats_proj = {}
for age in available_ages_proj:
    age_data = df_proj_norm[df_proj_norm['Age'] == age][projcount_cols]
    age_stats_proj[age] = {
        'mean': age_data.mean(),
        'sem': age_data.sem(),
        'individual': age_data.values  # All individual samples for this age
    }

# Aggregate by age (take mean of normalized percentages)
df_proj_percent = df_proj_norm.groupby('Age')[projcount_cols].mean()

# Set categorical ordering for the index
df_proj_percent.index = pd.Categorical(df_proj_percent.index, categories=available_ages_proj, ordered=True)
df_proj_percent = df_proj_percent.sort_index()

# Plot ProjCount composition
fig, ax = plt.subplots(figsize=(8, 3))

bottom = [0] * len(df_proj_percent)
# Map column names back to area labels for legend
proj_area_labels = []
for col in projcount_cols:
    # Extract area name from column (e.g., "ProjCount_rsp" -> "RSP")
    area_name = col.split('_')[-1].upper()
    proj_area_labels.append(area_name)

# Get y positions for each age (numeric for plotting)
y_positions_proj = {age: i for i, age in enumerate(df_proj_percent.index)}

colors_proj = plt.cm.tab10.colors[:len(projcount_cols)]
for i, col in enumerate(projcount_cols):
    values = df_proj_percent[col]
    ax.barh(df_proj_percent.index, values, left=bottom, color=colors_proj[i], label=proj_area_labels[i], alpha=0.8, zorder=1)
    
    # Add SEM error bars for the segment width (centered on segment)
    for j, age in enumerate(df_proj_percent.index):
        y_pos = y_positions_proj[age]
        segment_center = bottom[j] + values.iloc[j] / 2  # Center of this segment
        sem_val = age_stats_proj[age]['sem'][col]
        # Horizontal error bar showing uncertainty in this segment's width
        ax.errorbar(segment_center, y_pos, xerr=sem_val, fmt='none', 
                   color='black', capsize=2, capthick=1, linewidth=1, zorder=3, alpha=0.7)
    
    # Add individual points within each segment
    # Each point represents one sample's value for this region, positioned at segment center
    for j, age in enumerate(df_proj_percent.index):
        y_pos = y_positions_proj[age]
        individual_samples = age_stats_proj[age]['individual']
        segment_center = bottom[j] + values.iloc[j] / 2
        
        # Get each sample's value for this specific region
        region_values = [sample_row[i] for sample_row in individual_samples]
        
        if len(region_values) > 0:
            # Position points at segment center with slight horizontal jitter
            # The jitter helps visualize that these are individual samples
            x_jitter = np.random.normal(0, values.iloc[j] * 0.1, len(region_values))  # 10% of segment width
            y_jitter = np.random.normal(0, 0.08, len(region_values))
            
            ax.scatter([segment_center + xj for xj in x_jitter], 
                      [y_pos + yj for yj in y_jitter],
                      color='black', s=4, alpha=0.6, zorder=4, edgecolors='white', linewidths=0.5)
    
    bottom = [sum(x) for x in zip(bottom, values)]

ax.set_xlim(0, 100)
ax.set_xticks([])
ax.set_yticks(range(len(df_proj_percent)))
ax.set_yticklabels(df_proj_percent.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(False)

legend_elements = [Patch(facecolor=colors_proj[i], label=proj_area_labels[i]) for i in range(len(projcount_cols))]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
# Save in multiple formats for compatibility
for ext in ['png', 'pdf', 'svg']:
    output_file = HELPER_OUTPUT_DIR / f"ProjCount_composition_by_age.{ext}"
    plt.savefig(output_file, format=ext, dpi=300)
plt.close()