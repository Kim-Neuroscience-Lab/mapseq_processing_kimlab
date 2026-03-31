import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent opening windows
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import re
import os
from collections import defaultdict

# Set font preferences
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'

# Input file
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="All motif divergence analysis")
parser.add_argument('--input_dir', type=str, default=None,
                   help='Directory containing output from 05_motif_analysis.py (default: helpers/outputs/05_motif_analysis)')
parser.add_argument('--helper_output_dir', type=str, default=None,
                   help='Directory for helper script outputs (default: helpers/outputs/06_all_motif_divergence)')
args = parser.parse_args()

script_dir = Path(__file__).parent
# Look for input file in the 05_motif_analysis output directory
if args.input_dir:
    output_01_dir = Path(args.input_dir)
else:
    output_01_dir = script_dir.parent / "outputs" / "05_motif_analysis"
real_comparison_file = script_dir / "motif_real_p12vsP60.tsv"

# Output directory
if args.helper_output_dir:
    base_output_dir = Path(args.helper_output_dir)
else:
    base_output_dir = script_dir.parent / "outputs" / "06_all_motif_divergence"
os.makedirs(base_output_dir, exist_ok=True)

# Process all three models separately
models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                     'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                     'bayesian_hierarchical', 'ml_nonparametric']

for model_type in models_to_process:
    print("\n" + "="*80)
    print(f"Processing {model_type.upper()} MODEL")
    print("="*80)
    
    # Look for model-specific input file
    input_path = output_01_dir / f"motif_transition_significance_summary_{model_type}.txt"
    
    # If not found in output directory, try script directory
    if not input_path.exists():
        alt_path = script_dir / f"motif_transition_significance_summary_{model_type}.txt"
        if alt_path.exists():
            input_path = alt_path
        else:
            print(f"Warning: Could not find motif_transition_significance_summary_{model_type}.txt")
            print(f"  Expected location: {output_01_dir / f'motif_transition_significance_summary_{model_type}.txt'}")
            print(f"  Skipping {model_type} model")
            continue
    
    # Create model-specific output directory
    model_output_dir = base_output_dir / model_type
    os.makedirs(model_output_dir, exist_ok=True)
    
    input_path_str = str(input_path)
    real_comparison_file_str = str(real_comparison_file) if real_comparison_file.exists() else None

    # Parse the text file into a usable structure
    with open(input_path_str, 'r') as f:
        lines = f.readlines()

    # Check if file is essentially empty (only headers/whitespace)
    non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    if len(non_empty_lines) <= 2:  # Only header lines, no actual data
        print(f"Warning: Input file {input_path_str} appears to be empty or contains only headers.")
        print("This is expected when 05_motif_analysis.py is run with a single age group (no cross-age transitions possible).")
        print(f"Skipping {model_type} model plot generation.")
        continue

    # Collect data by transition
    true_data = defaultdict(dict)
    false_data = defaultdict(dict)
    current_transition = None
    data_found = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"P\d+ vs P\d+", line):
            current_transition = line.replace(" ", "")
            continue
        if 'JS Divergence' in line:
            motif_match = re.match(r"\[(.*?)\]: .*?JS Divergence = ([\d\.]+|nan), Significant = (True|False)", line)
            if motif_match:
                motif_str, divergence, significant = motif_match.groups()
                if divergence != 'nan':
                    divergence = float(divergence)
                    motif_clean = motif_str.replace("'", "").replace(",", "+")
                    data_found = True
                    if significant == 'True':
                        true_data[current_transition][motif_clean] = divergence
                    else:
                        false_data[current_transition][motif_clean] = divergence

    # Check if we found any actual divergence data
    if not data_found:
        print(f"Warning: No divergence data found in {input_path_str}.")
        print("This is expected when 05_motif_analysis.py is run with a single age group (no cross-age transitions possible).")
        print(f"Skipping {model_type} model plot generation.")
        continue

    # Override P12vsP60 with real divergence values
    if real_comparison_file_str and os.path.exists(real_comparison_file_str):
        real_df = pd.read_csv(real_comparison_file_str, sep='\t')
        real_df = real_df.dropna(subset=['JS_Divergence'])

        true_data['P12vsP60'] = {}
        false_data['P12vsP60'] = {}

        for _, row in real_df.iterrows():
            motif = row['Motif_Label']
            divergence = float(row['JS_Divergence'])
            significant = bool(row['Significant']) if 'Significant' in row else False
            if significant:
                true_data['P12vsP60'][motif] = divergence
            else:
                false_data['P12vsP60'][motif] = divergence

    # Plotting function
    def plot_true_and_false(transition, true_motifs_dict, false_motifs_dict, max_global_divergence, output_dir, model_type):
        true_sorted = sorted(true_motifs_dict.items(), key=lambda x: x[1], reverse=True) if true_motifs_dict else []
        false_sorted = sorted(false_motifs_dict.items(), key=lambda x: x[1], reverse=True) if false_motifs_dict else []
        combined = true_sorted + false_sorted

        if not combined:
            print(f"Warning: No data for transition {transition}. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [m for m, _ in combined]
        values = [d for _, d in combined]
        colors = ['red'] * len(true_sorted) + ['blue'] * len(false_sorted)

        ax.bar(labels, values, color=colors)
        ax.set_title(f"JS Divergences: {transition} ({model_type} model) (True=Red, False=Blue)")
        ax.set_ylabel("JS Divergence")
        ax.set_xticklabels(labels, rotation=90)

        # Set standard Y-axis range (based on global max)
        if max_global_divergence > 0:
            ax.set_ylim(0, max_global_divergence)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"divergence_{transition}_{model_type}.svg"), format="svg")
        plt.close()

    # Get global Y limit
    all_divergences = [d for motifs in list(true_data.values()) + list(false_data.values()) for d in motifs.values()]

    # Check if we have any data
    if not all_divergences:
        print("Warning: No divergence data found. This may indicate that 05_motif_analysis.py did not produce the expected output.")
        print(f"Expected input file: {input_path_str}")
        print(f"Skipping {model_type} model plot generation.")
        continue

    max_global_divergence = max(all_divergences)

    # Generate one plot per transition (use union of true_data and false_data keys)
    all_transitions = set(list(true_data.keys()) + list(false_data.keys()))

    if not all_transitions:
        print(f"Warning: No transitions found in data. Skipping {model_type} model plot generation.")
        continue

    for transition in all_transitions:
        plot_true_and_false(transition, true_data.get(transition, {}), false_data.get(transition, {}), max_global_divergence, model_output_dir, model_type)
    
    print(f"✅ Completed processing for {model_type} model")

print(f"\n📁 All results saved to: {base_output_dir}")
print("   - Uniform model: {}/uniform/".format(base_output_dir))
print("   - Region-specific model: {}/region_specific/".format(base_output_dir))
