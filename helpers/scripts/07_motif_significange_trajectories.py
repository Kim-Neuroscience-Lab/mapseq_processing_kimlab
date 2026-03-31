#Takes the upsetplot.csv files from the process-nbcm.tsv pipeline output and produces an effect size trajectory for each motif.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, linregress
from statsmodels.stats.multitest import fdrcorrection

# Set font to Helvetica with Arial fallback, and ensure SVG text is editable
plt.rcParams["font.family"] = ['Helvetica', 'Arial']
plt.rcParams["svg.fonttype"] = "none"

def motif_label(motif_str):
    try:
        items = eval(motif_str)
        return "+".join(sorted(items)) if items else "<null>"
    except:
        return "<parse_error>"

def load_and_process_files(input_dir, output_dir, model_type=None):
    """
    Load and process upsetplot files, optionally filtering by model type.
    
    Args:
        input_dir: Directory containing upsetplot CSV files
        output_dir: Directory to save combined output
        model_type: 'uniform', 'region_specific', or None (process all)
    
    Returns:
        Combined DataFrame with all processed data
    """
    all_data = []
    for fname in sorted(os.listdir(input_dir)):
        # Detect model type from filename
        all_models = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                      'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                      'bayesian_hierarchical', 'ml_nonparametric', 'proportional_effectsize']
        
        detected_model = None
        for m in all_models:
            if fname.endswith(f"upsetplot_{m}.csv"):
                detected_model = m
                break
        
        is_backward_compat = fname.endswith("upsetplot.csv") and detected_model is None
        
        # Filter by model type if specified
        if model_type:
            if detected_model != model_type:
                continue
        elif is_backward_compat:
            # If no model_type specified, skip backward compat files if model-specific files exist
            # (prefer model-specific files)
            continue
        
        # Extract stage from filename (first part before underscore)
        stage = fname.split("_")[0].upper()
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath)
        df["Motif_Label"] = df["Motifs"].apply(motif_label)
        df["Stage"] = stage
        df["Significant"] = df["P-value"].apply(lambda p: float(p) <= 0.05)
        df["Observed"] = df["Observed"].astype(int)
        all_data.append(df[["Motif_Label", "Effect Size", "Stage", "Significant", "Observed"]])
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data)
    output_filename = f"combined_effect_sizes_{model_type}.csv" if model_type else "combined_effect_sizes.csv"
    combined_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    return combined_df

def _se_effect_size(obs):
    """SE of effect size d = log2(obs/exp), treating expected as fixed: SE(d) = 1/(ln(2)*sqrt(obs))."""
    if obs is None or pd.isna(obs) or obs <= 0:
        return np.nan
    return 1.0 / (np.log(2) * np.sqrt(float(obs)))


def compute_transition_significance(df, stage_order, output_dir, use_fdr_for_significant=False):
    """
    Test whether effect size changed significantly between consecutive stages (z-test on difference).
    Writes transition_significance.csv with Motif, Transition, P-value, Significant, Delta_Effect_Size, SE_delta, P_value_adjusted.
    If use_fdr_for_significant is True, Significant is set from FDR-adjusted p <= 0.05; otherwise from raw p <= 0.05.
    """
    rows = []
    for motif in df["Motif_Label"].unique():
        motif_data = df[df["Motif_Label"] == motif].set_index("Stage")
        motif_pvals = []
        for i in range(len(stage_order) - 1):
            s1, s2 = stage_order[i], stage_order[i + 1]
            delta_d = np.nan
            se_delta = np.nan
            p = np.nan
            if s1 not in motif_data.index or s2 not in motif_data.index:
                rows.append({
                    "Motif": motif,
                    "Transition": f"{s1}_to_{s2}",
                    "P-value": p,
                    "Significant": False,
                    "Delta_Effect_Size": delta_d,
                    "SE_delta": se_delta,
                    "P_value_adjusted": np.nan,
                })
                continue
            d1 = motif_data.loc[s1, "Effect Size"]
            d2 = motif_data.loc[s2, "Effect Size"]
            obs1 = motif_data.loc[s1, "Observed"]
            obs2 = motif_data.loc[s2, "Observed"]
            if pd.isna(d1) or pd.isna(d2) or obs1 is None or obs2 is None or obs1 <= 0 or obs2 <= 0:
                pass
            else:
                obs1, obs2 = int(obs1), int(obs2)
                se1 = _se_effect_size(obs1)
                se2 = _se_effect_size(obs2)
                delta_d = float(d2) - float(d1)
                se_delta = np.sqrt(se1**2 + se2**2)
                z = delta_d / se_delta if se_delta > 0 else 0
                p = 2 * (1 - norm.cdf(np.abs(z)))
            rows.append({
                "Motif": motif,
                "Transition": f"{s1}_to_{s2}",
                "P-value": p,
                "Significant": bool(p <= 0.05) if not pd.isna(p) else False,
                "Delta_Effect_Size": delta_d,
                "SE_delta": se_delta,
                "P_value_adjusted": np.nan,
            })
            if not pd.isna(p):
                motif_pvals.append((len(rows) - 1, p))
        # FDR within motif (3 transitions)
        if motif_pvals:
            indices = [idx for idx, _ in motif_pvals]
            pvals = [p for _, p in motif_pvals]
            try:
                _, p_adj = fdrcorrection(pvals, alpha=0.05)
                for k, idx in enumerate(indices):
                    rows[idx]["P_value_adjusted"] = p_adj[k]
                    if use_fdr_for_significant:
                        rows[idx]["Significant"] = bool(p_adj[k] <= 0.05) if not pd.isna(p_adj[k]) else False
            except Exception:
                pass
    df_out = pd.DataFrame(rows)
    # Ensure column order for compatibility
    cols = ["Motif", "Transition", "P-value", "Significant", "Delta_Effect_Size", "SE_delta", "P_value_adjusted"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    df_out.to_csv(os.path.join(output_dir, "transition_significance.csv"), index=False)
    return df_out


def compute_motif_trajectory_summary(df, stage_order, output_dir, include_exploratory_trend_p=False):
    """
    Compute per-motif trajectory statistics: trend, total change P3->P60, N_stages_significant.
    Writes motif_trajectory_summary.csv (one row per motif).
    
    NOTE: The trend p-value from linear regression is EXPLORATORY ONLY due to very low
    statistical power with only n=4 observations per motif. It should not be used for
    formal hypothesis testing. Set include_exploratory_trend_p=True to include it in output.
    
    NOTE: Kruskal-Wallis test was removed because it is statistically invalid with n=1
    observation per group (each stage has only one effect size value per motif).
    """
    summary_rows = []
    stage_idx = {s: i for i, s in enumerate(stage_order)}
    for motif in df["Motif_Label"].unique():
        motif_data = df[df["Motif_Label"] == motif].set_index("Stage").reindex(stage_order)
        effect_sizes = motif_data["Effect Size"].values
        observed = motif_data["Observed"].values
        significant = motif_data["Significant"].fillna(False).values

        # Trend: regress Effect Size on stage index (1, 2, 3, 4)
        # NOTE: This is EXPLORATORY due to very low power with n=4 points.
        # The p-value should NOT be used for formal hypothesis testing.
        valid = ~pd.isna(effect_sizes)
        n_valid = np.sum(valid)
        trend_slope = np.nan
        trend_p_exploratory = np.nan  # Renamed to emphasize exploratory nature
        trend_direction = "none"
        if n_valid >= 2:
            x = np.array([stage_idx[s] + 1 for s in stage_order])[valid]
            y = effect_sizes[valid].astype(float)
            try:
                res = linregress(x, y)
                trend_slope = res.slope
                trend_p_exploratory = res.pvalue  # EXPLORATORY ONLY - low power with n=4
                trend_direction = "increasing" if trend_slope > 0 else ("decreasing" if trend_slope < 0 else "none")
            except Exception:
                pass

        # NOTE: Kruskal-Wallis test was REMOVED (previously global_change_p).
        # Reason: The test requires multiple observations per group, but we have
        # only n=1 effect size value per stage per motif. With n=1 per group,
        # there is no within-group variance to test against, making the test
        # statistically invalid. See: Kruskal & Wallis (1952), JASA.

        # Total change P3 -> P60
        delta_p3_p60 = np.nan
        delta_se = np.nan
        delta_ci_lower = np.nan
        delta_ci_upper = np.nan
        delta_p = np.nan
        if "P3" in stage_order and "P60" in stage_order and "P3" in motif_data.index and "P60" in motif_data.index:
            d_p3 = motif_data.loc["P3", "Effect Size"]
            d_p60 = motif_data.loc["P60", "Effect Size"]
            obs_p3 = motif_data.loc["P3", "Observed"]
            obs_p60 = motif_data.loc["P60", "Observed"]
            if not pd.isna(d_p3) and not pd.isna(d_p60) and obs_p3 and obs_p60 and obs_p3 > 0 and obs_p60 > 0:
                se_p3 = _se_effect_size(int(obs_p3))
                se_p60 = _se_effect_size(int(obs_p60))
                delta_p3_p60 = float(d_p60) - float(d_p3)
                delta_se = np.sqrt(se_p3**2 + se_p60**2)
                if delta_se > 0:
                    z = delta_p3_p60 / delta_se
                    delta_p = 2 * (1 - norm.cdf(np.abs(z)))
                delta_ci_lower = delta_p3_p60 - 1.96 * delta_se if not pd.isna(delta_se) else np.nan
                delta_ci_upper = delta_p3_p60 + 1.96 * delta_se if not pd.isna(delta_se) else np.nan

        # N_stages_significant
        n_stages_significant = int(np.sum(significant))

        # Build summary row
        row = {
            "Motif": motif,
            "trend_slope": trend_slope,
            "trend_direction": trend_direction,
            "Delta_P3_to_P60": delta_p3_p60,
            "Delta_P3_to_P60_SE": delta_se,
            "Delta_P3_to_P60_CI_lower": delta_ci_lower,
            "Delta_P3_to_P60_CI_upper": delta_ci_upper,
            "Delta_P3_to_P60_p": delta_p,
            "N_stages_significant": n_stages_significant,
        }
        # Only include exploratory trend p-value if explicitly requested
        if include_exploratory_trend_p:
            row["trend_p_EXPLORATORY"] = trend_p_exploratory
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "motif_trajectory_summary.csv"), index=False)
    return df_summary


def plot_motif_set(df, stage_order, trans_sig_df, motifs, output_pdf_path, output_dir, prefix="", unified_ymin=None, unified_ymax=None):
    if unified_ymin is not None and unified_ymax is not None:
        global_ymin = unified_ymin
        global_ymax = unified_ymax
    else:
        global_ymin = df["Effect Size"].min() - 0.5
        global_ymax = df["Effect Size"].max() + 0.5
    stage_to_x = {s: i for i, s in enumerate(stage_order)}
    x_numeric = np.arange(len(stage_order))

    with PdfPages(output_pdf_path) as pdf:
        for motif in motifs:
            motif_data = df[df["Motif_Label"] == motif].set_index("Stage").reindex(stage_order)
            y = motif_data["Effect Size"].values
            sig = motif_data["Significant"].fillna(False).values

            if np.all(pd.isna(y)):
                continue

            # Per-stage 95% CI for effect size (SE = 1.44/sqrt(obs)); draw error bars
            obs = motif_data["Observed"].reindex(stage_order).values
            se_arr = np.array([_se_effect_size(obs[i]) if i < len(obs) and obs[i] is not None and not pd.isna(obs[i]) and obs[i] > 0 else np.nan for i in range(len(stage_order))])
            yerr = np.where(np.isnan(se_arr) | (se_arr <= 0), 0, 1.96 * se_arr)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(x_numeric, y, yerr=yerr, fmt='none', ecolor='gray', capsize=2, capthick=1, zorder=1)
            ax.plot(x_numeric, y, linestyle='-', color='black', marker='o', markersize=6, zorder=2)
            for xi, yi, s in zip(x_numeric, y, sig):
                if s:
                    ax.plot(xi, yi, marker='o', color='red', markersize=8)

            for i in range(len(stage_order) - 1):
                s1, s2 = stage_order[i], stage_order[i + 1]
                trans = trans_sig_df[
                    (trans_sig_df["Motif"] == motif) &
                    (trans_sig_df["Transition"] == f"{s1}_to_{s2}")
                ]
                if not trans.empty:
                    pval = trans["P-value"].values[0]
                    if not pd.isna(pval):
                        if pval < 1e-4:
                            stars = "****"
                        elif pval < 1e-3:
                            stars = "***"
                        elif pval < 0.01:
                            stars = "**"
                        elif pval < 0.05:
                            stars = "*"
                        else:
                            stars = None
                        if stars:
                            xm = (x_numeric[i] + x_numeric[i + 1]) / 2
                            ym = (y[i] + y[i + 1]) / 2 if not (np.isnan(y[i]) or np.isnan(y[i + 1])) else 0
                            ax.annotate(stars, xy=(xm, ym), ha='center', va='bottom', color='blue', fontsize=12)

            ax.set_title(f"Motif: {motif}")
            ax.set_ylabel("Effect Size\nlog2(Observed / Expected)")
            ax.set_xlabel("Developmental Stage")
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_xticks(x_numeric)
            ax.set_xticklabels(stage_order)
            ax.grid(True, linestyle='--', alpha=0.5)

            pdf.savefig(fig)
            svg_path = os.path.join(output_dir, f"{prefix}{motif.replace('+','_')}_effect_size.svg")
            fig.savefig(svg_path, format='svg')
            plt.close(fig)

def calculate_unified_yaxis_range(input_dir, models_to_process):
    """
    Calculate unified y-axis range across all models.
    
    Args:
        input_dir: Directory containing upsetplot CSV files
        models_to_process: List of model types to process
    
    Returns:
        unified_ymin, unified_ymax: Unified y-axis range (rounded to multiple of 5)
    """
    all_effect_sizes = []
    
    for model_type in models_to_process:
        # Load all upsetplot files for this model
        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(f"upsetplot_{model_type}.csv"):
                fpath = os.path.join(input_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    if "Effect Size" in df.columns:
                        all_effect_sizes.extend(df["Effect Size"].dropna().tolist())
                except Exception as e:
                    print(f"Warning: Could not read {fpath}: {e}")
                    continue
    
    if not all_effect_sizes:
        return None, None
    
    global_min = min(all_effect_sizes)
    global_max = max(all_effect_sizes)
    max_abs = max(abs(global_min), abs(global_max))
    
    # Round up to next multiple of 5
    unified_range = ((int(max_abs) // 5) + 1) * 5
    
    return -unified_range, unified_range

def plot_motif_trajectories(df, stage_order, output_dir, unified_ymin=None, unified_ymax=None, use_fdr_for_significant=False, include_exploratory_trend_p=False):
    trans_sig_df = compute_transition_significance(df, stage_order, output_dir, use_fdr_for_significant=use_fdr_for_significant)
    compute_motif_trajectory_summary(df, stage_order, output_dir, include_exploratory_trend_p=include_exploratory_trend_p)
    motif_stage_counts = df.groupby("Motif_Label")["Stage"].nunique()
    full_motifs = motif_stage_counts[motif_stage_counts == len(stage_order)].index
    partial_motifs = motif_stage_counts[motif_stage_counts < len(stage_order)].index

    # Sort full motifs by subset size
    def subset_size(label):
        return len(label.split('+'))

    full_motif_sizes = pd.Series(full_motifs).apply(lambda x: (subset_size(x), x))
    sorted_full_motifs = [x[1] for x in sorted(full_motif_sizes, key=lambda x: (x[0], x[1]))]

    # Full motif plots
    pdf_full = os.path.join(output_dir, "motif_effect_trajectories.pdf")
    plot_motif_set(df, stage_order, trans_sig_df, sorted_full_motifs, pdf_full, output_dir, unified_ymin=unified_ymin, unified_ymax=unified_ymax)

    # Partial motif plots
    pdf_partial = os.path.join(output_dir, "motif_effect_partial_trajectories.pdf")
    plot_motif_set(df, stage_order, trans_sig_df, partial_motifs, pdf_partial, output_dir, prefix="partial_", unified_ymin=unified_ymin, unified_ymax=unified_ymax)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Plot motif effect size trajectories across stages.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_upsetplot.csv files")
    parser.add_argument("--helper_output_dir", type=str, default=None,
                       help="Directory for helper script outputs (default: helpers/outputs/07_motif_significange_trajectories)")
    parser.add_argument("--unified_yaxis", action="store_true",
                       help="Use unified y-axis range across all models (rounded to nearest multiple of 5)")
    parser.add_argument("--use_fdr_for_significant", action="store_true",
                       help="Set Significant from FDR-adjusted p <= 0.05 in transition_significance.csv (default: raw p <= 0.05)")
    parser.add_argument("--exploratory_trend_pvalue", action="store_true",
                       help="Include exploratory trend p-value in motif_trajectory_summary.csv. "
                            "WARNING: This p-value has very low statistical power (n=4) and should "
                            "NOT be used for formal hypothesis testing. Slope is always included.")
    args = parser.parse_args()

    # Set output directory
    script_dir = Path(__file__).parent
    if args.helper_output_dir:
        base_output_dir = Path(args.helper_output_dir)
    else:
        base_output_dir = script_dir.parent / "outputs" / "07_motif_significange_trajectories"
    os.makedirs(base_output_dir, exist_ok=True)

    stage_order = ["P3", "P12", "P20", "P60"]
    
    # Process all three models separately
    models_to_process = ['uniform', 'region_specific', 'correlated', 'empirical', 'smoothed_empirical', 
                     'max_entropy', 'hierarchical_correlations', 'negative_binomial', 'zero_inflated',
                     'bayesian_hierarchical', 'ml_nonparametric', 'proportional_effectsize']
    
    # Calculate unified y-axis range if requested
    unified_ymin = None
    unified_ymax = None
    if args.unified_yaxis:
        print("\n" + "="*80)
        print("Calculating unified y-axis range across all models...")
        print("="*80)
        unified_ymin, unified_ymax = calculate_unified_yaxis_range(args.input_dir, models_to_process)
        if unified_ymin is not None and unified_ymax is not None:
            print(f"✅ Unified y-axis range: [{unified_ymin}, {unified_ymax}]")
        else:
            print("⚠️ Warning: Could not calculate unified range, using per-model ranges")
            unified_ymin = None
            unified_ymax = None
    
    for model_type in models_to_process:
        print("\n" + "="*80)
        print(f"Processing {model_type.upper()} MODEL")
        print("="*80)
        
        # Create model-specific output directory
        model_output_dir = base_output_dir / model_type
        os.makedirs(model_output_dir, exist_ok=True)
        
        df_all = load_and_process_files(args.input_dir, str(model_output_dir), model_type=model_type)
        
        if df_all.empty:
            print(f"Warning: No data found for {model_type} model")
            continue
        
        plot_motif_trajectories(df_all, stage_order, str(model_output_dir), unified_ymin=unified_ymin, unified_ymax=unified_ymax, use_fdr_for_significant=args.use_fdr_for_significant, include_exploratory_trend_p=args.exploratory_trend_pvalue)
        print(f"✅ Completed processing for {model_type} model")
    
    print(f"\n📁 All results saved to: {base_output_dir}")
    print("   - Uniform model: {}/uniform/".format(base_output_dir))
    print("   - Region-specific model: {}/region_specific/".format(base_output_dir))
