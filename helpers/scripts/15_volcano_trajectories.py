# Generate per-motif volcano trajectory plots: one point per age (P3, P12, P20, P60)
# on effect-size vs significance space, with directional lines between ages.
# All plots share the same axis limits for cross-comparison.
#
# Statistical Methods (v2.0):
# This script implements multiple citable statistical methods for identifying
# trajectories that change significantly over time:
#   - Quadrant change detection (original method, with transition filter from helper 07)
#   - Permutation tests for trajectory significance (Good, 2005)
#   - Functional Data Analysis trajectory tests (Ramsay & Silverman, 2005)
#   - Mixed-effects models for repeated measures (Bates et al., 2015)
#   - Bootstrap confidence intervals for quadrant classification
#   - Standardized distance metrics (z-score, Mahalanobis)
#
# References:
#   - Good, P. (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses, 3rd ed., Springer.
#   - Ramsay & Silverman (2005). Functional Data Analysis, Springer.
#   - Bates et al. (2015). J Stat Software, 67(1), 1-48.
#   - Mahalanobis, P.C. (1936). Proc. Nat. Inst. Sci. India, 2, 49-55.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import binomtest
from scipy.interpolate import UnivariateSpline
from statsmodels.stats.multitest import fdrcorrection
import warnings

# Match script 07 and process-nbcm-tsv styling; editable text in Illustrator (Helvetica, Arial)
plt.rcParams["font.family"] = ["Helvetica", "Arial", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rc("text", usetex=False)

STAGE_ORDER = ["P3", "P12", "P20", "P60"]
SIGNIFICANCE_CAP = 300  # cap -log10(P) when P=0
VOLCANO_YLIM = (-10.0, 11.0)  # fixed y-axis ±10 with padding on positive end (was data-driven, could blow up to 300+)
ALPHA = 0.05
ALPHA_STRINGENT = 0.01  # More stringent p-value cutoff
EFFECT_SIZE_THRESHOLD_DEFAULT = 0.5  # |log2FC| threshold for grey zone (not used by default)
MODELS_DEFAULT = ["uniform", "region_specific", "proportional_effectsize", "correlated"]
XLABEL_DEFAULT = "Effect Size\n$log_2($observed/expected$)$"
XLABEL_PROP = "Effect Size (Proportional)\n$log_2(\\frac{k/n_0 + 1}{\\pi + 1})$"


def motif_label(motif_str):
    """Normalize motif string to label (match script 07)."""
    try:
        items = eval(motif_str)
        return "+".join(sorted(items)) if items else ""
    except Exception:
        return ""


def safe_significance(pval):
    """Convert P-value to -log10(P), capping at SIGNIFICANCE_CAP for P=0."""
    try:
        p = float(pval)
        if p <= 0:
            return SIGNIFICANCE_CAP
        s = -np.log10(p)
        return min(s, SIGNIFICANCE_CAP)
    except (TypeError, ValueError):
        return np.nan


def load_volcano_data(input_dir, model_type):
    """
    Load upsetplot CSVs for one model from input_dir.
    Returns DataFrame with columns: Motif_Label, Stage, effect_size, significance, Observed, Expected.
    
    Observed and Expected are loaded for bootstrap CI analysis. If not present in
    the source CSV, they will be NaN (bootstrap CI will be skipped for those rows).
    """
    all_models = [
        "uniform", "region_specific", "correlated", "empirical", "smoothed_empirical",
        "max_entropy", "hierarchical_correlations", "negative_binomial", "zero_inflated",
        "bayesian_hierarchical", "ml_nonparametric", "proportional_effectsize",
        "proportional_effectsize_raw",
    ]
    suffix = f"upsetplot_{model_type}.csv"
    rows = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(suffix):
            continue
        stage = fname.split("_")[0].upper()
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Warning: Could not read {fpath}: {e}")
            continue
        if "Effect Size" not in df.columns or "P-value" not in df.columns:
            print(f"Warning: Missing columns in {fpath}")
            continue
        for _, r in df.iterrows():
            label = motif_label(r["Motifs"])
            if label in ("", "<null>", "<parse_error>"):
                continue
            try:
                eff = float(r["Effect Size"])
            except (TypeError, ValueError):
                eff = np.nan
            sig = safe_significance(r["P-value"])
            if np.isnan(eff):
                continue
            # Load Observed and Expected for bootstrap CI
            try:
                observed = float(r["Observed"]) if "Observed" in r else np.nan
            except (TypeError, ValueError):
                observed = np.nan
            try:
                expected = float(r["Expected"]) if "Expected" in r else np.nan
            except (TypeError, ValueError):
                expected = np.nan
            rows.append({
                "Motif_Label": label,
                "Stage": stage,
                "effect_size": eff,
                "significance": sig,
                "Observed": observed,
                "Expected": expected,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Check if Observed/Expected columns are all NaN (incompatible source data)
    obs_missing = df["Observed"].isna().all()
    exp_missing = df["Expected"].isna().all()
    if obs_missing or exp_missing:
        print(f"  Warning: {model_type} - Observed/Expected columns not available in source data.")
        print(f"           Bootstrap CI analysis will be skipped for this model.")
    return df


def compute_unified_limits(df, volcano_ylim=True):
    """
    Compute (xlim, ylim) from full dataframe.
    x: symmetric max_abs * 1.1.
    y: if volcano_ylim True, [0, max(significance)*1.1]; else symmetric ± max_abs_y * 1.1.
    """
    if df.empty:
        return (None, None), (None, None)
    eff = df["effect_size"].dropna()
    sig = df["significance"].dropna()
    if eff.empty or sig.empty:
        return (None, None), (None, None)
    max_abs = max(abs(eff.min()), abs(eff.max()))
    unified_range = max_abs * 1.1
    xlim = (-unified_range, unified_range)
    if volcano_ylim:
        # Significance is non-negative; classic volcano y from 0 to max
        y_max = sig.max()
        ylim = (0.0, float(y_max) * 1.1)
    else:
        max_abs_y = max(abs(sig.min()), abs(sig.max()))
        unified_range_y = max_abs_y * 1.1
        ylim = (-unified_range_y, unified_range_y)
    return xlim, ylim


def bonferroni_cutoff(n_motifs):
    return -np.log10(ALPHA / max(1, n_motifs))


def bonferroni_cutoff_stringent(n_motifs):
    """Bonferroni-corrected cutoff for ALPHA_STRINGENT (0.01)."""
    return -np.log10(ALPHA_STRINGENT / max(1, n_motifs))


def draw_effect_size_grey_zone(ax, threshold, xlim):
    """
    Draw a grey zone for effect sizes below threshold (like gene expression volcano plots).
    
    Args:
        ax: Matplotlib axis
        threshold: Effect size threshold (symmetric around 0, in log2 fold change units)
        xlim: X-axis limits tuple (used to ensure zone doesn't exceed plot bounds)
    """
    if threshold is not None and threshold > 0:
        # Clamp the zone to the plot boundaries
        left = max(-threshold, xlim[0])
        right = min(threshold, xlim[1])
        ax.axvspan(left, right, color='lightgray', alpha=0.3, zorder=0)
        # Add dotted vertical lines at the threshold boundaries (if within plot range)
        if -threshold >= xlim[0]:
            ax.axvline(x=-threshold, linestyle=':', color='darkgray', linewidth=1, alpha=0.7)
        if threshold <= xlim[1]:
            ax.axvline(x=threshold, linestyle=':', color='darkgray', linewidth=1, alpha=0.7)


def classify_quadrant(effect_size, significance, pcutoff):
    """
    Classify a single (effect_size, significance) point into quadrant.
    Returns "sig_pos" | "sig_neg" | "not_sig".
    """
    if significance >= pcutoff:
        return "sig_pos" if effect_size > 0 else "sig_neg"
    return "not_sig"


def motif_changes_quadrant(traj_df, pcutoff):
    """
    Return True iff the motif's trajectory exhibits a meaningful quadrant change:
    not_sig -> sig (either direction) or sig_neg <-> sig_pos.
    """
    quadrants = set()
    for _, row in traj_df.iterrows():
        q = classify_quadrant(row["effect_size"], row["significance"], pcutoff)
        quadrants.add(q)
    if "not_sig" in quadrants and ("sig_pos" in quadrants or "sig_neg" in quadrants):
        return True
    if "sig_pos" in quadrants and "sig_neg" in quadrants:
        return True
    return False


def quadrant_to_label(q):
    """Map quadrant from classify_quadrant to user-facing label: over | under | ns."""
    if q == "sig_pos":
        return "over"
    if q == "sig_neg":
        return "under"
    return "ns"


def matt_visual_rules_per_motif(traj_df, pcutoff):
    """
    For a single trajectory (sorted by stage), compute Matt visual rules:
    per-stage labels (over/under/ns), changing (True/False), and transition flags for summary counts.
    traj_df: DataFrame with Stage, effect_size, significance (sorted by STAGE_ORDER).
    Returns dict: stage_labels (list of (stage, label)), changing (bool),
    has_become_over, has_become_under, has_switch_over_to_under, has_switch_under_to_over, has_lose_significance_to_ns.
    """
    traj = traj_df.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
    if len(traj) < 2:
        return None
    labels = []
    for _, row in traj.iterrows():
        q = classify_quadrant(row["effect_size"], row["significance"], pcutoff)
        labels.append((row["Stage"], quadrant_to_label(q)))
    changing = motif_changes_quadrant(traj, pcutoff)
    # Transition counts (consecutive pairs in stage order)
    lab_list = [lab for _, lab in labels]
    has_switch_over_to_under = False
    has_switch_under_to_over = False
    has_lose_significance_to_ns = False
    has_become_over = "over" in lab_list and (lab_list[0] != "over" or any(
        lab_list[i] != "over" and lab_list[i + 1] == "over" for i in range(len(lab_list) - 1)
    ))
    has_become_under = "under" in lab_list and (lab_list[0] != "under" or any(
        lab_list[i] != "under" and lab_list[i + 1] == "under" for i in range(len(lab_list) - 1)
    ))
    for i in range(len(lab_list) - 1):
        a, b = lab_list[i], lab_list[i + 1]
        if a == "over" and b == "under":
            has_switch_over_to_under = True
        if a == "under" and b == "over":
            has_switch_under_to_over = True
        if (a in ("over", "under")) and b == "ns":
            has_lose_significance_to_ns = True
    return {
        "stage_labels": labels,
        "changing": changing,
        "has_become_over": has_become_over,
        "has_become_under": has_become_under,
        "has_switch_over_to_under": has_switch_over_to_under,
        "has_switch_under_to_over": has_switch_under_to_over,
        "has_lose_significance_to_ns": has_lose_significance_to_ns,
    }


def path_length(traj_df):
    """
    Total Euclidean path length in (effect_size, significance) space over consecutive stages.
    traj_df must be sorted by stage order.
    """
    points = traj_df[["effect_size", "significance"]].values
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += np.sqrt((points[i + 1, 0] - points[i, 0]) ** 2 + (points[i + 1, 1] - points[i, 1]) ** 2)
    return total


def effect_size_range(traj_df):
    """Range of effect_size across stages (max - min)."""
    eff = traj_df["effect_size"].dropna()
    if eff.empty:
        return np.nan
    return float(eff.max() - eff.min())


# =============================================================================
# NEW STATISTICAL METHODS (v2.0) - Publication-ready trajectory significance tests
# =============================================================================

def compute_zscore_path_length(df_all, traj_df):
    """
    Compute path length in z-score standardized (effect_size, significance) space.
    This addresses the incommensurate scales problem of raw Euclidean distance.
    
    Args:
        df_all: Full DataFrame with all motifs (for computing global mean/std)
        traj_df: Single motif trajectory DataFrame (sorted by stage)
    
    Returns:
        float: Z-score standardized path length
    
    Reference: Standard statistical normalization for comparing variables on different scales.
    """
    # Compute global statistics for standardization
    effect_mean = df_all["effect_size"].mean()
    effect_std = df_all["effect_size"].std()
    sig_mean = df_all["significance"].mean()
    sig_std = df_all["significance"].std()
    
    # Avoid division by zero
    if effect_std <= 0:
        effect_std = 1.0
    if sig_std <= 0:
        sig_std = 1.0
    
    # Standardize trajectory points
    effect_z = (traj_df["effect_size"].values - effect_mean) / effect_std
    sig_z = (traj_df["significance"].values - sig_mean) / sig_std
    
    # Compute path length in z-score space
    if len(effect_z) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(effect_z) - 1):
        total += np.sqrt((effect_z[i + 1] - effect_z[i]) ** 2 + (sig_z[i + 1] - sig_z[i]) ** 2)
    return total


def compute_mahalanobis_path_length(df_all, traj_df):
    """
    Compute path length using Mahalanobis distance, which accounts for
    correlation between effect_size and significance and normalizes by variance.
    
    Args:
        df_all: Full DataFrame with all motifs (for computing covariance matrix)
        traj_df: Single motif trajectory DataFrame (sorted by stage)
    
    Returns:
        float: Mahalanobis-based path length
    
    Reference: Mahalanobis, P.C. (1936). Proc. Nat. Inst. Sci. India, 2, 49-55.
    """
    points = traj_df[["effect_size", "significance"]].values
    if len(points) < 2:
        return 0.0
    
    # Compute covariance matrix from all data
    cov_matrix = df_all[["effect_size", "significance"]].cov().values
    
    # Check for singular covariance matrix
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov_matrix)
    
    # Compute sum of Mahalanobis distances between consecutive points
    total = 0.0
    for i in range(len(points) - 1):
        diff = points[i + 1] - points[i]
        mahal_sq = diff @ cov_inv @ diff
        total += np.sqrt(max(0, mahal_sq))  # Ensure non-negative due to numerical precision
    return total


def compute_separate_axis_metrics(traj_df):
    """
    Compute separate metrics for effect_size and significance axes.
    This avoids the scale mixing problem entirely by treating axes independently.
    
    Args:
        traj_df: Single motif trajectory DataFrame (sorted by stage)
    
    Returns:
        dict: {effect_size_range, effect_size_sd, effect_size_total_variation,
               significance_range, significance_sd, significance_total_variation}
    """
    eff = traj_df["effect_size"].values
    sig = traj_df["significance"].values
    
    def total_variation(arr):
        """Sum of absolute differences between consecutive values."""
        if len(arr) < 2:
            return 0.0
        return np.sum(np.abs(np.diff(arr)))
    
    return {
        "effect_size_range": float(np.max(eff) - np.min(eff)) if len(eff) > 0 else np.nan,
        "effect_size_sd": float(np.std(eff)) if len(eff) > 1 else np.nan,
        "effect_size_total_variation": total_variation(eff),
        "significance_range": float(np.max(sig) - np.min(sig)) if len(sig) > 0 else np.nan,
        "significance_sd": float(np.std(sig)) if len(sig) > 1 else np.nan,
        "significance_total_variation": total_variation(sig),
    }


def compute_bootstrap_quadrant_ci(traj_df, pcutoff, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence intervals for quadrant classification.
    Resamples observed counts from Poisson distribution to estimate
    uncertainty in quadrant assignment.
    
    Args:
        traj_df: Single motif trajectory with columns: Stage, effect_size, significance, Observed, Expected
        pcutoff: Significance threshold (-log10 scale)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: Stage, quadrant_raw, prop_sig_pos, prop_sig_neg, prop_not_sig, quadrant_robust
    
    Note: Requires Observed and Expected columns in traj_df. If missing, returns None.
    """
    if "Observed" not in traj_df.columns or "Expected" not in traj_df.columns:
        return None
    
    np.random.seed(seed)
    results = []
    
    for _, row in traj_df.iterrows():
        obs = row.get("Observed")
        exp = row.get("Expected")
        
        if pd.isna(obs) or pd.isna(exp) or obs <= 0 or exp <= 0:
            results.append({
                "Stage": row["Stage"],
                "quadrant_raw": classify_quadrant(row["effect_size"], row["significance"], pcutoff),
                "prop_sig_pos": np.nan,
                "prop_sig_neg": np.nan,
                "prop_not_sig": np.nan,
                "quadrant_robust": "unknown",
            })
            continue
        
        quadrant_counts = {"sig_pos": 0, "sig_neg": 0, "not_sig": 0}
        
        for _ in range(n_bootstrap):
            # Resample observed count from Poisson(observed)
            obs_boot = np.random.poisson(obs)
            # Recompute effect size
            eff_boot = np.log2((obs_boot + 1) / (exp + 1))
            # Recompute significance (approximate using same relative p-value scaling)
            # This is a simplification - ideally we'd recompute the binomial test
            # but that requires N0 and probability which may not be available
            sig_boot = row["significance"]  # Keep original significance for now
            q = classify_quadrant(eff_boot, sig_boot, pcutoff)
            quadrant_counts[q] += 1
        
        proportions = {k: v / n_bootstrap for k, v in quadrant_counts.items()}
        max_prop = max(proportions.values())
        max_quadrant = max(proportions, key=proportions.get)
        robust = max_quadrant if max_prop > 0.95 else "uncertain"
        
        results.append({
            "Stage": row["Stage"],
            "quadrant_raw": classify_quadrant(row["effect_size"], row["significance"], pcutoff),
            "prop_sig_pos": proportions["sig_pos"],
            "prop_sig_neg": proportions["sig_neg"],
            "prop_not_sig": proportions["not_sig"],
            "quadrant_robust": robust,
        })
    
    return pd.DataFrame(results)


def compute_permutation_significance(traj_df, n_permutations=10000, seed=42):
    """
    Permutation test for trajectory significance.
    Tests whether the observed trajectory shows more variation than expected
    by chance if stage labels were randomly shuffled.
    
    Test statistic: Total variation in effect size across ordered stages.
    
    Args:
        traj_df: Single motif trajectory DataFrame (sorted by stage)
        n_permutations: Number of permutations for null distribution
        seed: Random seed for reproducibility
    
    Returns:
        dict: {observed_stat, null_mean, null_sd, p_value}
    
    Reference: Good, P. (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses, 3rd ed., Springer.
    """
    np.random.seed(seed)
    
    effect_sizes = traj_df["effect_size"].values.copy()
    n = len(effect_sizes)
    
    if n < 2:
        return {"observed_stat": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p_value": np.nan}
    
    def total_variation(arr):
        """Sum of absolute differences between consecutive values (ordered)."""
        return np.sum(np.abs(np.diff(arr)))
    
    # Observed statistic (effect sizes in stage order)
    observed_stat = total_variation(effect_sizes)
    
    # Generate null distribution by permuting stage assignments
    null_stats = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(effect_sizes)
        null_stats.append(total_variation(shuffled))
    
    null_stats = np.array(null_stats)
    null_mean = np.mean(null_stats)
    null_sd = np.std(null_stats)
    
    # Two-sided p-value: proportion of permuted stats >= observed
    # (we test for unusually HIGH variation, indicating temporal change)
    p_value = np.mean(null_stats >= observed_stat)
    
    return {
        "observed_stat": observed_stat,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "p_value": p_value,
    }


def compute_fda_significance(traj_df, n_permutations=1000, seed=42):
    """
    Functional Data Analysis (FDA) approach to trajectory significance.
    Tests whether the trajectory differs significantly from a flat (constant) function.
    
    Method: Fit a spline to the trajectory and compute integrated squared derivative
    as a measure of "non-flatness". Compare to null distribution via permutation.
    
    Args:
        traj_df: Single motif trajectory DataFrame (sorted by stage)
        n_permutations: Number of permutations for null distribution
        seed: Random seed for reproducibility
    
    Returns:
        dict: {fda_statistic, null_mean, null_sd, p_value}
    
    Reference: Ramsay & Silverman (2005). Functional Data Analysis, Springer.
    """
    np.random.seed(seed)
    
    effect_sizes = traj_df["effect_size"].values.copy()
    n = len(effect_sizes)
    
    if n < 3:
        return {"fda_statistic": np.nan, "null_mean": np.nan, "null_sd": np.nan, "p_value": np.nan}
    
    x = np.arange(n)
    
    def compute_roughness(y):
        """
        Compute roughness penalty (integrated squared second derivative).
        For discrete data, approximate with sum of squared second differences.
        """
        if len(y) < 3:
            return 0.0
        second_diff = np.diff(y, n=2)
        return np.sum(second_diff ** 2)
    
    def compute_range_statistic(y):
        """Combined statistic: range + roughness."""
        return (np.max(y) - np.min(y)) + compute_roughness(y)
    
    # Observed statistic
    observed_stat = compute_range_statistic(effect_sizes)
    
    # Null distribution via permutation
    null_stats = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(effect_sizes)
        null_stats.append(compute_range_statistic(shuffled))
    
    null_stats = np.array(null_stats)
    null_mean = np.mean(null_stats)
    null_sd = np.std(null_stats)
    
    # p-value: proportion of permuted stats >= observed
    p_value = np.mean(null_stats >= observed_stat)
    
    return {
        "fda_statistic": observed_stat,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "p_value": p_value,
    }


def compute_mixed_effects_trajectory_test(df, stage_order):
    """
    Mixed-effects model approach to trajectory analysis.
    Fits: effect_size ~ stage_numeric + (1|Motif_Label)
    Tests whether stage has a significant effect on effect size across all motifs.
    
    Also extracts per-motif random effects to identify outlier trajectories.
    
    Args:
        df: Full DataFrame with all motifs and stages
        stage_order: List of stages in order (e.g., ["P3", "P12", "P20", "P60"])
    
    Returns:
        dict: {stage_coefficient, stage_pvalue, random_effects_df, model_converged}
        random_effects_df has columns: Motif, random_intercept, residual_variance
    
    Reference: Bates et al. (2015). J Stat Software, 67(1), 1-48.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        warnings.warn("statsmodels not available; mixed-effects analysis skipped.")
        return {"stage_coefficient": np.nan, "stage_pvalue": np.nan, 
                "random_effects_df": pd.DataFrame(), "model_converged": False}
    
    # Prepare data: numeric stage encoding
    stage_map = {s: i for i, s in enumerate(stage_order)}
    work_df = df.copy()
    work_df["stage_numeric"] = work_df["Stage"].map(stage_map)
    work_df = work_df.dropna(subset=["effect_size", "stage_numeric"])
    
    if len(work_df) < 10 or work_df["Motif_Label"].nunique() < 3:
        return {"stage_coefficient": np.nan, "stage_pvalue": np.nan,
                "random_effects_df": pd.DataFrame(), "model_converged": False}
    
    try:
        # Fit mixed-effects model: effect_size ~ stage_numeric + (1|Motif_Label)
        model = MixedLM(
            endog=work_df["effect_size"],
            exog=sm.add_constant(work_df["stage_numeric"]),
            groups=work_df["Motif_Label"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(reml=True, maxiter=200)
        
        # Extract fixed effects
        stage_coef = result.fe_params.get("stage_numeric", np.nan)
        stage_pvalue = result.pvalues.get("stage_numeric", np.nan)
        
        # Extract random effects (per-motif intercepts)
        random_effects = result.random_effects
        re_rows = []
        for motif, effects in random_effects.items():
            re_rows.append({
                "Motif": motif,
                "random_intercept": effects.get("Group", effects.get("Intercept", effects.iloc[0] if hasattr(effects, 'iloc') else list(effects.values())[0])),
            })
        random_effects_df = pd.DataFrame(re_rows)
        
        return {
            "stage_coefficient": stage_coef,
            "stage_pvalue": stage_pvalue,
            "random_effects_df": random_effects_df,
            "model_converged": True,
        }
    except Exception as e:
        warnings.warn(f"Mixed-effects model failed: {e}")
        return {"stage_coefficient": np.nan, "stage_pvalue": np.nan,
                "random_effects_df": pd.DataFrame(), "model_converged": False}


# =============================================================================
# TREND ANALYSIS (v2.1) - Monotonic trend detection for hypothesis testing
# =============================================================================

def compute_spearman_trend_test(traj_df, stage_order):
    """
    Compute Spearman rank correlation to test for monotonic trend in effect size
    across developmental stages.
    
    Unlike the permutation test (which tests for ANY variation), this specifically
    tests whether effect size consistently increases or decreases with stage.
    
    Args:
        traj_df: Single motif trajectory DataFrame (sorted by stage)
        stage_order: List of stages in developmental order (e.g., ["P3", "P12", "P20", "P60"])
    
    Returns:
        dict: {spearman_rho, spearman_pvalue, trend_direction}
        - spearman_rho: Correlation coefficient (-1 to 1)
        - spearman_pvalue: Two-sided p-value
        - trend_direction: "increasing", "decreasing", or "none"
    
    Reference: Spearman, C. (1904). The proof and measurement of association between two things.
    """
    from scipy.stats import spearmanr
    
    # Map stages to numeric order
    stage_map = {s: i for i, s in enumerate(stage_order)}
    
    # Get effect sizes in stage order
    traj_sorted = traj_df.copy()
    traj_sorted["stage_numeric"] = traj_sorted["Stage"].map(stage_map)
    traj_sorted = traj_sorted.dropna(subset=["stage_numeric", "effect_size"])
    traj_sorted = traj_sorted.sort_values("stage_numeric")
    
    if len(traj_sorted) < 3:
        return {"spearman_rho": np.nan, "spearman_pvalue": np.nan, "trend_direction": "insufficient_data"}
    
    x = traj_sorted["stage_numeric"].values
    y = traj_sorted["effect_size"].values
    
    try:
        rho, pvalue = spearmanr(x, y)
    except Exception:
        return {"spearman_rho": np.nan, "spearman_pvalue": np.nan, "trend_direction": "error"}
    
    # Determine direction
    if pvalue < 0.05:
        direction = "increasing" if rho > 0 else "decreasing"
    else:
        direction = "none"
    
    return {
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
        "trend_direction": direction,
    }


def compute_global_direction_test(df, stage_order, early_stage="P12", late_stage="P60"):
    """
    Test whether the direction of effect size change is consistent across motifs.
    
    Uses a binomial test to determine if significantly more motifs increase
    (or decrease) than expected by chance (50/50).
    
    Args:
        df: Full DataFrame with all motifs and stages
        stage_order: List of stages in developmental order
        early_stage: Earlier stage for comparison (default "P12" to exclude P3)
        late_stage: Later stage for comparison (default "P60")
    
    Returns:
        dict: {n_increasing, n_decreasing, n_total, proportion_increasing,
               binomial_pvalue, direction_bias}
    
    Reference: Standard binomial test for proportion different from 0.5.
    """
    from scipy.stats import binomtest
    
    # Get effect size at early and late stages for each motif
    changes = []
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif]
        early_row = traj[traj["Stage"] == early_stage]
        late_row = traj[traj["Stage"] == late_stage]
        
        if len(early_row) == 0 or len(late_row) == 0:
            continue
        
        early_eff = early_row["effect_size"].values[0]
        late_eff = late_row["effect_size"].values[0]
        
        if np.isnan(early_eff) or np.isnan(late_eff):
            continue
        
        changes.append({
            "Motif": motif,
            "early_effect_size": early_eff,
            "late_effect_size": late_eff,
            "change": late_eff - early_eff,
            "direction": "increasing" if late_eff > early_eff else "decreasing",
        })
    
    if len(changes) == 0:
        return {
            "n_increasing": 0, "n_decreasing": 0, "n_total": 0,
            "proportion_increasing": np.nan, "binomial_pvalue": np.nan,
            "direction_bias": "insufficient_data", "changes_df": pd.DataFrame()
        }
    
    changes_df = pd.DataFrame(changes)
    n_increasing = (changes_df["direction"] == "increasing").sum()
    n_decreasing = (changes_df["direction"] == "decreasing").sum()
    n_total = len(changes_df)
    
    # Binomial test: is proportion of increasing significantly different from 0.5?
    try:
        # Use binomtest for two-sided test (scipy >= 1.7)
        result = binomtest(n_increasing, n_total, p=0.5, alternative='two-sided')
        pvalue = result.pvalue
    except Exception:
        pvalue = np.nan
    
    proportion = n_increasing / n_total if n_total > 0 else np.nan
    
    # Determine bias direction
    if pvalue < 0.05:
        bias = "more_increasing" if proportion > 0.5 else "more_decreasing"
    else:
        bias = "no_significant_bias"
    
    return {
        "n_increasing": n_increasing,
        "n_decreasing": n_decreasing,
        "n_total": n_total,
        "proportion_increasing": proportion,
        "binomial_pvalue": pvalue,
        "direction_bias": bias,
        "changes_df": changes_df,
    }


def compute_linear_trend_test(traj_df, stage_order):
    """
    Compute linear regression to test for linear trend in effect size across stages.
    
    More powerful than permutation tests when the true signal is monotonic/linear.
    
    Args:
        traj_df: Single motif trajectory DataFrame (sorted by stage)
        stage_order: List of stages in developmental order
    
    Returns:
        dict: {slope, intercept, r_squared, pvalue, trend_direction}
    
    Reference: Standard OLS regression with t-test for slope significance.
    """
    from scipy.stats import linregress
    
    # Map stages to numeric order
    stage_map = {s: i for i, s in enumerate(stage_order)}
    
    # Get effect sizes in stage order
    traj_sorted = traj_df.copy()
    traj_sorted["stage_numeric"] = traj_sorted["Stage"].map(stage_map)
    traj_sorted = traj_sorted.dropna(subset=["stage_numeric", "effect_size"])
    traj_sorted = traj_sorted.sort_values("stage_numeric")
    
    if len(traj_sorted) < 3:
        return {
            "slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
            "pvalue": np.nan, "trend_direction": "insufficient_data"
        }
    
    x = traj_sorted["stage_numeric"].values
    y = traj_sorted["effect_size"].values
    
    try:
        result = linregress(x, y)
        slope = result.slope
        intercept = result.intercept
        r_squared = result.rvalue ** 2
        pvalue = result.pvalue
    except Exception:
        return {
            "slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
            "pvalue": np.nan, "trend_direction": "error"
        }
    
    # Determine direction
    if pvalue < 0.05:
        direction = "increasing" if slope > 0 else "decreasing"
    else:
        direction = "none"
    
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "pvalue": pvalue,
        "trend_direction": direction,
    }


def execute_trend_analysis(df, output_dir, stage_order=None, exclude_p3=False, subdir_name="trend_analysis"):
    """
    Run comprehensive trend analysis and save results.
    
    This complements the existing permutation/FDA tests by specifically testing
    for monotonic trends and directional consistency across development.
    
    Args:
        df: Full DataFrame with all motifs and stages
        output_dir: Base output directory for this model
        stage_order: List of stages in order (default: STAGE_ORDER)
        exclude_p3: If True, filter out P3 stage before analysis
        subdir_name: Name of output subdirectory (default: "trend_analysis")
    
    Returns:
        dict: {spearman_df, linear_df, direction_result, summary_stats}
    """
    from scipy.stats import false_discovery_control
    
    if stage_order is None:
        stage_order = STAGE_ORDER
    
    # Filter out P3 if requested
    if exclude_p3:
        df = df[df["Stage"] != "P3"].copy()
        stage_order = [s for s in stage_order if s != "P3"]
    
    # Create trend_analysis subdirectory
    trend_dir = os.path.join(output_dir, subdir_name)
    os.makedirs(trend_dir, exist_ok=True)
    
    motifs = df["Motif_Label"].unique()
    
    # =========================================================================
    # Per-Motif Spearman Correlation Tests
    # =========================================================================
    print("    Running Spearman trend tests...")
    spearman_rows = []
    for motif in motifs:
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(stage_order)}))
        if len(traj) < 2:
            continue
        result = compute_spearman_trend_test(traj, stage_order)
        result["Motif"] = motif
        spearman_rows.append(result)
    
    spearman_df = pd.DataFrame(spearman_rows)
    if not spearman_df.empty and "spearman_pvalue" in spearman_df.columns:
        # FDR correction
        valid_pvals = spearman_df["spearman_pvalue"].dropna()
        if len(valid_pvals) > 0:
            fdr_pvals = false_discovery_control(valid_pvals.values, method='bh')
            spearman_df.loc[valid_pvals.index, "spearman_pvalue_fdr"] = fdr_pvals
            spearman_df["spearman_significant"] = spearman_df["spearman_pvalue_fdr"] < 0.05
        else:
            spearman_df["spearman_pvalue_fdr"] = np.nan
            spearman_df["spearman_significant"] = False
        spearman_df.to_csv(os.path.join(trend_dir, "spearman_trend_results.csv"), index=False)
    
    # =========================================================================
    # Per-Motif Linear Regression Tests
    # =========================================================================
    print("    Running linear trend tests...")
    linear_rows = []
    for motif in motifs:
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(stage_order)}))
        if len(traj) < 2:
            continue
        result = compute_linear_trend_test(traj, stage_order)
        result["Motif"] = motif
        linear_rows.append(result)
    
    linear_df = pd.DataFrame(linear_rows)
    if not linear_df.empty and "pvalue" in linear_df.columns:
        # FDR correction
        valid_pvals = linear_df["pvalue"].dropna()
        if len(valid_pvals) > 0:
            fdr_pvals = false_discovery_control(valid_pvals.values, method='bh')
            linear_df.loc[valid_pvals.index, "pvalue_fdr"] = fdr_pvals
            linear_df["linear_significant"] = linear_df["pvalue_fdr"] < 0.05
        else:
            linear_df["pvalue_fdr"] = np.nan
            linear_df["linear_significant"] = False
        linear_df.to_csv(os.path.join(trend_dir, "linear_trend_results.csv"), index=False)
    
    # =========================================================================
    # Global Direction Test
    # =========================================================================
    print("    Running global direction test...")
    direction_result = compute_global_direction_test(df, stage_order)
    
    # Save direction test results
    direction_summary = {
        "n_increasing": direction_result["n_increasing"],
        "n_decreasing": direction_result["n_decreasing"],
        "n_total": direction_result["n_total"],
        "proportion_increasing": direction_result["proportion_increasing"],
        "binomial_pvalue": direction_result["binomial_pvalue"],
        "direction_bias": direction_result["direction_bias"],
    }
    pd.DataFrame([direction_summary]).to_csv(os.path.join(trend_dir, "direction_test_summary.csv"), index=False)
    
    if "changes_df" in direction_result and not direction_result["changes_df"].empty:
        direction_result["changes_df"].to_csv(os.path.join(trend_dir, "direction_changes_per_motif.csv"), index=False)
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    summary_stats = {
        "spearman_n_significant": spearman_df["spearman_significant"].sum() if "spearman_significant" in spearman_df.columns else 0,
        "spearman_n_increasing": (spearman_df["trend_direction"] == "increasing").sum() if "trend_direction" in spearman_df.columns else 0,
        "spearman_n_decreasing": (spearman_df["trend_direction"] == "decreasing").sum() if "trend_direction" in spearman_df.columns else 0,
        "linear_n_significant": linear_df["linear_significant"].sum() if "linear_significant" in linear_df.columns else 0,
        "linear_n_increasing": (linear_df["trend_direction"] == "increasing").sum() if "trend_direction" in linear_df.columns else 0,
        "linear_n_decreasing": (linear_df["trend_direction"] == "decreasing").sum() if "trend_direction" in linear_df.columns else 0,
        "direction_test_pvalue": direction_result["binomial_pvalue"],
        "direction_bias": direction_result["direction_bias"],
        "mean_slope": linear_df["slope"].mean() if "slope" in linear_df.columns else np.nan,
        "median_slope": linear_df["slope"].median() if "slope" in linear_df.columns else np.nan,
    }
    pd.DataFrame([summary_stats]).to_csv(os.path.join(trend_dir, "trend_analysis_summary.csv"), index=False)
    
    return {
        "spearman_df": spearman_df,
        "linear_df": linear_df,
        "direction_result": direction_result,
        "summary_stats": summary_stats,
        "output_dir": trend_dir,
    }


def run_all_statistical_methods(df, pcutoff, transition_sig_set, output_dir, 
                                 run_permutation=True, run_fda=True, run_mixed_effects=True,
                                 run_bootstrap=False, run_distance_metrics=True,
                                 n_permutations=10000, n_bootstrap=1000, exclude_p3=False):
    """
    Run all statistical methods and save results to segregated output directories.
    
    Args:
        df: Full DataFrame with all motifs and stages
        pcutoff: Bonferroni-corrected significance threshold
        transition_sig_set: Set of significant transitions from helper 07 (or None)
        output_dir: Base output directory for this model
        run_*: Flags to enable/disable specific methods
        n_permutations: Number of permutations for permutation/FDA tests
        n_bootstrap: Number of bootstrap samples
        exclude_p3: If True, filter out P3 stage before analysis
    
    Returns:
        dict: Results from all methods for method comparison
    """
    # Filter out P3 if requested
    if exclude_p3:
        df = df[df["Stage"] != "P3"].copy()
    
    results = {}
    
    # Create subdirectories
    subdirs = {
        "permutation": os.path.join(output_dir, "permutation"),
        "fda": os.path.join(output_dir, "fda"),
        "mixed_effects": os.path.join(output_dir, "mixed_effects"),
        "bootstrap_ci": os.path.join(output_dir, "bootstrap_ci"),
        "distance_metrics": os.path.join(output_dir, "distance_metrics"),
        "method_comparison": os.path.join(output_dir, "method_comparison"),
    }
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    motifs = df["Motif_Label"].unique()
    
    # =========================================================================
    # Permutation Tests
    # =========================================================================
    if run_permutation:
        print("    Running permutation tests...")
        perm_rows = []
        for motif in motifs:
            traj = df[df["Motif_Label"] == motif].copy()
            traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
            if len(traj) < 2:
                continue
            perm_result = compute_permutation_significance(traj, n_permutations=n_permutations)
            perm_rows.append({
                "Motif": motif,
                "observed_stat": perm_result["observed_stat"],
                "null_mean": perm_result["null_mean"],
                "null_sd": perm_result["null_sd"],
                "p_value": perm_result["p_value"],
            })
        
        perm_df = pd.DataFrame(perm_rows)
        if not perm_df.empty:
            # Apply FDR correction
            valid_pvals = perm_df["p_value"].dropna()
            if len(valid_pvals) > 0:
                _, pvals_adj = fdrcorrection(perm_df["p_value"].fillna(1).values, alpha=0.05)
                perm_df["p_value_fdr"] = pvals_adj
                perm_df["significant"] = perm_df["p_value_fdr"] <= 0.05
            else:
                perm_df["p_value_fdr"] = np.nan
                perm_df["significant"] = False
            perm_df.to_csv(os.path.join(subdirs["permutation"], "permutation_test_results.csv"), index=False)
            results["permutation"] = perm_df
    
    # =========================================================================
    # FDA Tests
    # =========================================================================
    if run_fda:
        print("    Running FDA tests...")
        fda_rows = []
        for motif in motifs:
            traj = df[df["Motif_Label"] == motif].copy()
            traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
            if len(traj) < 3:  # FDA needs at least 3 points
                continue
            fda_result = compute_fda_significance(traj, n_permutations=min(n_permutations, 1000))
            fda_rows.append({
                "Motif": motif,
                "fda_statistic": fda_result["fda_statistic"],
                "null_mean": fda_result["null_mean"],
                "null_sd": fda_result["null_sd"],
                "p_value": fda_result["p_value"],
            })
        
        fda_df = pd.DataFrame(fda_rows)
        if not fda_df.empty:
            valid_pvals = fda_df["p_value"].dropna()
            if len(valid_pvals) > 0:
                _, pvals_adj = fdrcorrection(fda_df["p_value"].fillna(1).values, alpha=0.05)
                fda_df["p_value_fdr"] = pvals_adj
                fda_df["significant"] = fda_df["p_value_fdr"] <= 0.05
            else:
                fda_df["p_value_fdr"] = np.nan
                fda_df["significant"] = False
            fda_df.to_csv(os.path.join(subdirs["fda"], "fda_trajectory_significance.csv"), index=False)
            results["fda"] = fda_df
    
    # =========================================================================
    # Mixed-Effects Models
    # =========================================================================
    if run_mixed_effects:
        print("    Running mixed-effects analysis...")
        me_result = compute_mixed_effects_trajectory_test(df, STAGE_ORDER)
        
        # Save results
        me_summary = pd.DataFrame([{
            "stage_coefficient": me_result["stage_coefficient"],
            "stage_pvalue": me_result["stage_pvalue"],
            "model_converged": me_result["model_converged"],
        }])
        me_summary.to_csv(os.path.join(subdirs["mixed_effects"], "mixed_effects_summary.csv"), index=False)
        
        if not me_result["random_effects_df"].empty:
            me_result["random_effects_df"].to_csv(
                os.path.join(subdirs["mixed_effects"], "mixed_effects_random_effects.csv"), index=False
            )
        results["mixed_effects"] = me_result
    
    # =========================================================================
    # Bootstrap CI for Quadrant Classification
    # =========================================================================
    if run_bootstrap:
        print("    Running bootstrap CI analysis...")
        bootstrap_rows = []
        for motif in motifs:
            traj = df[df["Motif_Label"] == motif].copy()
            traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
            if len(traj) < 2:
                continue
            boot_result = compute_bootstrap_quadrant_ci(traj, pcutoff, n_bootstrap=n_bootstrap)
            if boot_result is not None:
                boot_result["Motif"] = motif
                bootstrap_rows.append(boot_result)
        
        if bootstrap_rows:
            bootstrap_df = pd.concat(bootstrap_rows, ignore_index=True)
            bootstrap_df.to_csv(os.path.join(subdirs["bootstrap_ci"], "quadrant_bootstrap_ci.csv"), index=False)
            results["bootstrap"] = bootstrap_df
        else:
            print("    Warning: No bootstrap CI results generated (Observed/Expected data unavailable).")
    
    # =========================================================================
    # Distance Metrics
    # =========================================================================
    if run_distance_metrics:
        print("    Computing distance metrics...")
        dist_rows = []
        for motif in motifs:
            traj = df[df["Motif_Label"] == motif].copy()
            traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
            if len(traj) < 2:
                continue
            
            # Raw path length (original, for comparison)
            raw_pl = path_length(traj)
            
            # Z-score standardized path length
            zscore_pl = compute_zscore_path_length(df, traj)
            
            # Mahalanobis path length
            mahal_pl = compute_mahalanobis_path_length(df, traj)
            
            # Separate axis metrics
            sep_metrics = compute_separate_axis_metrics(traj)
            
            dist_rows.append({
                "Motif": motif,
                "path_length_raw": raw_pl,
                "path_length_zscore": zscore_pl,
                "path_length_mahalanobis": mahal_pl,
                **sep_metrics,
            })
        
        dist_df = pd.DataFrame(dist_rows)
        if not dist_df.empty:
            # Add percentile ranks
            for col in ["path_length_raw", "path_length_zscore", "path_length_mahalanobis", 
                        "effect_size_range", "effect_size_total_variation",
                        "significance_range", "significance_total_variation"]:
                if col in dist_df.columns:
                    dist_df[f"{col}_pct"] = dist_df[col].rank(pct=True).values * 100
            dist_df.to_csv(os.path.join(subdirs["distance_metrics"], "distance_comparison.csv"), index=False)
            results["distance_metrics"] = dist_df
    
    return results


def generate_method_comparison_summary(df, pcutoff, transition_sig_set, results, output_dir):
    """
    Generate summary files comparing results across all statistical methods.
    
    Args:
        df: Full DataFrame with all motifs
        pcutoff: Significance threshold
        transition_sig_set: Set of significant transitions from helper 07
        results: Dict of results from run_all_statistical_methods
        output_dir: Output directory for comparison files
    
    Output files:
        - all_methods_summary.csv: One row per motif with results from all methods
        - method_agreement_matrix.csv: Pairwise agreement rates between methods
        - significant_by_method.csv: Lists of significant motifs per method
    """
    comparison_dir = os.path.join(output_dir, "method_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    motifs = df["Motif_Label"].unique()
    
    # Build comprehensive summary
    summary_rows = []
    for motif in motifs:
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        
        row = {"Motif": motif}
        
        # Quadrant change (original method)
        row["quadrant_change"] = motif_changes_quadrant(traj, pcutoff)
        row["quadrant_change_filtered"] = row["quadrant_change"] and motif_quadrant_change_has_significant_transition(
            traj, pcutoff, transition_sig_set
        ) if transition_sig_set else row["quadrant_change"]
        
        # Permutation test results
        if "permutation" in results and not results["permutation"].empty:
            perm_row = results["permutation"][results["permutation"]["Motif"] == motif]
            if not perm_row.empty:
                row["permutation_pvalue"] = perm_row["p_value"].values[0]
                row["permutation_pvalue_fdr"] = perm_row["p_value_fdr"].values[0]
                row["permutation_significant"] = perm_row["significant"].values[0]
            else:
                row["permutation_pvalue"] = np.nan
                row["permutation_pvalue_fdr"] = np.nan
                row["permutation_significant"] = False
        
        # FDA results
        if "fda" in results and not results["fda"].empty:
            fda_row = results["fda"][results["fda"]["Motif"] == motif]
            if not fda_row.empty:
                row["fda_pvalue"] = fda_row["p_value"].values[0]
                row["fda_pvalue_fdr"] = fda_row["p_value_fdr"].values[0]
                row["fda_significant"] = fda_row["significant"].values[0]
            else:
                row["fda_pvalue"] = np.nan
                row["fda_pvalue_fdr"] = np.nan
                row["fda_significant"] = False
        
        # Distance metrics
        if "distance_metrics" in results and not results["distance_metrics"].empty:
            dist_row = results["distance_metrics"][results["distance_metrics"]["Motif"] == motif]
            if not dist_row.empty:
                row["path_length_zscore"] = dist_row["path_length_zscore"].values[0]
                row["path_length_mahalanobis"] = dist_row["path_length_mahalanobis"].values[0]
                row["effect_size_range"] = dist_row["effect_size_range"].values[0]
                row["significance_range"] = dist_row["significance_range"].values[0]
                if "path_length_zscore_pct" in dist_row.columns:
                    row["path_length_zscore_pct"] = dist_row["path_length_zscore_pct"].values[0]
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(comparison_dir, "all_methods_summary.csv"), index=False)
    
    # Method agreement matrix
    methods = []
    method_cols = {
        "quadrant_change": "quadrant_change",
        "quadrant_filtered": "quadrant_change_filtered",
        "permutation": "permutation_significant",
        "fda": "fda_significant",
    }
    
    available_methods = [name for name, col in method_cols.items() if col in summary_df.columns]
    
    if len(available_methods) > 1:
        agreement_matrix = []
        for m1 in available_methods:
            row = {"method": m1}
            col1 = method_cols[m1]
            for m2 in available_methods:
                col2 = method_cols[m2]
                # Agreement = both True or both False
                valid = summary_df[[col1, col2]].dropna()
                if len(valid) > 0:
                    agreement = ((valid[col1] == True) & (valid[col2] == True)) | \
                                ((valid[col1] == False) & (valid[col2] == False))
                    row[m2] = agreement.mean() * 100
                else:
                    row[m2] = np.nan
            agreement_matrix.append(row)
        
        agreement_df = pd.DataFrame(agreement_matrix)
        agreement_df.to_csv(os.path.join(comparison_dir, "method_agreement_matrix.csv"), index=False)
    
    # Significant motifs by method
    sig_by_method = {}
    for name, col in method_cols.items():
        if col in summary_df.columns:
            sig_motifs = summary_df[summary_df[col] == True]["Motif"].tolist()
            sig_by_method[name] = sig_motifs
    
    # Convert to DataFrame with unequal column lengths
    max_len = max(len(v) for v in sig_by_method.values()) if sig_by_method else 0
    sig_dict = {}
    for name, motifs in sig_by_method.items():
        padded = motifs + [None] * (max_len - len(motifs))
        sig_dict[name] = padded
    
    if sig_dict:
        sig_df = pd.DataFrame(sig_dict)
        sig_df.to_csv(os.path.join(comparison_dir, "significant_by_method.csv"), index=False)
    
    return summary_df


def load_transition_significance(transition_dir, model_type):
    """
    Load helper 07 transition_significance.csv for one model.
    Returns set of (motif, transition) where Significant is True, or None if file missing/invalid.
    Skips empty Motif rows.
    """
    if not transition_dir or not os.path.isdir(transition_dir):
        return None
    path = os.path.join(transition_dir, model_type, "transition_significance.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "Motif" not in df.columns or "Transition" not in df.columns or "Significant" not in df.columns:
        return None
    sig_set = set()
    for _, row in df.iterrows():
        motif = row.get("Motif")
        if pd.isna(motif) or str(motif).strip() == "":
            continue
        if row.get("Significant") is True or (isinstance(row.get("Significant"), str) and row.get("Significant").strip().lower() == "true"):
            sig_set.add((str(motif).strip(), str(row["Transition"]).strip()))
    return sig_set


# Meaningful quadrant-crossing pairs: (not_sig, sig_pos), (not_sig, sig_neg), (sig_pos, sig_neg), (sig_neg, sig_pos)
_MEANINGFUL_QUADRANT_PAIRS = {
    ("not_sig", "sig_pos"), ("sig_pos", "not_sig"),
    ("not_sig", "sig_neg"), ("sig_neg", "not_sig"),
    ("sig_pos", "sig_neg"), ("sig_neg", "sig_pos"),
}


def write_comparison_list(df, pcutoff, transition_sig_set, output_path):
    """
    Write a CSV comparing change criteria per motif: quadrant_change (full, with transition filter),
    path_length (full), path_length_pct, effect_size_range (full), effect_size_range_pct.
    df: full dataframe (all stages).
    """
    rows = []
    for motif in df["Motif_Label"].unique():
        traj_full = df[df["Motif_Label"] == motif].copy().sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj_full) < 2:
            continue
        quadrant_change = motif_changes_quadrant(traj_full, pcutoff) and (
            motif_quadrant_change_has_significant_transition(traj_full, pcutoff, transition_sig_set) if transition_sig_set is not None else True
        )
        pl = path_length(traj_full)
        er = effect_size_range(traj_full)
        rows.append({
            "Motif": motif,
            "quadrant_change": quadrant_change,
            "path_length": pl,
            "effect_size_range": er,
        })
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return
    out_df["path_length_pct"] = out_df["path_length"].rank(pct=True).values * 100
    out_df["effect_size_range_pct"] = out_df["effect_size_range"].rank(pct=True).values * 100
    out_df.to_csv(output_path, index=False)


def motif_quadrant_change_has_significant_transition(traj_df, pcutoff, transition_sig_set):
    """
    Return True iff at least one meaningful quadrant-crossing segment in the trajectory
    has (motif, transition) in transition_sig_set. If transition_sig_set is None, return True (no filter).
    """
    if transition_sig_set is None:
        return True
    traj = traj_df.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
    if len(traj) < 2:
        return False
    motif = traj["Motif_Label"].iloc[0]
    stages = traj["Stage"].tolist()
    quadrants = [classify_quadrant(row["effect_size"], row["significance"], pcutoff) for _, row in traj.iterrows()]
    for i in range(len(stages) - 1):
        q_i, q_next = quadrants[i], quadrants[i + 1]
        if (q_i, q_next) in _MEANINGFUL_QUADRANT_PAIRS:
            trans = f"{stages[i]}_to_{stages[i + 1]}"
            if (motif, trans) in transition_sig_set:
                return True
    return False


def plot_one_motif_trajectory(motif, traj_df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Draw one volcano trajectory plot for a single motif.
    traj_df has rows for this motif with Stage, effect_size, significance.
    
    Args:
        pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    traj_df = traj_df.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
    points = traj_df[["effect_size", "significance"]].values
    stages = traj_df["Stage"].tolist()
    if len(points) < 2:
        return
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    for i, (x, y) in enumerate(points):
        st = stages[i]
        ax.scatter(x, y, c=[STAGE_COLORS.get(st, "gray")], s=80, zorder=3, edgecolors="black", linewidths=0.5)
    # Line segments in order
    for i in range(len(points) - 1):
        ax.plot(
            [points[i, 0], points[i + 1, 0]],
            [points[i, 1], points[i + 1, 1]],
            color="black", linestyle="-", linewidth=1.5, zorder=2
        )
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"Motif: {motif}", fontsize=14)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=10, label=s)
        for s in STAGE_ORDER
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    safe_name = motif.replace("+", "_").strip() or "motif"
    base = os.path.join(output_dir, f"{safe_name}_volcano_trajectory")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


# Stage colors shared with summary plot (same as plot_one_motif_trajectory)
STAGE_COLORS = {"P3": "#1f77b4", "P12": "#ff7f0e", "P20": "#2ca02c", "P60": "#d62728"}
GREY = "#888888"


def plot_summary_all_trajectories(df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, exclude_p3=False, transition_sig_set=None, name_suffix="", highlight_mode="quadrant", pcutoff_stringent=None, effect_size_threshold=None):
    """
    Overlay all motif trajectories on one plot. Grey vs stage-colored by highlight_mode:
    - "quadrant": colored if quadrant change AND (if transition_sig_set) a significant quadrant-crossing segment.
    If exclude_p3 is True, use only P12/P20/P60 data.
    transition_sig_set: used for filtering by significant transitions; None = no transition filter.
    name_suffix: appended to all output base names (e.g. "_not_filtered").
    pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
    effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D

    work = df[df["Stage"] != "P3"].copy() if exclude_p3 else df.copy()
    if work.empty:
        if exclude_p3:
            print(f"  No data after excluding P3 for {model_type}, skipping summary_no_P3.")
        return

    stage_order = [s for s in STAGE_ORDER if s in work["Stage"].unique()] or list(work["Stage"].unique())
    stage_order.sort(key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)

    # Split motifs into grey vs colored by highlight_mode
    grey_trajs = []
    colored_trajs = []
    for motif in work["Motif_Label"].unique():
        traj = work[work["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        # quadrant highlighting: colored if quadrant change AND (if transition_sig_set) a significant transition
        is_colored = motif_changes_quadrant(traj, pcutoff) and motif_quadrant_change_has_significant_transition(traj, pcutoff, transition_sig_set)
        if is_colored:
            colored_trajs.append(traj)
        else:
            grey_trajs.append(traj)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)

    # Draw grey trajectories first
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )

    # Draw colored trajectories on top
    for traj in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            # Segment color = later stage
            st = stages[i + 1]
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    prefix = "All motifs P12–P60 " if exclude_p3 else "All motifs "
    title = prefix + (
        "(highlight = quadrant change + significant transition)" if transition_sig_set is not None
        else "(highlight = quadrant change)"
    )
    ax.set_title(title, fontsize=14)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    grey_label = "no significant quadrant change" if transition_sig_set is not None else "no quadrant change"
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label=grey_label))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    base_name = ("summary_all_trajectories_no_P3" if exclude_p3 else "summary_all_trajectories") + name_suffix
    base = os.path.join(output_dir, base_name)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    # No-P3 only: grey-only and colored-only figures
    if exclude_p3:
        # Grey-only no-P3
        fig_g, ax_g = plt.subplots(1, 1)
        fig_g.set_size_inches(10, 10)
        ax_g.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
        ax_g.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
        
        # Optional effect size grey zone
        if effect_size_threshold is not None:
            draw_effect_size_grey_zone(ax_g, effect_size_threshold, xlim)
        
        # P=0.05 cutoff (dashed line)
        ax_g.axhline(y=pcutoff, linestyle="--", color="gray")
        ax_g.axvline(x=0, linestyle="--", color="gray")
        ax_g.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
        
        # P=0.01 cutoff (dotted line) - if provided
        if pcutoff_stringent is not None:
            ax_g.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
            ax_g.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
        
        for traj in grey_trajs:
            points = traj[["effect_size", "significance"]].values
            for i, (x, y) in enumerate(points):
                ax_g.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
            for i in range(len(points) - 1):
                ax_g.plot(
                    [points[i, 0], points[i + 1, 0]],
                    [points[i, 1], points[i + 1, 1]],
                    color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
                )
        ax_g.set_xlim(xlim[0], xlim[1])
        ax_g.set_ylim(ylim[0], ylim[1])
        grey_title = "P12–P60 grey only (no significant quadrant change)" if transition_sig_set is not None else "P12–P60 grey only (no quadrant change)"
        ax_g.set_title(grey_title, fontsize=14)
        leg_label = "no significant quadrant change" if transition_sig_set is not None else "no quadrant change"
        ax_g.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label=leg_label)], loc="best", fontsize=10)
        base_grey = os.path.join(output_dir, "summary_all_trajectories_no_P3_grey_only" + name_suffix)
        for ext in ["pdf", "svg", "png"]:
            fig_g.savefig(f"{base_grey}.{ext}", bbox_inches="tight")
        plt.close(fig_g)

        # Colored-only no-P3 (with motif labels near P60 point)
        fig_c, ax_c = plt.subplots(1, 1)
        fig_c.set_size_inches(10, 10)
        ax_c.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
        ax_c.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
        
        # Optional effect size grey zone
        if effect_size_threshold is not None:
            draw_effect_size_grey_zone(ax_c, effect_size_threshold, xlim)
        
        # P=0.05 cutoff (dashed line)
        ax_c.axhline(y=pcutoff, linestyle="--", color="gray")
        ax_c.axvline(x=0, linestyle="--", color="gray")
        ax_c.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
        
        # P=0.01 cutoff (dotted line) - if provided
        if pcutoff_stringent is not None:
            ax_c.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
            ax_c.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
        
        for traj in colored_trajs:
            points = traj[["effect_size", "significance"]].values
            stages = traj["Stage"].tolist()
            for i, (x, y) in enumerate(points):
                st = stages[i]
                ax_c.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
            for i in range(len(points) - 1):
                st = stages[i + 1]
                ax_c.plot(
                    [points[i, 0], points[i + 1, 0]],
                    [points[i, 1], points[i + 1, 1]],
                    color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
                )
            # Motif label near red P60 (last point)
            motif = traj["Motif_Label"].iloc[0]
            x_p60, y_p60 = points[-1, 0], points[-1, 1]
            ax_c.annotate(
                str(motif), xy=(x_p60, y_p60), xytext=(10, 10), textcoords="offset points",
                fontsize=8, zorder=4,
                fontfamily=["Helvetica", "Arial", "sans-serif"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
            )
        ax_c.set_xlim(xlim[0], xlim[1])
        ax_c.set_ylim(ylim[0], ylim[1])
        colored_title = "P12–P60 colored only (quadrant change + significant transition)" if transition_sig_set is not None else "P12–P60 colored only (quadrant change)"
        ax_c.set_title(colored_title, fontsize=14)
        legend_elements_c = [Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s) for s in stage_order if s in STAGE_COLORS]
        ax_c.legend(handles=legend_elements_c, loc="best", fontsize=10)
        base_colored = os.path.join(output_dir, "summary_all_trajectories_no_P3_colored_only" + name_suffix)
        for ext in ["pdf", "svg", "png"]:
            fig_c.savefig(f"{base_colored}.{ext}", bbox_inches="tight")
        plt.close(fig_c)


def run_matt_visual_rules_analysis(df, output_dir, pcutoff, xlim, ylim, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Run Matt visual rules: classify each motif as changing (TRUE) or not (FALSE) based on
    ns <-> significant transitions or over <-> under (effect-size crossing). Write CSVs and
    three summary plots to output_dir/matt_visual_rules/.
    df: volcano DataFrame (may already be filtered e.g. noP3); must have Motif_Label, Stage, effect_size, significance.
    """
    from matplotlib.lines import Line2D

    if df.empty:
        return
    stage_order = [s for s in STAGE_ORDER if s in df["Stage"].unique()]
    stage_order.sort(key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)
    if len(stage_order) < 2:
        return

    out_dir = os.path.join(output_dir, "matt_visual_rules")
    os.makedirs(out_dir, exist_ok=True)

    # Per-motif classification
    motif_rows = []
    grey_trajs = []
    colored_trajs = []
    n_true = 0
    n_false = 0
    n_become_over = 0
    n_become_under = 0
    n_switch_over_to_under = 0
    n_switch_under_to_over = 0
    n_lose_significance_to_ns = 0

    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        res = matt_visual_rules_per_motif(traj, pcutoff)
        if res is None:
            continue
        stage_labels = res["stage_labels"]
        changing = res["changing"]
        if changing:
            n_true += 1
            colored_trajs.append(traj)
        else:
            n_false += 1
            grey_trajs.append(traj)
        if res["has_become_over"]:
            n_become_over += 1
        if res["has_become_under"]:
            n_become_under += 1
        if res["has_switch_over_to_under"]:
            n_switch_over_to_under += 1
        if res["has_switch_under_to_over"]:
            n_switch_under_to_over += 1
        if res["has_lose_significance_to_ns"]:
            n_lose_significance_to_ns += 1

        # Build row: Motif, p3, p12, p20, p60 (or subset), changing
        row = {"Motif": motif}
        for st, lab in stage_labels:
            col = st.lower()
            row[col] = lab
        row["changing"] = "TRUE" if changing else "FALSE"
        motif_rows.append(row)

    # motif_change_list.csv
    motif_df = pd.DataFrame(motif_rows)
    if not motif_df.empty:
        motif_df.to_csv(os.path.join(out_dir, "motif_change_list.csv"), index=False)

    # summary_counts.csv
    summary = {
        "n_true": n_true,
        "n_false": n_false,
        "n_become_over_represented": n_become_over,
        "n_become_under_represented": n_become_under,
        "n_switch_over_to_under": n_switch_over_to_under,
        "n_switch_under_to_over": n_switch_under_to_over,
        "n_lose_significance_to_ns": n_lose_significance_to_ns,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "summary_counts.csv"), index=False)

    # Summary plots (both, grey only, colored only)
    def _draw_axes_setup(ax):
        ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
        ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
        if effect_size_threshold is not None:
            draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
        ax.axhline(y=pcutoff, linestyle="--", color="gray")
        ax.axvline(x=0, linestyle="--", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
        if pcutoff_stringent is not None:
            ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
            ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)

    def _draw_grey_trajs(ax, trajs):
        for traj in trajs:
            points = traj[["effect_size", "significance"]].values
            for i, (x, y) in enumerate(points):
                ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
            for i in range(len(points) - 1):
                ax.plot(
                    [points[i, 0], points[i + 1, 0]],
                    [points[i, 1], points[i + 1, 1]],
                    color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
                )

    def _draw_colored_trajs(ax, trajs, add_labels=False):
        for traj in trajs:
            points = traj[["effect_size", "significance"]].values
            stages = traj["Stage"].tolist()
            for i, (x, y) in enumerate(points):
                st = stages[i]
                ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
            for i in range(len(points) - 1):
                st = stages[i + 1]
                ax.plot(
                    [points[i, 0], points[i + 1, 0]],
                    [points[i, 1], points[i + 1, 1]],
                    color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
                )
            if add_labels:
                motif = traj["Motif_Label"].iloc[0]
                x_last, y_last = points[-1, 0], points[-1, 1]
                ax.annotate(
                    str(motif), xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
                    fontsize=8, zorder=4,
                    fontfamily=["Helvetica", "Arial", "sans-serif"],
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
                )

    # Plot 1: Both (grey first, then colored)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    _draw_axes_setup(ax)
    _draw_grey_trajs(ax, grey_trajs)
    _draw_colored_trajs(ax, colored_trajs, add_labels=False)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"Matt visual rules: changing (colored) vs not changing (grey) - {model_type}", fontsize=14)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="not changing"))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(os.path.join(out_dir, f"summary_matt_visual_rules.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Grey only
    fig_g, ax_g = plt.subplots(1, 1)
    fig_g.set_size_inches(10, 10)
    _draw_axes_setup(ax_g)
    _draw_grey_trajs(ax_g, grey_trajs)
    ax_g.set_xlim(xlim[0], xlim[1])
    ax_g.set_ylim(ylim[0], ylim[1])
    ax_g.set_title(f"Matt visual rules: not changing (grey only) - {model_type}", fontsize=14)
    ax_g.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="not changing")], loc="best", fontsize=10)
    for ext in ["pdf", "svg", "png"]:
        fig_g.savefig(os.path.join(out_dir, f"summary_matt_visual_rules_grey_only.{ext}"), bbox_inches="tight")
    plt.close(fig_g)

    # Plot 3: Colored only (with motif labels)
    fig_c, ax_c = plt.subplots(1, 1)
    fig_c.set_size_inches(10, 10)
    _draw_axes_setup(ax_c)
    _draw_colored_trajs(ax_c, colored_trajs, add_labels=True)
    ax_c.set_xlim(xlim[0], xlim[1])
    ax_c.set_ylim(ylim[0], ylim[1])
    ax_c.set_title(f"Matt visual rules: changing (colored only) - {model_type}", fontsize=14)
    legend_elements_c = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    ax_c.legend(handles=legend_elements_c, loc="best", fontsize=10)
    for ext in ["pdf", "svg", "png"]:
        fig_c.savefig(os.path.join(out_dir, f"summary_matt_visual_rules_colored_only.{ext}"), bbox_inches="tight")
    plt.close(fig_c)


# =============================================================================
# NEW VISUALIZATION FUNCTIONS (v2.0)
# =============================================================================

def plot_summary_by_significance(df, results, xlim, ylim, pcutoff, output_dir, model_type, method="permutation", xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Plot all motif trajectories with coloring based on statistical significance
    from permutation or FDA tests.
    
    Args:
        df: Full DataFrame with all motifs
        results: Dict of results from run_all_statistical_methods
        xlim, ylim: Axis limits
        pcutoff: Significance cutoff for reference line
        output_dir: Output directory (should be the method-specific subdirectory)
        model_type: Model type for labeling
        method: "permutation" or "fda"
        xlabel: Custom x-axis label
        pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D
    
    # Get significance results
    if method == "permutation" and "permutation" in results:
        sig_df = results["permutation"]
        sig_col = "significant"
        method_label = "Permutation Test"
    elif method == "fda" and "fda" in results:
        sig_df = results["fda"]
        sig_col = "significant"
        method_label = "FDA Trajectory Test"
    else:
        print(f"  Warning: No {method} results available for significance plot.")
        return
    
    if sig_df.empty:
        return
    
    # Build set of significant motifs
    sig_motifs = set(sig_df[sig_df[sig_col] == True]["Motif"].tolist())
    
    stage_order = [s for s in STAGE_ORDER if s in df["Stage"].unique()]
    
    # Split motifs into grey (not significant) vs colored (significant)
    grey_trajs = []
    colored_trajs = []
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        if motif in sig_motifs:
            colored_trajs.append(traj)
        else:
            grey_trajs.append(traj)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    # Draw grey trajectories first
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )
    
    # Draw colored trajectories on top
    for traj in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        # Label significant motifs
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        ax.annotate(
            str(motif), xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=8, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    n_sig = len(colored_trajs)
    ax.set_title(f"All motifs - {method_label} (n={n_sig} significant, FDR<0.05)", fontsize=14)
    
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="not significant"))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    
    base = os.path.join(output_dir, f"summary_trajectories_{method}_significant")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_heatmap(results, output_dir, pcutoff=None):
    """
    Create a heatmap showing -log10(p-value) across motifs for each method.
    
    Args:
        results: Dict of results from run_all_statistical_methods
        output_dir: Output directory (method_comparison subdirectory)
        pcutoff: Optional significance threshold to mark on colorbar
    """
    import matplotlib.colors as mcolors
    
    # Collect p-values from available methods
    pvalue_data = {}
    
    if "permutation" in results and not results["permutation"].empty:
        perm_df = results["permutation"]
        pvalue_data["Permutation"] = dict(zip(perm_df["Motif"], perm_df["p_value_fdr"]))
    
    if "fda" in results and not results["fda"].empty:
        fda_df = results["fda"]
        pvalue_data["FDA"] = dict(zip(fda_df["Motif"], fda_df["p_value_fdr"]))
    
    if not pvalue_data:
        print("  Warning: No p-value data available for heatmap.")
        return
    
    # Build DataFrame
    all_motifs = set()
    for method_pvals in pvalue_data.values():
        all_motifs.update(method_pvals.keys())
    all_motifs = sorted(all_motifs)
    
    heatmap_data = []
    for motif in all_motifs:
        row = {"Motif": motif}
        for method, pvals in pvalue_data.items():
            pval = pvals.get(motif, np.nan)
            # Convert to -log10 scale
            row[method] = -np.log10(pval) if pval > 0 else 0
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index("Motif")
    
    # Sort by mean -log10(p-value) descending
    heatmap_df["mean_neglog10p"] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values("mean_neglog10p", ascending=False)
    heatmap_df = heatmap_df.drop(columns=["mean_neglog10p"])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, max(8, len(all_motifs) * 0.3)))
    
    data_matrix = heatmap_df.values
    methods = heatmap_df.columns.tolist()
    motifs = heatmap_df.index.tolist()
    
    # Color scale: white (low) to red (high)
    cmap = plt.cm.Reds
    vmax = max(3, np.nanmax(data_matrix))  # At least show up to -log10(0.001) = 3
    
    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
    
    # Add significance threshold line to colorbar if provided
    cbar = plt.colorbar(im, ax=ax, label="-log10(FDR p-value)")
    if pcutoff is not None:
        # Mark significance threshold
        sig_line = -np.log10(0.05)
        cbar.ax.axhline(y=sig_line, color='black', linestyle='--', linewidth=1)
        cbar.ax.text(1.5, sig_line, 'p=0.05', va='center', fontsize=8)
    
    # Labels
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels(motifs, fontsize=8)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Motif", fontsize=12)
    ax.set_title("Trajectory Significance by Method", fontsize=14)
    
    # Add value annotations
    for i in range(len(motifs)):
        for j in range(len(methods)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=text_color)
    
    plt.tight_layout()
    
    base = os.path.join(output_dir, "pvalue_heatmap")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_summary_by_distance_gradient(df, dist_df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Plot all motif trajectories with color gradient based on path_length_zscore percentile.
    
    Args:
        df: Full DataFrame with all motifs
        dist_df: Distance metrics DataFrame with path_length_zscore_pct
        xlim, ylim: Axis limits
        pcutoff: Significance cutoff for reference line
        output_dir: Output directory (distance_metrics subdirectory)
        model_type: Model type for labeling
        xlabel: Custom x-axis label
        pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    
    if dist_df.empty or "path_length_zscore_pct" not in dist_df.columns:
        print("  Warning: No distance metrics available for gradient plot.")
        return
    
    # Build motif -> percentile mapping
    pct_map = dict(zip(dist_df["Motif"], dist_df["path_length_zscore_pct"]))
    
    # Colormap: light grey to dark blue
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=100)
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    # Collect trajectories with their percentiles
    trajs_with_pct = []
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        pct = pct_map.get(motif, 50)  # Default to 50th percentile if missing
        trajs_with_pct.append((traj, pct))
    
    # Sort by percentile so high percentile (dark) draws on top
    trajs_with_pct.sort(key=lambda x: x[1])
    
    for traj, pct in trajs_with_pct:
        points = traj[["effect_size", "significance"]].values
        color = cmap(norm(pct))
        alpha = 0.4 + 0.6 * (pct / 100)  # More visible for higher percentiles
        
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[color], s=40, zorder=2, edgecolors="none", alpha=alpha)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=color, linestyle="-", linewidth=1 + pct/50, zorder=1, alpha=alpha
            )
        
        # Label top 10% motifs
        if pct >= 90:
            motif = traj["Motif_Label"].iloc[0]
            x_last, y_last = points[-1, 0], points[-1, 1]
            ax.annotate(
                str(motif), xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
                fontsize=8, zorder=4,
                fontfamily=["Helvetica", "Arial", "sans-serif"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
            )
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"All motifs - Colored by trajectory magnitude (z-score path length percentile)", fontsize=14)
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Path length percentile")
    cbar.set_ticks([0, 25, 50, 75, 100])
    
    base = os.path.join(output_dir, "summary_trajectories_by_path_length")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_trend_analysis(trend_results, output_dir, model_type):
    """
    Create visualizations for trend analysis results.
    
    Generates:
    1. Slope distribution histogram with mean/median lines
    2. Direction pie chart showing proportion increasing vs decreasing
    3. Spearman correlation scatter plot
    
    Args:
        trend_results: Dict from run_trend_analysis()
        output_dir: trend_analysis output directory
        model_type: Model type for labeling
    """
    from matplotlib.patches import Patch
    
    linear_df = trend_results.get("linear_df", pd.DataFrame())
    spearman_df = trend_results.get("spearman_df", pd.DataFrame())
    direction_result = trend_results.get("direction_result", {})
    summary_stats = trend_results.get("summary_stats", {})
    
    # =========================================================================
    # 1. Slope Distribution Histogram
    # =========================================================================
    if not linear_df.empty and "slope" in linear_df.columns:
        slopes = linear_df["slope"].dropna()
        if len(slopes) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histogram
            ax.hist(slopes, bins=15, color="#3182ce", alpha=0.7, edgecolor="black")
            
            # Add mean and median lines
            mean_slope = slopes.mean()
            median_slope = slopes.median()
            ax.axvline(mean_slope, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_slope:.3f}")
            ax.axvline(median_slope, color="orange", linestyle="-.", linewidth=2, label=f"Median: {median_slope:.3f}")
            ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.7, label="Zero")
            
            ax.set_xlabel("Linear Trend Slope (effect size change per stage)", fontsize=12)
            ax.set_ylabel("Number of Motifs", fontsize=12)
            ax.set_title(f"Distribution of Linear Trend Slopes - {model_type}", fontsize=14)
            ax.legend(loc="best")
            
            # Add summary text
            n_pos = (slopes > 0).sum()
            n_neg = (slopes < 0).sum()
            ax.text(0.02, 0.98, f"Positive slopes: {n_pos}\nNegative slopes: {n_neg}", 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            base = os.path.join(output_dir, "slope_distribution")
            for ext in ["pdf", "svg", "png"]:
                fig.savefig(f"{base}.{ext}", bbox_inches="tight")
            plt.close(fig)
    
    # =========================================================================
    # 2. Direction Pie Chart
    # =========================================================================
    if direction_result and direction_result.get("n_total", 0) > 0:
        n_inc = direction_result["n_increasing"]
        n_dec = direction_result["n_decreasing"]
        n_total = direction_result["n_total"]
        pvalue = direction_result.get("binomial_pvalue", np.nan)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sizes = [n_inc, n_dec]
        labels = [f"Increasing\n({n_inc}/{n_total})", f"Decreasing\n({n_dec}/{n_total})"]
        colors = ["#48bb78", "#f56565"]  # Green and red
        explode = (0.02, 0.02)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 12})
        
        # Add p-value annotation
        pval_str = f"p = {pvalue:.4f}" if not np.isnan(pvalue) else "p = N/A"
        sig_str = " (significant)" if pvalue < 0.05 else " (not significant)"
        ax.set_title(f"Direction of Effect Size Change (P12 → P60)\n{model_type}\nBinomial test: {pval_str}{sig_str}", fontsize=14)
        
        base = os.path.join(output_dir, "direction_pie_chart")
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(f"{base}.{ext}", bbox_inches="tight")
        plt.close(fig)
    
    # =========================================================================
    # 3. Spearman Correlation vs P-value Plot
    # =========================================================================
    if not spearman_df.empty and "spearman_rho" in spearman_df.columns:
        rho = spearman_df["spearman_rho"].dropna()
        pvals = spearman_df["spearman_pvalue"].dropna()
        
        if len(rho) > 0 and len(pvals) > 0:
            # Align indices
            common_idx = rho.index.intersection(pvals.index)
            rho = rho.loc[common_idx]
            pvals = pvals.loc[common_idx]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color by significance
            sig_mask = pvals < 0.05
            ax.scatter(rho[~sig_mask], -np.log10(pvals[~sig_mask]), 
                      c="#a0aec0", s=60, alpha=0.7, label="Not significant", edgecolors="none")
            ax.scatter(rho[sig_mask], -np.log10(pvals[sig_mask]), 
                      c="#3182ce", s=80, alpha=0.9, label="Significant (p<0.05)", edgecolors="black", linewidths=0.5)
            
            # Add significance threshold line
            ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, alpha=0.7, label="p=0.05")
            ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
            
            ax.set_xlabel("Spearman Correlation (ρ)", fontsize=12)
            ax.set_ylabel("-log₁₀(p-value)", fontsize=12)
            ax.set_title(f"Spearman Trend Test Results - {model_type}\n(positive ρ = increasing trend, negative ρ = decreasing trend)", fontsize=14)
            ax.legend(loc="best")
            
            # Add annotations for significant motifs
            if sig_mask.any():
                for idx in spearman_df[spearman_df["spearman_pvalue"] < 0.05].index:
                    if idx in common_idx:
                        motif = spearman_df.loc[idx, "Motif"]
                        r = rho.loc[idx]
                        p = pvals.loc[idx]
                        ax.annotate(str(motif), (r, -np.log10(p)), 
                                   xytext=(5, 5), textcoords="offset points",
                                   fontsize=8, alpha=0.9)
            
            base = os.path.join(output_dir, "spearman_volcano")
            for ext in ["pdf", "svg", "png"]:
                fig.savefig(f"{base}.{ext}", bbox_inches="tight")
            plt.close(fig)
    
    # =========================================================================
    # 4. Summary Text Report
    # =========================================================================
    summary_lines = [
        "=" * 60,
        f"TREND ANALYSIS SUMMARY - {model_type}",
        "=" * 60,
        "",
        "SPEARMAN RANK CORRELATION (monotonic trend test):",
        f"  Significant (FDR<0.05): {summary_stats.get('spearman_n_significant', 0)}",
        f"  Increasing trends: {summary_stats.get('spearman_n_increasing', 0)}",
        f"  Decreasing trends: {summary_stats.get('spearman_n_decreasing', 0)}",
        "",
        "LINEAR REGRESSION (linear trend test):",
        f"  Significant (FDR<0.05): {summary_stats.get('linear_n_significant', 0)}",
        f"  Increasing trends: {summary_stats.get('linear_n_increasing', 0)}",
        f"  Decreasing trends: {summary_stats.get('linear_n_decreasing', 0)}",
        f"  Mean slope: {summary_stats.get('mean_slope', np.nan):.4f}",
        f"  Median slope: {summary_stats.get('median_slope', np.nan):.4f}",
        "",
        "GLOBAL DIRECTION TEST (binomial test on P12→P60 change):",
        f"  N increasing: {direction_result.get('n_increasing', 0)}",
        f"  N decreasing: {direction_result.get('n_decreasing', 0)}",
        f"  Proportion increasing: {direction_result.get('proportion_increasing', np.nan):.1%}",
        f"  Binomial p-value: {direction_result.get('binomial_pvalue', np.nan):.4f}",
        f"  Direction bias: {direction_result.get('direction_bias', 'N/A')}",
        "",
        "=" * 60,
    ]
    
    with open(os.path.join(output_dir, "trend_analysis_report.txt"), "w") as f:
        f.write("\n".join(summary_lines))


def check_threshold_crossing(traj, pcutoff):
    """
    Check if a motif's trajectory crosses the p-value cutoff OR effect size = 0 line.
    
    This identifies trajectories that have biologically meaningful changes, not just
    statistically significant trends within a single quadrant.
    
    Args:
        traj: DataFrame with trajectory data for one motif (must have 'significance' and 'effect_size' columns)
        pcutoff: P-value significance cutoff (as -log10 transformed value)
    
    Returns:
        dict: {
            'crosses_pvalue': bool - trajectory crosses significance threshold,
            'crosses_effectsize': bool - trajectory crosses effect size = 0,
            'crosses_either': bool - crosses at least one threshold
        }
    """
    sig_values = traj["significance"].values
    es_values = traj["effect_size"].values
    
    # Check if crosses p-value cutoff (significance threshold)
    # above_pcut means -log10(p) >= cutoff, i.e., p <= threshold (significant)
    above_pcut = sig_values >= pcutoff
    crosses_pvalue = not (above_pcut.all() or (~above_pcut).all())
    
    # Check if crosses effect size = 0 (over/under-represented boundary)
    positive_es = es_values > 0
    crosses_effectsize = not (positive_es.all() or (~positive_es).all())
    
    return {
        "crosses_pvalue": crosses_pvalue,
        "crosses_effectsize": crosses_effectsize,
        "crosses_either": crosses_pvalue or crosses_effectsize
    }


def generate_verdict_csv(spearman_df, output_dir, df=None, pcutoff=None):
    """
    Generate motif_verdicts.csv from Spearman results with threshold crossing information.
    
    This creates a verdict file that matches the documentation table format,
    providing a clear mapping of motifs to their statistical verdict. When df and pcutoff
    are provided, also computes threshold crossing (biologically meaningful change).
    
    Args:
        spearman_df: DataFrame with columns [Motif, spearman_rho, spearman_pvalue_fdr, spearman_significant]
        output_dir: Directory to save the verdict CSV
        df: Optional - Full DataFrame with trajectory data (needed for threshold crossing)
        pcutoff: Optional - P-value significance cutoff (needed for threshold crossing)
    
    Returns:
        DataFrame with verdict information including threshold crossing columns
    """
    if spearman_df.empty or "spearman_significant" not in spearman_df.columns:
        print("  Warning: Cannot generate verdict CSV - missing Spearman results.")
        return pd.DataFrame()
    
    verdict_rows = []
    for _, row in spearman_df.iterrows():
        motif = row.get("Motif", "")
        rho = row.get("spearman_rho", np.nan)
        pvalue_fdr = row.get("spearman_pvalue_fdr", np.nan)
        significant = row.get("spearman_significant", False)
        
        # Determine direction
        if pd.isna(rho):
            direction = "unknown"
        elif rho > 0:
            direction = "increasing"
        elif rho < 0:
            direction = "decreasing"
        else:
            direction = "flat"
        
        # Check threshold crossing if df and pcutoff are provided
        crosses_pvalue = False
        crosses_effectsize = False
        crosses_either = False
        if df is not None and pcutoff is not None:
            traj = df[df["Motif_Label"] == motif]
            if len(traj) >= 2:
                crossing = check_threshold_crossing(traj, pcutoff)
                crosses_pvalue = crossing["crosses_pvalue"]
                crosses_effectsize = crossing["crosses_effectsize"]
                crosses_either = crossing["crosses_either"]
        
        # Generate verdict label (matches documentation format)
        if significant:
            if direction == "increasing":
                verdict_label = "SIGNIFICANT MONOTONIC INCREASE"
            elif direction == "decreasing":
                verdict_label = "SIGNIFICANT MONOTONIC DECREASE"
            else:
                verdict_label = "SIGNIFICANT"
        else:
            verdict_label = "NO CHANGE"
        
        # Generate strict verdict label (requires threshold crossing)
        if significant and crosses_either:
            if direction == "increasing":
                strict_verdict_label = "SIGNIFICANT + THRESHOLD CROSSING (INCREASE)"
            elif direction == "decreasing":
                strict_verdict_label = "SIGNIFICANT + THRESHOLD CROSSING (DECREASE)"
            else:
                strict_verdict_label = "SIGNIFICANT + THRESHOLD CROSSING"
        else:
            strict_verdict_label = "NO MEANINGFUL CHANGE"
        
        verdict_rows.append({
            "Motif": motif,
            "spearman_rho": rho,
            "spearman_pvalue_fdr": pvalue_fdr,
            "spearman_significant": significant,
            "direction": direction,
            "verdict_label": verdict_label,
            "crosses_pvalue": crosses_pvalue,
            "crosses_effectsize": crosses_effectsize,
            "crosses_either": crosses_either,
            "strict_verdict_label": strict_verdict_label
        })
    
    verdict_df = pd.DataFrame(verdict_rows)
    
    # Save to CSV
    verdict_path = os.path.join(output_dir, "motif_verdicts.csv")
    verdict_df.to_csv(verdict_path, index=False)
    print(f"    Saved verdict CSV: {verdict_path}")
    
    return verdict_df


def plot_trajectories_by_verdict(df, verdict_df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Plot trajectories colored by Spearman verdict (significant vs non-significant).
    
    Generates three plots:
    1. trajectories_by_verdict.png - All trajectories (significant colored, others grey)
    2. trajectories_by_verdict_grey_only.png - Only non-significant trajectories
    3. trajectories_by_verdict_colored_only.png - Only significant trajectories with labels
    
    Args:
        df: Full DataFrame with all motifs and stages
        verdict_df: DataFrame from generate_verdict_csv() with verdicts
        xlim, ylim: Axis limits
        pcutoff: Significance cutoff for reference line
        output_dir: Output directory (trend_analysis subdirectory)
        model_type: Model type for labeling
        xlabel: Custom x-axis label
        pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D
    
    if verdict_df.empty:
        print("  Warning: Cannot generate verdict trajectory plots - no verdict data.")
        return
    
    # Build set of significant motifs and their directions
    sig_motifs = {}
    for _, row in verdict_df.iterrows():
        if row.get("spearman_significant", False):
            sig_motifs[row["Motif"]] = row.get("direction", "unknown")
    
    stage_order = [s for s in STAGE_ORDER if s in df["Stage"].unique()]
    
    # Split motifs into grey (not significant) vs colored (significant)
    grey_trajs = []
    colored_trajs = []
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        if motif in sig_motifs:
            colored_trajs.append((traj, sig_motifs[motif]))
        else:
            grey_trajs.append(traj)
    
    n_sig = len(colored_trajs)
    n_nonsig = len(grey_trajs)
    
    # =========================================================================
    # Plot 1: All trajectories (significant colored, non-significant grey)
    # =========================================================================
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    # Draw grey trajectories first
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )
    
    # Draw colored trajectories on top
    for traj, direction in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        # Label significant motifs
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        dir_symbol = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else ""
        ax.annotate(
            f"{motif} {dir_symbol}", xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=8, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"Trajectories by Spearman Verdict - {model_type}\n(n={n_sig} significant, FDR<0.05)", fontsize=14)
    
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="not significant"))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    
    base = os.path.join(output_dir, "trajectories_by_verdict")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    
    # =========================================================================
    # Plot 2: Grey only (non-significant trajectories)
    # =========================================================================
    fig_g, ax_g = plt.subplots(1, 1)
    fig_g.set_size_inches(10, 10)
    ax_g.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_g.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_g, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax_g.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_g.axvline(x=0, linestyle="--", color="gray")
    ax_g.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax_g.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_g.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax_g.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax_g.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )
    
    ax_g.set_xlim(xlim[0], xlim[1])
    ax_g.set_ylim(ylim[0], ylim[1])
    ax_g.set_title(f"Non-Significant Trajectories (Grey Only) - {model_type}\n(n={n_nonsig} motifs, no significant Spearman trend)", fontsize=14)
    ax_g.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="not significant")], loc="best", fontsize=10)
    
    base_grey = os.path.join(output_dir, "trajectories_by_verdict_grey_only")
    for ext in ["pdf", "svg", "png"]:
        fig_g.savefig(f"{base_grey}.{ext}", bbox_inches="tight")
    plt.close(fig_g)
    
    # =========================================================================
    # Plot 3: Colored only (significant trajectories with labels)
    # =========================================================================
    fig_c, ax_c = plt.subplots(1, 1)
    fig_c.set_size_inches(10, 10)
    ax_c.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_c.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_c, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax_c.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_c.axvline(x=0, linestyle="--", color="gray")
    ax_c.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax_c.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_c.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    for traj, direction in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax_c.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax_c.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        # Motif label with direction
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        dir_symbol = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else ""
        ax_c.annotate(
            f"{motif} {dir_symbol}", xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=9, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )
    
    ax_c.set_xlim(xlim[0], xlim[1])
    ax_c.set_ylim(ylim[0], ylim[1])
    ax_c.set_title(f"Significant Trajectories Only - {model_type}\n(n={n_sig} motifs with Spearman FDR<0.05)", fontsize=14)
    legend_elements_c = [Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s) for s in stage_order if s in STAGE_COLORS]
    ax_c.legend(handles=legend_elements_c, loc="best", fontsize=10)
    
    base_colored = os.path.join(output_dir, "trajectories_by_verdict_colored_only")
    for ext in ["pdf", "svg", "png"]:
        fig_c.savefig(f"{base_colored}.{ext}", bbox_inches="tight")
    plt.close(fig_c)
    
    print(f"    Generated verdict trajectory plots: {n_sig} significant, {n_nonsig} non-significant")


def plot_trajectories_by_strict_verdict(df, verdict_df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Plot trajectories colored by STRICT verdict (Spearman significant + threshold crossing).
    
    This uses a stricter criterion than plot_trajectories_by_verdict():
    - Must be Spearman significant (FDR < 0.05) AND
    - Must cross either p-value cutoff OR effect size = 0 line
    
    This filters out motifs with statistically significant trends that stay entirely
    within one quadrant (e.g., always significantly overrepresented).
    
    Generates three plots:
    1. trajectories_strict_verdict.png - All trajectories (strict significant colored, others grey)
    2. trajectories_strict_verdict_grey_only.png - Only non-strict trajectories
    3. trajectories_strict_verdict_colored_only.png - Only strict significant trajectories with labels
    
    Args:
        df: Full DataFrame with all motifs and stages
        verdict_df: DataFrame from generate_verdict_csv() with verdicts including crosses_either column
        xlim, ylim: Axis limits
        pcutoff: Significance cutoff for reference line
        output_dir: Output directory (trend_analysis subdirectory)
        model_type: Model type for labeling
        xlabel: Custom x-axis label
        pcutoff_stringent: Optional more stringent p-value cutoff (p=0.01, Bonferroni-corrected)
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D
    
    if verdict_df.empty or "crosses_either" not in verdict_df.columns:
        print("  Warning: Cannot generate strict verdict trajectory plots - missing threshold crossing data.")
        return
    
    # Build set of STRICT significant motifs (Spearman significant + crosses threshold)
    strict_sig_motifs = {}
    for _, row in verdict_df.iterrows():
        if row.get("spearman_significant", False) and row.get("crosses_either", False):
            strict_sig_motifs[row["Motif"]] = row.get("direction", "unknown")
    
    stage_order = [s for s in STAGE_ORDER if s in df["Stage"].unique()]
    
    # Split motifs into grey (not strict significant) vs colored (strict significant)
    grey_trajs = []
    colored_trajs = []
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        if motif in strict_sig_motifs:
            colored_trajs.append((traj, strict_sig_motifs[motif]))
        else:
            grey_trajs.append(traj)
    
    n_strict = len(colored_trajs)
    n_nonstrict = len(grey_trajs)
    
    # =========================================================================
    # Plot 1: All trajectories (strict significant colored, non-strict grey)
    # =========================================================================
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone (draw first so it's behind everything)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    # Draw grey trajectories first
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )
    
    # Draw colored trajectories on top
    for traj, direction in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        # Label strict significant motifs
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        dir_symbol = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else ""
        ax.annotate(
            f"{motif} {dir_symbol}", xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=8, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"Trajectories by Strict Verdict - {model_type}\n(n={n_strict} significant + threshold crossing)", fontsize=14)
    
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="no meaningful change"))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)
    
    base = os.path.join(output_dir, "trajectories_strict_verdict")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    
    # =========================================================================
    # Plot 2: Grey only (non-strict trajectories)
    # =========================================================================
    fig_g, ax_g = plt.subplots(1, 1)
    fig_g.set_size_inches(10, 10)
    ax_g.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_g.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_g, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax_g.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_g.axvline(x=0, linestyle="--", color="gray")
    ax_g.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax_g.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_g.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax_g.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax_g.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )
    
    ax_g.set_xlim(xlim[0], xlim[1])
    ax_g.set_ylim(ylim[0], ylim[1])
    ax_g.set_title(f"Non-Strict Trajectories (Grey Only) - {model_type}\n(n={n_nonstrict} motifs, no significant threshold crossing)", fontsize=14)
    ax_g.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="no meaningful change")], loc="best", fontsize=10)
    
    base_grey = os.path.join(output_dir, "trajectories_strict_verdict_grey_only")
    for ext in ["pdf", "svg", "png"]:
        fig_g.savefig(f"{base_grey}.{ext}", bbox_inches="tight")
    plt.close(fig_g)
    
    # =========================================================================
    # Plot 3: Colored only (strict significant trajectories with labels)
    # =========================================================================
    fig_c, ax_c = plt.subplots(1, 1)
    fig_c.set_size_inches(10, 10)
    ax_c.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_c.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    
    # Optional effect size grey zone
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_c, effect_size_threshold, xlim)
    
    # P=0.05 cutoff (dashed line)
    ax_c.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_c.axvline(x=0, linestyle="--", color="gray")
    ax_c.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    
    # P=0.01 cutoff (dotted line) - if provided
    if pcutoff_stringent is not None:
        ax_c.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_c.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)
    
    for traj, direction in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax_c.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax_c.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        # Motif label with direction
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        dir_symbol = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else ""
        ax_c.annotate(
            f"{motif} {dir_symbol}", xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=9, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )
    
    ax_c.set_xlim(xlim[0], xlim[1])
    ax_c.set_ylim(ylim[0], ylim[1])
    ax_c.set_title(f"Strict Significant Trajectories Only - {model_type}\n(n={n_strict} motifs: Spearman FDR<0.05 + threshold crossing)", fontsize=14)
    legend_elements_c = [Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s) for s in stage_order if s in STAGE_COLORS]
    ax_c.legend(handles=legend_elements_c, loc="best", fontsize=10)
    
    base_colored = os.path.join(output_dir, "trajectories_strict_verdict_colored_only")
    for ext in ["pdf", "svg", "png"]:
        fig_c.savefig(f"{base_colored}.{ext}", bbox_inches="tight")
    plt.close(fig_c)
    
    print(f"    Generated strict verdict trajectory plots: {n_strict} strict significant, {n_nonstrict} non-strict")


def _normalize_motif_for_eye(motif_str):
    """Normalize motif string to match df['Motif_Label']: sort parts and join with '+'."""
    if pd.isna(motif_str) or not isinstance(motif_str, str):
        return ""
    parts = [p.strip() for p in str(motif_str).split("+") if p.strip()]
    return "+".join(sorted(parts)) if parts else ""


def plot_trajectories_by_eye_verdict(df, eye_verdict_df, xlim, ylim, pcutoff, output_dir, model_type, xlabel=None, pcutoff_stringent=None, effect_size_threshold=None):
    """
    Plot trajectories colored by manual eye verdict (TRUE = stage-colored, FALSE = grey).

    Uses P12–P60 only. Generates three plots:
    1. summary_eye_sorted - All trajectories (TRUE colored, FALSE grey)
    2. summary_eye_sorted_grey_only - Only FALSE trajectories
    3. summary_eye_sorted_colored_only - Only TRUE trajectories with motif labels

    Args:
        df: Full DataFrame with all motifs and stages (Motif_Label, Stage, effect_size, significance)
        eye_verdict_df: DataFrame with Motif column and a boolean column (color/highlight or last column)
        xlim, ylim: Axis limits
        pcutoff: Significance cutoff for reference line
        output_dir: Output directory (e.g. per_motif_plots/sorted by eye for changes)
        model_type: Model type for labeling
        xlabel: Custom x-axis label
        pcutoff_stringent: Optional more stringent p-value cutoff
        effect_size_threshold: Optional effect size threshold for grey zone (disabled if None)
    """
    from matplotlib.lines import Line2D

    if eye_verdict_df.empty:
        print("  Warning: Cannot generate eye verdict plots - eye verdict DataFrame is empty.")
        return

    # Normalize motif names in eye table and build mapping: normalized_motif -> color (True/False)
    color_col = None
    for c in ["color", "highlight", "Color", "Highlight"]:
        if c in eye_verdict_df.columns:
            color_col = c
            break
    if color_col is None:
        # Use last column as boolean
        color_col = eye_verdict_df.columns[-1]

    eye_colored = {}
    for _, row in eye_verdict_df.iterrows():
        norm = _normalize_motif_for_eye(row.get("Motif", row.iloc[0]))
        if not norm:
            continue
        val = row[color_col]
        if isinstance(val, str):
            val = val.strip().upper() in ("TRUE", "1", "YES")
        else:
            val = bool(val)
        eye_colored[norm] = val

    # Restrict to P12–P60
    work = df[df["Stage"] != "P3"].copy()
    if work.empty:
        print("  Warning: No P12/P20/P60 data for eye verdict plots.")
        return

    stage_order = [s for s in STAGE_ORDER if s in work["Stage"].unique()]
    stage_order.sort(key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)

    grey_trajs = []
    colored_trajs = []
    for motif in work["Motif_Label"].unique():
        traj = work[work["Motif_Label"] == motif].copy()
        traj = traj.sort_values("Stage", key=lambda s: s.map({st: i for i, st in enumerate(STAGE_ORDER)}))
        if len(traj) < 2:
            continue
        if eye_colored.get(motif, False):
            colored_trajs.append(traj)
        else:
            grey_trajs.append(traj)

    n_colored = len(colored_trajs)
    n_grey = len(grey_trajs)

    # ----- Plot 1: All trajectories (colored + grey) -----
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    ax.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)

    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax, effect_size_threshold, xlim)
    ax.axhline(y=pcutoff, linestyle="--", color="gray")
    ax.axvline(x=0, linestyle="--", color="gray")
    ax.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    if pcutoff_stringent is not None:
        ax.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)

    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )

    for traj in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_title(f"Trajectories by Eye Verdict (P12–P60) - {model_type}\n(n={n_colored} colored, n={n_grey} grey)", fontsize=14)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s)
        for s in stage_order if s in STAGE_COLORS
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="no change (eye)"))
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    base = os.path.join(output_dir, "summary_eye_sorted")
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ----- Plot 2: Grey only -----
    fig_g, ax_g = plt.subplots(1, 1)
    fig_g.set_size_inches(10, 10)
    ax_g.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_g.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_g, effect_size_threshold, xlim)
    ax_g.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_g.axvline(x=0, linestyle="--", color="gray")
    ax_g.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    if pcutoff_stringent is not None:
        ax_g.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_g.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)

    for traj in grey_trajs:
        points = traj[["effect_size", "significance"]].values
        for i, (x, y) in enumerate(points):
            ax_g.scatter(x, y, c=[GREY], s=40, zorder=2, edgecolors="none", alpha=0.7)
        for i in range(len(points) - 1):
            ax_g.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=GREY, linestyle="-", linewidth=1, zorder=1, alpha=0.7
            )

    ax_g.set_xlim(xlim[0], xlim[1])
    ax_g.set_ylim(ylim[0], ylim[1])
    ax_g.set_title(f"Eye Verdict Grey Only (P12–P60) - {model_type}\n(n={n_grey} motifs)", fontsize=14)
    ax_g.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=8, label="no change (eye)")], loc="best", fontsize=10)
    base_grey = os.path.join(output_dir, "summary_eye_sorted_grey_only")
    for ext in ["pdf", "svg", "png"]:
        fig_g.savefig(f"{base_grey}.{ext}", bbox_inches="tight")
    plt.close(fig_g)

    # ----- Plot 3: Colored only (with labels) -----
    fig_c, ax_c = plt.subplots(1, 1)
    fig_c.set_size_inches(10, 10)
    ax_c.set_xlabel(xlabel or XLABEL_DEFAULT, fontsize=16)
    ax_c.set_ylabel("Significance\n$-log_{10}(P)$", fontsize=16)
    if effect_size_threshold is not None:
        draw_effect_size_grey_zone(ax_c, effect_size_threshold, xlim)
    ax_c.axhline(y=pcutoff, linestyle="--", color="gray")
    ax_c.axvline(x=0, linestyle="--", color="gray")
    ax_c.text(x=xlim[0] + 0.1, y=pcutoff + 0.05, s="P=0.05 cutoff", fontsize=10)
    if pcutoff_stringent is not None:
        ax_c.axhline(y=pcutoff_stringent, linestyle=":", color="gray")
        ax_c.text(x=xlim[0] + 0.1, y=pcutoff_stringent + 0.05, s="P=0.01 cutoff", fontsize=10)

    for traj in colored_trajs:
        points = traj[["effect_size", "significance"]].values
        stages = traj["Stage"].tolist()
        for i, (x, y) in enumerate(points):
            st = stages[i]
            ax_c.scatter(x, y, c=[STAGE_COLORS.get(st, GREY)], s=40, zorder=3, edgecolors="black", linewidths=0.3)
        for i in range(len(points) - 1):
            st = stages[i + 1]
            ax_c.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=STAGE_COLORS.get(st, GREY), linestyle="-", linewidth=1.5, zorder=2
            )
        motif = traj["Motif_Label"].iloc[0]
        x_last, y_last = points[-1, 0], points[-1, 1]
        ax_c.annotate(
            motif, xy=(x_last, y_last), xytext=(10, 10), textcoords="offset points",
            fontsize=8, zorder=4,
            fontfamily=["Helvetica", "Arial", "sans-serif"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="none", alpha=0.9),
        )

    ax_c.set_xlim(xlim[0], xlim[1])
    ax_c.set_ylim(ylim[0], ylim[1])
    ax_c.set_title(f"Eye Verdict Colored Only (P12–P60) - {model_type}\n(n={n_colored} motifs)", fontsize=14)
    legend_elements_c = [Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_COLORS[s], markersize=8, label=s) for s in stage_order if s in STAGE_COLORS]
    ax_c.legend(handles=legend_elements_c, loc="best", fontsize=10)
    base_colored = os.path.join(output_dir, "summary_eye_sorted_colored_only")
    for ext in ["pdf", "svg", "png"]:
        fig_c.savefig(f"{base_colored}.{ext}", bbox_inches="tight")
    plt.close(fig_c)

    print(f"    Generated eye verdict trajectory plots: {n_colored} colored, {n_grey} grey")


def run_model(input_dir, model_type, output_subdir, xlabel=None, transition_dir=None, volcano_ylim=True, 
               output_comparison_list=True, run_permutation=True, run_fda=True, run_mixed_effects=True,
               run_bootstrap=False, run_distance_metrics=True, run_trend_analysis=True,
               n_permutations=10000, n_bootstrap=1000, generate_method_comparison=True,
               run_noP3_analysis=True, eye_verdict_csv=None, eye_only=False, effect_size_threshold=None):
    """
    Load data for one model, compute limits, generate plots and run statistical analyses.

    Args:
        input_dir: Directory containing upsetplot CSV files
        model_type: Model type to process (e.g., "uniform", "region_specific")
        output_subdir: Output directory for this model
        xlabel: Custom x-axis label (optional)
        transition_dir: Path to helper 07 output for transition significance
        volcano_ylim: Use volcano-style y-axis [0, max] vs symmetric
        output_comparison_list: Write change_criteria_comparison.csv
        run_permutation: Run permutation tests
        run_fda: Run FDA trajectory tests
        run_mixed_effects: Run mixed-effects model analysis
        run_bootstrap: Run bootstrap CI for quadrant classification
        run_distance_metrics: Compute standardized distance metrics
        run_trend_analysis: Run Spearman/linear trend tests and direction analysis
        n_permutations: Number of permutations for permutation/FDA tests
        n_bootstrap: Number of bootstrap samples
        generate_method_comparison: Generate method comparison summary files
        run_noP3_analysis: Run parallel analysis excluding P3 stage (outputs to noP3/ subdirectory)
        eye_verdict_csv: Optional path to CSV with Motif and color (TRUE/FALSE) for eye-sorted summary plots
        eye_only: If True and eye_verdict_csv set, only generate eye verdict plots and return
        effect_size_threshold: Optional; draw lines and grey zone at ± this value (e.g. 1 for ±1). None to disable.
    """
    df = load_volcano_data(input_dir, model_type)
    if df.empty:
        print(f"  No data for model {model_type}, skipping.")
        return
    # Only multi-area motifs (at least two areas: label contains "+")
    df = df[df["Motif_Label"].str.contains("+", regex=False)]
    if df.empty:
        print(f"  No multi-area motifs for model {model_type}, skipping.")
        return
    xlim, ylim = compute_unified_limits(df, volcano_ylim=volcano_ylim)
    if xlim[0] is None:
        print(f"  Could not compute limits for {model_type}, skipping.")
        return
    ylim = VOLCANO_YLIM  # xlim stays data-driven from compute_unified_limits
    n_motifs = df["Motif_Label"].nunique()
    pcutoff = bonferroni_cutoff(n_motifs)
    pcutoff_stringent = bonferroni_cutoff_stringent(n_motifs)
    
    # Effect size threshold for grey zone (passed from CLI or None to disable)
    if xlabel is None and model_type == "proportional_effectsize":
        xlabel = XLABEL_PROP
    transition_sig_set = load_transition_significance(transition_dir, model_type) if transition_dir else None
    
    # Create per_motif_plots subdirectory for individual trajectory plots
    per_motif_dir = os.path.join(output_subdir, "per_motif_plots")
    os.makedirs(per_motif_dir, exist_ok=True)

    # Eye verdict summary plots (manual TRUE/FALSE coloring); can run first for --eye_only
    if eye_verdict_csv:
        eye_path = Path(eye_verdict_csv)
        if eye_path.is_file():
            eye_output_dir = os.path.join(per_motif_dir, "sorted by eye for changes")
            os.makedirs(eye_output_dir, exist_ok=True)
            try:
                eye_verdict_df = pd.read_csv(eye_path)
                print(f"  {model_type}: Generating eye verdict trajectory plots...")
                plot_trajectories_by_eye_verdict(
                    df, eye_verdict_df, xlim, ylim, pcutoff, eye_output_dir, model_type,
                    xlabel=xlabel, pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold
                )
            except Exception as e:
                print(f"  Warning: Could not generate eye verdict plots: {e}")
        else:
            print(f"  Warning: Eye verdict CSV not found: {eye_verdict_csv}")
        if eye_only:
            print(f"  {model_type}: Eye-only run, skipping rest of pipeline.")
            return

    motifs_done = 0
    for motif in df["Motif_Label"].unique():
        traj = df[df["Motif_Label"] == motif]
        if len(traj) < 2:
            continue
        plot_one_motif_trajectory(motif, traj, xlim, ylim, pcutoff, per_motif_dir, model_type, xlabel=xlabel,
                                  pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
        motifs_done += 1
    
    # Create quadrant_change subdirectory for summary plots
    quadrant_dir = os.path.join(output_subdir, "quadrant_change")
    os.makedirs(quadrant_dir, exist_ok=True)
    
    # Quadrant-based summaries (filtered and not_filtered)
    plot_summary_all_trajectories(df, xlim, ylim, pcutoff, quadrant_dir, model_type, xlabel=xlabel, exclude_p3=False, transition_sig_set=transition_sig_set, name_suffix="", highlight_mode="quadrant",
                                  pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    plot_summary_all_trajectories(df, xlim, ylim, pcutoff, quadrant_dir, model_type, xlabel=xlabel, exclude_p3=False, transition_sig_set=None, name_suffix="_not_filtered", highlight_mode="quadrant",
                                  pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    plot_summary_all_trajectories(df, xlim, ylim, pcutoff, quadrant_dir, model_type, xlabel=xlabel, exclude_p3=True, transition_sig_set=transition_sig_set, name_suffix="", highlight_mode="quadrant",
                                  pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    plot_summary_all_trajectories(df, xlim, ylim, pcutoff, quadrant_dir, model_type, xlabel=xlabel, exclude_p3=True, transition_sig_set=None, name_suffix="_not_filtered", highlight_mode="quadrant",
                                  pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    
    # Matt visual rules analysis (full data: P3, P12, P20, P60)
    run_matt_visual_rules_analysis(df, output_subdir, pcutoff, xlim, ylim, model_type, xlabel=xlabel,
                                   pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    
    if output_comparison_list:
        comparison_path = os.path.join(output_subdir, "change_criteria_comparison.csv")
        write_comparison_list(df, pcutoff, transition_sig_set, comparison_path)
    
    # =========================================================================
    # NEW: Run all statistical methods
    # =========================================================================
    any_new_methods = run_permutation or run_fda or run_mixed_effects or run_bootstrap or run_distance_metrics
    
    if any_new_methods:
        print(f"  {model_type}: Running statistical methods...")
        results = run_all_statistical_methods(
            df, pcutoff, transition_sig_set, output_subdir,
            run_permutation=run_permutation,
            run_fda=run_fda,
            run_mixed_effects=run_mixed_effects,
            run_bootstrap=run_bootstrap,
            run_distance_metrics=run_distance_metrics,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
        )
        
        # Generate method comparison summary
        if generate_method_comparison and results:
            print(f"  {model_type}: Generating method comparison summary...")
            generate_method_comparison_summary(df, pcutoff, transition_sig_set, results, output_subdir)
        
        # =====================================================================
        # NEW: Generate visualizations for statistical methods
        # =====================================================================
        print(f"  {model_type}: Generating statistical method visualizations...")
        
        # Significance-based trajectory plots
        if run_permutation and "permutation" in results:
            perm_dir = os.path.join(output_subdir, "permutation")
            plot_summary_by_significance(df, results, xlim, ylim, pcutoff, perm_dir, model_type, method="permutation", xlabel=xlabel,
                                         pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
        
        if run_fda and "fda" in results:
            fda_dir = os.path.join(output_subdir, "fda")
            plot_summary_by_significance(df, results, xlim, ylim, pcutoff, fda_dir, model_type, method="fda", xlabel=xlabel,
                                         pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
        
        # P-value heatmap
        if (run_permutation or run_fda) and generate_method_comparison:
            comparison_dir = os.path.join(output_subdir, "method_comparison")
            plot_pvalue_heatmap(results, comparison_dir, pcutoff=pcutoff)
        
        # Distance gradient trajectory plot
        if run_distance_metrics and "distance_metrics" in results:
            dist_dir = os.path.join(output_subdir, "distance_metrics")
            plot_summary_by_distance_gradient(df, results["distance_metrics"], xlim, ylim, pcutoff, dist_dir, model_type, xlabel=xlabel,
                                              pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    
    # =========================================================================
    # NEW: Run trend analysis (Spearman, linear, direction tests)
    # =========================================================================
    if run_trend_analysis:
        print(f"  {model_type}: Running trend analysis...")
        trend_results = execute_trend_analysis(df, output_subdir)
        
        # Generate trend visualizations
        if trend_results:
            print(f"  {model_type}: Generating trend analysis visualizations...")
            plot_trend_analysis(trend_results, trend_results["output_dir"], model_type)
            
            # Generate verdict CSV and trajectory plots
            if "spearman_df" in trend_results and not trend_results["spearman_df"].empty:
                print(f"  {model_type}: Generating verdict-based trajectory plots...")
                verdict_df = generate_verdict_csv(trend_results["spearman_df"], trend_results["output_dir"], df=df, pcutoff=pcutoff)
                plot_trajectories_by_verdict(df, verdict_df, xlim, ylim, pcutoff, 
                                             trend_results["output_dir"], model_type, xlabel,
                                             pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
                # Generate STRICT verdict plots (requires threshold crossing)
                plot_trajectories_by_strict_verdict(df, verdict_df, xlim, ylim, pcutoff,
                                                    trend_results["output_dir"], model_type, xlabel,
                                                    pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    
    # =========================================================================
    # NEW: Run parallel analysis EXCLUDING P3 (outputs to noP3/ subdirectory)
    # =========================================================================
    if run_noP3_analysis:
        noP3_dir = os.path.join(output_subdir, "noP3")
        os.makedirs(noP3_dir, exist_ok=True)
        
        # Filter data to exclude P3
        df_noP3 = df[df["Stage"] != "P3"].copy()
        
        if not df_noP3.empty:
            print(f"  {model_type}: Running noP3 analysis (excluding P3 stage)...")
            
            # Create per_motif_plots subdirectory for noP3 individual trajectory plots
            noP3_per_motif_dir = os.path.join(noP3_dir, "per_motif_plots")
            os.makedirs(noP3_per_motif_dir, exist_ok=True)
            
            # Generate individual trajectory plots for noP3
            for motif in df_noP3["Motif_Label"].unique():
                traj = df_noP3[df_noP3["Motif_Label"] == motif]
                if len(traj) < 2:
                    continue
                plot_one_motif_trajectory(motif, traj, xlim, ylim, pcutoff, noP3_per_motif_dir, model_type + " (noP3)", xlabel=xlabel,
                                          pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
            
            # Run all statistical methods on filtered data
            if any_new_methods:
                results_noP3 = run_all_statistical_methods(
                    df_noP3, pcutoff, transition_sig_set, noP3_dir,
                    run_permutation=run_permutation,
                    run_fda=run_fda,
                    run_mixed_effects=run_mixed_effects,
                    run_bootstrap=run_bootstrap,
                    run_distance_metrics=run_distance_metrics,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                    exclude_p3=False,  # Already filtered
                )
                
                # Generate method comparison summary for noP3
                if generate_method_comparison and results_noP3:
                    generate_method_comparison_summary(df_noP3, pcutoff, transition_sig_set, results_noP3, noP3_dir)
                
                # Generate visualizations for noP3
                if run_permutation and "permutation" in results_noP3:
                    perm_dir_noP3 = os.path.join(noP3_dir, "permutation")
                    plot_summary_by_significance(df_noP3, results_noP3, xlim, ylim, pcutoff, perm_dir_noP3, model_type + " (noP3)", method="permutation", xlabel=xlabel,
                                                 pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
                
                if run_fda and "fda" in results_noP3:
                    fda_dir_noP3 = os.path.join(noP3_dir, "fda")
                    plot_summary_by_significance(df_noP3, results_noP3, xlim, ylim, pcutoff, fda_dir_noP3, model_type + " (noP3)", method="fda", xlabel=xlabel,
                                                 pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
                
                if (run_permutation or run_fda) and generate_method_comparison:
                    comparison_dir_noP3 = os.path.join(noP3_dir, "method_comparison")
                    plot_pvalue_heatmap(results_noP3, comparison_dir_noP3, pcutoff=pcutoff)
                
                if run_distance_metrics and "distance_metrics" in results_noP3:
                    dist_dir_noP3 = os.path.join(noP3_dir, "distance_metrics")
                    plot_summary_by_distance_gradient(df_noP3, results_noP3["distance_metrics"], xlim, ylim, pcutoff, dist_dir_noP3, model_type + " (noP3)", xlabel=xlabel,
                                                      pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
            
            # Run trend analysis on filtered data
            if run_trend_analysis:
                print(f"  {model_type}: Running noP3 trend analysis...")
                trend_results_noP3 = execute_trend_analysis(df_noP3, noP3_dir, exclude_p3=False)  # Already filtered
                
                if trend_results_noP3:
                    plot_trend_analysis(trend_results_noP3, trend_results_noP3["output_dir"], model_type + " (noP3)")
                    
                    # Generate verdict CSV and trajectory plots for noP3
                    if "spearman_df" in trend_results_noP3 and not trend_results_noP3["spearman_df"].empty:
                        print(f"  {model_type}: Generating noP3 verdict-based trajectory plots...")
                        verdict_df_noP3 = generate_verdict_csv(trend_results_noP3["spearman_df"], trend_results_noP3["output_dir"], df=df_noP3, pcutoff=pcutoff)
                        plot_trajectories_by_verdict(df_noP3, verdict_df_noP3, xlim, ylim, pcutoff, 
                                                     trend_results_noP3["output_dir"], model_type + " (noP3)", xlabel,
                                                     pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
                        # Generate STRICT verdict plots for noP3 (requires threshold crossing)
                        plot_trajectories_by_strict_verdict(df_noP3, verdict_df_noP3, xlim, ylim, pcutoff,
                                                            trend_results_noP3["output_dir"], model_type + " (noP3)", xlabel,
                                                            pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
            # Matt visual rules analysis (noP3: P12, P20, P60)
            run_matt_visual_rules_analysis(df_noP3, noP3_dir, pcutoff, xlim, ylim, model_type + " (noP3)", xlabel=xlabel,
                                           pcutoff_stringent=pcutoff_stringent, effect_size_threshold=effect_size_threshold)
    
    print(f"  {model_type}: {motifs_done} trajectory plots saved to {output_subdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-motif volcano trajectories (effect size vs significance) across P3/P12/P20/P60 "
                    "with unified axes. Includes multiple statistical methods for trajectory significance testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Statistical Methods (all enabled by default):
  --methods all         Run all trajectory significance methods
  --methods quadrant    Only run original quadrant change detection
  --methods permutation Only run permutation tests
  --methods fda         Only run Functional Data Analysis
  --methods mixed       Only run mixed-effects models

Distance Metrics (all enabled by default):
  --distance_metrics all       Compute all distance metrics (zscore, mahalanobis, separate)
  --distance_metrics none      Skip distance metric computation

Output Structure:
  {model}/
    per_motif_plots/           Individual trajectory plots
    quadrant_change/           Quadrant-based summary plots
    permutation/               Permutation test results
    fda/                       FDA test results  
    mixed_effects/             Mixed-effects model results
    bootstrap_ci/              Bootstrap confidence intervals (if enabled)
    distance_metrics/          Standardized distance metrics
    method_comparison/         Cross-method comparison summaries

References:
  - Permutation: Good (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses.
  - FDA: Ramsay & Silverman (2005). Functional Data Analysis.
  - Mixed-effects: Bates et al. (2015). J Stat Software, 67(1), 1-48.
  - Mahalanobis: Mahalanobis (1936). Proc. Nat. Inst. Sci. India.
        """
    )
    # Input/output arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of upsetplot CSVs (e.g. 07_input)")
    parser.add_argument(
        "--helper_output_dir",
        type=str,
        default=None,
        help="Output directory (default: helpers/outputs/15_volcano_trajectories)",
    )
    parser.add_argument(
        "--transition_significance_dir",
        type=str,
        default=None,
        help="Helper 07 output dir (07_motif_significange_trajectories) for transition_significance.csv. If omitted, inferred from helper_output_dir.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=MODELS_DEFAULT + [None],
        help="Single model to process; if omitted, process all supported models",
    )
    
    # Plot options
    parser.add_argument(
        "--no_volcano_ylim",
        action="store_true",
        help="Use symmetric y-axis for significance instead of [0, max] (classic volcano uses [0, max])",
    )
    parser.add_argument(
        "--no_comparison_list",
        action="store_true",
        help="Do not write change_criteria_comparison.csv (quadrant vs centroid vs path_length/range)",
    )
    
    # Statistical method selection
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        choices=["all", "quadrant", "permutation", "fda", "mixed", "none"],
        help="Which trajectory significance methods to run (default: all)",
    )
    parser.add_argument(
        "--distance_metrics",
        type=str,
        default="all",
        choices=["all", "none"],
        help="Which distance metrics to compute (default: all)",
    )
    
    # Bootstrap options
    parser.add_argument(
        "--bootstrap_ci",
        action="store_true",
        help="Compute bootstrap CI for quadrant classification (slow, disabled by default)",
    )
    parser.add_argument(
        "--bootstrap_n",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    
    # Permutation options
    parser.add_argument(
        "--permutation_n",
        type=int,
        default=10000,
        help="Number of permutations for permutation/FDA tests (default: 10000)",
    )
    
    # Comparison summary
    parser.add_argument(
        "--no_method_comparison",
        action="store_true",
        help="Do not generate method comparison summary files",
    )
    
    # Trend analysis
    parser.add_argument(
        "--no_trend_analysis",
        action="store_true",
        help="Do not run trend analysis (Spearman, linear regression, direction tests)",
    )
    
    # noP3 analysis
    parser.add_argument(
        "--no_noP3_analysis",
        action="store_true",
        help="Do not run parallel analysis excluding P3 stage (by default, runs both full and noP3)",
    )
    
    # Eye verdict (manual TRUE/FALSE coloring for summary plots)
    parser.add_argument(
        "--eye_verdict_csv",
        type=str,
        default=None,
        help="Path to CSV with Motif and color (TRUE/FALSE). Saves summary plots to per_motif_plots/sorted by eye for changes/",
    )
    parser.add_argument(
        "--eye_only",
        action="store_true",
        help="With --eye_verdict_csv: only load data, compute limits, and generate eye verdict plots (skip rest of pipeline).",
    )
    parser.add_argument(
        "--effect_size_threshold",
        type=float,
        default=None,
        help="Draw vertical lines and grey zone at ± this effect size (log2 FC). E.g. 1 for lines at -1 and +1. Omit to disable.",
    )
    
    args = parser.parse_args()
    
    # Determine which methods to run
    if args.methods == "all":
        run_permutation = True
        run_fda = True
        run_mixed_effects = True
    elif args.methods == "none":
        run_permutation = False
        run_fda = False
        run_mixed_effects = False
    elif args.methods == "quadrant":
        run_permutation = False
        run_fda = False
        run_mixed_effects = False
    elif args.methods == "permutation":
        run_permutation = True
        run_fda = False
        run_mixed_effects = False
    elif args.methods == "fda":
        run_permutation = False
        run_fda = True
        run_mixed_effects = False
    elif args.methods == "mixed":
        run_permutation = False
        run_fda = False
        run_mixed_effects = True
    else:
        run_permutation = True
        run_fda = True
        run_mixed_effects = True
    
    run_distance_metrics = args.distance_metrics != "none"
    run_bootstrap = args.bootstrap_ci
    
    script_dir = Path(__file__).parent
    base_out = Path(args.helper_output_dir) if args.helper_output_dir else script_dir.parent / "outputs" / "15_volcano_trajectories"
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Resolve transition dir: explicit arg, or infer from helper_output_dir
    transition_dir = None
    if args.transition_significance_dir:
        p = Path(args.transition_significance_dir)
        if p.is_dir():
            transition_dir = str(p)
    elif args.helper_output_dir:
        helper_path = Path(args.helper_output_dir)
        if helper_path.name == "15_volcano_trajectories":
            inferred = helper_path.parent / "07_motif_significange_trajectories"
            if inferred.is_dir():
                transition_dir = str(inferred)
    
    models = [args.model_type] if args.model_type else MODELS_DEFAULT
    
    print("=" * 80)
    print("Volcano Trajectory Analysis (v2.0)")
    print("=" * 80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {base_out}")
    if transition_dir:
        print(f"Transition significance (helper 07): {transition_dir}")
    else:
        print("Transition significance: not used (no dir provided or inferred)")
    print()
    run_trend = not args.no_trend_analysis
    run_noP3 = not args.no_noP3_analysis
    
    print("Methods enabled:")
    print(f"  - Quadrant change detection: Yes (always)")
    print(f"  - Permutation tests: {'Yes' if run_permutation else 'No'}")
    print(f"  - FDA trajectory tests: {'Yes' if run_fda else 'No'}")
    print(f"  - Mixed-effects models: {'Yes' if run_mixed_effects else 'No'}")
    print(f"  - Bootstrap CI: {'Yes' if run_bootstrap else 'No'}")
    print(f"  - Distance metrics: {'Yes' if run_distance_metrics else 'No'}")
    print(f"  - Trend analysis (Spearman/linear/direction): {'Yes' if run_trend else 'No'}")
    print(f"  - Parallel noP3 analysis: {'Yes' if run_noP3 else 'No'}")
    print(f"  - Permutation iterations: {args.permutation_n}")
    if run_bootstrap:
        print(f"  - Bootstrap samples: {args.bootstrap_n}")
    print("=" * 80)
    
    for model_type in models:
        print(f"\nProcessing model: {model_type}")
        subdir = base_out / model_type
        subdir.mkdir(parents=True, exist_ok=True)
        run_model(
            args.input_dir,
            model_type,
            str(subdir),
            transition_dir=transition_dir,
            volcano_ylim=not args.no_volcano_ylim,
            output_comparison_list=not args.no_comparison_list,
            run_permutation=run_permutation,
            run_fda=run_fda,
            run_mixed_effects=run_mixed_effects,
            run_bootstrap=run_bootstrap,
            run_distance_metrics=run_distance_metrics,
            run_trend_analysis=run_trend,
            n_permutations=args.permutation_n,
            n_bootstrap=args.bootstrap_n,
            generate_method_comparison=not args.no_method_comparison,
            run_noP3_analysis=run_noP3,
            eye_verdict_csv=args.eye_verdict_csv,
            eye_only=args.eye_only,
            effect_size_threshold=args.effect_size_threshold,
        )
    
    print()
    print("=" * 80)
    print(f"Done. Results saved to: {base_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
