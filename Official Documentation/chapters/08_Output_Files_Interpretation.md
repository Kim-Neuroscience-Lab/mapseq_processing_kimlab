# Chapter 8: Output Files and Structure

## Overview

This chapter lists the main files and directories produced by the MAPseq processing pipeline and helper scripts: **locations, names, and column semantics**. Biological or study-specific interpretation depends on your design, thresholds, and model choice; use [Chapter 5: Statistical Methods](05_Statistical_Methods.md) for definitions of tests and effect sizes, and [Chapter 11: Stability Analysis](11_Stability_Analysis.md) for a generic framework (without project-specific verdicts).

Optional batch utilities (figure aggregation, conclusions generation) are described in [Chapter 14: Experimental Features](14_Experimental_Features.md).

## Output Directory Structure

```
02_output/
├── {age}/                          # P3, P12, P20, P60
│   └── {parameterization}/         # Filter parameter set
│       ├── analysis/
│       │   ├── uniform/           # Uniform model results
│       │   ├── region_specific/   # Region-specific model results
│       │   └── correlated/       # Correlated model results
│       ├── {sample}_*.csv         # Per-sample outputs
│       └── {sample}_*.png         # Visualization files
└── {parameterization}_helpers/    # Helper script outputs
    ├── 01_motif_analysis_per_animal/
    ├── 05_motif_analysis/
    └── ...
```

Additional model subdirectories under `analysis/` may appear when using extended `--model-type` values (see [Chapter 4](04_Main_Processing_Pipeline.md)).

## Main Processing Outputs

### Per-Sample Files

**Location**: `{out_dir}/`

#### Filtered Matrix

**File**: `{sample}_Filtered_Matrix.csv`

**Content**: Quality-filtered UMI count matrix

**Columns**:
- Barcode identifiers (if preserved)
- Target region columns with filtered UMI counts
- Values below the configured threshold set to zero

#### Normalized Matrix

**File**: `{sample}_Normalized_Matrix.csv`

**Content**: Row-normalized matrix (each row normalized by its max value)

**Columns**: Same as filtered matrix

#### UMI Total Counts

**File**: `{sample}_UMI_Total_Counts.csv`

**Content**: Summed UMI counts per region

**Columns**:
- `Region`: Region name
- `UMI_Sum`: Total UMI counts for that region

#### Region-Specific Probabilities (Anchor Model)

**File**: `{sample}_Region-specific_Probabilities_N0based.csv`

**Content**: Region-specific projection probabilities (if `--is-anchor-model`)

**Columns**:
- `Region`: Region name
- `Probability`: pᵢ = Nᵢ/N₀
- `N0`: Estimated population size

#### Conditional Probability Matrix (Anchor Model)

**File**: `{sample}_Conditional_Probability_Matrix.csv`

**Content**: P(B|A) matrix for correlated model (if `--is-anchor-model`)

**Structure**:
- Rows: Region A (given)
- Columns: Region B (target)
- Values: P(B|A) = probability of projecting to B given projection to A

### Analysis Outputs

**Location**: `{out_dir}/analysis/{model}/`

#### Upsetplot CSV

**File**: `{sample}_upsetplot_{model}.csv`

**Content**: Core motif analysis results

**Columns**:
- `Motifs`: List of regions in motif (e.g., "['al', 'lm']")
- `Observed`: Count of neurons with this motif
- `Expected`: Model-predicted count
- `Expected SD`: Standard deviation of expected count
- `Effect Size`: log₂(Observed+1) − log₂(Expected+1) (see Chapter 5)
- `P-value`: Two-tailed binomial test p-value
- `Degree`: Number of regions in motif
- `Group`: Integer significance/visual group (1–4); volcano plots map these to colors as documented in processing outputs

#### Effect Significance Plot

**File**: `{sample}_effect_significance_{model}.png`

**Content**: Volcano plot (effect size vs. −log₁₀(P-value))

#### Per-Cell Projection Strength

**File**: `{sample}_per_cell_proj_strength_{model}.png`

**Content**: Per-cell projection pattern visualization

#### 2-Region Motif Graph

**File**: `{sample}_panel_g_broadcasting_from_canonical_{model}.svg`

**Content**: Network graph of pairwise region relationships (2-region motifs)

#### Projection Summary

**File**: `projection_summary.csv`

**Content**: Summary metrics for each sample

**Columns** (non-exhaustive):
- `Sample`, `Model`, filtering thresholds, `N0`, `p_e` (uniform, if applicable), `consensus_k`, `motif_counts`, and other run metadata

## Helper Script Outputs

### Script 01: Per-Animal Analysis

**Location**: `02_output/{parameterization}_helpers/01_motif_analysis_per_animal/{model}/`

**Key outputs**: Kruskal-Wallis and related tables, JSD tables, plots (see script and [Chapter 7](07_Helper_Scripts.md)).

### Script 05: Motif Analysis

**Location**: `02_output/{parameterization}_helpers/05_motif_analysis/`

**Key files**:
- `motif_percent_matrix_by_age_{model}.csv` — rows: motifs; columns: ages; values: percentages
- `motif_transition_significance_summary_{model}.txt` — per-motif transition summaries (including JSD lines used by downstream helpers)

### Script 07: Trajectories

**Location**: `02_output/{parameterization}_helpers/07_motif_significange_trajectories/{model}/`

**Key files**:
- `combined_effect_sizes_{model}.csv` — columns include `Motif_Label`, `Effect Size`, `Stage`, `Significant`, `Observed`
- `transition_significance.csv` — columns include `Motif`, `Transition`, `P-value`, `Significant`, `Delta_Effect_Size`, `SE_delta`, `P_value_adjusted`. The **Significant** flag follows helper 07 options (raw vs FDR; see [Chapter 7](07_Helper_Scripts.md)).

### Script 15: Volcano Trajectories (v2.0)

**Location**: `02_output/{parameterization}_helpers/15_volcano_trajectories/{model}/`

Subdirectories include `per_motif_plots/`, `quadrant_change/`, `permutation/`, `fda/`, `mixed_effects/`, `bootstrap_ci/` (if run), `distance_metrics/`, `method_comparison/`, and legacy `change_criteria_comparison.csv`. For a concise guide to outputs and methods (without study-specific results), see [Chapter 15: Trajectory Results Interpretation](15_Trajectory_Results_Interpretation.md).

#### Permutation Test Results

**File**: `permutation/permutation_test_results.csv`

**Columns**: `Motif`, `observed_stat`, `null_mean`, `null_sd`, `p_value`, `p_value_fdr`, `significant`

#### FDA Test Results

**File**: `fda/fda_trajectory_significance.csv`

**Columns**: `Motif`, `fda_statistic`, `null_mean`, `null_sd`, `p_value`, `p_value_fdr`, `significant`

#### Mixed-Effects Model Results

**Files**: `mixed_effects/mixed_effects_summary.csv`, `mixed_effects/mixed_effects_random_effects.csv`

**Summary columns** (typical): `stage_coefficient`, `stage_pvalue`, `model_converged`

**Random effects columns** (typical): `Motif`, `random_intercept`

#### Bootstrap CI Results

**File**: `bootstrap_ci/quadrant_bootstrap_ci.csv` (if `--bootstrap_ci` enabled)

**Columns** (typical): `Motif`, `Stage`, `quadrant_raw`, bootstrap proportions, `quadrant_robust`

#### Distance Metrics Results

**File**: `distance_metrics/distance_comparison.csv`

**Columns** (typical): `Motif`, path length and effect-size / significance variation metrics and percentile columns (`*_pct`)

#### Method Comparison Summary

**Files**: `method_comparison/all_methods_summary.csv`, `method_comparison/method_agreement_matrix.csv`, `method_comparison/significant_by_method.csv`

**Purpose**: Combine flags and p-values from multiple trajectory methods in one table for comparison.

#### Legacy Files

**File**: `change_criteria_comparison.csv` — older quadrant/centroid/path metrics; prefer `method_comparison/` for new analyses unless reproducing legacy workflows.

## Quality Control

### Check Output Completeness

1. **Main processing**: Verify expected CSV/PNG files exist under `{out_dir}` and `analysis/{model}/`.
2. **Helper scripts**: Confirm numbered helper directories exist; inspect logs if a step failed.
3. **File sizes**: Ensure outputs are non-empty where data should be present.
4. **Columns**: Spot-check that key columns listed above exist.

### Sanity Checks

1. **Effect sizes**: Values are finite; typical magnitudes depend on depth and model.
2. **P-values**: In [0, 1] where reported.
3. **Expected counts**: Non-negative as produced by the pipeline.

### Common Issues

**Empty output files**: Often filtering removed all rows; relax thresholds or verify input paths.

**Missing helper outputs**: Wrong execution order or missing inputs; see [Chapter 7](07_Helper_Scripts.md).

**QC script**: `postprocessing_checks.py` can summarize run health; run from repo root with `--help` for arguments.

---

*For troubleshooting, see [Chapter 12: Troubleshooting and Best Practices](12_Troubleshooting_Best_Practices.md). Maintainer-only batch tools are in [Chapter 14: Experimental Features](14_Experimental_Features.md).*
