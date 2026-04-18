# Chapter 15: Trajectory Results (File Reference and Methods)

## Purpose

Helper **15** (`helpers/scripts/15_volcano_trajectories.py`) produces trajectory summaries and optional formal tests from upsetplot/effect-size inputs. This chapter describes **outputs and column semantics** and gives **generic** guidance on which outputs correspond to formal tests versus descriptive summaries. It does **not** report results for any particular dataset.

For directory layout and filenames, see [Chapter 8: Output Files and Structure](08_Output_Files_Interpretation.md). For statistical definitions of helper **07** transitions, see [Chapter 5: Statistical Methods](05_Statistical_Methods.md).

## Methods vs. descriptive outputs

| Method / output | Role | Typical CSV |
|-----------------|------|-------------|
| Quadrant change (legacy) | Descriptive classification | `change_criteria_comparison.csv` |
| Permutation | Formal test (null permutes stage labels) | `permutation/permutation_test_results.csv` |
| FDA | Formal test (non-constant trajectory) | `fda/fda_trajectory_significance.csv` |
| Mixed-effects | Population-level regression summary | `mixed_effects/mixed_effects_summary.csv` |
| Trend module (if enabled in your run) | Spearman / linear / direction summaries | `trend_analysis/*.csv` |
| Distance metrics | Continuous ranking of path variation | `distance_metrics/distance_comparison.csv` |
| Method comparison | Cross-method agreement | `method_comparison/all_methods_summary.csv`, `method_agreement_matrix.csv` |

**Generic guidance**: Pre-specify a **primary** formal method for publication-style claims; use others for robustness. Quadrant and centroid-style rules are exploratory unless you justify them.

## Key files and columns

### `permutation/permutation_test_results.csv`

Typical columns: `Motif`, `observed_stat`, `null_mean`, `null_sd`, `p_value`, `p_value_fdr`, `significant` (per your run’s definition, often FDR ≤ 0.05).

### `fda/fda_trajectory_significance.csv`

Typical columns: `Motif`, `fda_statistic`, `null_mean`, `null_sd`, `p_value`, `p_value_fdr`, `significant`.

### `mixed_effects/mixed_effects_summary.csv` and `mixed_effects/mixed_effects_random_effects.csv`

Summary file: global stage coefficients and convergence flags. Random-effects file: per-motif deviations from the population mean.

### `bootstrap_ci/quadrant_bootstrap_ci.csv` (if run with `--bootstrap_ci`)

Bootstrap proportions for quadrant labels; useful for classification uncertainty.

### `distance_metrics/distance_comparison.csv`

Path-length and variation metrics (raw and standardized). Prefer standardized columns when ranking across motifs.

### `method_comparison/`

- `all_methods_summary.csv`: one row per motif with flags/p-values from multiple methods.
- `method_agreement_matrix.csv`: pairwise agreement rates between method binary calls.
- `significant_by_method.csv`: which motifs each method flags.

### Legacy `change_criteria_comparison.csv`

Older quadrant/centroid/path summaries; kept for backward compatibility. Prefer `method_comparison/` for new work unless reproducing a legacy figure.

## Quadrant-based outputs (descriptive)

Quadrant summaries describe crossings in effect-size vs significance space. **They are not substitutes** for permutation/FDA p-values unless you explicitly adopt them as research outcomes.

## Trend analysis outputs (if present)

When the script is run with trend analysis enabled, look under `trend_analysis/` for files such as:

- `spearman_trend_results.csv` — monotonic association of effect size with stage order.
- `linear_trend_results.csv` — linear slope of effect size on stage index (low power with few stages).
- `direction_test_summary.csv` — population-level counts of increasing vs decreasing motifs.

With **very few stages**, Spearman ρ can take only discrete values; interpret with care and report stage count.

## “Strict verdict” style filters (optional)

Some workflows define a **strict** motif call as: a formal trend test passes **and** the trajectory crosses a pre-defined boundary in effect-size or significance space (e.g. crosses effect size zero or a significance threshold). That is a **study-specific** rule: if you use it, define it in the protocol before viewing results. Outputs such as `motif_verdicts.csv` and related plots may exist in custom runs; column names follow the same idea as in your script version—use `--help` on the installed helper for the exact schema.

## Filtering for visualization (generic)

| Tier | Idea |
|------|------|
| Conservative | Filter to rows where your chosen formal method’s FDR flag is true. |
| Exploratory | Add quadrant or path-percentile highlights with clear labeling as non-confirmatory. |
| Descriptive | Color by a continuous metric (e.g. standardized path length) without a binary claim. |

## Reporting template (fill in with your own numbers)

Use neutral language tied to your pre-specified primary test, for example:

- “Motif *M* showed significant trajectory variation by permutation test (FDR = …).”
- “No motif exceeded the FDR threshold under the permutation null.”

Avoid copying numeric summaries from example analyses in older versions of this document; they are not part of the pipeline distribution.

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289-300.
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer.
- Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer.

---

*Cross-anchor comparison at a conceptual level: [Chapter 16: Cross-Anchor Comparative Analysis](16_Cross_Anchor_Comparative_Analysis.md).*
