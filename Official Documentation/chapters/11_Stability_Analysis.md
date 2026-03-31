# Chapter 11: Stability Analysis

## Overview

This chapter describes the stability analysis framework for assessing whether projection motif frequencies change significantly across developmental timepoints. The analysis uses both model-independent and model-dependent metrics.

## Stability Analysis Framework

### Central Question

**Do projection motif frequencies change significantly over developmental time?**

This question is addressed using multiple complementary approaches:
1. **Model-independent analysis**: Direct comparison of observed frequencies
2. **Model-dependent analysis**: Comparison against statistical model predictions
3. **Transition analysis**: Specific stage-to-stage comparisons

### Key Metrics

| Metric | Type | What It Measures | Interpretation |
|--------|------|------------------|----------------|
| **Kruskal-Wallis** | Model-independent | Distribution of relative proportions across animals | Individual-level stability |
| **Effect Size Range** | Model-dependent | Change in aggregate deviation from model | Model-fit stability |
| **Transition Significance** | Model-independent | Proportion changes between consecutive stages | Proportional stability |
| **Jensen-Shannon Divergence** | Model-independent | Distribution similarity between ages | Magnitude of changes |

## Model-Independent Analysis

### Kruskal-Wallis Test

**Purpose**: Tests whether relative proportions across individual animals change over time.

**Null Hypothesis**: The distribution of normalized frequencies across animals is the same at all ages.

**Interpretation**:
- **p < 0.05**: Relative proportions across animals differ significantly across ages (UNSTABLE)
- **p ≥ 0.05**: Relative proportions across animals remain stable across ages (STABLE)

**Key Insight**: Tests whether animals maintain the same relative ranking of motif frequencies across development.

**Code Reference**: `helpers/scripts/01_motif_analysis_per_animal.py` (line 95)

**Example Results**:
- 0/31 motifs (0.0%) show significant changes → **STABLE**
- Mean p-value: 0.59, Median: 0.66

### Jensen-Shannon Divergence

**Purpose**: Measures distribution similarity between ages.

**Range**: 0 (identical) to 1 (maximally different)

**Interpretation**:
- **JSD < 0.05**: Very similar distributions (stable)
- **JSD 0.05-0.2**: Moderate difference
- **JSD > 0.2**: Large difference (unstable)

**Example Results**:
- P14→P16 vs P22→P24: JSD = 0.0013
- P22→P24 vs P60+→P62+: JSD = 0.0035
- P14→P16 vs P60+→P62+: JSD = 0.0019

All values < 0.05 → **STABLE**

## Model-Dependent Analysis

### Effect Size Trajectories

**Purpose**: Tests whether aggregate counts deviate from model predictions and how these deviations change.

**Calculation**: Effect size calculated at each developmental timepoint:

$$Effect\ Size = \log_2\left(\frac{Observed + 1}{Expected + 1}\right)$$

**Range**: max(effect_size) - min(effect_size) across ages

**Interpretation**:
- **Range > 1.0**: Large changes in deviation from model (UNSTABLE)
- **Range < 1.0**: Relatively stable deviation from model (STABLE)

**Key Insight**: Tests whether aggregate counts shift relative to model predictions across development.

**Example Results**:
- Uniform model: 71-90% of motifs show range > 1.0 → **UNSTABLE**
- Region-specific model: 74-87% of motifs show range > 1.0 → **UNSTABLE**

### Transition Significance

**Purpose**: Tests whether proportions change between consecutive stages.

**Method**: Fisher's exact test on 2×2 contingency tables

**Null Hypothesis**: The proportion of the motif (relative to all motifs) is the same in both stages.

**Interpretation**:
- **p < 0.05**: Proportion changed significantly between stages (UNSTABLE)
- **p ≥ 0.05**: Proportion remained stable between stages (STABLE)

**Code Reference**: `helpers/scripts/07_motif_significange_trajectories.py` (line 84)

**Example Results**:
- 13/52 transitions (25.0%) are significant → **MODERATE CHANGES**

## Logical Consistency

### The Apparent Paradox

**Finding 1**: Kruskal-Wallis shows 0% significant → **STABLE**

**Finding 2**: Effect size trajectories show 71-87% with large changes → **UNSTABLE**

**Finding 3**: Transition significance shows 25% significant → **MODERATE**

### Why This Is Logically Consistent

These tests measure **different aspects** of stability:

1. **Kruskal-Wallis**: Individual-level relative proportions
   - Tests whether animals maintain same relative ranking
   - Can be stable even if aggregate counts shift

2. **Effect Size**: Population-level model deviations
   - Tests whether aggregate counts shift relative to model
   - Can change even if relative proportions stay same

3. **Transition Significance**: Proportional changes
   - Tests whether proportions change between stages
   - Intermediate between complete stability and high instability

### Example Scenario

**Scenario**: Consistent Relative Proportions, Shifting Model Fit

- **P12**: Animal A has 10% of motif X, Animal B has 5% → Aggregate = 7.5%
- **P60**: Animal A has 10% of motif X, Animal B has 5% → Aggregate = 7.5%
- **Kruskal-Wallis**: Non-significant (relative proportions unchanged)
- **But if model's expected value changes**:
  - P12 effect size = log₂(7.5/5) = 0.585
  - P60 effect size = log₂(7.5/10) = -0.415
  - Range = 1.0 (large change!)

**Conclusion**: Relative proportions across animals can be stable while aggregate counts shift relative to model predictions.

## Interpretation Guidelines

### Overall Stability Assessment

1. **Start with Kruskal-Wallis results**:
   - Count motifs with p < 0.05: High count = unstable system
   - Count motifs with p ≥ 0.05: High count = stable system

2. **Examine transition significance**:
   - Which transitions have the most significant changes?
   - Are changes concentrated in specific transitions?

3. **Check effect size magnitudes**:
   - Large effect sizes (>1 or <-1) indicate substantial changes
   - Small effect sizes (<0.5) indicate minor changes

4. **Compare models**:
   - Do both models agree on stability?
   - Agreement suggests robust findings

### Motif-Specific Stability

**Stable Motifs** (p ≥ 0.05 in Kruskal-Wallis):
- Consistent representation across ages
- Small effect size changes
- No significant transitions

**Unstable Motifs** (p < 0.05 in Kruskal-Wallis):
- Changing representation across ages
- Large effect size changes
- Significant transitions

**Early-Changing Motifs** (significant P3→P12 transition):
- Changes occur early in development
- May stabilize later

**Late-Changing Motifs** (significant P20→P60 transition):
- Changes occur late in development
- May be maturation effects

### Magnitude of Changes

**JSD Interpretation**:
- < 0.05: Very stable (minimal change)
- 0.05-0.1: Moderately stable (small changes)
- 0.1-0.2: Moderately unstable (noticeable changes)
- > 0.2: Unstable (large changes)

**Effect Size Interpretation**:
- |Effect Size| < 0.5: Small change
- |Effect Size| 0.5-1.0: Moderate change
- |Effect Size| > 1.0: Large change

## Trajectory Significance Testing (v2.0)

Script 15 now implements multiple citable statistical methods for identifying which trajectories change significantly over time. These methods are appropriate for publication-quality analysis.

### Recommended Methods for Publication

| Priority | Method | Use Case | Citation |
|----------|--------|----------|----------|
| 1 | **Permutation test** | Primary trajectory significance test | Good (2005) |
| 2 | **FDA test** | Tests for non-constant trajectories | Ramsay & Silverman (2005) |
| 3 | **Mixed-effects model** | Population-level analysis; random effects | Bates et al. (2015) |
| Support | Bootstrap CI | Identify uncertain quadrant classifications | Efron & Tibshirani (1993) |
| Support | Z-score distance | Standardized path length metric | Standard |

### Method Selection Guide

**For Individual Motif Significance**:
- Use permutation test as primary method
- FDA test as secondary confirmation
- Report motifs significant by both (robust findings)

**For Population-Level Analysis**:
- Use mixed-effects model to test if stage affects effect sizes overall
- Extract random intercepts to identify outlier motifs

**For Uncertainty Quantification**:
- Enable bootstrap CI (`--bootstrap_ci`) to identify uncertain classifications
- Focus on motifs where most stages have robust quadrant assignments

### Interpreting Method Agreement

Check `method_comparison/method_agreement_matrix.csv`:

| Agreement Rate | Interpretation |
|----------------|----------------|
| >90% | Very robust: methods highly consistent |
| 80-90% | Robust: minor disagreements |
| 60-80% | Moderate: results depend on method |
| <60% | Low: investigate individual motifs carefully |

**When Methods Disagree**:
1. Check bootstrap CI for classification uncertainty
2. Examine individual trajectory plots
3. Consider biological plausibility
4. Report as "borderline" or "uncertain" in publication

### Methods to Avoid for Publication

**Centroid Rule (Script 15 Legacy)**:
- Non-standard, ad-hoc method
- No published precedent
- Unknown false positive/negative rates
- Use only for exploratory analysis, not formal testing

**Kruskal-Wallis on Single Observations (Script 07 Legacy)**:
- Removed in v2.0 as statistically invalid
- Was applied with n=1 per group (invalid)

**Linear Regression Trend P-value (Script 07)**:
- Flagged as exploratory only
- Very low power with n=4 points
- Report slope as descriptive statistic only

### Publication Recommendations

1. **Primary Analysis**: Report permutation test results
   - "Trajectory significance was assessed using permutation tests (Good, 2005; 10,000 permutations)"
   - Report FDR-adjusted p-values

2. **Robustness Check**: Report method agreement
   - "Results were robust across methods (X% agreement between permutation and FDA tests)"

3. **Supplementary**: Include method comparison table
   - Show which motifs are significant by each method
   - Include `method_agreement_matrix.csv` in supplementary materials

4. **Uncertainty**: Report bootstrap CI where relevant
   - "X motifs had uncertain quadrant classifications (bootstrap CI <95% in same quadrant)"

## Recommended Analysis Workflow

### Step 1: Model-Independent Analysis

1. Run Script 01: Get Kruskal-Wallis results
2. Calculate JSD between age pairs
3. Assess overall stability

### Step 2: Model-Dependent Analysis

1. Run Script 07: Get effect size trajectories
2. Calculate effect size ranges
3. Assess model-fit stability

### Step 3: Transition Analysis

1. Run Script 07: Get transition significance (z-test)
2. Identify which transitions show changes
3. Assess when changes occur

### Step 4: Trajectory Significance Testing (NEW)

1. Run Script 15 with all methods: `--methods all`
2. Check `permutation/permutation_test_results.csv` for primary results
3. Check `fda/fda_trajectory_significance.csv` for confirmation
4. Check `method_comparison/method_agreement_matrix.csv` for robustness
5. Optionally enable `--bootstrap_ci` for uncertainty quantification

### Step 5: Integration

1. Compare model-independent and model-dependent findings
2. Identify motifs significant by multiple methods (robust)
3. Flag motifs with method disagreement for careful review
4. Interpret biological significance

## Key Files for Stability Analysis

### Primary Data Sources

1. **Script 01: Kruskal-Wallis Results** ⭐ MOST IMPORTANT
   - Location: `01_motif_analysis_per_animal/{model}/`
   - Direct statistical test of temporal stability

2. **Script 07: Combined Effect Sizes CSV** ⭐ MOST IMPORTANT
   - Location: `07_motif_significange_trajectories/{model}/combined_effect_sizes_{model}.csv`
   - Effect size trajectories for each motif

3. **Script 07: Transition Significance CSV** ⭐ MOST IMPORTANT
   - Location: `07_motif_significange_trajectories/{model}/transition_significance.csv`
   - P-values for each transition (z-test, FDR-adjusted)

4. **Script 15: Permutation Test Results** ⭐ PUBLICATION-READY
   - Location: `15_volcano_trajectories/{model}/permutation/permutation_test_results.csv`
   - Non-parametric trajectory significance test
   - FDR-adjusted p-values

5. **Script 15: Method Comparison Summary** ⭐ PUBLICATION-READY
   - Location: `15_volcano_trajectories/{model}/method_comparison/all_methods_summary.csv`
   - Cross-method comparison for robustness

### Secondary Data Sources

6. **Script 05: Motif Percentage Matrix** ⚠️ SECONDARY
   - Location: `05_motif_analysis/motif_percent_matrix_by_age_{model}.csv`
   - Quantitative percentages for each motif at each age

7. **Script 15: FDA Test Results** ⚠️ SECONDARY
   - Location: `15_volcano_trajectories/{model}/fda/fda_trajectory_significance.csv`
   - Functional data analysis trajectory test

8. **Script 15: Distance Metrics** ⚠️ SECONDARY
   - Location: `15_volcano_trajectories/{model}/distance_metrics/distance_comparison.csv`
   - Standardized path length and axis-specific metrics

9. **Age-Specific Upsetplot CSVs** ⚠️ SECONDARY
   - Location: `{age}/{parameterization}/analysis/{model}/*_upsetplot_{model}.csv`
   - Core motif data for each age

## Summary

### Answering "Do Results Change Significantly Over Time?"

**Primary Evidence**:
1. **Kruskal-Wallis Test**: Model-independent test of frequency distributions
2. **Transition Significance**: Tests specific stage transitions
3. **Effect Size Trajectories**: Quantifies magnitude of changes

**Interpretation**:
- **Many significant results** = unstable system
- **Few significant results** = stable system
- **Mixed results** = selective changes in specific motifs

**Key Insight**: Model-independent and model-dependent metrics can give different results without contradiction—they measure different aspects of stability.

---

*For detailed statistical methods, see [Chapter 5: Statistical Methods](05_Statistical_Methods.md). For helper script details, see [Chapter 7: Helper Scripts](07_Helper_Scripts.md).*
