# Chapter 5: Statistical Methods

## Overview

This chapter describes the statistical methods used in the MAPseq processing pipeline, including population estimation, binomial testing, multiple testing correction, and effect size calculation.

## Population Estimation (N₀)

### The Problem

We observe **N_obs** neurons that project to at least one target region. However, some neurons may project only to regions we did not sample, meaning the true population **N₀ ≥ N_obs**.

### Mathematical Model

Let **sₖ** = number of neurons observed projecting to region k (for k = 0, 1, ..., m regions).

We model the probability that a neuron is detected (projects to ≥1 sampled region) as:

$$\pi = 1 - \prod_{k=0}^{m}\left(1 - \frac{s_k}{N_0}\right)$$

**Interpretation:**
- Each term $\frac{s_k}{N_0}$ is the probability a neuron projects to region k
- $\prod(1 - \frac{s_k}{N_0})$ is the probability of projecting to *no* regions (under independence)
- $\pi$ is therefore the probability of being detected

### Solving for N₀

We solve the constraint equation:

$$\pi \cdot N_0 = N_{obs}$$

This is a polynomial equation in N₀ (degree = number of regions + 1). We use symbolic algebra (SymPy) to find roots and select the **largest real root > N_obs**.

**Code Implementation**: `process-nbcm-tsv.py::solve_for_roots()`

```python
def solve_for_roots(projections, observed_cells):
    N0, k = symbols('N_0 k')
    m = len(projections) - 1
    s = Array(list(projections.values()))
    pi = (1 - Product((1 - (s[k]/N0)), (k, 0, m)).doit())
    soln = sympy.solve(pi * N0 - observed_cells)
    roots = [N(x).as_real_imag()[0] for x in soln]
    return roots, pi

# Root selection (in main script after solve_for_roots)
valid_N0 = [root for root in roots if root.is_real and root > observed_cells]
N0_value = max(valid_N0)  # Largest valid root
```

### Key Assumption

**Independence:** Projections to different regions are statistically independent. This may not hold biologically if certain projection patterns co-occur—which is why we developed Model 3 (Correlated Binomial). See [Chapter 6: Probability Models](06_Probability_Models.md) for details.

## Binomial Testing Framework

### Test Structure

For each motif, we perform a two-tailed binomial test:

**Null Hypothesis**: Observed motif count follows a binomial distribution with:
- **n** = N₀ (estimated total population)
- **p** = P(motif) from probability model
- **k** = observed count for motif

**Alternative Hypothesis**: Observed count differs significantly from expected.

### Test Implementation

**Code Reference**: `process-nbcm-tsv.py` (binomial testing)

```python
from scipy.stats import binomtest

p_value = binomtest(int(observed), n=int(N0_value), p=max(prob, 1e-10)).pvalue
```

**Note**: The test uses N₀ (estimated population), not N_obs (observed cells). This accounts for neurons that may project only to unsampled regions.

### Two-Tailed Test

The binomial test is two-tailed, meaning it tests for both over-representation and under-representation:

- **Over-representation**: Observed > Expected
- **Under-representation**: Observed < Expected

## Multiple Testing Correction

### Bonferroni Correction

With multiple motifs tested simultaneously, we apply Bonferroni correction to control family-wise error rate:

$$\alpha_{corrected} = \frac{\alpha}{n_{motifs}}$$

Where:
- $\alpha$ = significance threshold (default: 0.05)
- $n_{motifs}$ = number of motifs tested

**Example**: With 32 motifs and α = 0.05:
$$\alpha_{corrected} = \frac{0.05}{32} \approx 0.00156$$

**Code Implementation**: `process-nbcm-tsv.py` (significance grouping)

```python
bonferroni_thr = alpha / len(motif_labels)
```

### Significance Criteria

A motif is considered **significant** if:
$$p\text{-value} < \alpha_{corrected}$$

## Effect Size Calculation

### Formula

Effect size quantifies the magnitude of deviation from model expectations:

$$Effect\ Size = \log_2\left(\frac{Observed + 1}{Expected + 1}\right)$$

**Code Reference**: `process-nbcm-tsv.py` (effect size calculation)

```python
effect_size = np.log2((observed + 1) / (expected + 1))
```

### Interpretation

| Effect Size | Interpretation |
|-------------|----------------|
| > 0 | Motif is **over-represented** (observed > expected) |
| < 0 | Motif is **under-represented** (observed < expected) |
| ≈ 0 | Motif frequency matches expectation |
| > 1.0 | Large over-representation (2× expected) |
| < -1.0 | Large under-representation (< 0.5× expected) |

### Pseudocount

The +1 pseudocount prevents:
- Division by zero (when Expected = 0)
- Log of zero (when Observed = 0)

## Significance Grouping

Motifs are classified into 4 groups based on effect size and significance:

| Group | Description | Criteria | Color (plots) |
|-------|-------------|----------|---------------|
| 1 | Significantly over-represented | Effect > 0 AND p < α_corrected | Red |
| 2 | Significantly under-represented | Effect ≤ 0 AND p < α_corrected | Blue |
| 3 | Non-significantly over-represented | Effect > 0 AND p ≥ α_corrected | Black |
| 4 | Non-significantly under-represented | Effect ≤ 0 AND p ≥ α_corrected | Black |

**Code Implementation**: `process-nbcm-tsv.py` (significance grouping)

```python
if effect_size > 0:  # over-represented
    if p_value < bonferroni_thr:  # statistically significant
        grp = 1  # significant + over-represented
    else:
        grp = 3  # non-significant + over-represented
else:  # under-represented
    if p_value < bonferroni_thr:  # statistically significant
        grp = 2  # significant + under-represented
    else:
        grp = 4  # non-significant + under-represented
```

## Anchor Model System

### Purpose

The anchor model system enables **cross-age comparative analysis** by establishing P60 (adult) as the baseline for comparing developmental cohorts (P3, P12, P20).

### Mathematical Formulation

For a developmental cohort (e.g., P3):

$$Expected_{motif} = P_{P60}(motif) \times N_0^{P3}$$

Where:
- **P_P60(motif)** = probability of motif from P60 anchor model
- **N₀^P3** = estimated population size for P3 cohort

### Key Insight

This separates **what projection patterns are biologically "expected"** (from P60 adult baseline) from **how many neurons exist in each developmental stage** (local N₀). Deviations indicate developmental differences in projection patterns.

**Code Reference**: `process-nbcm-tsv.py::load_anchor_model()`

## Statistical Assumptions

### Model Assumptions

1. **Independence** (Models 1 & 2): Projections to different regions are independent
2. **Binomial Distribution**: Motif counts follow binomial distribution
3. **Population Estimation**: N₀ estimation assumes independence (may not hold biologically)

### Violations and Alternatives

- **Correlated Projections**: Model 3 (Correlated Binomial) accounts for pairwise correlations
- **Model Misspecification**: Multiple models tested to assess robustness
- **Non-Binomial Distributions**: Consider alternative models (negative binomial, zero-inflated)

## Helper Script Statistical Methods

### Kruskal-Wallis Test

**Script**: `01_motif_analysis_per_animal.py`

**Purpose**: Non-parametric test comparing motif frequencies across age groups

**Null Hypothesis**: No difference in motif frequency distributions across ages

**Code Reference**: `helpers/scripts/01_motif_analysis_per_animal.py`

```python
from scipy.stats import kruskal
h_stat, p_value = kruskal(*valid_groups)
```

### Fisher's Exact Test

**Script**: `07_motif_significange_trajectories.py`

**Purpose**: Tests significance of transitions between consecutive stages

**Null Hypothesis**: No change in motif representation between stages

**Code Reference**: `helpers/scripts/07_motif_significange_trajectories.py`

```python
from scipy.stats import fisher_exact
_, p = fisher_exact([[a, b], [c, d]])
```

### Jensen-Shannon Divergence

**Scripts**: `01_motif_analysis_per_animal.py`, `05_motif_analysis.py`

**Purpose**: Measures distribution similarity between ages

**Range**: 0 (identical) to 1 (maximally different)

**Interpretation**:
- JSD < 0.05: Very similar distributions (stable)
- JSD 0.05-0.2: Moderate difference
- JSD > 0.2: Large difference (unstable)

**Code Reference**: Uses `scipy.spatial.distance.jensenshannon`

## Trajectory Significance Testing (Script 15)

Script 15 (`15_volcano_trajectories.py`) implements multiple citable statistical methods for identifying motif trajectories that change significantly across developmental stages. These methods address the fundamental question: "Does this motif's effect size/significance change meaningfully over time?"

### Transition Z-Test (Script 07)

**Script**: `07_motif_significange_trajectories.py`

**Purpose**: Tests whether effect size changed significantly between consecutive developmental stages.

**Method**: Two-sample z-test for the difference of log-transformed counts.

**Standard Error Derivation**:
For effect size $d = \log_2(\text{observed}/\text{expected})$, treating expected as fixed:

$$SE(d) = \frac{1}{\ln(2) \cdot \sqrt{\text{observed}}}$$

This follows from the delta method applied to Poisson-distributed counts.

**Test Statistic**:
$$z = \frac{d_2 - d_1}{\sqrt{SE_1^2 + SE_2^2}}$$

**P-value**: Two-sided p-value from standard normal distribution.

**Reference**: Oehlert, G.W. (1992). "A Note on the Delta Method." *The American Statistician*, 46(1), 27-29.

**Note**: FDR correction (Benjamini-Hochberg) is applied within each motif across the 3 transitions.

### Permutation Tests for Trajectory Significance

**Script**: `15_volcano_trajectories.py`

**Purpose**: Non-parametric test for whether a trajectory shows more temporal variation than expected by chance.

**Method**:
1. Compute observed test statistic: total variation in effect size across ordered stages
2. Generate null distribution by randomly permuting stage labels (10,000 permutations)
3. P-value = proportion of permuted statistics ≥ observed statistic

**Test Statistic**:
$$TV = \sum_{i=1}^{n-1} |d_{i+1} - d_i|$$

where $d_i$ is the effect size at stage $i$ in temporal order.

**Advantages**:
- Distribution-free (no normality assumptions)
- Exact p-values
- Robust to outliers

**Reference**: Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses*, 3rd ed., Springer.

**Code Reference**: `helpers/scripts/15_volcano_trajectories.py::compute_permutation_significance()`

### Functional Data Analysis (FDA) Trajectory Tests

**Script**: `15_volcano_trajectories.py`

**Purpose**: Tests whether a trajectory differs significantly from a flat (constant) function.

**Method**:
1. Treat each motif's effect size trajectory as a functional observation
2. Compute a "roughness" statistic combining range and integrated squared second derivative
3. Compare to null distribution via permutation (shuffled stage labels)

**Roughness Statistic**:
$$R = \text{range}(d) + \sum_{i=1}^{n-2} (\Delta^2 d_i)^2$$

where $\Delta^2 d_i = d_{i+2} - 2d_{i+1} + d_i$ is the discrete second difference.

**Interpretation**: Higher roughness indicates more non-constant trajectory behavior.

**Reference**: Ramsay, J.O. & Silverman, B.W. (2005). *Functional Data Analysis*, 2nd ed., Springer.

**Code Reference**: `helpers/scripts/15_volcano_trajectories.py::compute_fda_significance()`

### Mixed-Effects Models for Repeated Measures

**Script**: `15_volcano_trajectories.py`

**Purpose**: Model effect size as a function of developmental stage while accounting for motif-level variation.

**Model**:
$$\text{effect\_size}_{ij} = \beta_0 + \beta_1 \cdot \text{stage}_j + u_i + \varepsilon_{ij}$$

Where:
- $\beta_1$ = fixed effect of stage (tests: does stage affect effect size overall?)
- $u_i \sim N(0, \sigma_u^2)$ = random intercept for motif $i$
- $\varepsilon_{ij} \sim N(0, \sigma^2)$ = residual error

**Output**:
- Stage coefficient ($\beta_1$) and its p-value
- Per-motif random intercepts (identify outlier motifs)
- Model convergence status

**Reference**: Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). "Fitting Linear Mixed-Effects Models Using lme4." *Journal of Statistical Software*, 67(1), 1-48.

**Code Reference**: `helpers/scripts/15_volcano_trajectories.py::compute_mixed_effects_trajectory_test()`

### Bootstrap Confidence Intervals for Quadrant Classification

**Script**: `15_volcano_trajectories.py`

**Purpose**: Quantify uncertainty in volcano plot quadrant assignments.

**Problem**: Points near the significance threshold may flip quadrants due to sampling noise.

**Method**:
1. For each (effect_size, observed_count) pair, resample observed count from Poisson(observed)
2. Recompute effect size for each bootstrap sample
3. Classify quadrant for each sample
4. Report proportion of samples in each quadrant
5. "Robust" classification if >95% of samples agree

**Output**:
- `quadrant_raw`: Original classification
- `prop_sig_pos`, `prop_sig_neg`, `prop_not_sig`: Bootstrap proportions
- `quadrant_robust`: Robust classification ("uncertain" if <95% agreement)

**Reference**: Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*, Chapman & Hall.

**Code Reference**: `helpers/scripts/15_volcano_trajectories.py::compute_bootstrap_quadrant_ci()`

### Standardized Distance Metrics

The original path length calculation in (effect_size, significance) space combined incommensurate scales. Three alternatives are now provided:

#### Z-Score Standardized Path Length

**Method**: Standardize both axes to z-scores before computing Euclidean distance.

$$z_{\text{effect}} = \frac{d - \bar{d}}{\sigma_d}, \quad z_{\text{sig}} = \frac{s - \bar{s}}{\sigma_s}$$

$$PL_z = \sum_{i=1}^{n-1} \sqrt{(z_{\text{effect},i+1} - z_{\text{effect},i})^2 + (z_{\text{sig},i+1} - z_{\text{sig},i})^2}$$

**Advantage**: Puts both axes on comparable scales.

#### Mahalanobis Distance Path Length

**Method**: Use Mahalanobis distance, which accounts for correlation and variance structure.

$$PL_M = \sum_{i=1}^{n-1} \sqrt{(\mathbf{x}_{i+1} - \mathbf{x}_i)^T \Sigma^{-1} (\mathbf{x}_{i+1} - \mathbf{x}_i)}$$

where $\Sigma$ is the covariance matrix of (effect_size, significance).

**Advantage**: Accounts for correlation between effect size and significance.

**Reference**: Mahalanobis, P.C. (1936). "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 2, 49-55.

#### Separate Axis Metrics

**Method**: Analyze effect_size and significance independently.

**Output**:
- `effect_size_range`: max - min of effect sizes
- `effect_size_sd`: Standard deviation of effect sizes
- `effect_size_total_variation`: Sum of absolute differences
- Same metrics for significance axis

**Advantage**: Avoids scale mixing entirely; more interpretable.

**Code Reference**: `helpers/scripts/15_volcano_trajectories.py::compute_separate_axis_metrics()`

### Method Comparison and Recommendations

| Method | Best For | Citation |
|--------|----------|----------|
| Permutation test | Primary analysis; no distributional assumptions | Good (2005) |
| FDA | Detecting non-constant trajectories | Ramsay & Silverman (2005) |
| Mixed-effects | Pooling across experiments; random effects | Bates et al. (2015) |
| Bootstrap CI | Quantifying quadrant uncertainty | Efron & Tibshirani (1993) |
| Z-score path | Standardized distance metric | Standard |
| Mahalanobis | Covariance-aware distance | Mahalanobis (1936) |

**Recommendation for Publication**:
1. Use permutation tests or FDA as primary trajectory significance test
2. Report method agreement matrix to show robustness
3. Use bootstrap CI to identify uncertain quadrant classifications
4. Prefer separate axis metrics over raw path length for interpretability

### Methods Removed or Deprecated

**Kruskal-Wallis Test (Script 07)**: Removed. The original implementation applied Kruskal-Wallis with n=1 observation per group (each stage had one effect size per motif), which is statistically invalid. Kruskal-Wallis requires multiple observations per group to estimate within-group variance.

**Linear Regression Trend P-value (Script 07)**: Retained but flagged as **exploratory only**. With only n=4 observations per trajectory, the p-value from `linregress` has extremely low statistical power and should not be used for formal hypothesis testing. The slope itself remains useful as a descriptive statistic.

**Centroid Rule (Script 15)**: Retained for backward compatibility but flagged as **ad-hoc**. This method has no published precedent and unknown false positive/negative rates. Use permutation tests or FDA for publication-quality analysis.

## Summary

| Method | Purpose | Code Reference |
|--------|---------|----------------|
| N₀ Estimation | Population size estimation | `solve_for_roots()` in process-nbcm-tsv.py |
| Binomial Test | Motif significance testing | `binomtest()` in process-nbcm-tsv.py |
| Bonferroni Correction | Multiple testing correction | Lines 800+ |
| Effect Size | Magnitude of deviation | Line 1438 |
| Anchor Model | Cross-cohort comparison | `load_anchor_model()` in process-nbcm-tsv.py |

---

*For probability model details, see [Chapter 6: Probability Models](06_Probability_Models.md). For mathematical formulas, see [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions.md).*
