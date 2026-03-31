# Chapter 10: Mathematical Functions Reference

## Overview

This chapter provides a comprehensive reference for all mathematical formulas used in the MAPseq processing pipeline, with code implementation references and interpretations.

## Population Estimation

### N₀ Estimation Formula

**Mathematical Formulation**:

$$\pi = 1 - \prod_{k=0}^{m}\left(1 - \frac{s_k}{N_0}\right)$$

$$\pi \cdot N_0 = N_{obs}$$

Where:
- $\pi$ = detection probability
- $s_k$ = number of neurons projecting to region k
- $N_0$ = true (latent) population size
- $N_{obs}$ = observed neurons (project to ≥1 region)
- $m$ = number of regions

**Code Implementation**: `process-nbcm-tsv.py::solve_for_roots()`

```python
pi = (1 - Product((1 - (s[k]/N0)), (k, 0, m)).doit())
soln = sympy.solve(pi * N0 - observed_cells)
```

**Interpretation**: Uses inclusion-exclusion principle to estimate true population accounting for neurons projecting only to unsampled regions.

**Assumptions**: Projections to different regions are statistically independent.

## Uniform Edge Probability Model

### pₑ Calculation

**Mathematical Formulation**:

$$P(\text{detected}) = 1 - (1 - p_e)^n$$

$$\left(1 - (1-p_e)^n\right) \cdot N_0 = N_{obs}$$

**Code Implementation**: `process-nbcm-tsv.py` (uniform pₑ)

```python
pe_solutions = sympy.solve(
    (1 - (1 - pe)**len(projections)) * N0_value - observed_cells, pe, force=True
)
```

**Interpretation**: Solves for uniform edge probability pₑ assuming all regions equally likely.

### Motif Probability (Uniform)

**Mathematical Formulation**:

$$P(k \text{ regions}) = p_e^k \cdot (1-p_e)^{(R-k)}$$

Where:
- $k$ = number of regions in motif
- $R$ = total number of regions

**Code Implementation**: `process-nbcm-tsv.py` (uniform model)

```python
motif_probs_uniform = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n))
    for n in range(1, total_regions + 1)
}
```

## Region-Specific Probability Model

### Region Probability

**Mathematical Formulation**:

$$p_i = \frac{N_i}{N_0}$$

Where:
- $N_i$ = number of neurons projecting to region i
- $N_0$ = estimated total population

**Code Implementation**: `process-nbcm-tsv.py` (region probabilities)

```python
psdict_region_specific = {
    region: (count / float(N0_value)) 
    for region, count in projections.items()
}
```

### Motif Probability (Region-Specific)

**Mathematical Formulation**:

$$P(\text{motif } M) = \prod_{i \in M} p_i \cdot \prod_{j \notin M} (1 - p_j)$$

Where:
- $M$ = set of regions in motif
- $p_i$ = probability of projecting to region i

**Code Implementation**: `process-nbcm-tsv.py::compute_motif_probabilities_region_specific()`

```python
for region in region_names:
    if region in motif_regions:
        prob *= region_probs_dict.get(region, 0.0)  # p_i
    else:
        prob *= (1.0 - region_probs_dict.get(region, 0.0))  # (1 - p_i)
```

**Interpretation**: Treats each region's projection as independent Bernoulli trial.

## Correlated Binomial Model

### Conditional Probability

**Mathematical Formulation**:

$$P(B|A) = \frac{\text{neurons projecting to both A and B}}{\text{neurons projecting to A}}$$

**Code Implementation**: `process-nbcm-tsv.py::gen_prob_matrix()`

```python
prob = ids_row.shape[0] / ids_col.shape[0]  # P(B|A)
```

### Motif Probability (Correlated)

**Mathematical Formulation**:

$$P(A, B, C, ...) = P(A) \cdot P(B|A) \cdot P(C|B) \cdot ...$$

**Code Implementation**: `process-nbcm-tsv.py::compute_motif_probabilities_correlated()`

```python
prob = region_probs_dict.get(sorted_regions[0], 0.0)  # P(A)
for j in range(1, len(sorted_regions)):
    prev_region = sorted_regions[j-1]
    curr_region = sorted_regions[j]
    cond_prob = cond_prob_matrix.loc[prev_region, curr_region]  # P(B|A)
    prob *= cond_prob
```

**Interpretation**: Uses chain rule of probability with pairwise conditional probabilities.

## Effect Size

### Effect Size Formula

**Mathematical Formulation**:

$$Effect\ Size = \log_2\left(\frac{Observed + 1}{Expected + 1}\right)$$

**Code Implementation**: `process-nbcm-tsv.py` (effect size)

```python
effect_size = np.log2((observed + 1) / (expected + 1))
```

**Interpretation**:
- Effect Size > 0: Over-represented
- Effect Size < 0: Under-represented
- Effect Size ≈ 0: Matches expectation

**Pseudocount**: +1 prevents division by zero and log of zero.

## Statistical Tests

### Binomial Test

**Mathematical Formulation**:

Two-tailed binomial test:
- **n** = N₀ (estimated population)
- **k** = observed count
- **p** = P(motif) from model

**Code Implementation**: `process-nbcm-tsv.py` (binomtest)

```python
p_value = binomtest(int(observed), n=int(N0_value), p=max(prob, 1e-10)).pvalue
```

### Bonferroni Correction

**Mathematical Formulation**:

$$\alpha_{corrected} = \frac{\alpha}{n_{motifs}}$$

**Code Implementation**: `process-nbcm-tsv.py` (significance grouping)

```python
bonferroni_thr = alpha / len(motif_labels)
```

## Entropy Calculation

### Normalized Entropy

**Mathematical Formulation**:

$$H_{norm} = \frac{H}{H_{max}} = \frac{-\sum p_i \log p_i}{\log n}$$

Where:
- $p_i$ = probability of region i
- $n$ = number of regions
- $H_{max} = \log n$ (entropy of uniform distribution)

**Code Implementation**: Based on `docs/Entropy Calculation Notes.txt`

```python
probs = counts / counts.sum()
norm_entropy = entropy(probs) / np.log(len(probs))
```

**Interpretation**:
- 0.0: All projections to one region (highly focused)
- 1.0: Evenly distributed across all regions (highly distributed)
- 0.2-0.5: Moderate focus

## Helper Script Formulas

### Kruskal-Wallis Test

**Mathematical Formulation**:

$$H = \frac{12}{N(N+1)}\sum_{i=1}^{k}\frac{R_i^2}{n_i} - 3(N+1)$$

Where:
- $N$ = total number of observations
- $k$ = number of groups
- $R_i$ = sum of ranks in group i
- $n_i$ = number of observations in group i

**Code Implementation**: `helpers/scripts/01_motif_analysis_per_animal.py`

```python
from scipy.stats import kruskal
h_stat, p_value = kruskal(*valid_groups)
```

### Fisher's Exact Test

**Mathematical Formulation**:

For 2×2 contingency table:
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

$$p = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}} = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{a!b!c!d!n!}$$

**Code Implementation**: `helpers/scripts/07_motif_significange_trajectories.py`

```python
from scipy.stats import fisher_exact
_, p = fisher_exact([[a, b], [c, d]])
```

### Jensen-Shannon Divergence

**Mathematical Formulation**:

$$JSD(P||Q) = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)$$

Where:
- $M = \frac{1}{2}(P + Q)$ (mixture distribution)
- $D_{KL}$ = Kullback-Leibler divergence

**Range**: 0 (identical) to 1 (maximally different)

**Code Implementation**: Uses `scipy.spatial.distance.jensenshannon`

```python
from scipy.spatial.distance import jensenshannon
js_div = jensenshannon(p, q)**2
```

**Interpretation**:
- JSD < 0.05: Very similar (stable)
- JSD 0.05-0.2: Moderate difference
- JSD > 0.2: Large difference (unstable)

### Welch's t-test

**Mathematical Formulation**:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Where:
- $\bar{X}_i$ = sample mean
- $s_i$ = sample standard deviation
- $n_i$ = sample size

**Code Implementation**: `helpers/scripts/05_motif_analysis.py`

```python
from scipy.stats import ttest_ind
t_stat, t_p = ttest_ind(vec1, vec2, equal_var=False)
```

### Kolmogorov-Smirnov Test

**Mathematical Formulation**:

$$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$

Where:
- $F_{1,n}$ = empirical CDF of sample 1
- $F_{2,m}$ = empirical CDF of sample 2

**Code Implementation**: `helpers/scripts/05_motif_analysis.py`

```python
from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(vec1, vec2)
```

## Anchor Model System

### Expected Count Calculation

**Mathematical Formulation**:

$$Expected_{motif} = P_{P60}(motif) \times N_0^{local}$$

Where:
- $P_{P60}(motif)$ = probability from P60 anchor model
- $N_0^{local}$ = local population estimate

**Code Implementation**: `process-nbcm-tsv.py` (expected count with anchor)

```python
psdict_for_expected = anchor_probs_loaded if anchor_probs_loaded else psdict_region_specific
exp_counts = n0 * motif_probs_array
```

## Additional Model Formulas

### Smoothed Empirical

**Mathematical Formulation**:

$$P(motif) = \frac{observed\_count(motif) + \alpha}{N0\_P60 + \alpha \times total\_motifs}$$

**Code Implementation**: `process-nbcm-tsv.py::compute_motif_probabilities_smoothed_empirical()`

### Maximum Entropy

**Mathematical Formulation**:

Maximize: $H = -\sum P(motif) \times \log P(motif)$

Subject to:
- $\sum P(motif) = 1$ (normalization)
- $\sum P(motif) \times I(region_i \in motif) = p_i$ (marginals)
- $\sum P(motif) \times I(region_i, region_j \in motif) = p_{ij}$ (pairwise correlations)

**Code Implementation**: `process-nbcm-tsv.py::compute_motif_probabilities_max_entropy()`

## Mathematical Notation Glossary

| Symbol | Definition |
|--------|------------|
| $N_0$ | True (latent) neuron population size |
| $N_{obs}$ | Observed neurons (project to ≥1 region) |
| $n$ | Number of target brain regions |
| $s_k$ | Neurons projecting to region k |
| $\pi$ | Detection probability |
| $p_e$ | Uniform edge probability (Model 1) |
| $p_i$ | Region-specific probability (Model 2) |
| $P(B\|A)$ | Conditional probability: B given A (Model 3) |
| $H$ | Entropy |
| $JSD$ | Jensen-Shannon Divergence |
| $\alpha$ | Significance threshold |
| $\alpha_{corrected}$ | Bonferroni-corrected threshold |

---

*For code implementations, see [Chapter 9: Code Review](09_Code_Review.md). For statistical methods, see [Chapter 5: Statistical Methods](05_Statistical_Methods.md).*
