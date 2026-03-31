# Chapter 6: Probability Models

## Overview

The MAPseq pipeline implements multiple probability models to identify non-random projection patterns. This chapter describes all available models, their assumptions, mathematical formulations, and implementations.

## Model Comparison

| Model | Parameters | Independence | Use Case |
|-------|------------|--------------|----------|
| **Uniform (pₑ)** | 1 | Full independence | Baseline comparison |
| **Region-Specific (pᵢ)** | n (one per region) | Independence between regions | Region-specific rates |
| **Correlated** | n + n² | Pairwise correlations | Captures correlations |
| **Empirical** | Observed frequencies | None | Perfect P60 fit |
| **Smoothed Empirical** | Observed + smoothing | None | Regularized empirical |
| **Max Entropy** | Constraints (marginals + correlations) | None | Principled approach |
| **Hierarchical Correlations** | Higher-order correlations | None | Complex interactions |
| **Negative Binomial** | pᵢ + dispersion | Independence | Overdispersion |
| **Zero-Inflated** | pᵢ + zero-inflation | Independence | Excess zeros |
| **Bayesian Hierarchical** | Dirichlet prior | Independence | Uncertainty quantification |
| **ML Non-Parametric** | Learned function | None | Maximum flexibility |

## Model 1: Uniform Edge Probability (pₑ)

### Assumption

Every neuron has the same probability **pₑ** of projecting to any given region, independent of which region.

### Derivation of pₑ

Given N₀ and n regions, if each projection has probability pₑ:

$$P(\text{detected}) = 1 - (1 - p_e)^n$$

Since we observe N_obs neurons out of N₀:

$$\left(1 - (1-p_e)^n\right) \cdot N_0 = N_{obs}$$

We solve this symbolically for pₑ, selecting solutions in (0, 1).

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities()` (uniform)

```python
pe_solutions = sympy.solve(
    (1 - (1 - pe)**len(projections)) * N0_value - observed_cells, pe, force=True
)
```

### Motif Probability

Under this model, the probability of a motif with k regions:

$$P(k \text{ regions}) = p_e^k \cdot (1-p_e)^{(R-k)}$$

Where R = total number of regions.

**Code Reference**: `process-nbcm-tsv.py` (uniform pₑ calculation)

```python
motif_probs_uniform = {
    n: (pe_num ** n) * ((1 - pe_num) ** (total_regions - n))
    for n in range(1, total_regions + 1)
}
```

### Limitations

- Assumes all regions equally likely (biologically unrealistic)
- Ignores region-specific differences in projection rates

## Model 2: Region-Specific Probability (pᵢ)

### Assumption

Each region i has its own projection probability based on observed data:

$$p_i = \frac{N_i}{N_0}$$

Where:
- $N_i$ = number of neurons projecting to region i
- $N_0$ = estimated total population

### Motif Probability Formula

For a specific motif defined by subset M of regions:

$$P(\text{motif } M) = \prod_{i \in M} p_i \cdot \prod_{j \notin M} (1 - p_j)$$

**Interpretation:** Treats each region's projection as an independent Bernoulli trial with region-specific success probability.

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_region_specific()`

```python
def compute_motif_probabilities_region_specific(motif_labels, region_probs_dict, region_names):
    motif_probs_rs = {}
    for i, motif_regions in enumerate(motif_labels):
        prob = 1.0
        for region in region_names:
            if region in motif_regions:
                prob *= region_probs_dict.get(region, 0.0)  # p_i
            else:
                prob *= (1.0 - region_probs_dict.get(region, 0.0))  # (1 - p_i)
        motif_probs_rs[i] = prob
    return motif_probs_rs
```

### Limitations

- Assumes independence between regions (may not hold biologically)
- May underestimate multi-region motifs if correlations exist

## Model 3: Correlated Binomial

### Motivation

Models 1 and 2 assume **independence** between projections to different regions. However, biological evidence suggests projection patterns may be correlated—neurons projecting to region A may be more (or less) likely to also project to region B.

### Conditional Probability Matrix

We compute **P(B|A)** = probability of projecting to region B, given projection to region A:

$$P(B|A) = \frac{\text{neurons projecting to both A and B}}{\text{neurons projecting to A}}$$

**Code Reference**: `process-nbcm-tsv.py::gen_prob_matrix()`

```python
def gen_prob_matrix(df: pd.DataFrame):
    data = df.to_numpy(copy=True)
    cells, regions = data.shape
    mat = np.zeros((regions, regions))  # area B x area A
    
    for col in range(regions):  # For each region A
        ids_col = np.where(data[:, col] != 0)[0]
        sub_col = data[ids_col]
        
        for row in range(regions):  # For each region B
            ids_row = np.where(sub_col[:, row] != 0)[0]
            if ids_col.shape[0] == 0:
                prob = 0
            else:
                prob = ids_row.shape[0] / ids_col.shape[0]  # P(B|A)
            mat[col, row] = prob
    
    return mat
```

### Motif Probability Using Chain Rule

For multi-region motifs, we use the chain rule of probability:

$$P(A, B, C, ...) = P(A) \cdot P(B|A) \cdot P(C|B) \cdot ...$$

Regions are sorted alphabetically for consistent ordering.

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_correlated()`

```python
def compute_motif_probabilities_correlated(motif_labels, region_probs_dict, 
                                           cond_prob_matrix, region_names):
    motif_probs_corr = {}
    
    for i, motif_regions in enumerate(motif_labels):
        if len(motif_regions) == 1:
            # Single region: use marginal probability
            region = motif_regions[0]
            motif_probs_corr[i] = region_probs_dict.get(region, 0.0)
        else:
            # Multi-region: use chain rule
            sorted_regions = sorted(motif_regions)
            
            # Start with first region's marginal probability
            prob = region_probs_dict.get(sorted_regions[0], 0.0)
            
            # Multiply by conditional probabilities
            for j in range(1, len(sorted_regions)):
                prev_region = sorted_regions[j-1]
                curr_region = sorted_regions[j]
                cond_prob = cond_prob_matrix.loc[prev_region, curr_region]
                prob *= cond_prob
            
            motif_probs_corr[i] = prob
    
    return motif_probs_corr
```

### Limitations

- Only captures **pairwise** correlations
- Chain rule assumes **ordering matters** (may not be optimal)
- May not capture higher-order interactions

## Additional Models

### Empirical Model

**Description**: Uses observed P60 motif frequencies directly as expected values.

**Formula**: 
$$P(motif) = \frac{observed\_count(motif)}{N0\_P60}$$

**Pros**: Perfect fit to P60 (0% significant)
**Cons**: Overfitting risk, no generalization

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_empirical()`

### Smoothed Empirical Model

**Description**: Empirical frequencies with additive smoothing (Laplace).

**Formula**:
$$P(motif) = \frac{observed\_count(motif) + \alpha}{N0\_P60 + \alpha \times total\_motifs}$$

**Pros**: Reduces overfitting, handles sparse data
**Cons**: Arbitrary smoothing parameter α

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_smoothed_empirical()`

### Maximum Entropy Model

**Description**: Finds probability distribution that matches P60 constraints (marginals, pairwise correlations) while being maximally uniform.

**Mathematical Formulation**:
- Maximize entropy: $H = -\Sigma P(motif) \times \log P(motif)$
- Subject to constraints:
  - $\Sigma P(motif) = 1$ (normalization)
  - $\Sigma P(motif) \times I(region_i \in motif) = p_i$ (marginals)
  - $\Sigma P(motif) \times I(region_i, region_j \in motif) = p_{ij}$ (pairwise correlations)

**Pros**: Theoretically principled, matches constraints
**Cons**: Computationally complex, may not fit perfectly

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_max_entropy()`

### Hierarchical Correlations Model

**Description**: Explicitly models higher-order interactions (3-way, 4-way correlations).

**Pros**: Captures complex interactions
**Cons**: Very complex, many parameters, overfitting risk

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_hierarchical_correlations()`

### Negative Binomial Model

**Description**: Accounts for overdispersion (variance > mean) in motif counts.

**Model**: $Count \sim NegativeBinomial(mean, dispersion)$

**Pros**: Handles overdispersion, better uncertainty quantification
**Cons**: May not solve systematic bias

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_negative_binomial()`

### Zero-Inflated Model

**Description**: Accounts for excess zeros (many motifs never observed).

**Model**: Two-component:
- With probability π: motif count = 0 (structural zero)
- With probability (1-π): motif count ~ Binomial/NegativeBinomial

**Pros**: Handles excess zeros, biologically plausible
**Cons**: Additional complexity

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_zero_inflated()`

### Bayesian Hierarchical Model

**Description**: Models motif probabilities with hierarchical structure and uncertainty.

**Prior**: $P(motif) \sim Dirichlet(\alpha_1, \alpha_2, ..., \alpha_n)$

**Pros**: Quantifies uncertainty, handles sparse data
**Cons**: Computationally intensive (MCMC), requires prior specification

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_bayesian_hierarchical()`

### ML Non-Parametric Model

**Description**: Uses machine learning to learn motif probability function from P60 data.

**Pros**: Flexible, no explicit assumptions
**Cons**: Black box, overfitting risk, less interpretable

**Code Reference**: `process-nbcm-tsv.py::compute_motif_probabilities_ml_nonparametric()`

## Anchor Model Integration

All models (except Uniform) can use the anchor model system:

**For P60 (anchor)**:
- Calculate probabilities from P60 data
- Save probabilities and correlation matrices

**For developmental cohorts (P3, P12, P20)**:
- Load P60 anchor probabilities
- Use anchor probabilities with local N₀:
  $$Expected_{motif} = P_{P60}(motif) \times N_0^{local}$$

**Code Reference**: `process-nbcm-tsv.py` (region-specific probability calculation)

```python
psdict_for_expected = anchor_probs_loaded if anchor_probs_loaded else psdict_region_specific
corr_matrix_for_expected = anchor_corr_loaded
```

## Model Selection

### Recommendations

**For Strict Fit (<10% significant) with Cross-Cohort Comparability:**
- **Smoothed Empirical Model**: Balances fit quality with overfitting concerns
- **Maximum Entropy Model**: Theoretically principled, matches constraints

**For Perfect P60 Fit:**
- **Pure Empirical Model**: Guarantees 0% significant in P60

**For Biological Interpretation:**
- **Region-Specific Model**: Most interpretable, region-specific rates
- **Correlated Model**: Captures pairwise correlations

**For Robustness Assessment:**
- Compare results across multiple models
- Agreement suggests robust findings

## Model Comparison Table

| Model | Fit Quality | Complexity | Interpretability | Cross-Cohort |
|-------|-------------|------------|-------------------|--------------|
| Uniform | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Region-Specific | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Correlated | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Empirical | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Smoothed Empirical | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Max Entropy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Hierarchical Correlations | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Negative Binomial | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Zero-Inflated | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Bayesian Hierarchical | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| ML Non-Parametric | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |

---

*For detailed mathematical formulations, see [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions.md). For statistical testing, see [Chapter 5: Statistical Methods](05_Statistical_Methods.md).*
