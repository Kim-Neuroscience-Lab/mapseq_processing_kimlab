# Chapter 13: References and Appendices

## Code File References

### Main Scripts

| File | Purpose | Key Functions |
|------|---------|---------------|
| `process-nbcm-tsv.py` | Main processing pipeline | `solve_for_roots()`, `clean_and_filter()`, `compute_motif_probabilities_*()` |
| `preprocess_and_aggregate.py` | Data preprocessing | `preprocess_file()`, `get_column_mapping()` |
| `postprocessing_checks.py` | Quality control | Main QC checking functions |

### Helper Scripts

| File | Purpose | Key Statistical Tests |
|------|---------|----------------------|
| `01_motif_analysis_per_animal.py` | Per-animal analysis | Kruskal-Wallis, JSD |
| `02_projection_analysis.py` | Projection analysis | PCA, clustering |
| `03_composition.py` | Composition analysis | Descriptive statistics |
| `04_proportions_over_time_stats.py` | Proportions over time | Chi-square, CLR |
| `05_motif_analysis.py` | Motif analysis | Welch's t-test, KS test, JSD |
| `06_all_motif_divergence.py` | Motif divergence | Visualization |
| `07_motif_significange_trajectories.py` | Trajectory analysis | Fisher's exact test |
| `08_motif_clustering.py` | Motif clustering | Clustering algorithms |
| `09_plot_normalized_projection_strength_data.py` | Projection strength | Visualization |
| `13_aggregate_projection_summaries.py` | Aggregation | Data aggregation |

### Code Line References

**Note:** Line numbers are approximate and may change; locate implementations by function name (e.g. `load_df`, `plot_effect_significance`) in `process-nbcm-tsv.py`.

#### Population Estimation

- **N₀ Estimation**: `process-nbcm-tsv.py::solve_for_roots()`
- **Root Selection**: `process-nbcm-tsv.py` (valid_N0 selection after `solve_for_roots()`)

#### Probability Models

- **Uniform Model**: `process-nbcm-tsv.py::compute_motif_probabilities()`
- **Region-Specific Model**: `process-nbcm-tsv.py::compute_motif_probabilities_region_specific()`
- **Correlated Model**: `process-nbcm-tsv.py::compute_motif_probabilities_correlated()`
- **Conditional Probability Matrix**: `process-nbcm-tsv.py::gen_prob_matrix()`

#### Statistical Testing

- **Binomial Test**: `process-nbcm-tsv.py` (uses `scipy.stats.binomtest`)
- **Effect Size**: `process-nbcm-tsv.py` (log₂(Observed+1)/(Expected+1))
- **Bonferroni Correction**: `process-nbcm-tsv.py`

#### Helper Scripts

- **Kruskal-Wallis**: `helpers/scripts/01_motif_analysis_per_animal.py`
- **Fisher's Exact Test**: `helpers/scripts/07_motif_significange_trajectories.py`

## Mathematical Notation Glossary

| Symbol | Definition | Chapter Reference |
|--------|------------|-------------------|
| $N_0$ | True (latent) neuron population size | [Chapter 5](05_Statistical_Methods.md), [Chapter 10](10_Mathematical_Functions.md) |
| $N_{obs}$ | Observed neurons (project to ≥1 region) | [Chapter 5](05_Statistical_Methods.md) |
| $n$ | Number of target brain regions | [Chapter 5](05_Statistical_Methods.md) |
| $s_k$ | Neurons projecting to region k | [Chapter 5](05_Statistical_Methods.md) |
| $\pi$ | Detection probability | [Chapter 5](05_Statistical_Methods.md) |
| $p_e$ | Uniform edge probability (Model 1) | [Chapter 6](06_Probability_Models.md) |
| $p_i$ | Region-specific probability (Model 2) | [Chapter 6](06_Probability_Models.md) |
| $P(B\|A)$ | Conditional probability: B given A (Model 3) | [Chapter 6](06_Probability_Models.md) |
| $H$ | Entropy | [Chapter 10](10_Mathematical_Functions.md) |
| $JSD$ | Jensen-Shannon Divergence | [Chapter 10](10_Mathematical_Functions.md) |
| $\alpha$ | Significance threshold | [Chapter 5](05_Statistical_Methods.md) |
| $\alpha_{corrected}$ | Bonferroni-corrected threshold | [Chapter 5](05_Statistical_Methods.md) |

## Statistical Test Reference Table

| Test | Purpose | Null Hypothesis | Code Reference |
|------|---------|-----------------|----------------|
| **Binomial Test** | Motif significance | Observed = Expected | `scipy.stats.binomtest` |
| **Kruskal-Wallis** | Frequency distribution comparison | No difference across ages | `scipy.stats.kruskal` |
| **Fisher's Exact Test** | Transition significance | No change between stages | `scipy.stats.fisher_exact` |
| **Welch's t-test** | Mean comparison | No difference in means | `scipy.stats.ttest_ind` |
| **Kolmogorov-Smirnov** | Distribution comparison | Distributions identical | `scipy.stats.ks_2samp` |
| **Chi-square** | Independence test | Proportions independent of age | `scipy.stats.chi2_contingency` |

## Model Comparison Table

| Model | Parameters | Independence | Fit Quality | Complexity |
|-------|------------|--------------|-------------|------------|
| **Uniform (pₑ)** | 1 | Full | ⭐⭐ | ⭐ |
| **Region-Specific (pᵢ)** | n | Between regions | ⭐⭐⭐ | ⭐⭐ |
| **Correlated** | n + n² | Pairwise correlations | ⭐⭐⭐ | ⭐⭐⭐ |
| **Empirical** | Observed frequencies | None | ⭐⭐⭐⭐⭐ | ⭐ |
| **Smoothed Empirical** | Observed + smoothing | None | ⭐⭐⭐⭐ | ⭐⭐ |
| **Max Entropy** | Constraints | None | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hierarchical Correlations** | Higher-order | None | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Negative Binomial** | pᵢ + dispersion | Independence | ⭐⭐ | ⭐⭐⭐ |
| **Zero-Inflated** | pᵢ + zero-inflation | Independence | ⭐⭐⭐ | ⭐⭐⭐ |
| **Bayesian Hierarchical** | Dirichlet prior | Independence | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ML Non-Parametric** | Learned function | None | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## External Dependencies

### Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| `numpy` | Numerical computations | Latest |
| `pandas` | Data manipulation | Latest |
| `scipy` | Statistical functions | Latest |
| `sympy` | Symbolic mathematics | ≥1.9 |
| `matplotlib` | Plotting | Latest |
| `seaborn` | Statistical visualization | Latest |
| `scikit-learn` | Machine learning | Latest |
| `upsetplot` | Set visualization | Latest |

### Optional Dependencies

| Package | Purpose | When Needed |
|---------|---------|-------------|
| `customtkinter` | GUI framework | GUI wizard |
| `pyyaml` | YAML parsing | Configuration files |
| `markdown` | Markdown processing | Documentation generation |

## Related Publications

### MAPseq Methodology

- **Han et al. 2018**: Original MAPseq methodology and filtering parameters
- **Klingler et al. 2018**: Alternative filtering parameters
- **CSHL Pipeline**: [mapseq-processing](https://github.com/ZadorLaboratory/mapseq-processing)

### Statistical Methods

- **Inclusion-Exclusion Principle**: Standard probability theory
- **Binomial Testing**: Standard statistical testing
- **Bonferroni Correction**: Multiple testing correction
- **Kruskal-Wallis Test**: Non-parametric ANOVA
- **Fisher's Exact Test**: Exact test for 2×2 tables
- **Jensen-Shannon Divergence**: Information-theoretic distance

## Change Log

### Version History

**Current Version**: As of January 2026

**Key Features**:
- Multiple probability models (uniform, region-specific, correlated, and additional models)
- Anchor model system for cross-cohort comparison
- Comprehensive helper script suite
- GUI wizard interface
- HTML documentation generation

**Recent Updates**:
- Added correlated binomial model
- Implemented anchor model system
- Added additional probability models (empirical, smoothed empirical, max entropy, etc.)
- Enhanced helper scripts for stability analysis
- Improved documentation

## Appendix A: File Naming Conventions

### Main Processing Outputs

- `{sample}_Filtered_Matrix.csv`: Filtered UMI matrix
- `{sample}_Normalized_Matrix.csv`: Normalized matrix
- `{sample}_upsetplot_{model}.csv`: Motif results
- `{sample}_effect_significance_{model}.png`: Volcano plot
- `{sample}_per_cell_proj_strength_{model}.png`: Per-cell plots

### Helper Script Outputs

- `{parameterization}_helpers/{script_number}/`: Helper script outputs
- `combined_effect_sizes_{model}.csv`: Effect size trajectories
- `transition_significance.csv`: Transition p-values
- `motif_percent_matrix_by_age_{model}.csv`: Percentage matrix

## Appendix B: Command-Line Argument Reference

### Required Arguments

- `-o, --out_dir`: Output directory
- `-s, --sample_name`: Sample name
- `-d, --data_file`: Input data file
- `-l, --labels`: Column labels

### Optional Arguments

- `-i, --injection_umi_min`: Injection UMI threshold (default: 1)
- `-t, --min_target_count`: Minimum target UMI (default: 10)
- `-r, --min_body_to_target_ratio`: Injection/target ratio (default: 10)
- `-u, --target_umi_min`: Target UMI threshold (default: 2)
- `-a, --alpha`: Significance threshold (default: 0.05)
- `--is-anchor-model`: Mark as anchor model
- `--anchor-model-file`: Path to anchor probabilities
- `--anchor-correlation-file`: Path to anchor correlation matrix
- `--model-type`: Which models to run (default: all)

## Appendix C: Output File Structure

### Complete Directory Tree

```
02_output/
├── p3/
│   └── {parameterization}/
│       ├── analysis/
│       │   ├── uniform/
│       │   ├── region_specific/
│       │   └── correlated/
│       └── {sample}_*.csv
├── p12/
│   └── {parameterization}/
│       └── ...
├── p20/
│   └── {parameterization}/
│       └── ...
├── p60/
│   └── {parameterization}/
│       └── ...
└── {parameterization}_helpers/
    ├── 01_motif_analysis_per_animal/
    ├── 05_motif_analysis/
    ├── 07_motif_significange_trajectories/
    └── ...
```

## Appendix D: Quick Reference

### Key Formulas

**N₀ Estimation**:
$$\pi = 1 - \prod_{k=0}^{m}\left(1 - \frac{s_k}{N_0}\right)$$

**Effect Size**:
$$Effect\ Size = \log_2\left(\frac{Observed + 1}{Expected + 1}\right)$$

**Region-Specific Probability**:
$$P(motif) = \prod_{i \in M} p_i \cdot \prod_{j \notin M} (1 - p_j)$$

**Correlated Probability**:
$$P(A, B, C) = P(A) \cdot P(B|A) \cdot P(C|B)$$

### Key Interpretations

- **Effect Size > 0**: Over-represented
- **Effect Size < 0**: Under-represented
- **JSD < 0.05**: Very similar (stable)
- **JSD > 0.2**: Large difference (unstable)
- **Kruskal-Wallis p < 0.05**: Significant change over time

---

*For detailed information, refer to specific chapters. For mathematical details, see [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions.md).*
