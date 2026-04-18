# Chapter 13: References and Appendices

## Code File References

### Main Scripts

| File | Purpose | Key Functions |
|------|---------|---------------|
| `process-nbcm-tsv.py` | Main processing pipeline | `solve_for_roots()`, `clean_and_filter()`, `compute_motif_probabilities_*()` |
| `preprocess_and_aggregate.py` | Data preprocessing | `preprocess_file()`, `get_column_mapping()` |
| `postprocessing_checks.py` | Quality control | Main QC checking functions |

### Helper Scripts

| File | Purpose | Tier | Key methods |
|------|---------|------|-------------|
| `01_motif_analysis_per_animal.py` | Per-animal analysis | Core | Kruskal-Wallis, JSD |
| `02_projection_analysis.py` | Projection analysis | Core | PCA, clustering |
| `03_composition.py` | Composition | Core | Descriptive |
| `04_proportions_over_time_stats.py` | Proportions over time | Core | Chi-square, CLR |
| `05_motif_analysis.py` | Motif analysis | Core | Welch, KS, JSD |
| `06_all_motif_divergence.py` | Motif divergence plots | Core | Visualization |
| `07_motif_significange_trajectories.py` | Effect-size trajectories | Core | Transition **z-test** (log counts), optional FDR |
| `08_motif_clustering.py` | Motif clustering | Core | Clustering |
| `09_plot_normalized_projection_strength_data.py` | Projection strength | Core | Visualization |
| `10_plot_per_cell_projection_strength_across_ages.py` | Per-cell lines across ages | Optional | Visualization |
| `10_compare_datasets_pipeline.py` | Two-way dataset compare | Maintainer | External data |
| `11_compare_vsv_mapseq_two_way.py` | VSV vs MapSeq | Maintainer | External data |
| `12_compare_datasets_pipeline_mapseq.py` | Three-way compare | Maintainer | External data |
| `13_aggregate_projection_summaries.py` | Aggregate summaries | Core | Aggregation |
| `14_model_group_comparison.py` | Model comparison | Optional | Plots / stats |
| `15_volcano_trajectories.py` | Trajectory methods | Core | Permutation, FDA, mixed-effects, etc. |
| `16_power_analysis.py` | Power / equivalence | Maintainer | See Chapter 14 |
| `17_jsd_cross_source_summary.py` | JSD / homogeneity summary | Optional | Chi-square, Monte Carlo |
| `18_mean_jsd_transition_tests.py` | Mean JS² transition tests | Optional | Permutation, bootstrap |
| `00_teleporting_barcode_detection.py` | Batch barcode QC | Maintainer | Project-specific mappings |

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
- **Transition z-test (helper 07)**: `helpers/scripts/07_motif_significange_trajectories.py`

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
| **Transition z-test (helper 07)** | Effect-size change between stages | No change in log-count–derived effect size | Normal approx.; see Chapter 5 |
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
- **Transition z-tests (helper 07)**: Consecutive-stage effect-size contrasts
- **Jensen-Shannon Divergence**: Information-theoretic distance

## Change Log

### Version History

**Current Version**: As of April 2026

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
- `-f, --apply_outlier_filtering`: Enable outlier filtering (mean + 2×std)
- `--force_user_threshold`: Use the user’s `-u` value without automatic overrides
- `--is-anchor-model`: Mark as anchor model
- `--anchor-model-file`: Path to anchor probabilities CSV
- `--anchor-correlation-file`: Path to anchor correlation matrix CSV
- `--model-type`: One of `uniform`, `region_specific`, `correlated`, `empirical`, `smoothed_empirical`, `max_entropy`, `hierarchical_correlations`, `negative_binomial`, `zero_inflated`, `bayesian_hierarchical`, `ml_nonparametric`, `all` (default: `all`)
- `--smoothing-alpha`: Smoothing α for `smoothed_empirical` (default: 1.0)
- `--skip-sections`: Comma-separated: `visualizations`, `clustering`, `heatmaps`
- `--illustrator-volcano-dir`: Directory for illustrator-ready uniform volcano SVG
- `--illustrator-report-ranges-only`: With illustrator dir: append ranges only, no SVG
- `--illustrator-xlim`, `--illustrator-ylim`: Axis limits for illustrator volcano
- `-A`, `-B`: Legacy special-area flags (deprecated for most workflows)

Run `python process-nbcm-tsv.py --help` for the authoritative list.

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
    ├── 07_input/                      # copied upsetplot CSVs for 07/15
    ├── 07_motif_significange_trajectories/
    ├── 15_volcano_trajectories/
    ├── 17_jsd_cross_source/           # optional
    ├── 18_mean_jsd_transition_tests/  # optional
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

### Key readings (definitions)

- **Effect Size > 0**: Observed exceeds expected under the model
- **Effect Size < 0**: Observed below expected
- **JSD**: Similarity metric; thresholds are context- and implementation-dependent (see helper docs)
- **Kruskal-Wallis p < α**: Reject equal distributions across groups (at chosen α)

---

*For detailed information, refer to specific chapters. For mathematical details, see [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions.md).*
