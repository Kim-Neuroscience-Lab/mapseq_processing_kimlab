# Chapter 11: Stability Analysis

## Overview

This chapter outlines a **generic** framework for asking whether projection motif frequencies (or model deviations) change across developmental stages. It does not prescribe verdicts for a particular dataset: combine metrics below with your pre-registered criteria and [Chapter 5: Statistical Methods](05_Statistical_Methods.md).

## Stability Analysis Framework

### Central Question

**Do projection motif frequencies or model-relative deviations change across the stages you include?**

Typical complementary angles:

1. **Model-independent**: Compare observed frequencies or distributions across ages (e.g., per-animal summaries, JSD between age pairs).
2. **Model-dependent**: Track effect sizes (observed vs. expected under a chosen model) across ages.
3. **Transition-focused**: Test or describe changes between consecutive stages.

### Key Metrics (what they stress)

| Metric | Type | What it stresses |
|--------|------|------------------|
| **Kruskal-Wallis** (helper 01) | Model-independent | Whether per-animal motif frequency distributions differ across ages |
| **Jensen-Shannon** (helpers 01 / 05) | Model-independent | Similarity of frequency distributions between age pairs (scale depends on implementation—see helper docs) |
| **Effect size trajectories** (helper 07) | Model-dependent | How log₂(Observed+1 / Expected+1) moves across stages |
| **Transition tests** (helper 07) | Model-dependent | Whether effect size changes between consecutive stages (primary implementation: z-test on differences; see Chapter 5) |
| **Trajectory methods** (helper 15) | Various | Path-level variation, mixed models, quadrant summaries, etc. |

## Model-Independent Analysis

### Kruskal-Wallis (helper 01)

**Purpose**: Non-parametric comparison of motif-related frequency structure across age groups (per-animal replicates when available).

**Null hypothesis**: Same distribution across ages.

**Reading**: Small p-values flag motifs/groups where the test rejects the null; thresholding (e.g. α or FDR) is your choice. See [Chapter 7](07_Helper_Scripts.md) for outputs.

**Code reference**: `helpers/scripts/01_motif_analysis_per_animal.py`

### Jensen-Shannon divergence

**Purpose**: Quantify similarity between empirical distributions (e.g. across ages).

**Range**: 0 (identical) to 1 (maximally different) for the standard JSD definition; helper 05 also reports squared JS on 2-bin vectors for motifs—do not mix scales across helpers without reading the helper docstrings.

## Model-Dependent Analysis

### Effect size trajectories (helper 07)

**Definition** (per motif, per stage):

$$Effect\ Size = \log_2\left(\frac{Observed + 1}{Expected + 1}\right)$$

**Descriptive range**: e.g. max(effect size) − min(effect size) across stages summarizes how much the deviation from the model moves; cutoffs are not universal.

**Code reference**: `helpers/scripts/07_motif_significange_trajectories.py`

### Transition significance (helper 07)

**Purpose**: Stage-to-stage changes in effect size.

**Implementation note**: The pipeline’s primary transition inference uses a **two-sample z-test** on differences of log-transformed counts (delta method SE), with optional FDR for the significance flag—**not** Fisher’s exact test. Details: [Chapter 5](05_Statistical_Methods.md).

## Why metrics can disagree (no paradox)

Different summaries answer different questions:

- **Per-animal distribution tests** can be insensitive to shared multiplicative shifts that leave ranks similar.
- **Population-level effect sizes** move when expected counts under the model change, even if raw proportions look similar.
- **Transition p-values** highlight local stage pairs; they need not align with global distribution tests.

Use this section to avoid forcing a single “stability score” when metrics intentionally stress different structure.

## Using multiple outputs (generic workflow)

1. **Model-independent**: Inspect helper 01 (and JSD tables) for cohort-level distribution shifts.
2. **Model-dependent**: Inspect helper 07 `combined_effect_sizes_*.csv` and `transition_significance.csv`.
3. **Trajectory / path analyses**: If you run helper 15, start from method-specific CSVs under `15_volcano_trajectories/{model}/` and pre-specify which method is primary for your study.

## Trajectory significance (helper 15)

Helper 15 bundles several approaches (permutation, FDA, mixed effects, distance metrics, legacy quadrant rules). For **file locations and column names**, see [Chapter 8](08_Output_Files_Interpretation.md) and [Chapter 15](15_Trajectory_Results_Interpretation.md).

**Publication hygiene (generic)**:

- Prefer methods with clear nulls (e.g. permutation, FDA) when claiming trajectory significance.
- Report multiple-testing handling as implemented in each CSV.
- Treat legacy quadrant/centroid rules as exploratory unless you justify them.

**Method agreement**: `method_comparison/method_agreement_matrix.csv` summarizes how often binary calls agree across methods—useful for robustness, not as a substitute for a pre-specified primary test.

## Key file locations (reference)

| Role | Typical path pattern |
|------|------------------------|
| Helper 01 outputs | `02_output/{parameterization}_helpers/01_motif_analysis_per_animal/{model}/` |
| Helper 05 outputs | `02_output/{parameterization}_helpers/05_motif_analysis/` |
| Helper 07 outputs | `02_output/{parameterization}_helpers/07_motif_significange_trajectories/{model}/` |
| Helper 15 outputs | `02_output/{parameterization}_helpers/15_volcano_trajectories/{model}/` |
| Per-age upsetplot CSVs | `02_output/{age}/{parameterization}/analysis/{model}/*_upsetplot_{model}.csv` |

---

*For formulas and test definitions, see [Chapter 5: Statistical Methods](05_Statistical_Methods.md). For run order and CLI, see [Chapter 7: Helper Scripts](07_Helper_Scripts.md).*
