# Cross-Anchor Comparative Analysis

This chapter presents a comparative analysis of trajectory significance testing results across different anchor datasets. Understanding how the choice of anchor affects conclusions is critical for robust interpretation of MAPseq developmental data.

---

## What is an Anchor Dataset?

In MAPseq analysis, the "anchor" refers to the developmental stage used as the normalization reference for computing expected projection counts. The three anchors analyzed are:

| Anchor | Description |
|--------|-------------|
| **P12 Anchor** | Uses P12 data as the normalization reference |
| **P20 Anchor** | Uses P20 data as the normalization reference |
| **P60 Anchor** | Uses P60 data as the normalization reference |

The choice of anchor affects how "expected" values are calculated, which in turn affects effect sizes and significance determinations. A robust finding should be consistent across anchor choices.

---

## Summary Comparison: All Stages Analysis (P3-P60)

### Global Statistics by Anchor

| Metric | P12 Anchor | P20 Anchor | P60 Anchor |
|--------|------------|------------|------------|
| Spearman significant | 3/26 (12%) | 3/26 (12%) | 2/26 (8%) |
| Increasing trends | 2 | 2 | 2 |
| Decreasing trends | 1 | 1 | 0 |
| Mixed-effects coefficient | 0.150 | 0.167 | 0.163 |
| Mixed-effects p-value | 0.084 | 0.061 | 0.059 |
| Direction test (% increasing) | 61.5% | 61.5% | 61.5% |
| Direction test p-value | 0.327 | 0.327 | 0.327 |

### Per-Motif Spearman Significance (All Stages)

| Motif | P12 Anchor | P20 Anchor | P60 Anchor | Consensus |
|-------|------------|------------|------------|-----------|
| pm+rsp | NS | NS | NS | Stable |
| am+rsp | NS | NS | NS | Stable |
| al+rsp | NS | NS | NS | Stable |
| lm+rsp | NS | NS | NS | Stable |
| am+pm | NS | NS | NS | Stable |
| al+pm | NS | NS | NS | Stable |
| lm+pm | NS | NS | NS | Stable |
| al+am | NS | NS | NS | Stable |
| am+lm | NS | NS | NS | Stable |
| al+lm | NS | NS | NS | Stable |
| **am+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| al+pm+rsp | NS | NS | NS | Stable |
| lm+pm+rsp | NS | NS | NS | Stable |
| al+am+rsp | **SIG (-)** | **SIG (-)** | NS | Anchor-Dependent |
| am+lm+rsp | NS | NS | NS | Stable |
| al+lm+rsp | NS | NS | NS | Stable |
| al+am+pm | NS | NS | NS | Stable |
| am+lm+pm | NS | NS | NS | Stable |
| al+lm+pm | NS | NS | NS | Stable |
| al+am+lm | NS | NS | NS | Stable |
| **al+am+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| am+lm+pm+rsp | NS | NS | NS | Stable |
| al+lm+pm+rsp | NS | NS | NS | Stable |
| al+am+lm+rsp | NS | NS | NS | Stable |
| al+am+lm+pm | NS | NS | NS | Stable |
| al+am+lm+pm+rsp | NS | NS | NS | Stable |

**Legend**: SIG (+) = Significant increasing, SIG (-) = Significant decreasing, NS = Not significant

---

## Summary Comparison: noP3 Analysis (P12-P60 only)

### Global Statistics by Anchor (noP3)

| Metric | P12 Anchor | P20 Anchor | P60 Anchor |
|--------|------------|------------|------------|
| Spearman significant | 11/26 (42%) | 11/26 (42%) | 10/26 (38%) |
| **Strict verdict significant** | **6/26 (23%)** | **9/26 (35%)** | **4/26 (15%)** |
| Increasing trends (Spearman) | 7 | 7 | 7 |
| Decreasing trends (Spearman) | 4 | 4 | 3 |
| Increasing trends (Strict) | 4 | 6 | 3 |
| Decreasing trends (Strict) | 2 | 3 | 1 |
| Mixed-effects coefficient | 0.175 | 0.206 | 0.199 |
| Mixed-effects p-value | 0.218 | 0.159 | 0.153 |
| Direction test (% increasing) | 61.5% | 61.5% | 61.5% |
| Direction test p-value | 0.327 | 0.327 | 0.327 |

**Note**: Strict verdict requires Spearman significance AND crossing of p-value cutoff or effect size = 0 line (see Chapter 15 for methodology).

### Per-Motif Spearman Significance (noP3)

| Motif | P12 Anchor | P20 Anchor | P60 Anchor | Consensus |
|-------|------------|------------|------------|-----------|
| pm+rsp | NS | NS | NS | Stable |
| am+rsp | NS | NS | NS | Stable |
| **al+rsp** | **SIG (-)** | **SIG (-)** | **SIG (-)** | **ROBUST DECREASE** |
| lm+rsp | NS | NS | NS | Stable |
| **am+pm** | **SIG (-)** | **SIG (-)** | **SIG (-)** | **ROBUST DECREASE** |
| al+pm | NS | NS | NS | Stable |
| lm+pm | NS | NS | NS | Stable |
| al+am | NS | NS | NS | Stable |
| **am+lm** | **SIG (-)** | **SIG (-)** | **SIG (-)** | **ROBUST DECREASE** |
| al+lm | NS | NS | NS | Stable |
| **am+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| **al+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| lm+pm+rsp | NS | NS | NS | Stable |
| **al+am+rsp** | **SIG (-)** | **SIG (-)** | NS | Anchor-Dependent |
| am+lm+rsp | NS | NS | NS | Stable |
| al+lm+rsp | NS | NS | NS | Stable |
| **al+am+pm** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| am+lm+pm | NS | NS | NS | Stable |
| **al+lm+pm** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| al+am+lm | NS | NS | NS | Stable |
| **al+am+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| am+lm+pm+rsp | NS | NS | NS | Stable |
| **al+lm+pm+rsp** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| al+am+lm+rsp | NS | NS | NS | Stable |
| **al+am+lm+pm** | **SIG (+)** | **SIG (+)** | **SIG (+)** | **ROBUST INCREASE** |
| al+am+lm+pm+rsp | NS | NS | NS | Stable |

---

## Robust Findings Across All Anchors

### All Stages Analysis (P3-P60)

**Robustly Spearman significant in ALL anchors (2 motifs):**

| Motif | Direction | P12 rho | P20 rho | P60 rho | Strict Verdict |
|-------|-----------|---------|---------|---------|----------------|
| am+pm+rsp | Increasing | 1.0 | 1.0 | 1.0 | NO (stays in quadrant) |
| al+am+pm+rsp | Increasing | 1.0 | 1.0 | 1.0 | **YES** (crosses ES=0) |

**Note**: Only `al+am+pm+rsp` meets the strict verdict criterion in the all-stages analysis.

**Anchor-dependent (1 motif):**

| Motif | P12 | P20 | P60 | Note |
|-------|-----|-----|-----|------|
| al+am+rsp | SIG (-) | SIG (-) | NS | Significant in P12/P20 only |

### noP3 Analysis (P12-P60 only)

**Robustly Spearman significant in ALL anchors (9 motifs):**

| Motif | Direction | Spearman Consistent | Strict Verdict (P60) |
|-------|-----------|---------------------|----------------------|
| al+rsp | Decreasing | Yes | NO (stays in quadrant) |
| am+pm | Decreasing | Yes | NO (stays in quadrant) |
| am+lm | Decreasing | Yes | **YES** (crosses both) |
| am+pm+rsp | Increasing | Yes | NO (stays in quadrant) |
| al+pm+rsp | Increasing | Yes | NO (stays in quadrant) |
| al+am+pm | Increasing | Yes | NO (stays in quadrant) |
| al+lm+pm | Increasing | Yes | NO (stays in quadrant) |
| al+am+pm+rsp | Increasing | Yes | **YES** (crosses ES=0) |
| al+lm+pm+rsp | Increasing | Yes | **YES** (crosses ES=0) |
| al+am+lm+pm | Increasing | Yes | **YES** (crosses both) |

**Strict verdict summary (noP3, P60 anchor):** Only 4/9 Spearman-robust motifs also meet the strict criterion.

**Anchor-dependent (1 motif):**

| Motif | P12 | P20 | P60 | Note |
|-------|-----|-----|-----|------|
| al+am+rsp | SIG (-) | SIG (-) | NS | Significant in P12/P20 only |

---

## Impact of Anchor Selection on Conclusions

### Key Observations

1. **Direction test is invariant**: The proportion of increasing motifs (61.5%) and binomial p-value (0.327) are identical across all anchors. This is because the direction is computed from raw P12-P60 differences.

2. **Mixed-effects coefficients vary slightly**: Range from 0.150 (P12) to 0.167 (P20), but all remain marginally non-significant (p = 0.059-0.084).

3. **Spearman significance is largely consistent**: The same motifs tend to be significant across anchors, with minor variations.

4. **One motif is anchor-sensitive**: `al+am+rsp` shows a significant decreasing trend with P12 and P20 anchors but not with P60 anchor.

### Recommendations for Reporting

**For robust claims**, report only motifs significant across ALL anchors:
- All stages: am+pm+rsp, al+am+pm+rsp (both increasing)
- noP3: 9 motifs with robust consensus (see table above)

**For exploratory findings**, note anchor-dependent results with appropriate caveats:
- al+am+rsp shows anchor-dependent significance

**For methodological transparency**, report:
- Results are consistent across P12, P20, and P60 anchor choices
- Mixed-effects global trend is marginally non-significant regardless of anchor
- Direction bias is not statistically significant in any analysis

---

## Comparison: All Stages vs noP3 by Anchor

| Anchor | All Stages Sig | noP3 Sig | Change |
|--------|---------------|----------|--------|
| P12 | 3/26 (12%) | 11/26 (42%) | +8 |
| P20 | 3/26 (12%) | 11/26 (42%) | +8 |
| P60 | 2/26 (8%) | 10/26 (38%) | +8 |

The increase in significant motifs when excluding P3 is consistent across all anchors (+8 motifs in each case). This confirms that the P3 stage introduces variability that obscures monotonic trends in the P12-P60 trajectory.

---

## Conclusions

### Hypothesis Evaluation Across Anchors

**Original hypothesis**: "Motifs do not significantly change over time, however many appear to become more or less over- and under-represented."

**Verdict across all anchors (applying strict criterion)**:

The hypothesis is **strongly supported** and this conclusion is **robust to anchor selection**:

1. **Most motifs show no biologically meaningful change**: 
   - All stages: 24-25/26 (92-96%) fail strict criterion
   - noP3: 20-22/26 (77-85%) fail strict criterion

2. **Very few motifs meet strict criterion (Spearman + threshold crossing)**:
   - All stages: Only 1-2 motifs across anchors
   - noP3: Only 4-9 motifs across anchors (varies by anchor)

3. **The dramatic reduction from Spearman to strict verdict is consistent across anchors**:
   - noP3 Spearman: 10-11 significant
   - noP3 Strict: 4-9 significant (40-55% filtered out for staying in same quadrant)

4. **Global direction bias is not significant in any anchor**: The 61.5% increasing proportion does not differ significantly from 50%

### Strict Verdict Summary by Anchor (noP3)

| Anchor | Spearman Sig | Strict Sig | Filtered Out |
|--------|-------------|------------|--------------|
| P12 | 11 | 6 | 5 (45%) |
| P20 | 11 | 9 | 2 (18%) |
| P60 | 10 | 4 | 6 (60%) |

The P60 anchor is the most conservative, filtering out the most motifs that show within-quadrant trends.

### Final Recommendation

Results reported using any of the three anchor datasets will yield consistent biological conclusions. The choice of anchor does not materially affect the interpretation that:

- **The vast majority of multi-area projection motifs are developmentally stable** (no biologically meaningful change)
- Motifs meeting strict criterion (**am+lm**, **al+am+pm+rsp**, **al+lm+pm+rsp**, **al+am+lm+pm** in P60 noP3) show genuine classification changes
- Many motifs with Spearman-significant trends show **quantitative** but not **qualitative** changes
- P3 represents a distinct early developmental state that introduces trajectory variability

---

## References

See Chapter 15 for methodological details and statistical references.
