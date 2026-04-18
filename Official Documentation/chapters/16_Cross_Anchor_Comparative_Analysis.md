# Chapter 16: Cross-Anchor Comparative Analysis (Conceptual)

## Purpose

Some projects fit MAPseq models using different **anchor** normalizations (e.g. anchor probabilities or correlation structure taken from different reference ages or cohorts). Effect sizes and trajectory statistics can shift when the reference changes. This chapter gives a **generic checklist** for comparing anchor variants—**without** embedding results from any particular experiment.

For anchor-related **CLI flags**, see [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md). For trajectory file layouts, see [Chapters 8](08_Output_Files_Interpretation.md) and [15](15_Trajectory_Results_Interpretation.md).

## What “anchor” means here

- The anchor supplies **reference probabilities** (and optionally a **correlation matrix**) used to define expected counts under anchored models.
- Changing anchor changes **expected** counts, hence **effect sizes** and any trajectory statistics derived from them.
- A **robust** scientific claim should be stated in terms that either (a) hold under every anchor you prespecified, or (b) explicitly acknowledge sensitivity to anchor choice.

## Suggested workflow (no numbers)

1. **Pre-register** which anchors you will compare and which outcome (e.g. permutation FDR, helper **07** transition table) is primary.
2. **Run the same samples** through each anchor configuration with identical filtering and `--model-type` choices.
3. **Align outputs** by `Motif` (and model) and compare:
   - Binary calls (significant vs not) from your primary test.
   - Direction of change (increasing vs decreasing) if applicable.
4. **Classify motifs** for reporting (example categories—define your own):
   - **Consensus**: same qualitative call across all anchors.
   - **Sensitive**: call changes when anchor changes.
   - **Borderline**: primary test near threshold under one anchor only.
5. **Document** stage sets (e.g. P3–P60 vs P12–P60) separately; removing stages changes power and discrete correlation behavior.

## Interpreting disagreement (generic)

- **Invariant tests** (e.g. some direction summaries that use only raw P12–P60 differences) may agree across anchors even when model-based tests shift.
- **Model-based tests** should be expected to move somewhat when expectations move.
- **Few time points** make monotonic trend statistics brittle; see [Chapter 15](15_Trajectory_Results_Interpretation.md).

## What not to do

- Do not treat a table of motif-level results copied from a single study as universal MAPseq behavior.
- Do not merge results across anchors without clear rules for ties and FDR scope.

---

*For stability framing without dataset-specific verdicts, see [Chapter 11: Stability Analysis](11_Stability_Analysis.md).*
