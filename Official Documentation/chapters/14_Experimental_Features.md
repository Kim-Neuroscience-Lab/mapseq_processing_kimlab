# Chapter 14: Experimental Features

## Disclaimer

The features described in this chapter are **experimental and have not been fully tested**. For production runs, use the supported workflow: make the script executable if needed (`chmod +x run_commands.sh`) and run **`./run_commands.sh`** with a command file (`all_commands.txt` or `all_commands_all-parameters.txt`). See [Chapter 1: Introduction](01_Introduction.md), [Chapter 2: Installation and Setup](02_Installation_Setup.md), and [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md) for the primary workflow.

---

## Simple GUI Launcher (MAPseq_Wizard.py)

A single-form GUI that runs the main processing script with user-entered parameters.

- **Technology**: PySimpleGUI
- **Scope**: Launches `process-nbcm-tsv.py` only (no preprocessing or helper scripts from the GUI)
- **Status**: Experimental, untested

### Usage

```bash
conda activate mapseq_processing
python MAPseq_Wizard.py
```

The form provides fields for sample name, data file, output directory, labels, filter parameters (injection UMI min, min target count, min body-to-target ratio, target UMI min), alpha, and optional flags (outlier filtering, force user threshold). Preprocessing still requires terminal access.

---

## Full Wizard (MAPseq_wizard Package)

A 7-step GUI wizard that guides users through the entire pipeline from project setup to execution.

- **Technology**: customtkinter
- **Scope**: Project setup, preprocessing (with GUI column mapping), main processing configuration, helper scripts, QC and figures, review and command file generation, and pipeline execution
- **Status**: Experimental, untested

### Usage

```bash
conda activate mapseq_processing
python -m MAPseq_wizard.main
```

### Dependencies

- `customtkinter` - GUI framework
- `pyyaml` - Configuration file handling

These are listed in the repository `requirements.txt`.

### Pipeline Steps (in the wizard)

1. **Project Setup** - Configure project name and directories
2. **Preprocessing** - Map TSV columns and configure preprocessing
3. **Main Processing** - Set up parameterizations and samples
4. **Helper Scripts** - Select which helper scripts to run
5. **QC & Figures** - Configure quality control and figure generation
6. **Review & Generate** - Review configuration and generate command files
7. **Execute** - Run the pipeline with progress monitoring

For more detail, see the [MAPseq_wizard package README](../../MAPseq_wizard/README.md) in the repository.

### Wizard helper checklist vs full repository

The **Helper Scripts** step in `MAPseq_wizard` exposes only a **subset** of helpers (e.g. 01–09, 13, 15 with dependency hints). It does **not** mirror every script under `helpers/scripts/` (e.g. optional **17** / **18**, per-cell projection plot **10**, model comparison **14**, or maintainer tools below). For the full list and batch order, use [Chapter 7: Helper Scripts](07_Helper_Scripts.md) and your edited `all_commands.txt`.

---

## Maintainer and lab batch utilities (not required for standard MAPseq)

These tools are **optional**. They are kept in the repository for lab workflows; end users can ignore them unless needed.

### Teleporting / batch barcode QC

- **`helpers/scripts/00_teleporting_barcode_detection.py`** — Sequencing-batch vs animal mapping and related QC plots. Hardcoded mappings are project-specific; read the script before running.

### Dataset comparison helpers (external data)

- **`helpers/scripts/10_compare_datasets_pipeline.py`** — Two-way pipeline comparison (external datasets).
- **`helpers/scripts/11_compare_vsv_mapseq_two_way.py`** — VSV vs MapSeq.
- **`helpers/scripts/12_compare_datasets_pipeline_mapseq.py`** — Three-way (e.g. Allen / VSV / MapSeq).

### Power and manuscript ancillary analyses

- **`helpers/scripts/16_power_analysis.py`** — Power / equivalence style analyses tied to manuscript claims; depends on other helper outputs. Run with `--help` for current arguments.

### Figure aggregation (multi-panel TIFF / matrices)

- **`figure_generation/generate_figure_from_outputs.py`** — Builds multi-panel figures from existing pipeline outputs (e.g. by age). Not part of the core statistical pipeline; run only if you maintain that figure workflow.
- Other scripts under `figure_generation/` may exist for one-off exports; use `--help` per script.

### Conclusions generation (lab reports)

Scripts under **`conclusions/scripts/`** (e.g. `extract_stability_data.py`, `generate_conclusions.py`) read processed outputs and write markdown/HTML conclusions for this repository’s stability narrative. Paths and arguments change with the repo; inspect the scripts and [`bash/all_commands.txt`](../../bash/all_commands.txt) for the current invocation. These are **not** general MAPseq user requirements.

---

## Recommendation

For production and reproducible runs, use **`./run_commands.sh`** (after `chmod +x run_commands.sh` if needed) with an edited command file (`all_commands.txt` or `all_commands_all-parameters.txt`) as described in Chapters 1, 2, and 4.

---

*Return to [Chapter 1: Introduction](01_Introduction.md) or [Chapter 2: Installation and Setup](02_Installation_Setup.md) for the supported workflow.*
