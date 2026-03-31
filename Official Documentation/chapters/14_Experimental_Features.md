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

---

## Recommendation

For production and reproducible runs, use **`./run_commands.sh`** (after `chmod +x run_commands.sh` if needed) with an edited command file (`all_commands.txt` or `all_commands_all-parameters.txt`) as described in Chapters 1, 2, and 4.

---

*Return to [Chapter 1: Introduction](01_Introduction.md) or [Chapter 2: Installation and Setup](02_Installation_Setup.md) for the supported workflow.*
