# MAPseq Pipeline Wizard

A comprehensive GUI wizard for configuring and executing the entire MAPseq processing pipeline.

## Features

- **Step-by-step wizard interface** - Guides users through the entire pipeline
- **Preprocessing with GUI column mapping** - Visual interface for mapping TSV columns with undo/redo
- **Main processing configuration** - Manage parameterizations and samples across age groups
- **Helper scripts management** - Enable/disable helper scripts with automatic dependency handling
- **Quality control and figure generation** - Configure QC checks and figure generation
- **Command file generation** - Automatically generates `all_commands.txt` style files
- **Pipeline execution** - Execute commands with real-time progress tracking

## Usage

```bash
# Activate conda environment
conda activate mapseq_processing

# Run the wizard
python -m MAPseq_wizard.main
```

## Requirements

See `requirements.txt` for all dependencies. Key dependencies:
- `customtkinter` - Modern GUI framework
- `pyyaml` - Configuration file handling
- All existing MAPseq pipeline dependencies

## Configuration

The wizard saves configurations as YAML files. You can:
- Save configurations for reuse
- Load previous configurations
- Export configurations for sharing

## Pipeline Steps

1. **Project Setup** - Configure project name and directories
2. **Preprocessing** - Map TSV columns and configure preprocessing
3. **Main Processing** - Set up parameterizations and samples
4. **Helper Scripts** - Select which helper scripts to run
5. **QC & Figures** - Configure quality control and figure generation
6. **Review & Generate** - Review configuration and generate command files
7. **Execute** - Run the pipeline with progress monitoring


