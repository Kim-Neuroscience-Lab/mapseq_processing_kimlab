# Chapter 2: Installation and Setup

## System Requirements

### Operating System

The MAPseq processing pipeline supports:
- **Linux** (tested on Ubuntu 20.04+)
- **macOS** (tested on macOS 10.15+)
- **Windows** (via WSL; for an experimental GUI installer see [Chapter 14: Experimental Features](14_Experimental_Features.md))

### Software Dependencies

- **Python 3.9** (required)
- **Conda** (Miniconda or Anaconda) for environment management
- **Git** (for cloning the repository)

### Hardware Requirements

- **RAM**: Minimum 8 GB, recommended 16 GB for large datasets
- **Storage**: ~2 GB for installation, additional space for output files
- **CPU**: Multi-core recommended for parallel processing

## Installation Methods

Use the following steps on all platforms (Linux, macOS, Windows with WSL). Windows users looking for an experimental GUI installer can see [Chapter 14: Experimental Features](14_Experimental_Features.md).

### Command-Line Installation

#### Step 1: Install Miniconda

Download and install Miniconda for your operating system:
- [Miniconda quick command-line install](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

#### Step 2: Create Conda Environment

```bash
conda create -n mapseq_processing python==3.9 pip
```

#### Step 3: Activate Environment

```bash
conda activate mapseq_processing
```

#### Step 4: Configure Conda Channels

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
```

#### Step 5: Clone Repository

```bash
cd /path/to/your/git/directory
git clone https://github.com/matsojr22/mapseq_processing_Jacobs.git
cd mapseq_processing_Jacobs
```

#### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

## Verification

After installation, verify the setup:

```bash
# Activate environment
conda activate mapseq_processing

# Test script execution
python process-nbcm-tsv.py --help
```

## Running the Pipeline (Recommended)

The primary way to run the pipeline is via the bash script and a command file:

1. **Edit the command file**: Copy or edit `all_commands.txt` (or use `all_commands_all-parameters.txt`) so paths and sample names match your project. The file contains one command per line (main processing and helper scripts).
2. **Run the script**: From the repository root, make the script executable once (if needed), then run it:
   ```bash
   chmod +x run_commands.sh
   ./run_commands.sh
   ```
   If you use the copy in `bash/`, run `chmod +x bash/run_commands.sh` and then `./bash/run_commands.sh`. The script reads the command file line by line, executes each command in order, and logs output to a timestamped file named `processing_YYYYMMDD_HHMMSS.log`. The script continues even if individual commands fail; check the log file to identify any failures.
3. **Review output**: Results are written to the output directories specified in the command file.

For experimental GUI options, see [Chapter 14: Experimental Features](14_Experimental_Features.md).

## Command-Line Interface

**Advantages:**
- Full control over all parameters
- Scriptable and automatable
- Better for batch processing
- Easier integration with workflows

**Usage:**
```bash
python process-nbcm-tsv.py -o output_dir -s sample_name -d data_file.tsv -l "RSP,PM,AM,AL,LM,neg,inj"
```

**See**: [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md) for detailed CLI usage.

## Dependencies

### Core Dependencies

Key packages required for the pipeline:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `scipy` | Statistical functions |
| `sympy` | Symbolic mathematics (N₀ estimation) |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualization |
| `scikit-learn` | Machine learning (clustering, PCA) |
| `upsetplot` | Set visualization |
| `markdown` | Documentation generation |

### Optional Dependencies

See `requirements.txt` for the complete list. For optional dependencies used by experimental GUI tools, see [Chapter 14: Experimental Features](14_Experimental_Features.md).

## Environment Management

### Activating Environment

Always activate the conda environment before running scripts:

```bash
conda activate mapseq_processing
```

### Deactivating Environment

```bash
conda deactivate
```

### Updating Dependencies

To update packages:

```bash
conda activate mapseq_processing
pip install --upgrade -r requirements.txt
```

### Removing Environment

If you need to start fresh:

```bash
conda deactivate
conda env remove -n mapseq_processing
```

Then follow installation steps again.

## Troubleshooting Installation

### Common Issues

**Issue**: `conda: command not found`
- **Solution**: Ensure conda is installed and in your PATH. Restart terminal after installation.

**Issue**: `pip install` fails with permission errors
- **Solution**: Use `pip install --user` or ensure you're in the conda environment.

**Issue**: Import errors after installation
- **Solution**: Verify environment is activated and packages are installed:
  ```bash
  conda activate mapseq_processing
  pip list | grep numpy
  ```

**Issue**: SymPy errors during N₀ estimation
- **Solution**: Ensure sympy version ≥ 1.9. Update with `pip install --upgrade sympy`.

### Getting Help

- Check [Chapter 12: Troubleshooting and Best Practices](12_Troubleshooting_Best_Practices.md)
- Review error messages carefully
- Verify all dependencies are installed
- Check Python version matches requirements

## Next Steps

After successful installation:

1. **Prepare your data**: See [Chapter 3: Data Preparation](03_Data_Preparation.md). Run preprocessing to produce an aggregated matrix:
   ```bash
   python preprocess_and_aggregate.py -i /path/to/input/nbcm_files/ -o /path/to/output/
   ```
2. **Run the pipeline**: Either use the command file (recommended) or run main processing manually. To use the command file: edit `all_commands.txt` (or `all_commands_all-parameters.txt`) so paths and sample names match your project, make the script executable if needed (`chmod +x run_commands.sh`), then from the repository root run `./run_commands.sh`. To run main processing once manually:
   ```bash
   python process-nbcm-tsv.py -o /path/to/output/ -s sample_name -d /path/to/data.tsv -l "RSP,PM,AM,AL,LM,neg,inj"
   ```
   See [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md) for all arguments.
3. **Run helper scripts**: See [Chapter 7: Helper Scripts](07_Helper_Scripts.md). The command file can include helper script commands in the correct order; otherwise run them individually after main processing.

---

*For detailed usage instructions, proceed to [Chapter 3: Data Preparation](03_Data_Preparation.md).*
