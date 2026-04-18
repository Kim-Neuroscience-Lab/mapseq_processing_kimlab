# MAPseq Analysis Script
MAPseq processing code based on previous works and designed to be used with the CSHL python pipeline.

Code found here is generally a work in progress until publication.

[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fmatsojr22%2Fmapseq_processing_Jacobs%2Flines)](https://tokei.kojix2.net/github/matsojr22/mapseq_processing_Jacobs)

## **Before you run:**
- Be sure that you have processed your fastq files using the [CSHL mapseq-processing Python Pipeline](https://github.com/ZadorLaboratory/mapseq-processing).
- A preprocessing and data aggregation script is provided to prepare a individaul and combined cohort level dataframe for analysis using the per-animal sample.nbcm.tsv files produced by the CSHL pipeline. This script requires the user to match the nbcm header labels to their own sample labels to ensure all the data is correctly aligned when concatenated.
- This script **process-nbcm-tsv.py** uses the aggregated_cleaned_matrix.tsv produced by the preprocessing and aggregation script (or individual sample.nbcm.tsv files from the CSHL pipeline if you do not have replicates). If you want to run a full analysis, you will need to ensure that the fastq processing parameters in the CSHL script have included: your samples, your negative control, and your injection columns in the output. Partial analysis is also possible at your discretion; there is a provided truncated "sample dataset" and associated "labels" which you can check out for guidance. You will need to check the arguments for each script if these requirements are unclear.
- The [WIKI](https://github.com/matsojr22/mapseq_processing_Jacobs/wiki) can provide many answers not listed in the Readme.
- If you are on Windows and want to **try the GUI Wizard**, then please download the most recent **setup_wizard.exe** from the releases page. Running this will automatically install the software necessary to run all the scripts, and will create a MAPseq_Wizard.exe in the installation directory that will provide a GUI for the main **process-nbcm-tsv.py** script. You will still need to preprocess in the terminal at the moment.
- Else, from the terminal you need to setup a new conda environment, repos, and dependencies as shown below.
- Run preprocessing then main analysis scripts.
<br/>

## EXE Installation (Windows Only)

1. Download and [install Git](https://gitforwindows.org/) if not already installed. 
  
2. Download the most recent Setup_Wizard.exe from the [releases page](https://github.com/matsojr22/mapseq_processing_Jacobs/releases).

3. Run the file and wait for it to complete the installation (Default location is the user directory).

## CLI Installation

1. Install mini-conda for your operating system. [mini-conda quick command line install](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

2.  With conda installed create a new environment preloaded with pip

```
conda create -n mapseq_processing python==3.9 pip
```

3. Activate your new environment

```
conda activate mapseq_processing
```

4. Install additional repositories

```
conda config --add channels conda-forge
conda config --add channels bioconda
```

5. Browse to your git directory and clone this project

```
cd /home/your_user/git/
git clone https://github.com/matsojr22/mapseq_processing_Jacobs.git
```

6. Browse into the project directory and install dependencies

```
cd /mapseq_processing_Jacobs/
pip install -r requirements.txt
```

---

## GUI Tools

### MAPseq_Wizard.py

A graphical user interface wrapper for the main `process-nbcm-tsv.py` script, providing an easy-to-use form for all processing parameters.

**Features:**
- **PySimpleGUI interface**: User-friendly form with file browsers and input fields
- **Auto-update**: Automatically updates the repository to the latest version on startup
- **Parameter validation**: All arguments from the command-line interface are available
- **Cross-platform**: Works on Windows, macOS, and Linux (requires Python and PySimpleGUI)

**Usage:**
```bash
python MAPseq_Wizard.py
```

**Requirements:**
- Python environment with `PySimpleGUI` installed
- Conda environment `mapseq_processing` (or modify the script to use your environment)

**Note:** The GUI currently requires preprocessing to be done in the terminal. Future versions may include preprocessing functionality.

### setup_wizard.py

Installation wizard for Windows users that automates the setup process.

**Features:**
- **Conda environment creation**: Automatically creates the `mapseq_processing` environment
- **Dependency installation**: Installs all required packages from `requirements.txt`
- **GUI executable download**: Downloads the latest `MAPseq_Wizard.exe` from GitHub releases
- **Repository cloning**: Optionally clones the repository if not already present

**Usage:**
```bash
python setup_wizard.py
```

**Windows Executable:**
For Windows users who prefer not to use Python directly, download `Setup_Wizard.exe` from the [releases page](https://github.com/matsojr22/mapseq_processing_Jacobs/releases). This executable:
- Installs all necessary dependencies
- Creates the conda environment
- Downloads and sets up the GUI executable
- Provides a complete installation without requiring manual Python setup

**Note:** The setup wizard detects the repository location automatically and can work with both local repositories and GitHub-hosted repositories.

---

7. Run the preprocessing script to clean and aggregate your replicate TSV files (see Preprocessing Scripts section below for details)

```
python preprocess_and_aggregate.py -i /home/mwjacobs/git/mapseq_processing_jacobs/predata/adults/ -o /home/mwjacobs/git/mapseq_processing_jacobs/data/adults/
```

8. Run the main analysis script on your sample.nbcm.tsv (command below shown using included sample dataset, but typically you would run using your aggregated data from the prior step)

```
python process-nbcm-tsv.py -o /home/mwjacobs/git/mapseq_processing_jacobs/jr0375_out/ -s JR0375 -d /home/mwjacobs/git/mapseq_processing_jacobs/sample_data/JR0375.nbcm.tsv -u 2 -l "RSP,PM,AM,A,RL,AL,LM,neg,inj"
```

<br/>

---

## Preprocessing Scripts

### preprocess_and_aggregate.py

This script preprocesses and aggregates replicate TSV files from the CSHL pipeline. It standardizes column names across files, applies thresholding based on negative controls, and creates a combined dataset for analysis.

**Purpose:**
- Clean and standardize column names across multiple replicate files
- Apply noise filtering based on negative control statistics
- Aggregate multiple cleaned files into a single matrix for cohort-level analysis

**Usage:**
```bash
python preprocess_and_aggregate.py -i <input_dir> -o <output_dir> [-t <fallback_threshold>]
```

**Arguments:**
- **-i, --input_dir** (required): Directory containing replicate `.tsv` files from the CSHL pipeline
- **-o, --output_dir** (required): Output directory for cleaned and aggregated files
- **-t, --fallback_threshold** (optional, default: 2.0): Threshold value used if negative control column has no data

**Features:**
- **Interactive column mapping**: For each file, the script prompts you to map original column names to standardized names. You must map at least one column to `barcodes`.
- **Automatic threshold calculation**: Uses mean + standard deviation of negative control values. Falls back to user-specified threshold if no negative control is available.
- **Column alignment**: Ensures all aggregated files have the same column structure, filling missing columns with NaN values.
- **Negative control filtering**: Removes rows where negative control values are non-zero.

**Output:**
- Individual cleaned files: `{basename}_cleaned.tsv` for each input file
- Aggregated matrix: `aggregated_cleaned_matrix.tsv` containing all cleaned data combined

**Example:**
```bash
python preprocess_and_aggregate.py -i /path/to/nbcm_files/ -o /path/to/output/ -t 2.0
```

**Note:** This script is interactive and will prompt you for column mappings. Ensure you have a terminal session that supports interactive input.

<br/>

## **REQUIRED Arguments**

**-o** = path to your output directory

**-d** = path to your sample.nbcm.tsv which was produced by the [CSHL mapseq-processing Python Pipeline](https://github.com/ZadorLaboratory/mapseq-processing) Or your group.aggregate.tsv file.

**-s** = prefix for your saved files

**-l** = A list of your human readable column names in the tsv (Example:"area,area,area,neg,area,inj") 
- Your list must use 'neg' for any columns containing negative controls and 'inj' for any injection site column.
- Your list can use whatever names you want for the target areas but avoid spaces and characters.
- The code will try to sort target areas if you have repeat values (visp1,visp2,visp3,audp1,audp2...).
- I do not know if you can use more than one neg and and inj in a matrix. My data does not look like that and I havent tested.
- Use the exact format shown, no spaces between your list or the code will error.

<br/>
<br/>

## **OPTIONAL Arguments**

**-i** = Sets a threshold value for filtering barcodes by minimim injection site UMI. (default: 1) Han et. al. sets this to 300, Klingler et. al. 2018 sets this to 100, you may set it to your desired value.

**-t** = Sets a threshold value for filtering barcodes by requiring a minimim UMI value in at least one target area or the barcode is removed. (default: 10) Han et. al. sets this to 10, you may set it to your desired value.

**-r** = Minimum fold-difference between 'inj' value and the highest target count. Rows not meeting this threshold are removed. (default: 10) Han et. al. sets this to 10, you may set it to your desired value.

**-f** = Enable outlier filtering of barcodes. Where any target value in a row is greater than the mean of all target values in the dataset plus two standard deviations, drop that barcode. We include this argument for microdissections which neighbor the injection site and there is no good way to know if very large UMI counts are from some kind of contamination. Use this at your own discretion.

**-a** = Value for alpha. This is the signifigance threshold (default 0.05) for Bonferroni correction, False Discovery Rate correction, and the Binomial Test.

**-u** = Changes the threshold filter for target area UMI counts where very small values (noise) will be set to zero. (default: 2) You may want to set this to the maximum value seen in your negative control as was done in Han et. al. 2018..

```
For example the default setting is 2 meaning that for every rown in your matrix the following logic will be applied

some_row_[0,1,0,35,12,1,0,120,1,0]

will be filtered with the default value of 2 to

some_row_[0,0,0,35,12,0,0,120,0,0].

Used for potential noise reduction of single UMI values in targets, but you can change this if you would like.
```

**--force_user_threshold** = enforce the value you set for **-u**, else the script will pick the largest value from the user input, the UMI KDE curve elbow value, the maximal value in the negative control column, or the default minimum of 2.

<br/>

## **BUGS**
There are a few bugs presently. 

1. The plots may experience formatting issues

2. The variables order_full and order_partial are not dynamically defined and their use is not currently implemented correctly. You may see these strings in the cli output, but they can be ignored.

<br/>

## **Old Arguments not yet removed**

**-A** = Label from your labels to match for the first "important area" (e.g., 'AL') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

**-B** = Label from your labels to match for the second "important area" (e.g., 'PM') Must match something in your labels! (updated to dynamically calculate using all labeled areas)

<br/>

---

## Pipeline Workflow

The MAPseq processing pipeline follows a structured workflow from raw data to publication-ready figures. Below is the recommended execution order:

### 1. Data Preparation
- **Input**: Raw `.nbcm.tsv` files from CSHL pipeline
- **Script**: `preprocess_and_aggregate.py`
- **Output**: `aggregated_cleaned_matrix.tsv`
- **Purpose**: Standardize column names, apply noise filtering, aggregate replicates

### 2. Main Processing
- **Input**: Aggregated cleaned matrix or individual `.nbcm.tsv` files
- **Script**: `process-nbcm-tsv.py`
- **Output**: Filtered matrices, normalized data, statistical analyses, motif analyses
- **Purpose**: Core analysis including filtering, normalization, statistical testing, and motif identification

### 3. Helper Scripts (Sequential Execution)
After main processing, run helper scripts in numerical order:

1. **01_motif_analysis_per_animal.py** → Per-animal motif statistics
2. **02_projection_analysis.py** → Projection pattern analysis
3. **03_composition.py** → Composition by age analysis
4. **04_proportions_over_time_stats.py** → Temporal proportion statistics
5. **05_motif_analysis.py** → Motif frequency analysis (required for step 6)
6. **06_all_motif_divergence.py** → Motif divergence (requires step 5)
7. **07_motif_significange_trajectories.py** → Trajectory analysis (uses step 5)
8. **08_motif_clustering.py** → Clustering analysis (uses step 7)
9. **09_plot_normalized_projection_strength_data.py** → Projection strength visualization
10. **13_aggregate_projection_summaries.py** → Aggregate summaries across parameterizations

**Optional/Comparison Scripts** (require external datasets):
- **10_compare_datasets_pipeline.py** → Two-way comparison
- **11_compare_vsv_mapseq_two_way.py** → VSV vs MapSeq
- **12_compare_datasets_pipeline_mapseq.py** → Three-way comparison

### 4. Quality Control
- **Script**: `postprocessing_checks.py`
- **Purpose**: Verify outputs, check for errors, generate QC reports
- **When**: After main processing and helper scripts

### 5. Data Extraction and Conclusions
- **Scripts**: 
  - `conclusions/scripts/extract_stability_data.py` → Extract analysis data
  - `conclusions/scripts/generate_conclusions.py` → Generate markdown report
- **Purpose**: Compile findings into structured conclusions document

### 6. Figure Generation
- **Script**: `figure_generation/generate_figure_from_outputs.py`
- **Purpose**: Create publication-ready multi-panel figures
- **When**: After all analyses are complete

### Execution Methods

**Option 1: Manual Execution**
Run scripts individually in the order listed above.

**Option 2: Batch Execution**
Use `run_commands.sh` to execute all commands from `all_commands.txt`. Make the script executable once, then run it:
```bash
chmod +x run_commands.sh
./run_commands.sh
```

**Option 3: GUI**
Use `MAPseq_Wizard.py` for the main processing step (preprocessing still requires terminal).

### Dependencies

- **Main processing** must complete before helper scripts
- **Script 05** must run before scripts 06 and 07
- **Script 07** must run before script 08
- **Quality control** can run at any time after main processing
- **Figure generation** requires outputs from multiple scripts
- **Conclusions scripts** require helper script outputs

---

## Helper Scripts

After running the main processing pipeline (`process-nbcm-tsv.py`), you can run helper scripts for additional analysis. These scripts are organized in the `helpers/` directory and are numbered by their execution order in the pipeline.

### Directory Structure

```
helpers/
├── scripts/              # All Python helper scripts (numbered by execution order)
├── logs/                # Processing log files
├── outputs/             # All script outputs (numbered to match scripts)
├── composition by age/  # Legacy directory (contains data files)
└── motif_proportion_analysis/  # Legacy directory (now empty, scripts moved to scripts/)
```

### Script Execution Order

Helper scripts are numbered by their execution order in the pipeline:

1. **01_motif_analysis_per_animal.py** - Per-animal motif analysis
2. **02_projection_analysis.py** - Projection analysis
3. **03_composition.py** - Composition by age analysis
4. **04_proportions_over_time_stats.py** - Proportions over time statistics
5. **05_motif_analysis.py** - Motif proportion analysis (step 1)
6. **06_all_motif_divergence.py** - Motif divergence analysis (step 2, requires step 5)
7. **07_motif_significange_trajectories.py** - Motif trajectory analysis
8. **08_motif_clustering.py** - Motif clustering (uses output from step 7)
9. **09_plot_normalized_projection_strength_data.py** - Normalized projection strength plots
10. **13_aggregate_projection_summaries.py** - Aggregates projection summaries across parameterizations
    - Finds all `projection_summary.csv` files in output directories
    - Filters for aggregate samples (containing "_ALL_" in sample name)
    - Extracts metadata (age, parameterization) from file paths
    - Combines all summaries into a single CSV file

#### Optional/Comparison Scripts

These scripts are typically commented out in `all_commands.txt` and require external datasets:

- **10_compare_datasets_pipeline.py** - Two-way dataset comparison
- **11_compare_vsv_mapseq_two_way.py** - VSV vs MapSeq comparison
- **12_compare_datasets_pipeline_mapseq.py** - Three-way comparison (Allen, VSV, MapSeq)

### Output Directories

All script outputs are organized in `helpers/outputs/` with numbered subdirectories matching script numbers:

- `helpers/outputs/01_motif_analysis_per_animal/`
- `helpers/outputs/02_projection_analysis/`
- `helpers/outputs/03_composition/`
- `helpers/outputs/04_proportions_over_time_stats/`
- `helpers/outputs/05_motif_analysis/`
- `helpers/outputs/06_all_motif_divergence/`
- `helpers/outputs/07_motif_significange_trajectories/`
- `helpers/outputs/08_motif_clustering/`
- `helpers/outputs/09_plot_normalized_projection_strength_data/`
- `helpers/outputs/10_compare_datasets_pipeline/`
- `helpers/outputs/11_compare_vsv_mapseq_two_way/`
- `helpers/outputs/12_compare_datasets_pipeline_mapseq/`
- `helpers/outputs/13_aggregate_projection_summaries/` (or in repository root)

### Running Helper Scripts

All scripts should be run from the repository root or from the `helpers/` directory:

```bash
# From repository root
python helpers/scripts/01_motif_analysis_per_animal.py
python helpers/scripts/02_projection_analysis.py
# ... etc

# Or from helpers directory
cd helpers
python scripts/01_motif_analysis_per_animal.py
python scripts/02_projection_analysis.py
# ... etc
```

For convenience, you can use the `all_commands.txt` or `all_commands_all-parameters.txt` file which contains all commands in the correct order. Make the script executable once (`chmod +x run_commands.sh`), then run:

```bash
./run_commands.sh
```

Or manually execute commands from `all_commands.txt`.

### Shell Scripts

#### run_commands.sh

Batch execution script that runs all commands from a command file sequentially with logging.

**Purpose:**
- Execute multiple processing commands in sequence
- Create timestamped log files for tracking execution
- Capture all output and errors for debugging

**Usage:**
```bash
chmod +x run_commands.sh   # one-time, if needed
./run_commands.sh
```

**Features:**
- **Automatic logging**: Creates a log file named `processing_YYYYMMDD_HHMMSS.log` with timestamp
- **Sequential execution**: Runs commands one at a time from `all_commands.txt`
- **Error capture**: Captures both stdout and stderr to the log file
- **Progress tracking**: Echoes each command to console before execution

**Requirements:**
- `all_commands.txt` file in the same directory (or modify script to specify path)
- All commands in the file should be valid and executable

**Example:**
```bash
# From repository root
chmod +x run_commands.sh   # one-time, if needed
./run_commands.sh

# Or from bash directory
cd bash
chmod +x run_commands.sh
./run_commands.sh
```

**Note:** The script will continue executing even if individual commands fail. Check the log file to identify any failures. The script is identical in both the root directory and `bash/` subdirectory.

### Notes

- Scripts use `Path(__file__).parent` to determine their location, so they work regardless of where they're called from
- Output directories are automatically created by each script
- Log files from processing are stored in `helpers/logs/`
- All dependencies are listed in the main `requirements.txt` file
- Script and output directory numbers correspond to execution order in the pipeline

---

## Utility Scripts

Additional utility scripts for quality control, figure generation, and data extraction.

### postprocessing_checks.py

Comprehensive quality control checker that analyzes processing outputs and generates detailed reports.

**Purpose:**
- Analyze log files for errors, warnings, and success indicators
- Verify output file structure and completeness
- Generate human-readable QC reports with statistics and recommendations

**Usage:**
```bash
python postprocessing_checks.py [--repo_root REPO_ROOT] [--log_file LOG_FILE] [--output OUTPUT] [--base_output_dir BASE_OUTPUT_DIR]
```

**Arguments:**
- **--repo_root** (optional): Repository root directory (default: script directory)
- **--log_file** (optional): Path to processing log file (default: find most recent)
- **--output** (optional): Output report file path (default: `qc_report_TIMESTAMP.txt`)
- **--base_output_dir** (optional): Base output directory for processing results

**Features:**
- **Log analysis**: Parses log files to identify successful completions, expected failures (low data quality), and unexpected errors
- **Output verification**: Checks for expected output files in main processing and helper script directories
- **Error categorization**: Categorizes errors by type (FileNotFoundError, KeyError, AssertionError, etc.)
- **Warning analysis**: Identifies and categorizes warnings (FutureWarnings, UserWarnings, processing warnings)
- **Statistics**: Provides summary statistics including success rates, file counts, and animal processing outcomes

**Output:**
- Console output with immediate feedback
- Detailed QC report file with sections for:
  - Executive summary
  - Findings (errors, warnings, successes, info)
  - Unexpected failures with detailed context
  - Recommendations for addressing issues

**Example:**
```bash
python postprocessing_checks.py --base_output_dir 02_output/p12/05.HAN_filter_parameters_i300_r10_t10_u5
```

### conclusions/scripts/extract_stability_data.py

Extracts stability analysis data from pipeline outputs for downstream analysis. (Project specific and not universal)

**Purpose:**
- Extract Kruskal-Wallis test results from script 01
- Extract transition significance data from script 07
- Extract motif frequency matrices from script 05
- Extract divergence metrics from script 06
- Aggregate upsetplot data from each age group

**Usage:**
```bash
python conclusions/scripts/extract_stability_data.py [--base_dir BASE_DIR] [--output_file OUTPUT_FILE]
```

**Arguments:**
- **--base_dir** (optional): Base directory containing pipeline outputs (default: repository root)
- **--output_file** (optional): Output JSON file path (default: `extracted_stability_data.json`)

**Output:**
- JSON file containing structured data for all extracted metrics
- Organized by parameterization, model type (uniform/region_specific), and age group

**Example:**
```bash
python conclusions/scripts/extract_stability_data.py --base_dir 02_output --output_file stability_data.json
```

### helpers/scripts/13_aggregate_projection_summaries.py

Aggregates projection summary files across all parameterizations and age groups.

**Purpose:**
- Find all `projection_summary.csv` files in output directories
- Filter for aggregate samples (containing "_ALL_" in sample name)
- Extract metadata (age, parameterization) from file paths
- Combine all summaries into a single CSV file

**Usage:**
```bash
python helpers/scripts/13_aggregate_projection_summaries.py [--output_dir OUTPUT_DIR] [--base_dir BASE_DIR]
```

**Arguments:**
- **--output_dir** (optional): Output directory for aggregated summary (default: repository root)
- **--base_dir** (optional): Base directory to search for projection_summary.csv files (default: `02_output/`)

**Output:**
- Single CSV file containing all aggregate projection summaries with metadata columns for age and parameterization

**Example:**
```bash
python helpers/scripts/13_aggregate_projection_summaries.py --base_dir 02_output --output_dir aggregated_summaries/
```
