# Chapter 3: Data Preparation

## Overview

Before running the main MAPseq processing pipeline, raw data from the [CSHL mapseq-processing Python Pipeline](https://github.com/ZadorLaboratory/mapseq-processing) must be preprocessed and aggregated. This chapter describes the data format requirements and preprocessing workflow.

## Input Data Format

### NBCM Files

The pipeline expects **NBCM (Neural Barcode Count Matrix)** files in TSV (tab-separated values) format. These files are produced by the CSHL mapseq-processing pipeline.

**File Structure:**
- **Rows**: Individual barcoded neurons (cells)
- **Columns**: Brain regions, injection site, and negative controls
- **Values**: UMI (Unique Molecular Identifier) counts

**Required Columns:**
- **`barcodes`**: Unique identifier for each neuron (required)
- **`inj`**: Injection site UMI counts (required)
- **`neg`**: Negative control UMI counts (recommended)
- **Target regions**: One or more columns with region names (e.g., `RSP`, `PM`, `AM`, `AL`, `LM`)

**Example NBCM File:**
```
barcodes    RSP    PM    AM    AL    LM    neg    inj
barcode_1   0      5     12    8     3     0     150
barcode_2   2      0     0     15    0     0     200
barcode_3   0      0     0     0     0     1     50
```

**Sample data and labels**: The repository includes example NBCM files and label files in `sample_data/` (e.g. `JR0375.nbcm.tsv`, `labels_for_jr0375`, `p60_aggregated_cleaned_matrix.tsv`, `labels_for_p60_aggregated_cleaned_matrix.txt`) for format guidance. Use these when in doubt about column order and the labels argument for main processing. See [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md) for the `-l` argument.

### Data Quality Requirements

- **Minimum injection UMI**: Typically ≥ 300 (Han et al. 2018) or ≥ 100 (Klingler et al. 2018)
- **Minimum target UMI**: At least one target region must have UMI count ≥ 10
- **Injection-to-target ratio**: Injection UMI should be ≥ 10× the highest target UMI
- **Negative control**: Should be zero or very low (used for thresholding)

## Preprocessing Workflow

### Step 1: Column Mapping

The preprocessing script standardizes column names across replicate files. This is necessary because different samples may use different column naming conventions.

**Interactive Mode:**
The script prompts you to map each original column name to a standardized name:

```
📂 Processing file: sample1.nbcm.tsv
Detected columns:
   RSP | PM | AM | neg | inj
🔤 Enter standardized name for column 'RSP': RSP
🔤 Enter standardized name for column 'PM': PM
...
```

**Mapping File Mode:**
You can provide a YAML or JSON file with column mappings to skip interactive prompts:

```yaml
column_mappings:
  sample1.nbcm.tsv:
    RSP: RSP
    PM: PM
    AM: AM
    neg: neg
    inj: inj
negative_columns:
  sample1.nbcm.tsv: neg
```

### Step 2: Threshold Calculation

The preprocessing script calculates a threshold for noise filtering based on negative control values:

$$threshold = \mu_{neg} + \sigma_{neg}$$

where:
- $\mu_{neg}$ = mean of negative control values
- $\sigma_{neg}$ = standard deviation of negative control values

**Fallback**: If no negative control column is available or has no data, uses a fallback threshold (default: 2.0).

### Step 3: Noise Filtering

Values below the threshold are set to zero:

```python
# Pseudo-code
for each column (except barcodes):
    if value < threshold:
        value = 0
```

This removes noise from single UMI counts in target regions.

### Step 4: Negative Control Filtering

Rows where the negative control column has non-zero values are removed:

```python
if neg_col in df.columns:
    df = df[df[neg_col] == 0]
```

This ensures only high-quality barcodes are retained.

### Step 5: Column Alignment

All cleaned files are aligned to have the same column structure:
- Missing columns are filled with NaN
- Column order is preserved from first file processed

### Step 6: Aggregation

All aligned cleaned dataframes are concatenated into a single aggregated matrix:

```python
final_df = pd.concat(aligned_dfs, axis=0)
```

**Output**: `aggregated_cleaned_matrix.tsv`

## Preprocessing Script Usage

### Command-Line Interface

```bash
python preprocess_and_aggregate.py \
    -i /path/to/input/directory \
    -o /path/to/output/directory \
    -t 2.0 \
    --mapping-file mappings.yaml
```

**Arguments:**
- **`-i, --input_dir`** (required): Directory containing replicate `.tsv` files
- **`-o, --output_dir`** (required): Output directory for cleaned and aggregated files
- **`-t, --fallback_threshold`** (optional, default: 2.0): Threshold if negative control unavailable
- **`--mapping-file`** (optional): YAML or JSON file with column mappings

### Code Implementation

**Main Function**: `preprocess_and_aggregate.py::main()`

**Key Functions:**
- `load_mapping_file()`: Loads column mappings from YAML/JSON
- `get_column_mapping()`: Gets mappings interactively or from file
- `identify_neg_column()`: Identifies negative control column
- `preprocess_file()`: Processes individual file

**Logic Flow:**
1. Load mapping file (if provided)
2. For each `.tsv` file in input directory:
   - Map columns to standardized names
   - Identify negative control column
   - Calculate threshold
   - Apply thresholding
   - Filter negative control rows
   - Save cleaned file
3. Align all cleaned files to same column structure
4. Concatenate into aggregated matrix
5. Save aggregated file

## Output Files

### Individual Cleaned Files

For each input file, a cleaned version is saved:
- **Format**: `{basename}_cleaned.tsv`
- **Location**: Output directory
- **Content**: Thresholded and filtered data

### Aggregated Matrix

**File**: `aggregated_cleaned_matrix.tsv`
**Location**: Output directory
**Content**: All cleaned data combined into single matrix

**Structure:**
- Same columns as cleaned files (aligned)
- All rows from all input files concatenated
- NaN values for missing columns in individual files

## Quality Control

### Preprocessing Checks

After preprocessing, verify:

1. **File exists**: Check `aggregated_cleaned_matrix.tsv` exists
2. **Row count**: Sum of rows should match sum of input files (minus filtered rows)
3. **Column alignment**: All files should have same columns
4. **Thresholding**: Check that low values were zeroed
5. **Negative filtering**: Verify negative control rows were removed

### Common Issues

**Issue**: "You must map a column to 'barcodes'"
- **Solution**: Ensure one column is mapped to `barcodes` during column mapping

**Issue**: All values zeroed after thresholding
- **Solution**: Threshold may be too high. Check negative control values or adjust fallback threshold

**Issue**: Column alignment errors
- **Solution**: Ensure consistent column naming across files or use mapping file

## Next Steps

After preprocessing:

1. **Verify output**: Check `aggregated_cleaned_matrix.tsv` exists and has expected structure
2. **Proceed to main processing**: Use aggregated file as input to `process-nbcm-tsv.py`
3. **See**: [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md) for next steps

---

*For detailed preprocessing code review, see [Chapter 9: Code Review](09_Code_Review.md).*
