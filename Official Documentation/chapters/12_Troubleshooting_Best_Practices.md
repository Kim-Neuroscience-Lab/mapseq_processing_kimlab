# Chapter 12: Troubleshooting and Best Practices

## Overview

This chapter provides solutions to common issues, parameter selection guidelines, quality control procedures, and best practices for using the MAPseq processing pipeline.

## Known Issues

- **Plot formatting**: Some plots may experience formatting issues (e.g. axis labels, layout). These are cosmetic and do not affect the underlying analysis.
- **order_full and order_partial**: The variables `order_full` and `order_partial` are not dynamically defined and their use is not fully implemented. You may see these strings in CLI output; they can be ignored.

## Common Errors and Solutions

### N₀ Estimation Errors

**Error**: "No valid positive real root found for N0"

**Causes**:
- Data quality issues (too few projections)
- All projections to single region
- Numerical instability in symbolic solving

**Solutions**:
1. Check projection counts: Verify sufficient neurons project to multiple regions
2. Verify data quality: Ensure filtering didn't remove too many cells
3. Check region labels: Ensure correct column mapping
4. Try different parameterization: Adjust filtering thresholds

**Code Reference**: `process-nbcm-tsv.py` (N₀ root selection after `solve_for_roots()`)

### Column Mismatch Errors

**Error**: "Mismatch: Normalized matrix columns X, headers Y"

**Causes**:
- Column labels don't match data columns
- Filtering removed columns unexpectedly
- Label argument format incorrect

**Solutions**:
1. Verify label argument: Check format (comma-separated, no spaces)
2. Check column names: Ensure labels match actual column names
3. Verify preprocessing: Check that preprocessing preserved columns

**Code Reference**: `process-nbcm-tsv.py` (column assert after normalization)

### Empty Matrix Errors

**Error**: "WARNING: Normalized matrix is empty"

**Causes**:
- All cells filtered out
- Filtering thresholds too strict
- Data quality issues

**Solutions**:
1. Check filtering parameters: Reduce thresholds if too strict
2. Verify input data: Ensure data has valid projections
3. Check negative control: Ensure negative control filtering not too aggressive

**Code Reference**: `process-nbcm-tsv.py` (`normalize_rows()` empty-matrix check)

### Import Errors

**Error**: "ModuleNotFoundError" or "ImportError"

**Causes**:
- Missing dependencies
- Wrong conda environment
- Package version conflicts

**Solutions**:
1. Activate environment: `conda activate mapseq_processing`
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python version: Ensure Python 3.9
4. Update packages: `pip install --upgrade package_name`

### File Not Found Errors

**Error**: "FileNotFoundError" or "No such file or directory"

**Causes**:
- Incorrect file paths
- Missing input files
- Output directory doesn't exist

**Solutions**:
1. Verify file paths: Use absolute paths or check relative paths
2. Check file existence: Ensure input files exist
3. Create output directory: Scripts create directories, but verify permissions

## Parameter Selection Guidelines

### Injection UMI Threshold (`-i, --injection_umi_min`)

**Default**: 1

**Recommendations**:
- **Han et al. 2018**: 300
- **Klingler et al. 2018**: 100
- **Typical range**: 50-300

**Considerations**:
- Higher values: More stringent filtering, fewer cells
- Lower values: Less filtering, more cells but potentially lower quality

### Target UMI Minimum (`-t, --min_target_count`)

**Default**: 10

**Recommendations**:
- **Han et al. 2018**: 10
- **Typical range**: 5-20

**Considerations**:
- Ensures cells have meaningful projections
- Too high: May remove valid low-projection cells
- Too low: May include noise

### Injection-to-Target Ratio (`-r, --min_body_to_target_ratio`)

**Default**: 10

**Recommendations**:
- **Han et al. 2018**: 10
- **Typical range**: 5-20

**Considerations**:
- Ensures injection site signal is strong
- Prevents contamination from neighboring regions
- Too high: May remove valid cells with strong projections

### Target UMI Threshold (`-u, --target_umi_min`)

**Default**: 2

**Recommendations**:
- Set to maximum value in negative control
- **Typical range**: 1-5

**Considerations**:
- Removes noise from single UMI counts
- Should match negative control statistics
- Use `--force_user_threshold` to override automatic calculation

### Significance Threshold (`-a, --alpha`)

**Default**: 0.05

**Recommendations**:
- Standard: 0.05
- Strict: 0.01
- Bonferroni correction applied automatically

**Considerations**:
- Lower values: More stringent significance
- Higher values: More permissive significance

## Quality Control Procedures

### Pre-Processing QC

1. **Verify Input Files**:
   - Check file format (TSV)
   - Verify column structure
   - Check for missing values

2. **Check Data Quality**:
   - Verify negative control values (should be low)
   - Check injection site values (should be high)
   - Ensure sufficient cell count (>100 recommended)

3. **Validate Column Mapping**:
   - Ensure `barcodes` column mapped
   - Verify `neg` and `inj` columns identified
   - Check target region columns

### Post-Processing QC

1. **Verify Output Files**:
   - Check all expected files exist
   - Verify file sizes (not empty)
   - Check file formats

2. **Validate Results**:
   - Effect sizes in reasonable range (-5 to +5)
   - P-values in range [0, 1]
   - Expected counts positive
   - Motif counts match observed

3. **Check for Errors**:
   - Review log files
   - Check for warning messages
   - Verify processing completed successfully

### Helper Script QC

1. **Check Dependencies**:
   - Script 05 before 06 and 07
   - Script 07 before 08
   - Main processing before all helpers

2. **Verify Outputs**:
   - Check expected output files exist
   - Verify file formats
   - Check for error messages

## Best Practices

### Data Preparation

1. **Preprocessing**:
   - Use consistent column naming
   - Apply appropriate thresholds
   - Verify aggregated matrix quality

2. **Parameter Selection**:
   - Start with default parameters
   - Adjust based on data quality
   - Document parameter choices

3. **Quality Control**:
   - Check data before processing
   - Verify outputs after processing
   - Keep processing logs

### Processing Workflow

1. **Execution Order**:
   - Preprocessing first
   - Main processing second
   - Helper scripts in numerical order

2. **Anchor Model**:
   - Process P60 first with `--is-anchor-model`
   - Use anchor files for developmental cohorts
   - Verify anchor files exist

3. **Batch Processing**:
   - Use `all_commands.txt` for consistency
   - Keep command files version-controlled
   - Document any manual modifications

### Result Interpretation

1. **Multiple Models**:
   - Compare results across models
   - Agreement suggests robust findings
   - Disagreement may indicate model sensitivity

2. **Stability Analysis**:
   - Use both model-independent and model-dependent metrics
   - Consider all evidence together
   - Don't rely on single metric

3. **Biological Interpretation**:
   - Consider biological context
   - Verify findings make biological sense
   - Consult domain experts

## Performance Optimization

### Computational Resources

1. **Memory**:
   - Large datasets may require 16+ GB RAM
   - Monitor memory usage during processing
   - Consider processing in batches if needed

2. **Processing Time**:
   - N₀ estimation can be slow for many regions
   - Visualization may take time for large motif sets
   - Consider parallel processing for multiple samples

3. **Storage**:
   - Output files can be large
   - Plan for sufficient disk space
   - Consider archiving old results

### Optimization Strategies

1. **Skip Sections**:
   - Use `--skip-sections` to skip visualization/clustering if not needed
   - Saves processing time

2. **Model Selection**:
   - Use `--model-type` to run specific models only
   - Reduces computation if not all models needed

3. **Batch Processing**:
   - Process multiple samples in parallel
   - Use job schedulers for cluster computing

## Data Quality Requirements

### Minimum Requirements

1. **Cell Count**: >100 cells recommended
2. **Regions**: At least 2 target regions
3. **Projections**: Cells should project to multiple regions
4. **Quality**: Negative control should be low, injection high

### Quality Indicators

1. **Good Quality**:
   - High injection UMI values
   - Low negative control values
   - Cells project to multiple regions
   - Reasonable N₀ estimates

2. **Poor Quality**:
   - Low injection UMI values
   - High negative control values
   - Most cells project to single region
   - N₀ estimation failures

## Getting Help

### Documentation

1. **This Documentation**: Comprehensive reference
2. **README.md**: Quick start guide
3. **Code Comments**: Inline documentation

### Common Resources

1. **Error Messages**: Read carefully, often contain solutions
2. **Log Files**: Check for detailed error information
3. **Example Data**: Use sample data to verify setup

### Support

1. **GitHub Issues**: Report bugs and ask questions
2. **Code Review**: Check code for implementation details
3. **Community**: Consult with other users

---

*For detailed usage instructions, see [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline.md). For output interpretation, see [Chapter 8: Output Files and Interpretation](08_Output_Files_Interpretation.md).*
