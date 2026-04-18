# MAPseq Processing Pipeline Documentation

## Overview

This documentation provides comprehensive coverage of the MAPseq (Multiplexed Analysis of Projections by Sequencing) processing pipeline, including installation, usage, statistical methods, code review, and mathematical functions.

## Documentation Structure

### Chapters

1. **[Introduction](chapters/01_Introduction.md)** - Overview, key concepts, and pipeline architecture
2. **[Installation and Setup](chapters/02_Installation_Setup.md)** - System requirements and installation instructions
3. **[Data Preparation](chapters/03_Data_Preparation.md)** - Preprocessing and data format requirements
4. **[Main Processing Pipeline](chapters/04_Main_Processing_Pipeline.md)** - Core analysis script usage and workflow
5. **[Statistical Methods](chapters/05_Statistical_Methods.md)** - N₀ estimation, binomial testing, and effect sizes
6. **[Probability Models](chapters/06_Probability_Models.md)** - All probability models explained in detail
7. **[Helper Scripts](chapters/07_Helper_Scripts.md)** - Cross-age analysis scripts and their usage
8. **[Output Files and Structure](chapters/08_Output_Files_Interpretation.md)** - Output paths, filenames, and column reference
9. **[Code Review](chapters/09_Code_Review.md)** - Architecture, functions, and implementation details
10. **[Mathematical Functions Reference](chapters/10_Mathematical_Functions.md)** - All formulas with code references
11. **[Stability Analysis](chapters/11_Stability_Analysis.md)** - Temporal stability framework and interpretation
12. **[Troubleshooting and Best Practices](chapters/12_Troubleshooting_Best_Practices.md)** - Common issues and solutions
13. **[References and Appendices](chapters/13_References_Appendices.md)** - Code references, notation glossary, and quick reference
14. **[Experimental Features](chapters/14_Experimental_Features.md)** - Untested GUI wizards, maintainer batch scripts (use bash + command file for production)
15. **[Trajectory Results](chapters/15_Trajectory_Results_Interpretation.md)** - Helper 15 output files and methods (reference; no dataset-specific results)
16. **[Cross-Anchor Analysis](chapters/16_Cross_Anchor_Comparative_Analysis.md)** - Conceptual checklist for comparing anchor configurations

## Quick Start

### For New Users

1. Start with [Chapter 1: Introduction](chapters/01_Introduction.md) for overview
2. Follow [Chapter 2: Installation and Setup](chapters/02_Installation_Setup.md) for installation
3. Review [Chapter 3: Data Preparation](chapters/03_Data_Preparation.md) for data format
4. **Run the pipeline**: Edit `all_commands.txt` (or use `all_commands_all-parameters.txt`) to match your paths and samples, then from the repository root make the script executable if needed (`chmod +x run_commands.sh`) and run **`./run_commands.sh`**. See [Chapter 4: Main Processing Pipeline](chapters/04_Main_Processing_Pipeline.md) for details

### For Understanding Methods

1. Read [Chapter 5: Statistical Methods](chapters/05_Statistical_Methods.md) for statistical framework
2. Review [Chapter 6: Probability Models](chapters/06_Probability_Models.md) for model details
3. Consult [Chapter 10: Mathematical Functions Reference](chapters/10_Mathematical_Functions.md) for formulas

### For outputs and file layout

1. See [Chapter 8: Output Files and Structure](chapters/08_Output_Files_Interpretation.md) for paths and columns
2. Check [Chapter 7: Helper Scripts](chapters/07_Helper_Scripts.md) for helper outputs and run order
3. Review [Chapter 11: Stability Analysis](chapters/11_Stability_Analysis.md) for a generic metrics framework (no study-specific verdicts)
4. Advanced: [Chapter 15](chapters/15_Trajectory_Results_Interpretation.md), [Chapter 16](chapters/16_Cross_Anchor_Comparative_Analysis.md)

### For Developers

1. Review [Chapter 9: Code Review](chapters/09_Code_Review.md) for architecture
2. Consult [Chapter 10: Mathematical Functions Reference](chapters/10_Mathematical_Functions.md) for implementations
3. Check [Chapter 13: References and Appendices](chapters/13_References_Appendices.md) for code references

## HTML Documentation

HTML versions of all chapters are available in the `html/` directory:

- [HTML Index](html/index.html) - Navigation page with links to all chapters
- Individual chapter HTML files in `html/` directory

## Documentation Features

### Comprehensive Coverage

- **Software User Level**: Step-by-step usage instructions
- **Manuscript Methods Level**: Detailed mathematical and statistical explanations
- **Code Review**: Function-by-function analysis with logic explanations
- **Mathematical Reference**: All formulas with code implementation references

### Style Guidelines

- Consistent formatting across all chapters
- Mathematical notation using LaTeX
- Code blocks with syntax highlighting
- Mermaid diagrams for workflows
- Cross-references between chapters

## Getting Help

### Documentation

- Browse chapters using the table of contents above
- Use HTML version for better navigation
- Search for specific topics using chapter headings

### Troubleshooting

- See [Chapter 12: Troubleshooting and Best Practices](chapters/12_Troubleshooting_Best_Practices.md)
- Check error messages carefully
- Review code references in [Chapter 13: References and Appendices](chapters/13_References_Appendices.md)

## Contributing

When updating documentation:

1. Update relevant chapter markdown files
2. Regenerate HTML files using `scripts/generate_html.py`
3. Update this README if structure changes
4. Verify cross-references between chapters

## Version

Documentation version: April 2026

Corresponds to MAPseq processing pipeline as of April 2026.

---

*For the latest updates, check the repository. For code implementation details, see [Chapter 9: Code Review](chapters/09_Code_Review.md).*
