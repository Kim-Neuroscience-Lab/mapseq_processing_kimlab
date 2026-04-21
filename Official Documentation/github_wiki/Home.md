# MAPseq Processing Pipeline Documentation

Comprehensive documentation for the MAPseq (Multiplexed Analysis of Projections by Sequencing) processing pipeline, including installation, usage, statistical methods, code review, and mathematical functions.

This wiki is generated from the main repository under `Official Documentation/chapters/`. To refresh, run `Official Documentation/scripts/export_github_wiki.py` and push the `github_wiki/` contents to this wiki repository.

## Documentation chapters

1. **[Chapter 1: Introduction](01_Introduction)** — Overview, key concepts, pipeline architecture, and research questions
2. **[Chapter 2: Installation and Setup](02_Installation_Setup)** — System requirements, installation methods, and environment setup
3. **[Chapter 3: Data Preparation](03_Data_Preparation)** — Input data format, preprocessing workflow, and quality control
4. **[Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline)** — Command-line interface, processing steps, and output structure
5. **[Chapter 5: Statistical Methods](05_Statistical_Methods)** — N₀ estimation, binomial testing, multiple testing correction, and effect sizes
6. **[Chapter 6: Probability Models](06_Probability_Models)** — Uniform, region-specific, correlated, and additional probability models
7. **[Chapter 7: Helper Scripts](07_Helper_Scripts)** — Cross-age analysis scripts, execution order, and dependencies
8. **[Chapter 8: Output Files and Structure](08_Output_Files_Interpretation)** — Output paths, file formats, and column reference
9. **[Chapter 9: Code Review](09_Code_Review)** — Architecture, key functions, data flow, and implementation details
10. **[Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions)** — Formulas with code references and interpretations
11. **[Chapter 11: Stability Analysis](11_Stability_Analysis)** — Generic stability metrics framework (not study-specific results)
12. **[Chapter 12: Troubleshooting and Best Practices](12_Troubleshooting_Best_Practices)** — Common errors, parameter selection, quality control, and best practices
13. **[Chapter 13: References and Appendices](13_References_Appendices)** — Code references, notation glossary, statistical test reference, and quick reference
14. **[Chapter 14: Experimental Features](14_Experimental_Features)** — GUI wizards, maintainer batch scripts; use bash + command file for production
15. **[Chapter 15: Trajectory Results](15_Trajectory_Results_Interpretation)** — Helper 15 outputs and methods (file reference; no bundled results)
16. **[Chapter 16: Cross-Anchor Analysis](16_Cross_Anchor_Comparative_Analysis)** — Conceptual workflow for comparing anchor configurations

## Quick start

### For new users

1. Start with [Chapter 1: Introduction](01_Introduction) for overview
2. Follow [Chapter 2: Installation and Setup](02_Installation_Setup) for installation
3. Review [Chapter 3: Data Preparation](03_Data_Preparation) for data format
4. **Run the pipeline**: Edit `all_commands.txt` (or `all_commands_all-parameters.txt`) to match your paths and samples, then from the repository root run `./run_commands.sh`. See [Chapter 4: Main Processing Pipeline](04_Main_Processing_Pipeline) for details.

### For understanding methods

1. Read [Chapter 5: Statistical Methods](05_Statistical_Methods) for the statistical framework
2. Review [Chapter 6: Probability Models](06_Probability_Models) for model details
3. Consult [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions) for formulas

### For outputs and file layout

1. See [Chapter 8: Output Files and Structure](08_Output_Files_Interpretation) for paths and columns
2. Check [Chapter 7: Helper Scripts](07_Helper_Scripts) for helper outputs and run order
3. Review [Chapter 11: Stability Analysis](11_Stability_Analysis) for a generic metrics framework
4. Advanced: [Chapter 15](15_Trajectory_Results_Interpretation), [Chapter 16](16_Cross_Anchor_Comparative_Analysis)

### For developers

1. Review [Chapter 9: Code Review](09_Code_Review) for architecture
2. Consult [Chapter 10: Mathematical Functions Reference](10_Mathematical_Functions) for implementations
3. Check [Chapter 13: References and Appendices](13_References_Appendices) for code references

## Main repository

Pipeline source and HTML documentation: `https://github.com/OWNER/REPO` (branch `main`).

### MAPseq_wizard (experimental)

See the [MAPseq_wizard README](https://github.com/OWNER/REPO/blob/main/MAPseq_wizard/README.md) in the main repository.

---

*Documentation version: April 2026 (wiki export).*
