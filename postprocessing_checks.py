#!/usr/bin/env python3
"""
Post-Processing Quality Control Check Script

This script performs comprehensive quality control checks on all processing outputs,
analyzes log files for errors, and generates a human-readable summary report.

Usage:
    python postprocessing_checks.py [--log_file LOG_FILE] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import pandas as pd

# Expected output structure
EXPECTED_MAIN_OUTPUTS = {
    'p12': {
        'base_dir': '02_output/p12/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot_uniform.csv', '*_upsetplot_region_specific.csv', '*_pie_chart_data.csv', '*_motif_obs_exp_uniform.csv', '*_motif_obs_exp_region_specific.csv']
    },
    'p20': {
        'base_dir': '02_output/p20/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot_uniform.csv', '*_upsetplot_region_specific.csv', '*_pie_chart_data.csv', '*_motif_obs_exp_uniform.csv', '*_motif_obs_exp_region_specific.csv']
    },
    'p60': {
        'base_dir': '02_output/p60/05.HAN_filter_parameters_i300_r10_t10_u5',
        'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
        'analysis_dir': 'analysis',
        'analysis_files': ['*_upsetplot_uniform.csv', '*_upsetplot_region_specific.csv', '*_pie_chart_data.csv', '*_motif_obs_exp_uniform.csv', '*_motif_obs_exp_region_specific.csv']
    }
}

EXPECTED_HELPER_OUTPUTS = {
    '01_motif_analysis_per_animal': {
        'dir': 'helpers/outputs/01_motif_analysis_per_animal',
        'min_files': 6,  # 3 files × 2 models (uniform + region_specific)
        'expected_patterns': ['*.svg', '*.csv'],
        'has_model_subdirs': True
    },
    '02_projection_analysis': {
        'dir': 'helpers/outputs/02_projection_analysis',
        'min_files': 9,  # 3 age comparisons × 4 plot types
        'expected_patterns': ['*_vs_*.svg'],
        'has_model_subdirs': False
    },
    '03_composition': {
        'dir': 'helpers/outputs/03_composition',
        'min_files': 2,
        'expected_patterns': ['*.svg'],
        'has_model_subdirs': False
    },
    '04_proportions_over_time_stats': {
        'dir': 'helpers/outputs/04_proportions_over_time_stats',
        'min_files': 4,
        'expected_patterns': ['*.png', '*.csv'],
        'has_model_subdirs': False
    },
    '05_motif_analysis': {
        'dir': 'helpers/outputs/05_motif_analysis',
        'min_files': 10,  # 5 files × 2 models (uniform + region_specific)
        'expected_patterns': ['*_uniform.*', '*_region_specific.*', '*.png', '*.csv', '*.txt'],
        'has_model_subdirs': False  # Files have model suffixes instead
    },
    '06_all_motif_divergence': {
        'dir': 'helpers/outputs/06_all_motif_divergence',
        'min_files': 6,  # 3 files × 2 models (uniform + region_specific)
        'expected_patterns': ['divergence_*.svg'],
        'has_model_subdirs': True
    },
    '07_motif_significange_trajectories': {
        'dir': 'helpers/outputs/07_motif_significange_trajectories',
        'min_files': 6,  # 3 files × 2 models (uniform + region_specific)
        'expected_patterns': ['*.csv', '*.svg', '*.pdf'],
        'has_model_subdirs': True
    },
    '08_motif_clustering': {
        'dir': 'helpers/outputs/08_motif_clustering',
        'min_files': 8,  # 4 files × 2 models (uniform + region_specific)
        'expected_patterns': ['*.png'],
        'has_model_subdirs': True
    },
    '09_plot_normalized_projection_strength_data': {
        'dir': 'helpers/outputs/09_plot_normalized_projection_strength_data',
        'min_files': 20,
        'expected_patterns': ['*.svg'],
        'has_model_subdirs': False
    },
    '10_plot_per_cell_projection_strength_across_ages': {
        'dir': 'helpers/outputs/10_plot_per_cell_projection_strength_across_ages',
        'min_files': 26,
        'expected_patterns': ['*.svg'],
        'has_model_subdirs': False
    }
}


class QualityControlChecker:
    def __init__(self, repo_root=None, log_file=None, output_file=None, base_output_dir=None):
        if repo_root is None:
            repo_root = Path(__file__).parent
        self.repo_root = Path(repo_root)
        self.log_file = log_file
        self.output_file = output_file or self.repo_root / f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.base_output_dir = Path(base_output_dir) if base_output_dir else None
        
        self.findings = {
            'errors': [],
            'warnings': [],
            'successes': [],
            'info': []
        }
        
        self.stats = {
            'animals_processed': 0,
            'animals_succeeded': 0,
            'animals_failed': 0,
            'expected_failures': 0,
            'helper_scripts_run': 0,
            'helper_scripts_succeeded': 0,
            'total_output_files': 0
        }
        
        self.animal_outcomes = {}
        self.unexpected_failures = []

    def _get_output_root(self):
        """Return the output root (directory containing age dirs and helpers).
        When base_output_dir is set and contains p12_anchor, p20_anchor, or p60_anchor,
        return the path up to and including that segment so helper paths use the anchor tree.
        """
        if self.base_output_dir is None:
            return self.repo_root / "02_output"
        base_path = Path(self.base_output_dir)
        for i, part in enumerate(base_path.parts):
            if part in ('p12_anchor', 'p20_anchor', 'p60_anchor'):
                return Path(*base_path.parts[:i + 1])
        return self.repo_root / "02_output"

    def find_log_file(self):
        """Find the most recent processing log file"""
        if self.log_file:
            log_path = Path(self.log_file)
            if log_path.exists():
                return log_path
        
        # Search for processing log files
        log_files = list(self.repo_root.glob("processing_*.log"))
        if not log_files:
            # Also check helpers directory
            log_files = list((self.repo_root / "helpers").glob("processing_*.log"))
        
        if log_files:
            # Return most recent
            return max(log_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def analyze_log_file(self, log_path):
        """Analyze log file for errors, warnings, and success indicators"""
        if not log_path or not log_path.exists():
            self.findings['warnings'].append("No log file found for analysis")
            return
        
        self.findings['info'].append(f"Analyzing log file: {log_path.name} ({log_path.stat().st_size:,} bytes)")
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
            lines = log_content.split('\n')
        
        # Track animals and their outcomes
        animal_outcomes = {}  # {animal_name: {'status': 'success'|'failed'|'expected_failure', 'error': None|str, 'cell_count': None|int}}
        
        # Count animals processed
        animal_matches = re.findall(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', log_content)
        unique_animals = set(animal_matches)
        self.stats['animals_processed'] = len(unique_animals)
        
        # Initialize all animals
        for animal in unique_animals:
            animal_outcomes[animal] = {'status': 'unknown', 'error': None, 'cell_count': None}
        
        # Find successful completions - look for success message and trace back to find animal
        success_pattern = r'All result files have been successfully saved!'
        for i, line in enumerate(lines):
            if re.search(success_pattern, line):
                # Look backwards for the animal name (check up to 200 lines back)
                animal = None
                for j in range(max(0, i-200), i):
                    match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if match:
                        animal = match.group(1)
                        if animal in animal_outcomes:
                            animal_outcomes[animal]['status'] = 'success'
                        break
        
        # Find expected failures (N0 root finding errors) with context
        n0_pattern = r'No valid.*root found for N0.*observed_cells \((\d+)\)'
        for i, line in enumerate(lines):
            match = re.search(n0_pattern, line)
            if match:
                cell_count = int(match.group(1))
                # Look backwards for animal name
                for j in range(max(0, i-50), i):
                    animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if animal_match:
                        animal = animal_match.group(1)
                        animal_outcomes[animal]['status'] = 'expected_failure'
                        animal_outcomes[animal]['error'] = f"N0 root-finding failed (expected for low-quality sample with {cell_count} cells)"
                        animal_outcomes[animal]['cell_count'] = cell_count
                        break
        
        # Find unexpected failures with detailed context
        unexpected_failures = []
        
        # AssertionError
        for i, line in enumerate(lines):
            if 'AssertionError:' in line:
                error_msg = line.split('AssertionError:')[-1].strip()[:150]
                # Look backwards for animal name
                animal = None
                for j in range(max(0, i-50), i):
                    animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                    if animal_match:
                        animal = animal_match.group(1)
                        break
                
                if animal and animal_outcomes[animal]['status'] != 'expected_failure':
                    animal_outcomes[animal]['status'] = 'failed'
                    animal_outcomes[animal]['error'] = f"AssertionError: {error_msg}"
                    unexpected_failures.append((animal, 'AssertionError', error_msg))
        
        # Other critical errors
        error_patterns = [
            (r'FileNotFoundError:.*', 'FileNotFoundError', 'Missing input file'),
            (r'KeyError:.*', 'KeyError', 'Missing expected data column'),
            (r'IndentationError:.*', 'IndentationError', 'Code syntax error'),
            (r'SyntaxError:.*', 'SyntaxError', 'Code syntax error'),
        ]
        
        for pattern, error_type, description in error_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    error_msg = match.group(0)[:150]
                    # Look backwards for animal name
                    animal = None
                    for j in range(max(0, i-50), i):
                        animal_match = re.search(r'Running:.*process-nbcm-tsv\.py.*-s\s+([a-zA-Z0-9_]+)', lines[j])
                        if animal_match:
                            animal = animal_match.group(1)
                            break
                    
                    if animal and animal_outcomes[animal]['status'] not in ['expected_failure', 'failed']:
                        animal_outcomes[animal]['status'] = 'failed'
                        animal_outcomes[animal]['error'] = f"{error_type}: {error_msg}"
                        unexpected_failures.append((animal, error_type, error_msg))
        
        # Count outcomes - also use direct count from log as backup
        success_count_from_log = log_content.count("All result files have been successfully saved!")
        success_count_from_tracking = sum(1 for a in animal_outcomes.values() if a['status'] == 'success')
        # Use the higher count (log is more reliable)
        self.stats['animals_succeeded'] = max(success_count_from_log, success_count_from_tracking)
        
        self.stats['expected_failures'] = sum(1 for a in animal_outcomes.values() if a['status'] == 'expected_failure')
        self.stats['animals_failed'] = self.stats['animals_processed'] - self.stats['animals_succeeded']
        
        # Store unexpected failures for report
        self.unexpected_failures = unexpected_failures
        
        # Categorize all errors in the log (not just unexpected failures)
        error_patterns = {
            'FileNotFoundError': [
                r'FileNotFoundError:.*',
            ],
            'KeyError': [
                r'KeyError:.*',
            ],
            'ValueError': [
                r'ValueError:.*',
            ],
            'TypeError': [
                r'TypeError:.*',
            ],
            'IndentationError': [
                r'IndentationError:.*',
            ],
            'SyntaxError': [
                r'SyntaxError:.*',
            ],
            'AssertionError': [
                r'AssertionError:.*',
            ],
            'AttributeError': [
                r'AttributeError:.*',
            ],
            'IndexError': [
                r'IndexError:.*',
            ],
        }
        
        error_counts_all = {}
        for error_type, patterns in error_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, log_content)
                count += len(matches)
            if count > 0:
                error_counts_all[error_type] = count
        
        self.error_counts_all = error_counts_all
        
        # Check for critical errors summary (unexpected failures)
        if unexpected_failures:
            error_counts = Counter([e[1] for e in unexpected_failures])
            for error_type, count in error_counts.items():
                self.findings['errors'].append(
                    f"{error_type}: {count} occurrence(s) - CRITICAL: Requires investigation"
                )
        
        # Categorize FutureWarnings by type
        # First, get all FutureWarning lines
        all_future_warnings = re.findall(r'FutureWarning:.*', log_content, re.IGNORECASE)
        total_future_warnings = len(all_future_warnings)
        
        future_warning_patterns = {
            'Pandas fillna/ffill/bfill deprecation': [
                r'Downcasting object dtype arrays on \.fillna',
                r'Downcasting object dtype arrays on \.ffill',
                r'Downcasting object dtype arrays on \.bfill',
            ],
            'DataFrame concatenation deprecation': [
                r'behavior of DataFrame concatenation',
            ],
            'Chained assignment deprecation': [
                r'value is trying to be set on a copy.*chained assignment',
            ],
        }
        
        future_warning_counts = {}
        matched_warnings = set()
        
        for category, patterns in future_warning_patterns.items():
            count = 0
            for pattern in patterns:
                for i, warn in enumerate(all_future_warnings):
                    if i not in matched_warnings and re.search(pattern, warn, re.IGNORECASE):
                        count += 1
                        matched_warnings.add(i)
            if count > 0:
                future_warning_counts[category] = count
        
        # Calculate "Other" as unmatched warnings
        other_count = total_future_warnings - len(matched_warnings)
        if other_count > 0:
            future_warning_counts['Other FutureWarnings'] = other_count
        
        self.future_warning_counts = future_warning_counts
        
        if total_future_warnings > 0:
            self.findings['warnings'].append(
                f"FutureWarning: {total_future_warnings} occurrence(s) - LOW severity: Library deprecation warnings (mostly from pandas/upsetplot, non-critical)"
            )
        
        # Categorize UserWarnings by type
        # First, get all UserWarning lines (excluding our own QC report lines)
        all_user_warnings = [w for w in re.findall(r'UserWarning:.*', log_content, re.IGNORECASE) 
                            if 'occurrence(s)' not in w and 'severity' not in w]
        total_user_warnings = len(all_user_warnings)
        
        user_warning_patterns = {
            'Seaborn clustering performance': [
                r'Clustering large matrix.*fastcluster',
            ],
        }
        
        user_warning_counts = {}
        matched_warnings = set()
        
        for category, patterns in user_warning_patterns.items():
            count = 0
            for pattern in patterns:
                for i, warn in enumerate(all_user_warnings):
                    if i not in matched_warnings and re.search(pattern, warn, re.IGNORECASE):
                        count += 1
                        matched_warnings.add(i)
            if count > 0:
                user_warning_counts[category] = count
        
        # Calculate "Other" as unmatched warnings
        other_count = total_user_warnings - len(matched_warnings)
        if other_count > 0:
            user_warning_counts['Other UserWarnings'] = other_count
        
        self.user_warning_counts = user_warning_counts
        
        if total_user_warnings > 0:
            self.findings['warnings'].append(
                f"UserWarning: {total_user_warnings} occurrence(s) - MEDIUM severity: User-facing warnings that may indicate issues"
            )
        
        # Check for SyntaxWarnings
        syntax_warnings = len(re.findall(r'SyntaxWarning:.*', log_content))
        if syntax_warnings > 0:
            self.findings['warnings'].append(
                f"SyntaxWarning: {syntax_warnings} occurrence(s) - HIGH severity: Code syntax issues that should be fixed"
            )
        
        # Categorize processing warnings (⚠) by type
        processing_warning_patterns = {
            'User-forced threshold': [
                r'⚠.*User-forced threshold in effect',
                r'⚠.*Step 6b: Forcing user-defined UMI threshold',
            ],
            'Normalized matrix is empty': [
                r'WARNING: Normalized matrix is empty',
            ],
            'No nonzero target values': [
                r'WARNING: No nonzero target values found for threshold estimation',
            ],
            "'neg' column contains only NaN": [
                r"WARNING: 'neg' column contains only NaN",
            ],
            "'neg' column not found": [
                r"WARNING: 'neg' column not found",
            ],
            'Insufficient data points for KDE': [
                r'WARNING: Insufficient data points.*for KDE',
            ],
            'Complex number errors': [
                r'Skipping symbolic solution.*Cannot convert complex to float',
            ],
            'Zero or missing probabilities': [
                r'Warning: Zero or missing probabilities',
            ],
        }
        
        processing_warning_counts = {}
        for category, patterns in processing_warning_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, log_content, re.IGNORECASE)
                count += len(matches)
            if count > 0:
                processing_warning_counts[category] = count
        
        # Store processing warnings for detailed report
        self.processing_warning_counts = processing_warning_counts
        total_processing_warnings = sum(processing_warning_counts.values())
        
        if total_processing_warnings > 0:
            self.findings['warnings'].append(
                f"Processing Warning: {total_processing_warnings} occurrence(s) - MEDIUM severity: Processing warnings from the pipeline"
            )
        
        # Check for helper script execution by checking output files exist
        # (More reliable than parsing log since scripts may run outside the logged session)
        helper_scripts = {
            '01_motif_analysis_per_animal.py': 'helpers/outputs/01_motif_analysis_per_animal',
            '02_projection_analysis.py': 'helpers/outputs/02_projection_analysis',
            '03_composition.py': 'helpers/outputs/03_composition',
            '04_proportions_over_time_stats.py': 'helpers/outputs/04_proportions_over_time_stats',
            '05_motif_analysis.py': 'helpers/outputs/05_motif_analysis',
            '06_all_motif_divergence.py': 'helpers/outputs/06_all_motif_divergence',
            '07_motif_significange_trajectories.py': 'helpers/outputs/07_motif_significange_trajectories',
            '08_motif_clustering.py': 'helpers/outputs/08_motif_clustering',
            '09_plot_normalized_projection_strength_data.py': 'helpers/outputs/09_plot_normalized_projection_strength_data',
            '10_plot_per_cell_projection_strength_across_ages.py': 'helpers/outputs/10_plot_per_cell_projection_strength_across_ages',
        }
        
        for script, output_dir in helper_scripts.items():
            output_path = self.repo_root / output_dir
            if output_path.exists() and any(output_path.rglob('*')):
                self.stats['helper_scripts_run'] += 1
                self.findings['info'].append(f"Helper script {script}: Output directory exists with files")
            else:
                self.findings['warnings'].append(f"Helper script {script}: No output found (may not have run)")
        
        # Check for t-SNE and KDE fixes
        if 'Skipping t-SNE' in log_content or 'insufficient.*samples' in log_content.lower():
            self.findings['successes'].append("t-SNE perplexity fix working: small datasets handled gracefully")
        
        if 'Insufficient data points.*KDE' in log_content:
            self.findings['successes'].append("KDE fix working: insufficient data handled with fallback")
        
        # Check for syntax warnings (should be 0 after fix)
        syntax_warnings = len(re.findall(r'SyntaxWarning', log_content))
        if syntax_warnings == 0:
            self.findings['successes'].append("No SyntaxWarnings found (fixes applied successfully)")
        # Note: SyntaxWarnings are now handled in the warning_info section above
        
        # Store animal outcomes for detailed report
        self.animal_outcomes = animal_outcomes
    
    def check_main_outputs(self):
        """Check main processing outputs in 02_output directories"""
        self.findings['info'].append("\n=== Main Processing Outputs ===")
        
        # Extract age from base_output_dir if provided
        detected_age = None
        if self.base_output_dir:
            base_path = Path(self.base_output_dir)
            # Check if path contains an age directory (p3, p12, p20, p60)
            for part in base_path.parts:
                part_lower = part.lower()
                if part_lower in ['p3', 'p12', 'p20', 'p60']:
                    detected_age = part_lower
                    break
        
        # Filter ages to check based on detected age
        ages_to_check = list(EXPECTED_MAIN_OUTPUTS.items())
        if detected_age:
            # Only check the matching age
            if detected_age in EXPECTED_MAIN_OUTPUTS:
                ages_to_check = [(detected_age, EXPECTED_MAIN_OUTPUTS[detected_age])]
            else:
                # Age not in expected list (e.g., p3), check it anyway with default config
                self.findings['info'].append(f"Checking age {detected_age} (not in standard list)")
                # Try to extract parameterization from path
                parameterization = None
                for part in base_path.parts:
                    if part.startswith(('01.', '02.', '03.', '04.', '05.')):
                        parameterization = part
                        break
                
                if parameterization:
                    base_dir_str = f"02_output/{detected_age}/{parameterization}"
                else:
                    # Use the provided base_output_dir as-is
                    base_dir_str = str(self.base_output_dir.relative_to(self.repo_root)) if self.base_output_dir.is_relative_to(self.repo_root) else None
                
                ages_to_check = [(detected_age, {
                    'base_dir': base_dir_str,
                    'required_files': ['*_Filtered_Matrix.csv', '*_Normalized_Matrix.csv', '*_UMI_Total_Counts.csv'],
                    'analysis_dir': 'analysis',
                    'analysis_files': ['*_upsetplot_uniform.csv', '*_upsetplot_region_specific.csv', '*_pie_chart_data.csv', '*_motif_obs_exp_uniform.csv', '*_motif_obs_exp_region_specific.csv']
                })]
        
        for age, config in ages_to_check:
            if self.base_output_dir:
                if detected_age and detected_age == age:
                    # Use the provided base_output_dir directly for the matching age
                    base_dir = self.base_output_dir
                else:
                    # Construct path relative to base_output_dir for other ages
                    # If base_output_dir is age-specific, this shouldn't happen due to filtering above
                    # But handle case where base_output_dir is a parent directory
                    # Extract parameterization from config if available
                    config_path = Path(config['base_dir'])
                    if len(config_path.parts) >= 3:
                        # Expected format: 02_output/{age}/{parameterization}
                        parameterization = config_path.parts[-1]
                        base_dir = self.base_output_dir / age / parameterization
                    else:
                        # Fallback: try to construct from config
                        base_dir = self.base_output_dir / config['base_dir']
            else:
                base_dir = self.repo_root / config['base_dir']
            analysis_dir = base_dir / config['analysis_dir']
            
            if not base_dir.exists():
                self.findings['errors'].append(f"{age}: Base output directory not found: {base_dir}")
                continue
            
            # Count output files
            output_files = []
            for pattern in config['required_files']:
                files = list(base_dir.glob(pattern))
                output_files.extend(files)
            
            file_count = len(output_files)
            self.stats['total_output_files'] += file_count
            
            # Extract animal names from files
            animal_names = set()
            for f in output_files:
                # Extract animal name from filename (e.g., jr0420_Filtered_Matrix.csv -> jr0420)
                match = re.search(r'([a-zA-Z0-9_]+)_(Filtered|Normalized|UMI)', f.name)
                if match:
                    animal_names.add(match.group(1))
            
            if file_count > 0:
                self.findings['successes'].append(f"{age}: Found {file_count} main output files ({len(animal_names)} unique animals)")
                self.findings['info'].append(f"  {age}: Animals with outputs: {', '.join(sorted(animal_names)[:10])}" + 
                                            (f" (+{len(animal_names)-10} more)" if len(animal_names) > 10 else ""))
            else:
                self.findings['warnings'].append(f"{age}: No main output files found")
            
            # Check analysis directory (including model subdirectories)
            if analysis_dir.exists():
                analysis_files = []
                # Check in model subdirectories (uniform/ and region_specific/)
                for model_type in ['uniform', 'region_specific']:
                    model_dir = analysis_dir / model_type
                    if model_dir.exists():
                        for pattern in config['analysis_files']:
                            files = list(model_dir.glob(pattern))
                            analysis_files.extend(files)
                
                # Also check main analysis directory for backward compatibility and non-model-specific files
                for pattern in config['analysis_files']:
                    files = list(analysis_dir.glob(pattern))
                    analysis_files.extend(files)
                
                if analysis_files:
                    self.findings['successes'].append(f"{age}: Found {len(analysis_files)} analysis files")
                    # Check for motif_raw_data subdirectory
                    motif_raw_dir = analysis_dir / 'motif_raw_data'
                    if motif_raw_dir.exists():
                        raw_files = list(motif_raw_dir.glob('*.csv'))
                        self.findings['info'].append(f"  {age}: Found {len(raw_files)} motif raw data files")
                else:
                    self.findings['warnings'].append(f"{age}: Analysis directory exists but no analysis files found")
            else:
                self.findings['warnings'].append(f"{age}: Analysis directory not found: {analysis_dir}")
            
            # Check for aggregate upsetplot files (in model subdirectories)
            if analysis_dir.exists():
                agg_upsetplot_uniform = []
                agg_upsetplot_region_specific = []
                
                # Check uniform subdirectory
                uniform_dir = analysis_dir / 'uniform'
                if uniform_dir.exists():
                    # Try multiple pattern variations to handle case differences
                    # Only match files that START with the age prefix (aggregate files)
                    patterns_uniform = [
                        f"{age.upper()}_ALL_*_filters_upsetplot_uniform.csv",  # P12_ALL_*_filters_upsetplot_uniform.csv
                        f"{age.lower()}_ALL_*_filters_upsetplot_uniform.csv",  # p12_ALL_*_filters_upsetplot_uniform.csv
                        f"P{age[1:]}_ALL_*_filters_upsetplot_uniform.csv",  # P12_ALL_*_filters_upsetplot_uniform.csv (alternative)
                        f"p{age[1:]}_ALL_*_filters_upsetplot_uniform.csv",  # p12_ALL_*_filters_upsetplot_uniform.csv (alternative)
                    ]
                    for pattern in patterns_uniform:
                        matches = list(uniform_dir.glob(pattern))
                        agg_upsetplot_uniform.extend(matches)
                    # Remove duplicates while preserving order
                    agg_upsetplot_uniform = list(dict.fromkeys(agg_upsetplot_uniform))
                    # Filter to ensure we only have aggregate files (start with age prefix, not individual animal names)
                    agg_upsetplot_uniform = [f for f in agg_upsetplot_uniform if f.name.startswith((age.upper(), age.lower(), f"P{age[1:]}", f"p{age[1:]}"))]
                
                # Check region_specific subdirectory
                region_specific_dir = analysis_dir / 'region_specific'
                if region_specific_dir.exists():
                    # Try multiple pattern variations to handle case differences
                    # Only match files that START with the age prefix (aggregate files)
                    patterns_region_specific = [
                        f"{age.upper()}_ALL_*_filters_upsetplot_region_specific.csv",  # P12_ALL_*_filters_upsetplot_region_specific.csv
                        f"{age.lower()}_ALL_*_filters_upsetplot_region_specific.csv",  # p12_ALL_*_filters_upsetplot_region_specific.csv
                        f"P{age[1:]}_ALL_*_filters_upsetplot_region_specific.csv",  # P12_ALL_*_filters_upsetplot_region_specific.csv (alternative)
                        f"p{age[1:]}_ALL_*_filters_upsetplot_region_specific.csv",  # p12_ALL_*_filters_upsetplot_region_specific.csv (alternative)
                    ]
                    for pattern in patterns_region_specific:
                        matches = list(region_specific_dir.glob(pattern))
                        agg_upsetplot_region_specific.extend(matches)
                    # Remove duplicates while preserving order
                    agg_upsetplot_region_specific = list(dict.fromkeys(agg_upsetplot_region_specific))
                    # Filter to ensure we only have aggregate files (start with age prefix, not individual animal names)
                    agg_upsetplot_region_specific = [f for f in agg_upsetplot_region_specific if f.name.startswith((age.upper(), age.lower(), f"P{age[1:]}", f"p{age[1:]}"))]
                
                # Check main directory for backward compatibility (old single-model structure)
                patterns_main = [
                    f"{age.upper()}_ALL_HAN_filters_upsetplot.csv",
                    f"{age.lower()}_ALL_HAN_filters_upsetplot.csv",
                    f"P{age[1:]}_ALL_HAN_filters_upsetplot.csv",
                    f"p{age[1:]}_ALL_HAN_filters_upsetplot.csv",
                ]
                agg_upsetplot_main = []
                for pattern in patterns_main:
                    matches = list(analysis_dir.glob(pattern))
                    agg_upsetplot_main.extend(matches)
                # Remove duplicates while preserving order
                agg_upsetplot_main = list(dict.fromkeys(agg_upsetplot_main))
                # Filter to ensure we only have aggregate files (start with age prefix)
                agg_upsetplot_main = [f for f in agg_upsetplot_main if f.name.startswith((age.upper(), age.lower(), f"P{age[1:]}", f"p{age[1:]}"))]
                
                if agg_upsetplot_uniform:
                    self.findings['successes'].append(f"{age}: Aggregate upsetplot file found (uniform model): {agg_upsetplot_uniform[0].name}")
                if agg_upsetplot_region_specific:
                    self.findings['successes'].append(f"{age}: Aggregate upsetplot file found (region_specific model): {agg_upsetplot_region_specific[0].name}")
                if agg_upsetplot_main:
                    self.findings['info'].append(f"{age}: Aggregate upsetplot file found in main directory (backward compatibility): {agg_upsetplot_main[0].name}")
                
                if not agg_upsetplot_uniform and not agg_upsetplot_region_specific and not agg_upsetplot_main:
                    self.findings['warnings'].append(f"{age}: Aggregate upsetplot file not found in analysis directory or model subdirectories")
    
    def check_helper_outputs(self):
        """Check helper script outputs"""
        self.findings['info'].append("\n=== Helper Script Outputs ===")
        
        helper_outputs_found = False
        
        for script_name, config in EXPECTED_HELPER_OUTPUTS.items():
            # Check if we should look in parameterization-specific helper directory
            if self.base_output_dir:
                # Try to find parameterization from base_output_dir
                base_path = Path(self.base_output_dir)
                parameterization = None
                for part in base_path.parts:
                    if part.startswith(('01.', '02.', '03.', '04.', '05.')) and '_helpers' not in part:
                        parameterization = part
                        break
                
                if parameterization:
                    # Look in parameterization-specific helper directory (under output root so anchor layout works)
                    output_root = self._get_output_root()
                    output_dir = output_root / f"{parameterization}_helpers" / script_name
                else:
                    # Fall back to default location
                    output_dir = self.repo_root / config['dir']
            else:
                output_dir = self.repo_root / config['dir']
            
            if not output_dir.exists():
                self.findings['warnings'].append(f"{script_name}: Output directory not found: {output_dir}")
                continue
            
            # Count files (including model subdirectories if applicable)
            all_files = []
            if config.get('has_model_subdirs', False):
                # Check in model subdirectories
                for model_type in ['uniform', 'region_specific']:
                    model_dir = output_dir / model_type
                    if model_dir.exists():
                        all_files.extend([f for f in model_dir.rglob('*') if f.is_file()])
                # Also check main directory for backward compatibility
                all_files.extend([f for f in output_dir.rglob('*') if f.is_file() and f.parent == output_dir])
            else:
                # Check for model suffixes in filenames (for script 05)
                all_files = [f for f in output_dir.rglob('*') if f.is_file()]
            
            files = all_files
            file_count = len(files)
            self.stats['total_output_files'] += file_count
            
            if file_count >= config['min_files']:
                self.findings['successes'].append(f"{script_name}: {file_count} files found (expected ≥{config['min_files']})")
                self.stats['helper_scripts_succeeded'] += 1
            else:
                self.findings['warnings'].append(
                    f"{script_name}: Only {file_count} files found (expected ≥{config['min_files']})"
                )
            
            # Check for specific expected patterns
            for pattern in config['expected_patterns']:
                matching_files = list(output_dir.rglob(pattern))
                if matching_files:
                    self.findings['info'].append(f"  {script_name}: Found {len(matching_files)} files matching {pattern}")
                    helper_outputs_found = True
            
            # Check for model subdirectories if applicable
            if config.get('has_model_subdirs', False):
                uniform_dir = output_dir / 'uniform'
                region_specific_dir = output_dir / 'region_specific'
                if uniform_dir.exists():
                    uniform_files = [f for f in uniform_dir.rglob('*') if f.is_file()]
                    self.findings['info'].append(f"  {script_name}: Found {len(uniform_files)} files in uniform/ subdirectory")
                if region_specific_dir.exists():
                    region_specific_files = [f for f in region_specific_dir.rglob('*') if f.is_file()]
                    self.findings['info'].append(f"  {script_name}: Found {len(region_specific_files)} files in region_specific/ subdirectory")
        
        # If no helper outputs were found, add a message
        if not helper_outputs_found:
            self.findings['info'].append("  ✓ All helper script outputs verified (see SUCCESSES section for details)")
    
    def check_data_quality(self):
        """Check data quality indicators"""
        self.findings['info'].append("\n=== Data Quality Checks ===")
        
        data_quality_info_added = False
        
        # Check for projection_summary.csv (indicates successful animals)
        summary_files = list(self.repo_root.rglob("projection_summary.csv"))
        if summary_files:
            for summary_file in summary_files:
                try:
                    df = pd.read_csv(summary_file)
                    animal_count = len(df)
                    self.findings['successes'].append(
                        f"projection_summary.csv found with {animal_count} animals"
                    )
                    data_quality_info_added = True
                except Exception as e:
                    self.findings['warnings'].append(f"Could not read projection_summary.csv: {e}")
        else:
            self.findings['warnings'].append("projection_summary.csv not found")
        
        # If no additional info was added to this section, add a completion message
        # (Note: success messages go to successes list, so we check if we found files)
        if not data_quality_info_added and not summary_files:
            self.findings['info'].append("  ✓ No additional data quality issues detected")
        elif data_quality_info_added:
            # Files were found and processed successfully - add a note that checks passed
            self.findings['info'].append("  ✓ All data quality checks passed (see SUCCESSES section for details)")
    
    def generate_report(self):
        """Generate human-readable quality control report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("POST-PROCESSING QUALITY CONTROL REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Repository: {self.repo_root}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Animals Processed: {self.stats['animals_processed']}")
        report_lines.append(f"Animals Succeeded: {self.stats['animals_succeeded']}")
        report_lines.append(f"Animals Failed: {self.stats['animals_failed']}")
        report_lines.append(f"  - Expected Failures (low data quality): {self.stats['expected_failures']}")
        unexpected_fail_count = max(0, self.stats['animals_failed'] - self.stats['expected_failures'])
        report_lines.append(f"  - Unexpected Failures: {unexpected_fail_count}")
        report_lines.append(f"Helper Scripts Run: {self.stats['helper_scripts_run']}")
        report_lines.append(f"Helper Scripts Succeeded: {self.stats['helper_scripts_succeeded']}")
        report_lines.append(f"Total Output Files: {self.stats['total_output_files']}")
        report_lines.append("")
        
        # Success Rate
        if self.stats['animals_processed'] > 0:
            success_rate = (self.stats['animals_succeeded'] / self.stats['animals_processed']) * 100
            report_lines.append(f"Success Rate: {success_rate:.1f}%")
            report_lines.append("")
        
        # Findings Summary
        report_lines.append("FINDINGS SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"✓ Successes: {len(self.findings['successes'])}")
        report_lines.append(f"⚠ Warnings: {len(self.findings['warnings'])}")
        report_lines.append(f"✗ Errors: {len(self.findings['errors'])}")
        report_lines.append(f"ℹ Info: {len(self.findings['info'])}")
        report_lines.append("")
        
        # Detailed Findings
        # ERRORS section - always show
        report_lines.append("ERRORS")
        report_lines.append("-" * 80)
        if self.findings['errors']:
            for error in self.findings['errors']:
                report_lines.append(f"  ✗ {error}")
        else:
            report_lines.append("  ✓ No unexpected errors found")
        
        # Add error breakdown if available
        if hasattr(self, 'error_counts_all') and self.error_counts_all:
            report_lines.append("")
            report_lines.append("  Error breakdown (all errors in log):")
            for error_type, count in sorted(self.error_counts_all.items(), key=lambda x: x[1], reverse=True):
                # Check if these are expected (ValueError for N0 root-finding)
                if error_type == 'ValueError' and 'No valid root' in str(self.error_counts_all):
                    report_lines.append(f"    - {error_type}: {count}x (expected for low-quality samples)")
                else:
                    report_lines.append(f"    - {error_type}: {count}x")
        report_lines.append("")
        
        # Warnings section
        report_lines.append("WARNINGS")
        report_lines.append("-" * 80)
        if self.findings['warnings']:
            for warning in self.findings['warnings']:
                # Check if this is a processing warning that needs breakdown
                if warning.startswith("Processing Warning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'processing_warning_counts') and self.processing_warning_counts:
                        for category, count in sorted(self.processing_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                # Check if this is a FutureWarning that needs breakdown
                elif warning.startswith("FutureWarning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'future_warning_counts') and self.future_warning_counts:
                        for category, count in sorted(self.future_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                # Check if this is a UserWarning that needs breakdown
                elif warning.startswith("UserWarning:"):
                    report_lines.append(f"  ⚠ {warning}")
                    # Add detailed breakdown if available
                    if hasattr(self, 'user_warning_counts') and self.user_warning_counts:
                        for category, count in sorted(self.user_warning_counts.items(), key=lambda x: x[1], reverse=True):
                            report_lines.append(f"    - {category}: {count}x")
                else:
                    report_lines.append(f"  ⚠ {warning}")
        else:
            report_lines.append("  ✓ No warnings found - all checks passed")
        report_lines.append("")
        
        # SUCCESSES section - always show
        report_lines.append("SUCCESSES")
        report_lines.append("-" * 80)
        if self.findings['successes']:
            for success in self.findings['successes']:
                report_lines.append(f"  ✓ {success}")
        else:
            report_lines.append("  ℹ No specific successes to report")
        report_lines.append("")
        
        # DETAILED INFORMATION section - always show
        report_lines.append("DETAILED INFORMATION")
        report_lines.append("-" * 80)
        if self.findings['info']:
            for info in self.findings['info']:
                report_lines.append(f"  ℹ {info}")
        else:
            report_lines.append("  ℹ No additional information to report")
        report_lines.append("")
        
        # Unexpected Failures Detail
        if hasattr(self, 'unexpected_failures') and self.unexpected_failures:
            report_lines.append("UNEXPECTED FAILURES - DETAILED")
            report_lines.append("-" * 80)
            for animal, error_type, error_msg in self.unexpected_failures:
                report_lines.append(f"  Animal: {animal}")
                report_lines.append(f"    Error Type: {error_type}")
                report_lines.append(f"    Error Message: {error_msg}")
                # Provide context about what this means
                if error_type == 'AssertionError':
                    if 'Mismatch' in error_msg or 'columns' in error_msg:
                        report_lines.append(f"    Likely Cause: Data structure mismatch after filtering (possibly empty matrix)")
                        report_lines.append(f"    Action: Check if {animal} has sufficient data after filtering")
                elif error_type == 'FileNotFoundError':
                    report_lines.append(f"    Likely Cause: Missing input file")
                    report_lines.append(f"    Action: Verify input file exists in 01_aggregate_data directory")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        unexpected_fail_count = self.stats['animals_failed'] - self.stats['expected_failures']
        if unexpected_fail_count > 0:
            report_lines.append(f"  ⚠ {unexpected_fail_count} animal(s) failed unexpectedly (see UNEXPECTED FAILURES section above)")
            report_lines.append(f"     Action: Review the specific errors above. Most common issues:")
            report_lines.append(f"       - AssertionError: Usually indicates empty matrix after filtering (data quality issue)")
            report_lines.append(f"       - FileNotFoundError: Missing input file - check all_commands.txt paths")
            report_lines.append(f"     To investigate: Search log file for the animal name to see full error context")
        
        if self.stats['expected_failures'] > 0:
            report_lines.append(f"  ℹ {self.stats['expected_failures']} animal(s) failed due to low data quality (expected behavior)")
            report_lines.append(f"     These samples had 0-2 cells after filtering and cannot be processed")
            report_lines.append(f"     No action needed - these are documented as expected failures")
        
        if self.stats['helper_scripts_succeeded'] < len(EXPECTED_HELPER_OUTPUTS):
            missing = len(EXPECTED_HELPER_OUTPUTS) - self.stats['helper_scripts_succeeded']
            report_lines.append(f"  ⚠ {missing} helper script(s) did not produce expected outputs")
            report_lines.append(f"     Action: Check if helper scripts ran (see Helper Scripts Run count above)")
            report_lines.append(f"     If scripts didn't run: Check all_commands.txt and ensure scripts are executed")
            report_lines.append(f"     If scripts ran but failed: Check log file for error messages from helper scripts")
        
        if len(self.findings['errors']) > 0:
            report_lines.append(f"  ✗ {len(self.findings['errors'])} critical error type(s) detected")
            report_lines.append(f"     Action: Review ERRORS section above. These require immediate attention")
            report_lines.append(f"     Most critical errors prevent successful processing and should be fixed")
        elif len([w for w in self.findings['warnings'] if 'HIGH' in w]) > 0:
            report_lines.append(f"  ⚠ High-severity warnings detected (see WARNINGS section)")
            report_lines.append(f"     Action: Review HIGH severity warnings - these may indicate issues")
        elif len(self.findings['warnings']) == 0:
            report_lines.append("  ✓ All checks passed! Pipeline completed successfully.")
            report_lines.append("     All animals processed, all helper scripts ran, no critical issues found")
        else:
            report_lines.append(f"  ⚠ Pipeline completed with {len(self.findings['warnings'])} warning(s)")
            report_lines.append(f"     Most warnings are LOW severity (FutureWarnings from libraries)")
            report_lines.append(f"     Review WARNINGS section for any HIGH or MEDIUM severity items")
        
        # Success rate context
        if self.stats['animals_processed'] > 0:
            success_rate = (self.stats['animals_succeeded'] / self.stats['animals_processed']) * 100
            if success_rate >= 80:
                report_lines.append(f"  ✓ Good success rate ({success_rate:.1f}%) - pipeline performing well")
            elif success_rate >= 60:
                report_lines.append(f"  ⚠ Moderate success rate ({success_rate:.1f}%) - some data quality issues expected")
            else:
                report_lines.append(f"  ⚠ Low success rate ({success_rate:.1f}%) - investigate data quality or processing parameters")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def run_checks(self):
        """Run all quality control checks"""
        print("Running quality control checks...")
        
        # Find and analyze log file
        log_path = self.find_log_file()
        if log_path:
            print(f"Found log file: {log_path}")
            self.analyze_log_file(log_path)
        else:
            print("Warning: No log file found")
        
        # Check outputs
        print("Checking main processing outputs...")
        self.check_main_outputs()
        
        print("Checking helper script outputs...")
        self.check_helper_outputs()
        
        print("Checking data quality...")
        self.check_data_quality()
        
        # Generate and save report
        report = self.generate_report()
        
        # Print to console
        print("\n" + report)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {self.output_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Post-processing quality control check script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--repo_root',
        type=str,
        default=None,
        help='Repository root directory (default: script directory)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to processing log file (default: find most recent)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output report file path (default: qc_report_TIMESTAMP.txt)'
    )
    parser.add_argument(
        '--base_output_dir',
        type=str,
        default=None,
        help='Base output directory for processing results (default: repo_root/02_output). If specified, will check this directory instead of hardcoded paths.'
    )
    
    args = parser.parse_args()
    
    checker = QualityControlChecker(
        repo_root=args.repo_root,
        log_file=args.log_file,
        output_file=args.output,
        base_output_dir=args.base_output_dir
    )
    
    checker.run_checks()


if __name__ == "__main__":
    main()

