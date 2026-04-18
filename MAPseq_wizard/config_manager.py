"""
Configuration management for MAPseq Pipeline Wizard
Handles loading, saving, and validation of YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from .utils.path_utils import normalize_path, get_repo_root


class ConfigManager:
    """Manages YAML configuration files for the pipeline wizard"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ConfigManager
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure"""
        repo_root = get_repo_root()
        return {
            'project': {
                'name': 'My MAPseq Analysis',
                'base_output_dir': str(repo_root / '02_output'),
                'repo_root': str(repo_root),
            },
            'preprocessing': {
                'input_dir': '',  # Legacy: single preprocessing
                'output_dir': '',  # Legacy: single preprocessing
                'fallback_threshold': 2.0,
                'column_mappings': {},  # Legacy format
                'negative_columns': {},  # Legacy format
                'cohorts': {},  # New format: {cohort_name: {input_dir, output_dir, ...}}
            },
            'main_processing': {
                'parameterizations': [],
                'age_groups': {
                    'p3': {'samples': []},
                    'p12': {'samples': []},
                    'p20': {'samples': []},
                    'p60': {'samples': []},
                },
            },
            'helper_scripts': {
                'enabled': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '13'],
                'base_output_dir': str(repo_root / '02_output'),
            },
            'quality_control': {
                'enabled': True,
                'base_output_dir': str(repo_root / '02_output'),
            },
            'figure_generation': {
                'enabled': True,
                'parameterization': '05.HAN_filter_parameters_i300_r10_t10_u5',
                'output_dir': str(repo_root / 'figure_generation' / 'generated_figures'),
            },
        }
    
    def load(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file (uses self.config_path if not provided)
            
        Returns:
            Configuration dictionary
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            # Return default config if file doesn't exist
            self.config = self._default_config.copy()
            return self.config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
            
            # Merge with defaults to ensure all keys exist
            self.config = self._merge_configs(self._default_config, loaded_config)
            return self.config
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config_path: Path to save config (uses self.config_path if not provided)
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("No config path specified")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize paths in config before saving
        normalized_config = self._normalize_paths(self.config.copy())
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(normalized_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'project.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'project.name')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def add_parameterization(self, name: str, injection_umi_min: float, min_body_to_target_ratio: float,
                            min_target_count: float, target_umi_min: float, force_user_threshold: bool = False) -> None:
        """Add a new parameterization"""
        param = {
            'name': name,
            'injection_umi_min': injection_umi_min,
            'min_body_to_target_ratio': min_body_to_target_ratio,
            'min_target_count': min_target_count,
            'target_umi_min': target_umi_min,
            'force_user_threshold': force_user_threshold,
        }
        
        params = self.config['main_processing']['parameterizations']
        # Check if parameterization with this name already exists
        for i, p in enumerate(params):
            if p['name'] == name:
                params[i] = param
                return
        
        params.append(param)
    
    def add_sample(self, age_group: str, sample_name: str, data_file: str, labels: str) -> None:
        """Add a sample to an age group"""
        age_groups = self.config['main_processing']['age_groups']
        
        if age_group not in age_groups:
            age_groups[age_group] = {'samples': []}
        
        sample = {
            'name': sample_name,
            'data_file': normalize_path(data_file),
            'labels': labels,
        }
        
        samples = age_groups[age_group]['samples']
        # Check if sample with this name already exists in this age group
        for i, s in enumerate(samples):
            if s['name'] == sample_name:
                samples[i] = sample
                return
        
        samples.append(sample)
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _normalize_paths(self, config: Dict) -> Dict:
        """Normalize all path strings in configuration"""
        if isinstance(config, dict):
            return {k: self._normalize_paths(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._normalize_paths(item) for item in config]
        elif isinstance(config, str) and ('/' in config or '\\' in config):
            # Try to normalize if it looks like a path
            try:
                return normalize_path(config) or config
            except:
                return config
        else:
            return config
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate configuration
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate project settings
        if not self.config.get('project', {}).get('name'):
            errors.append("Project name is required")
        
        # Validate preprocessing
        preprocess = self.config.get('preprocessing', {})
        if preprocess.get('input_dir') and not Path(preprocess['input_dir']).exists():
            errors.append(f"Preprocessing input directory does not exist: {preprocess['input_dir']}")
        
        # Validate main processing
        main_proc = self.config.get('main_processing', {})
        if not main_proc.get('parameterizations'):
            errors.append("At least one parameterization is required")
        
        # Validate samples have required fields
        age_groups = main_proc.get('age_groups', {})
        for age_group, data in age_groups.items():
            for sample in data.get('samples', []):
                if not sample.get('name'):
                    errors.append(f"Sample in {age_group} is missing name")
                if not sample.get('data_file'):
                    errors.append(f"Sample {sample.get('name', 'unknown')} in {age_group} is missing data_file")
                if not sample.get('labels'):
                    errors.append(f"Sample {sample.get('name', 'unknown')} in {age_group} is missing labels")
        
        return len(errors) == 0, errors


