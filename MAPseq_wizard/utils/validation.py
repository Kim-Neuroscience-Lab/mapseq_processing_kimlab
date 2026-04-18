"""
Input validation utilities for the MAPseq Pipeline Wizard
"""

import os
from pathlib import Path


def validate_directory(path_str, must_exist=True):
    """Validate that a path is a directory"""
    if not path_str:
        return False, "Path is empty"
    
    path = Path(path_str)
    
    if must_exist:
        if not path.exists():
            return False, f"Directory does not exist: {path}"
        if not path.is_dir():
            return False, f"Path is not a directory: {path}"
    
    return True, None


def validate_file(path_str, must_exist=True, extensions=None):
    """Validate that a path is a file, optionally with specific extensions"""
    if not path_str:
        return False, "Path is empty"
    
    path = Path(path_str)
    
    if must_exist:
        if not path.exists():
            return False, f"File does not exist: {path}"
        if not path.is_file():
            return False, f"Path is not a file: {path}"
    
    if extensions:
        if path.suffix.lower() not in [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]:
            return False, f"File must have one of these extensions: {extensions}"
    
    return True, None


def validate_sample_name(name):
    """Validate a sample name"""
    if not name:
        return False, "Sample name cannot be empty"
    
    if len(name) > 100:
        return False, "Sample name is too long (max 100 characters)"
    
    # Check for invalid characters (basic validation)
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in name:
            return False, f"Sample name contains invalid character: {char}"
    
    return True, None


def validate_labels(labels_str):
    """Validate labels string"""
    if not labels_str:
        return False, "Labels cannot be empty"
    
    labels = [l.strip() for l in labels_str.split(',')]
    
    if not labels:
        return False, "At least one label is required"
    
    # Check for required labels
    has_neg = any('neg' in label.lower() for label in labels)
    has_inj = any('inj' in label.lower() for label in labels)
    
    if not has_inj:
        return False, "Labels must include 'inj' (injection site)"
    
    return True, None


def validate_parameterization_name(name):
    """Validate a parameterization name"""
    if not name:
        return False, "Parameterization name cannot be empty"
    
    # Should start with number and dot (e.g., "01.minimal_filter_parameters...")
    if not name[0].isdigit():
        return False, "Parameterization name should start with a number (e.g., '01.')"
    
    return True, None


def validate_numeric(value, min_val=None, max_val=None, allow_float=True):
    """Validate a numeric value"""
    if value is None or value == "":
        return False, "Value cannot be empty"
    
    try:
        if allow_float:
            num_value = float(value)
        else:
            num_value = int(value)
    except ValueError:
        return False, "Value must be a number"
    
    if min_val is not None and num_value < min_val:
        return False, f"Value must be >= {min_val}"
    
    if max_val is not None and num_value > max_val:
        return False, f"Value must be <= {max_val}"
    
    return True, None


