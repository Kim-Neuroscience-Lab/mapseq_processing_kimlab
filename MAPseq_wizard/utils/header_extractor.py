"""
Extract column headers from TSV files for preprocessing configuration
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def extract_headers_from_file(file_path: Path) -> Optional[List[str]]:
    """
    Extract column headers from a single TSV file
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        List of column names, or None if error
    """
    try:
        # Read just the header row
        df = pd.read_csv(file_path, sep='\t', header=0, nrows=0)
        return list(df.columns)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_headers_from_directory(directory: Path) -> Dict[str, List[str]]:
    """
    Extract headers from all TSV files in a directory
    
    Args:
        directory: Directory containing TSV files
        
    Returns:
        Dictionary mapping filename to list of column names
        Format: {filename: [col1, col2, ...]}
    """
    headers = {}
    
    if not directory.exists() or not directory.is_dir():
        return headers
    
    # Find all .tsv files
    tsv_files = list(directory.glob("*.tsv"))
    
    for tsv_file in tsv_files:
        filename = tsv_file.name
        columns = extract_headers_from_file(tsv_file)
        if columns:
            headers[filename] = columns
    
    return headers


def validate_tsv_file(file_path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate that a file is a readable TSV file
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    if file_path.suffix.lower() != '.tsv':
        return False, f"File is not a TSV file: {file_path}"
    
    try:
        # Try to read first row
        df = pd.read_csv(file_path, sep='\t', header=0, nrows=1)
        if df.empty:
            return False, "File appears to be empty"
        return True, None
    except Exception as e:
        return False, f"Error reading TSV file: {e}"


