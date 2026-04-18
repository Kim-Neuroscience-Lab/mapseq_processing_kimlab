"""
Cross-platform path utilities for the MAPseq Pipeline Wizard
"""

import os
import platform
import shutil
from pathlib import Path


def find_conda_executable():
    """Find conda executable in PATH or common locations"""
    # First try to find in PATH
    conda_cmd = shutil.which("conda")
    if conda_cmd:
        return conda_cmd
    
    # Platform-specific fallbacks
    system = platform.system()
    if system == "Windows":
        # Common Windows conda locations
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "Miniconda3", "Scripts", "conda.exe"),
            os.path.join(os.path.expanduser("~"), "Anaconda3", "Scripts", "conda.exe"),
            os.path.join("C:", "ProgramData", "Anaconda3", "Scripts", "conda.exe"),
            os.path.join("C:", "ProgramData", "Miniconda3", "Scripts", "conda.exe"),
        ]
    else:
        # Unix-like systems
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "miniconda3", "bin", "conda"),
            os.path.join(os.path.expanduser("~"), "anaconda3", "bin", "conda"),
            os.path.join(os.path.expanduser("~"), ".conda", "bin", "conda"),
        ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "conda"  # Fallback to assuming it's in PATH


def normalize_path(path_str):
    """Normalize a path string for cross-platform compatibility"""
    if not path_str:
        return None
    return str(Path(path_str).resolve())


def ensure_dir(path):
    """Ensure a directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def get_repo_root():
    """Get the repository root directory"""
    # Try to find .git directory
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # Fallback to parent of MAPseq_wizard
    return Path(__file__).resolve().parent.parent.parent


def relative_to_repo(path):
    """Convert absolute path to relative path from repo root"""
    repo_root = get_repo_root()
    try:
        return Path(path).relative_to(repo_root)
    except ValueError:
        return Path(path)


def absolute_from_repo(relative_path):
    """Convert relative path to absolute path from repo root"""
    repo_root = get_repo_root()
    return (repo_root / relative_path).resolve()


