import PySimpleGUI as sg
import subprocess
import os
import sys
import platform
import shutil

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

def auto_update_repo(repo_path, branch="main"):
    """Auto-update repository if git is available"""
    try:
        # Check if git is available
        if not shutil.which("git"):
            return  # Silently skip if git not available
        
        print(f"🔄 Checking for updates in {repo_path}...")
        subprocess.run(
            ["git", "fetch", "--all"], 
            cwd=repo_path, 
            check=True, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"], 
            cwd=repo_path, 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Repository successfully updated to the latest version.")
    except Exception as e:
        print(f"⚠️ Auto-update failed: {e}")

# Detect the repo path (assumes this script is inside the repo)
if getattr(sys, 'frozen', False):
    # We're in a PyInstaller exe
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

repo_path = base_path  # adjust if the repo is in a subfolder

auto_update_repo(repo_path)


layout = [
    [sg.Text("Sample Naming Prefix:"), sg.Input(key="sample_name")],
    [sg.Text("Select your nbcm.tsv (individual or aggregated):"), sg.Input(key="data_file"), sg.FileBrowse(file_types=(("CSV/TSV Files", "*.csv *.tsv"),))],
    [sg.Text("Output Directory (will not prompt on overwrite):"), sg.Input(key="out_dir"), sg.FolderBrowse()],
    [sg.Text("Alpha (see readme):"), sg.Input(key="alpha", default_text="0.05")],
    [sg.Text("Labels (see sample data for example):"), sg.Input(key="labels")],
    [sg.Text("Filter: Minimum Injection site UMI"), sg.Input(key="injection_umi_min", default_text="1")],
    [sg.Text("Filter: At least one target UMI > X"), sg.Input(key="min_target_count", default_text="10")],
    [sg.Text("Filter: Min Injection-to-Target Ratio:"), sg.Input(key="min_body_to_target_ratio", default_text="10")],
    [sg.Text("Filter: Noise. Zero any matrix value less than X"), sg.Input(key="target_umi_min", default_text="2")],
    [sg.Checkbox("Experimental: Remove high-UMI outliers where value was > (mean+2*StdDev)", key="apply_outlier_filtering")],
    [sg.Checkbox("Force user-defined threshold (override automatic thresholding)", key="force_user_threshold")],
    [sg.Button("Run"), sg.Exit()]
]

window = sg.Window("NBCM Processing GUI Wizard", layout)

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == "Run":
        # Validate required fields
        if not values["sample_name"] or not values["data_file"] or not values["out_dir"] or not values["labels"]:
            sg.popup_error("Please fill in all required fields: Sample Name, Data File, Output Directory, and Labels.")
            continue
        
        # Normalize paths for cross-platform compatibility
        data_file = os.path.normpath(values["data_file"])
        out_dir = os.path.normpath(values["out_dir"])
        
        # Verify files exist
        if not os.path.exists(data_file):
            sg.popup_error(f"Data file not found: {data_file}")
            continue
        
        # Find conda executable
        conda_exe = find_conda_executable()
        
        # Get absolute path to process script
        process_script = os.path.join(repo_path, "process-nbcm-tsv.py")
        if not os.path.exists(process_script):
            sg.popup_error(f"Processing script not found: {process_script}")
            continue
        
        # Build command - use list format (more reliable cross-platform)
        cmd = [
            conda_exe, "run", "-n", "mapseq_processing", "python", process_script,
            "--sample_name", values["sample_name"],
            "--data_file", data_file,
            "--out_dir", out_dir,
            "--alpha", values["alpha"],
            "--labels", values["labels"],
            "--injection_umi_min", values["injection_umi_min"],
            "--min_target_count", values["min_target_count"],
            "--min_body_to_target_ratio", values["min_body_to_target_ratio"],
            "--target_umi_min", values["target_umi_min"]
        ]

        if values["apply_outlier_filtering"]:
            cmd += ["--apply_outlier_filtering"]
        
        if values["force_user_threshold"]:
            cmd += ["--force_user_threshold"]

        print("🔧 Running:", " ".join(cmd))
        
        # Run without shell=True for better cross-platform compatibility
        try:
            result = subprocess.run(
                cmd, 
                shell=False,  # More reliable cross-platform
                cwd=repo_path,  # Set working directory
                check=False  # Don't raise on error, we'll check return code
            )
            
            if result.returncode == 0:
                sg.popup("✅ Processing completed successfully!", title="Success")
            else:
                sg.popup_error(f"❌ Processing failed with exit code {result.returncode}.\nCheck the console for details.", title="Error")
        except FileNotFoundError:
            sg.popup_error(f"Conda not found. Please ensure conda is installed and in your PATH.\nTried: {conda_exe}", title="Conda Not Found")
        except Exception as e:
            sg.popup_error(f"Error running command: {e}", title="Error")

window.close()
