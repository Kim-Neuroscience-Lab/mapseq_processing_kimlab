import json
import os
import platform
import re
import shutil
import subprocess
import sys

import requests

ENV_NAME = "mapseq_processing"
GUI_VERSION = "v0.2.0-beta"

DEFAULT_CLONE_REPOSITORY_URL = "https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab.git"


def get_clone_repository_url():
    """URL for `git clone` during setup.

    Defaults to the public lab repo so running the wizard from a fork does not clone
    the fork. Override with MAPSEQ_WIZARD_CLONE_URL (non-empty).
    """
    override = os.environ.get("MAPSEQ_WIZARD_CLONE_URL", "").strip()
    return override if override else DEFAULT_CLONE_REPOSITORY_URL


def get_conda_path(install_path):
    """Get platform-specific conda executable path"""
    if platform.system() == "Windows":
        return os.path.join(install_path, "Scripts", "conda.exe")
    else:
        return os.path.join(install_path, "bin", "conda")

def get_default_install_path():
    """Get platform-specific default Miniconda install path"""
    system = platform.system()
    if system == "Windows":
        return os.path.expanduser("~\\Miniconda3")
    elif system == "Darwin":  # macOS
        return os.path.expanduser("~/miniconda3")
    else:  # Linux
        return os.path.expanduser("~/miniconda3")

def get_miniconda_installer_info():
    """Get platform-specific Miniconda installer URL and filename"""
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == "Windows":
        return (
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe",
            "Miniconda3.exe"
        )
    elif system == "Darwin":  # macOS
        if machine == "arm64" or machine == "aarch64":
            return (
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh",
                "Miniconda3-latest-MacOSX-arm64.sh"
            )
        else:
            return (
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh",
                "Miniconda3-latest-MacOSX-x86_64.sh"
            )
    else:  # Linux
        return (
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            "Miniconda3-latest-Linux-x86_64.sh"
        )

def is_git_repo(path):
    """Check if a directory is a git repository"""
    return os.path.exists(os.path.join(path, ".git"))

def _ensure_macos_git_on_path():
    """If /usr/bin/git exists (Xcode CLT) but PATH does not resolve `git`, prepend /usr/bin."""
    if platform.system() != "Darwin":
        return
    git_bin = "/usr/bin/git"
    if not (os.path.isfile(git_bin) and os.access(git_bin, os.X_OK)):
        return
    if shutil.which("git"):
        return
    os.environ["PATH"] = "/usr/bin" + os.pathsep + os.environ.get("PATH", "")


def check_git_installed():
    """Check if git is available in PATH"""
    _ensure_macos_git_on_path()
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _prepend_windows_git_to_path():
    """If Git for Windows is installed but not on PATH, prepend Git\\cmd."""
    if platform.system() != "Windows":
        return
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pfx = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for d in (
        os.path.join(pf, "Git", "cmd"),
        os.path.join(pfx, "Git", "cmd"),
    ):
        git_exe = os.path.join(d, "git.exe")
        if os.path.isfile(git_exe):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return


def auto_install_git():
    """Try to install Git using the OS package manager or GUI installer. Returns True if git works after."""
    if check_git_installed():
        return True

    system = platform.system()
    print("\n📥 Git was not found in PATH. Attempting automatic installation...")

    if system == "Darwin":
        # Homebrew requires git to run; "brew install git" fails when git is missing and can error on
        # broken legacy installs. Prefer Apple Command Line Tools (ships /usr/bin/git).
        try:
            print("   Launching Apple Command Line Tools installer (includes git)...")
            subprocess.run(["xcode-select", "--install"], check=False)
        except FileNotFoundError:
            pass
        print("   Approve the dialog if shown, wait for installation to finish, then continue here.")
        input("   Press Enter when Command Line Tools installation has finished (or if already installed)...")
        if check_git_installed():
            print("✅ Git is now available.")
            return True
        print(
            "⚠️  Git still not detected. Open a **new** Terminal window and run this wizard again, or install manually:\n"
            "   https://git-scm.com/download/mac\n"
            "   (Homebrew needs git before it can install packages—install Command Line Tools or git first.)"
        )
        return False

    if system == "Windows":
        winget = shutil.which("winget")
        if winget:
            try:
                print("   Running: winget install Git.Git")
                subprocess.run(
                    [
                        winget,
                        "install",
                        "--id",
                        "Git.Git",
                        "-e",
                        "--source",
                        "winget",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ],
                    check=False,
                    timeout=600,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        _prepend_windows_git_to_path()
        if check_git_installed():
            print("✅ Git is now available.")
            return True
        print(
            "⚠️  Automatic install did not make `git` available in this session.\n"
            "   Install from https://gitforwindows.org/ , restart the terminal, and run this wizard again."
        )
        return False

    if system == "Linux":
        if shutil.which("apt-get"):
            try:
                print("   Running: sudo apt-get install -y git (enter your password if prompted)")
                subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False)
                r = subprocess.run(["sudo", "apt-get", "install", "-y", "git"])
                if r.returncode == 0 and check_git_installed():
                    print("✅ Git is now available.")
                    return True
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        if shutil.which("dnf"):
            try:
                print("   Running: sudo dnf install -y git")
                r = subprocess.run(["sudo", "dnf", "install", "-y", "git"])
                if r.returncode == 0 and check_git_installed():
                    print("✅ Git is now available.")
                    return True
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        if shutil.which("yum"):
            try:
                print("   Running: sudo yum install -y git")
                r = subprocess.run(["sudo", "yum", "install", "-y", "git"])
                if r.returncode == 0 and check_git_installed():
                    print("✅ Git is now available.")
                    return True
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        if shutil.which("pacman"):
            try:
                print("   Running: sudo pacman -S --noconfirm git")
                r = subprocess.run(["sudo", "pacman", "-S", "--noconfirm", "git"])
                if r.returncode == 0 and check_git_installed():
                    print("✅ Git is now available.")
                    return True
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
        print("⚠️  Could not install git automatically. Install with your distribution's package manager.")
        return False

    print("⚠️  Automatic git installation is not implemented for this OS.")
    return False


def prompt_git_installation():
    """Prompt user to install git with platform-specific instructions"""
    system = platform.system()
    if system == "Windows":
        print("\n⚠️  Git is not installed. Please install Git for Windows:")
        print("   Download from: https://gitforwindows.org/")
        print("   After installation, restart this setup wizard.")
    elif system == "Darwin":  # macOS
        print("\n⚠️  Git is not installed. Install using one of these methods:")
        print("   Option 1: Homebrew: brew install git")
        print("   Option 2: Download Xcode Command Line Tools: xcode-select --install")
    else:  # Linux
        print("\n⚠️  Git is not installed. Install using your package manager:")
        print("   Ubuntu/Debian: sudo apt-get install git")
        print("   Fedora/RHEL: sudo yum install git")
        print("   Arch: sudo pacman -S git")
    
    response = input("\nContinue setup anyway? (You can clone the repo manually later) [y/N]: ")
    return response.lower() == 'y'

def get_git_remote_url(repo_path=None):
    """Resolve ``origin`` for the repo containing this script (e.g. GUI release URL via get_gui_exe_url).

    Not used for install-time ``git clone``; use get_clone_repository_url() for that.
    """
    if repo_path is None:
        repo_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DEFAULT_CLONE_REPOSITORY_URL

def get_gui_exe_url(git_url=None, version=GUI_VERSION):
    """Construct GUI exe download URL from git remote URL"""
    if git_url is None:
        git_url = get_git_remote_url()
    
    # Convert git URL to GitHub releases URL format
    # Handle both https://github.com/user/repo.git and git@github.com:user/repo.git
    if "github.com" in git_url:
        # Extract user/repo from URL
        match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', git_url)
        if match:
            user, repo = match.groups()
            return f"https://github.com/{user}/{repo}/releases/download/{version}/MAPseq_Wizard.exe"
    
    # Return None if we can't construct the URL (e.g., not GitHub or private repo)
    return None

def get_repo_path():
    """Get repository path - either current directory if in repo, or None"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if is_git_repo(script_dir):
        return script_dir
    return None

def prompt_install_path(default_path):
    print(f"\n📁 Default Miniconda install location: {default_path}")
    custom_path = input("Enter custom install path (or press Enter to use default): ").strip()
    return custom_path if custom_path else default_path

def install_miniconda(install_path):
    """Install Miniconda with platform-specific handling"""
    url, installer = get_miniconda_installer_info()
    system = platform.system()

    print("🔍 Downloading Miniconda...")
    try:
        if system == "Windows":
            # Windows: use curl if available, otherwise requests
            try:
                subprocess.run(["curl", "-L", "-o", installer, url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to requests for Windows
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(installer, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            # macOS/Linux: use curl or wget
            try:
                subprocess.run(["curl", "-L", "-o", installer, url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["wget", "-O", installer, url], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to requests
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(installer, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download Miniconda installer: {e}. Please check your internet connection.")

    print(f"🔧 Installing Miniconda to: {install_path}")
    
    if system == "Windows":
        # Windows silent installation
        subprocess.run([
            installer,
            "/InstallationType=JustMe",
            "/RegisterPython=0",
            "/AddToPath=1",
            "/S",
            f"/D={install_path}"
        ], check=True)
    else:
        # macOS/Linux: make executable and run
        os.chmod(installer, 0o755)
        subprocess.run([
            "bash", installer, "-b", "-p", install_path, "-f"
        ], check=True)
        # Clean up installer
        try:
            os.remove(installer)
        except OSError:
            pass

def conda_env_exists(conda_exe, env_name):
    """Return True if a named conda environment already exists."""
    try:
        r = subprocess.run(
            [conda_exe, "info", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(r.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return False
    for p in data.get("envs", []):
        if os.path.basename(os.path.normpath(p)) == env_name:
            return True
    return False


def conda_run(conda_exe, args):
    """Run conda; on failure print stdout/stderr (conda often prints errors to stdout)."""
    r = subprocess.run([conda_exe] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("\n--- conda output (for debugging) ---")
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        print("--- end conda output ---\n")
    r.check_returncode()
    return r

def download_gui_exe(url, target_path):
    """Download GUI exe with enhanced error handling"""
    print(f"⬇️  Downloading GUI .exe from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ GUI exe saved to: {target_path}")
    except requests.Timeout:
        raise RuntimeError(f"Download timeout. Please check your internet connection and try again.")
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download GUI exe: {e}. You can build it manually using PyInstaller if needed.")

def verify_installation(conda_exe, env_name):
    """Verify that critical packages can be imported"""
    print("\n🔍 Verifying installation...")
    test_imports = [
        "pandas", "numpy", "matplotlib", "scipy", 
        "sklearn", "PySimpleGUI", "seaborn", "statsmodels"
    ]
    
    failed_imports = []
    for package in test_imports:
        try:
            # Map package names to import names
            import_name = package
            if package == "sklearn":
                import_name = "sklearn"
            elif package == "PySimpleGUI":
                import_name = "PySimpleGUI"
            
            result = subprocess.run(
                [conda_exe, "run", "-n", env_name, "python", "-c", f"import {import_name}"],
                capture_output=True,
                check=True
            )
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            failed_imports.append(package)
            print(f"  ❌ {package} (failed to import)")
    
    if failed_imports:
        print(f"\n⚠️  Warning: Some packages failed to import: {', '.join(failed_imports)}")
        print("   You may need to reinstall dependencies manually.")
        return False
    else:
        print("\n✅ All critical packages verified successfully!")
        return True

def print_post_installation_instructions(repo_path):
    """Print comprehensive post-installation instructions"""
    print("\n" + "="*70)
    print("📚 POST-INSTALLATION INSTRUCTIONS")
    print("="*70)
    
    print("\n1️⃣  PREPROCESSING (if needed):")
    print("   Run the preprocessing script to clean and aggregate your data:")
    print("   conda activate mapseq_processing")
    print(f"   python {os.path.join(repo_path, 'preprocess_and_aggregate.py')} -i <input_dir> -o <output_dir>")
    
    print("\n2️⃣  MAIN PROCESSING:")
    if platform.system() == "Windows":
        print("   Option A - GUI (Windows):")
        print(f"   Run MAPseq_Wizard.exe from: {repo_path}")
        print("   Option B - Command Line:")
    else:
        print("   Command line:")
    print("   conda activate mapseq_processing")
    print(f"   python {os.path.join(repo_path, 'process-nbcm-tsv.py')} -o <out_dir> -s <sample> -d <data_file> -l <labels>")
    
    print("\n3️⃣  HELPER SCRIPTS (run in order after main processing):")
    print("   conda activate mapseq_processing")
    helper_scripts = [
        "01_motif_analysis_per_animal.py",
        "02_projection_analysis.py",
        "03_composition.py",
        "04_proportions_over_time_stats.py",
        "05_motif_analysis.py",
        "18_mean_jsd_transition_tests.py",
        "06_all_motif_divergence.py",
        "17_jsd_cross_source_summary.py",
        "07_motif_significange_trajectories.py",
        "08_motif_clustering.py",
        "09_plot_normalized_projection_strength_data.py",
        "10_plot_per_cell_projection_strength_across_ages.py",
        "13_aggregate_projection_summaries.py"
    ]
    for script in helper_scripts:
        print(f"   python {os.path.join(repo_path, 'helpers', 'scripts', script)}")
    
    print("\n   Note: Scripts 05 must run before 06 and 07.")
    print("         Script 17 requires outputs from 01 and 05 (run after both).")
    print("         Script 18 requires outputs from 05.")
    print("         Script 07 must run before 08.")
    
    print("\n4️⃣  BATCH EXECUTION (alternative to running scripts individually):")
    print("   conda activate mapseq_processing")
    rc = os.path.join(repo_path, "run_commands.sh")
    print(f"   bash {rc}")
    print(f"   bash {rc} all_commands.local.txt")
    print(
        "   (First form runs all_commands.txt by default. Use the second form if you have "
        "all_commands.local.txt from setup_wizard_with_sample path resolution.)"
    )
    
    print("\n5️⃣  QUALITY CONTROL:")
    print("   conda activate mapseq_processing")
    print(f"   python {os.path.join(repo_path, 'postprocessing_checks.py')}")
    
    print("\n" + "="*70)
    print("📖 For detailed documentation, see README.md")
    print("="*70 + "\n")

def create_env_and_setup(conda_exe, install_dir, repo_path=None):
    """Create conda environment and set up the repository"""
    print(f"\n📦 Creating environment '{ENV_NAME}'...")
    if conda_env_exists(conda_exe, ENV_NAME):
        print(
            f"   Environment '{ENV_NAME}' already exists — skipping conda create.\n"
            "   (Remove it first with: conda env remove -n "
            f"{ENV_NAME} — only if you want a clean reinstall.)"
        )
    else:
        # Use conda-forge only so we do not require accepting Anaconda, Inc. ToS for pkgs/main.
        conda_run(
            conda_exe,
            [
                "create",
                "-y",
                "-n",
                ENV_NAME,
                "-c",
                "conda-forge",
                "--override-channels",
                "python=3.9",
                "pip",
            ],
        )

    print("🔁 Adding channels: conda-forge, bioconda")
    conda_run(conda_exe, ["config", "--add", "channels", "conda-forge"])
    conda_run(conda_exe, ["config", "--add", "channels", "bioconda"])

    # Determine repository path
    if repo_path is None:
        # Not in a repo, need to clone
        print("🐙 Cloning project repository...")
        git_url = get_clone_repository_url()
        print(f"   {git_url}")
        repo_name = os.path.basename(git_url.rstrip('.git')) if git_url.endswith('.git') else os.path.basename(git_url)
        git_dir = os.path.join(install_dir, repo_name)
        
        if not os.path.exists(git_dir):
            if not check_git_installed():
                auto_install_git()
            if not check_git_installed():
                if not prompt_git_installation():
                    print("\n⚠️  Setup will continue, but you'll need to clone the repository manually.")
                    print(f"   Repository URL: {git_url}")
                    return None
            try:
                subprocess.run(["git", "clone", git_url], cwd=install_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️  Failed to clone repository: {e}")
                print(f"   Please clone manually: git clone {git_url}")
                return None
        else:
            print("📂 Repo already cloned.")
        
        repo_path = git_dir
    else:
        print(f"📂 Using existing repository at: {repo_path}")

    # Download the GUI exe into the repo directory (if available and on Windows)
    if platform.system() == "Windows":
        gui_exe_path = os.path.join(repo_path, "MAPseq_Wizard.exe")
        if not os.path.exists(gui_exe_path):
            gui_exe_url = get_gui_exe_url()
            if gui_exe_url:
                try:
                    download_gui_exe(gui_exe_url, gui_exe_path)
                except Exception as e:
                    print(f"⚠️ Could not download GUI exe: {e}")
                    print("   You can build it manually using PyInstaller if needed.")
            else:
                print("⚠️ GUI exe URL not available (may not be a GitHub repo or no releases available)")
                print("   You can build it manually using PyInstaller if needed.")
        else:
            print(f"✅ GUI exe already exists at: {gui_exe_path}")

    # Install dependencies
    requirements_path = os.path.join(repo_path, "requirements.txt")
    if os.path.exists(requirements_path):
        print("📄 Installing dependencies from requirements.txt...")
        try:
            subprocess.run([
                conda_exe, "run", "-n", ENV_NAME, "pip", "install", "-r", requirements_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Some dependencies may have failed to install: {e}")
            print("   You may need to install them manually.")
    else:
        print(f"⚠️ No requirements.txt found in {repo_path}")

    # Verify installation
    verify_installation(conda_exe, ENV_NAME)
    
    return repo_path

def main():
    try:
        # Check if already in a repository
        repo_path = get_repo_path()
        
        default_path = get_default_install_path()
        install_path = prompt_install_path(default_path)

        if not os.path.isdir(install_path):
            os.makedirs(install_path, exist_ok=True)

        conda_exe = get_conda_path(install_path)

        if not os.path.exists(conda_exe):
            print("\n❗ Conda not found. Installing Miniconda...")
            try:
                install_miniconda(install_path)
            except RuntimeError as e:
                print(f"\n❌ {e}")
                input("\n📝 Press Enter to exit...")
                return
        else:
            print("✅ Conda already installed.")

        if not os.path.exists(conda_exe):
            raise FileNotFoundError(f"conda executable not found at {conda_exe}")

        # Create environment and setup
        final_repo_path = create_env_and_setup(conda_exe, install_dir=install_path if repo_path is None else os.path.dirname(repo_path), repo_path=repo_path)
        
        if final_repo_path:
            print("\n✅ All steps completed successfully!")
            if platform.system() == "Windows":
                print(f"   You can now run MAPseq_Wizard.exe from: {final_repo_path}")
            print_post_installation_instructions(final_repo_path)
        else:
            print("\n⚠️  Setup completed with warnings. Please review the messages above.")

    except subprocess.CalledProcessError as e:
        print(f"\n🚨 Subprocess failed: {e}")
        print("   This usually indicates a problem with conda, git, or network connectivity.")
        print("   Please check the error message above and try again.")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Please check that Miniconda installed correctly.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
    except Exception as e:
        print(f"\n⚠️  Unexpected error: {e}")
        print("   Please check the error message and try again.")
        import traceback
        traceback.print_exc()

    input("\n📝 Press Enter to exit...")

if __name__ == "__main__":
    main()
