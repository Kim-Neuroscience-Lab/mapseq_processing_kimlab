"""
MAPseq setup wizard plus optional published sample batch run.

Based on setup_wizard.py (same Miniconda/env/pip flow). Use this entrypoint when you
want the post-setup prompt to process the published sample via run_commands.sh.
"""
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

# Used when `git remote get-url origin` cannot be read (e.g. script copied outside a clone).
DEFAULT_CLONE_REPOSITORY_URL = "https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab.git"


def get_clone_repository_url():
    """URL for `git clone` during setup.

    Defaults to the public lab repo so running the wizard from a fork does not clone
    the fork. Override with MAPSEQ_WIZARD_CLONE_URL (non-empty).
    """
    override = os.environ.get("MAPSEQ_WIZARD_CLONE_URL", "").strip()
    return override if override else DEFAULT_CLONE_REPOSITORY_URL


REPO_ROOT_PLACEHOLDER = "__REPO_ROOT__"
LOCAL_COMMANDS_BASENAME = "all_commands.local.txt"
COMMANDS_TEMPLATE_BASENAME = "all_commands.txt"

# Paths under raw_data_sources/ shipped with the published sample (see all_commands.txt)
EXPECTED_SAMPLE_RAW_FILES = (
    "raw_data_sources/p60/aggregated_cleaned_matrix.tsv",
    "raw_data_sources/p12/aggregated_cleaned_matrix.tsv",
    "raw_data_sources/p20/aggregated_cleaned_matrix.tsv",
    "raw_data_sources/p60/jr0695.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/jr0694.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/jr0692.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/JR0552.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/JR0548.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/JR0547.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/JR0546.nbcm_cleaned.tsv",
    "raw_data_sources/p60/JR0693.nbcm_cleaned.tsv",
    "raw_data_sources/p60/jr0448.nbcm.all_cleaned.tsv",
    "raw_data_sources/p60/jr0446.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/M777.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/jr0686.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/JR0671.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/jr0670.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/jr0422.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/jr0420.nbcm.all_cleaned.tsv",
    "raw_data_sources/p12/JR0685.nbcm_cleaned.tsv",
    "raw_data_sources/p12/JR0689.nbcm_cleaned.tsv",
    "raw_data_sources/p12/JR0690.nbcm_cleaned.tsv",
    "raw_data_sources/p20/jr0678.nbcm.all_cleaned.tsv",
    "raw_data_sources/p20/jr0674.nbcm.all_cleaned.tsv",
    "raw_data_sources/p20/jr0672.nbcm.all_cleaned.tsv",
    "raw_data_sources/p20/JR0883.nbcm_cleaned.tsv",
    "raw_data_sources/p20/JR0884.nbcm_cleaned.tsv",
    "raw_data_sources/p20/JR0887.nbcm_cleaned.tsv",
)


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
            "Miniconda3.exe",
        )
    elif system == "Darwin":  # macOS
        if machine == "arm64" or machine == "aarch64":
            return (
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh",
                "Miniconda3-latest-MacOSX-arm64.sh",
            )
        else:
            return (
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh",
                "Miniconda3-latest-MacOSX-x86_64.sh",
            )
    else:  # Linux
        return (
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
            "Miniconda3-latest-Linux-x86_64.sh",
        )


def is_git_repo(path):
    """Check if a directory is a git repository"""
    return os.path.exists(os.path.join(path, ".git"))


def _ensure_macos_git_on_path():
    """If git exists under CLT or /usr/bin but PATH does not resolve `git`, prepend that directory."""
    if platform.system() != "Darwin":
        return
    if shutil.which("git"):
        return
    for git_bin in (
        "/usr/bin/git",
        "/Library/Developer/CommandLineTools/usr/bin/git",
    ):
        if os.path.isfile(git_bin) and os.access(git_bin, os.X_OK):
            bindir = os.path.dirname(git_bin)
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            return


def _ensure_darwin_tool_path():
    """Prepend standard system dirs so bash/curl work under PyInstaller or minimal PATH."""
    if platform.system() != "Darwin":
        return
    p = os.environ.get("PATH", "")
    parts = [x for x in p.split(os.pathsep) if x]
    for d in (
        "/usr/bin",
        "/bin",
        "/Library/Developer/CommandLineTools/usr/bin",
        "/usr/sbin",
        "/sbin",
    ):
        if os.path.isdir(d) and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


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
        print("   Option 1: Xcode Command Line Tools (includes git): xcode-select --install")
        print("   Option 2: https://git-scm.com/download/mac")
        print("   (Homebrew needs git before it can install packages.)")
    else:  # Linux
        print("\n⚠️  Git is not installed. Install using your package manager:")
        print("   Ubuntu/Debian: sudo apt-get install git")
        print("   Fedora/RHEL: sudo yum install git")
        print("   Arch: sudo pacman -S git")

    response = input("\nContinue setup anyway? (You can clone the repo manually later) [y/N]: ")
    return response.lower() == "y"


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
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DEFAULT_CLONE_REPOSITORY_URL


def get_gui_exe_url(git_url=None, version=GUI_VERSION):
    """Construct GUI exe download URL from git remote URL"""
    if git_url is None:
        git_url = get_git_remote_url()

    if "github.com" in git_url:
        match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", git_url)
        if match:
            user, repo = match.groups()
            return f"https://github.com/{user}/{repo}/releases/download/{version}/MAPseq_Wizard.exe"

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
    install_path = os.path.abspath(install_path)
    cache_dir = os.path.join(install_path, ".setup_wizard_cache")
    os.makedirs(cache_dir, exist_ok=True)
    url, installer_name = get_miniconda_installer_info()
    installer = os.path.join(cache_dir, installer_name)
    system = platform.system()
    _req_timeout = (30, 600)

    print("🔍 Downloading Miniconda...")
    try:
        if system == "Windows":
            try:
                subprocess.run(["curl", "-L", "-o", installer, url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                response = requests.get(url, stream=True, timeout=_req_timeout)
                response.raise_for_status()
                with open(installer, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            try:
                subprocess.run(["curl", "-L", "-o", installer, url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                try:
                    subprocess.run(["wget", "-O", installer, url], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    response = requests.get(url, stream=True, timeout=_req_timeout)
                    response.raise_for_status()
                    with open(installer, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download Miniconda installer: {e}. Please check your internet connection."
        )

    print(f"🔧 Installing Miniconda to: {install_path}")

    if system == "Windows":
        subprocess.run(
            [
                installer,
                "/InstallationType=JustMe",
                "/RegisterPython=0",
                "/AddToPath=1",
                "/S",
                f"/D={install_path}",
            ],
            check=True,
        )
    else:
        os.chmod(installer, 0o755)
        subprocess.run(["bash", installer, "-b", "-p", install_path, "-f"], check=True)
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
        raise RuntimeError("Download timeout. Please check your internet connection and try again.")
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download GUI exe: {e}. You can build it manually using PyInstaller if needed."
        )


def verify_installation(conda_exe, env_name):
    """Verify that critical packages can be imported"""
    print("\n🔍 Verifying installation...")
    test_imports = [
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "sklearn",
        "PySimpleGUI",
        "seaborn",
        "statsmodels",
    ]

    failed_imports = []
    for package in test_imports:
        try:
            import_name = package
            if package == "sklearn":
                import_name = "sklearn"
            elif package == "PySimpleGUI":
                import_name = "PySimpleGUI"

            subprocess.run(
                [conda_exe, "run", "-n", env_name, "python", "-c", f"import {import_name}"],
                capture_output=True,
                check=True,
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


def _try_git_pull_existing_clone(repo_path):
    """Best-effort fast-forward pull so a previous wizard clone picks up new tracked files."""
    if not is_git_repo(repo_path):
        return
    print("🔄 Updating existing clone (git pull --ff-only)...")
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    r = subprocess.run(
        ["git", "-C", repo_path, "pull", "--ff-only"],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode == 0:
        print("✅ Repository is up to date.")
        return
    print(
        "⚠️  Could not fast-forward the existing clone (offline, diverged branches, or other git error)."
    )
    print(f"   Try from that folder: cd \"{repo_path}\" && git pull")
    print(f"   Or re-clone from:\n   {get_clone_repository_url()}")
    err = (r.stderr or r.stdout or "").strip()
    if err:
        snippet = err[:500] + ("…" if len(err) > 500 else "")
        print(f"   git said: {snippet}")


def print_post_installation_instructions(repo_path):
    """Print comprehensive post-installation instructions"""
    print("\n" + "=" * 70)
    print("📚 POST-INSTALLATION INSTRUCTIONS")
    print("=" * 70)

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
    print(
        f"   python {os.path.join(repo_path, 'process-nbcm-tsv.py')} -o <out_dir> -s <sample> -d <data_file> -l <labels>"
    )

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
        "13_aggregate_projection_summaries.py",
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
    print(f"   bash {rc} {LOCAL_COMMANDS_BASENAME}")
    print(
        f"   (First form uses {COMMANDS_TEMPLATE_BASENAME} by default. Template uses {REPO_ROOT_PLACEHOLDER}; "
        f"this wizard can write {LOCAL_COMMANDS_BASENAME} with your repo path — use the second form for that file.)"
    )

    print("\n5️⃣  QUALITY CONTROL:")
    print("   conda activate mapseq_processing")
    print(f"   python {os.path.join(repo_path, 'postprocessing_checks.py')}")

    print("\n" + "=" * 70)
    print("📖 For detailed documentation, see README.md")
    print("=" * 70 + "\n")


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
        # Use conda-forge only so we do not require accepting Anaconda, Inc. ToS for pkgs/main
        # (see https://www.anaconda.com/docs/tools/working-with-conda/channels).
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

    if repo_path is None:
        print("🐙 Cloning project repository...")
        git_url = get_clone_repository_url()
        print(f"   {git_url}")
        repo_name = (
            os.path.basename(git_url.rstrip(".git"))
            if git_url.endswith(".git")
            else os.path.basename(git_url)
        )
        git_dir = os.path.join(install_dir, repo_name)

        if not os.path.exists(git_dir):
            if not check_git_installed():
                auto_install_git()
            if not check_git_installed():
                if not prompt_git_installation():
                    print("\n⚠️  Setup will continue, but you'll need to clone the repository manually.")
                    print(f"   Repository URL: {git_url}")
                    return None, False
            try:
                subprocess.run(["git", "clone", git_url], cwd=install_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️  Failed to clone repository: {e}")
                print(f"   Please clone manually: git clone {git_url}")
                return None, False
        else:
            print("📂 Repo already cloned.")
            _try_git_pull_existing_clone(git_dir)

        repo_path = git_dir
    else:
        print(f"📂 Using existing repository at: {repo_path}")

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

    requirements_path = os.path.join(repo_path, "requirements.txt")
    if os.path.exists(requirements_path):
        print("📄 Installing dependencies from requirements.txt...")
        try:
            subprocess.run(
                [conda_exe, "run", "-n", ENV_NAME, "pip", "install", "-r", requirements_path],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Some dependencies may have failed to install: {e}")
            print("   You may need to install them manually.")
    else:
        print(f"⚠️ No requirements.txt found in {repo_path}")

    verify_ok = verify_installation(conda_exe, ENV_NAME)

    return repo_path, verify_ok


def normalize_repo_root_for_commands(repo_root):
    root = os.path.normpath(os.path.abspath(repo_root))
    return root.replace("\\", "/")


def write_resolved_commands_file(repo_root):
    template_path = os.path.join(repo_root, COMMANDS_TEMPLATE_BASENAME)
    out_path = os.path.join(repo_root, LOCAL_COMMANDS_BASENAME)
    with open(template_path, "r", encoding="utf-8") as f:
        text = f.read()
    root = normalize_repo_root_for_commands(repo_root)
    text = text.replace(REPO_ROOT_PLACEHOLDER, root)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def preflight_sample_raw_data(repo_root):
    missing = []
    for rel in EXPECTED_SAMPLE_RAW_FILES:
        path = os.path.join(repo_root, rel)
        if not os.path.isfile(path):
            missing.append(rel)
    return (len(missing) == 0, missing)


def _maybe_seed_sample_raw_data_sources(repo_root):
    """Optional: if MAPSEQ_WIZARD_SAMPLE_RAW_SOURCES is set, copy that tree into repo_root/raw_data_sources.

    The default clone (DEFAULT_CLONE_REPOSITORY_URL) already includes raw_data_sources/ (cleaned TSVs) in git; use this
    only to overlay TSVs when the clone is missing them (e.g. sparse checkout, fork without data, CI fixture).
    """
    src = os.environ.get("MAPSEQ_WIZARD_SAMPLE_RAW_SOURCES", "").strip()
    if not src:
        return
    src = os.path.abspath(os.path.expanduser(src))
    if not os.path.isdir(src):
        print(f"\n⚠️  MAPSEQ_WIZARD_SAMPLE_RAW_SOURCES is not a directory — skipping seed: {src}")
        return
    dst = os.path.join(repo_root, "raw_data_sources")
    print(f"\n📂 Seeding raw_data_sources from MAPSEQ_WIZARD_SAMPLE_RAW_SOURCES:\n   {src}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prompt_and_run_sample_batch(conda_exe, repo_root):
    """Run optional sample batch. Returns True if batch ran and succeeded, False if failed, None if skipped."""
    if platform.system() == "Windows":
        print("\n📌 Optional sample batch run is not started automatically on Windows.")
        print("   Use WSL or Git Bash from the repo root after activating the environment:")
        print(f"   conda activate {ENV_NAME}")
        print(f"   bash run_commands.sh {LOCAL_COMMANDS_BASENAME}")
        return None

    response = input(
        "\nWould you like to process the published sample dataset? (yes/no): "
    ).strip().lower()
    if response not in ("y", "yes"):
        print("Skipping sample batch run.")
        return None

    _maybe_seed_sample_raw_data_sources(repo_root)

    run_sh = os.path.join(repo_root, "run_commands.sh")
    template_path = os.path.join(repo_root, COMMANDS_TEMPLATE_BASENAME)
    if not os.path.isfile(run_sh):
        print(f"\n❌ Missing {run_sh}. Cannot run sample batch.")
        print(f"   This folder may be an old clone. Try: cd \"{repo_root}\" && git pull")
        print(f"   Or re-clone from:\n   {get_clone_repository_url()}")
        return False
    if not os.path.isfile(template_path):
        print(f"\n❌ Missing {template_path}. Cannot run sample batch.")
        return False

    ok, missing = preflight_sample_raw_data(repo_root)
    if not ok:
        print(
            "\n❌ Published sample data files are missing under raw_data_sources/ in the clone "
            "(see raw_data_sources/README.md). The upstream repo includes them; if your tree is "
            "incomplete, set MAPSEQ_WIZARD_SAMPLE_RAW_SOURCES to a directory containing the same layout."
        )
        print("   Missing:")
        for rel in missing[:20]:
            print(f"     - {rel}")
        if len(missing) > 20:
            print(f"     ... and {len(missing) - 20} more.")
        return False

    try:
        local_commands = write_resolved_commands_file(repo_root)
    except OSError as e:
        print(f"\n❌ Could not write resolved command file: {e}")
        return False

    print(f"\n▶️  Running batch via: bash run_commands.sh {LOCAL_COMMANDS_BASENAME}")
    print(f"   Resolved commands: {local_commands}")
    print(
        "   Tip: `all_commands.local.txt` is generated from `all_commands.txt`; "
        "run `git pull` for template updates, then delete the `.local` file or re-run this wizard step."
    )
    print(f"   Batch logs: processing_*.log under {repo_root}")
    try:
        proc = subprocess.run(
            [
                conda_exe,
                "run",
                "-n",
                ENV_NAME,
                "bash",
                run_sh,
                LOCAL_COMMANDS_BASENAME,
            ],
            cwd=repo_root,
        )
        if proc.returncode != 0:
            print(
                f"\n❌ Sample batch exited with status {proc.returncode}. "
                f"See processing_*.log in:\n   {repo_root}"
            )
            return False
        print("\n✅ Sample batch run finished.")
        return True
    except OSError as e:
        print(f"\n❌ Failed to start batch run: {e}")
        return False


def main():
    try:
        _ensure_darwin_tool_path()
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

        final_repo_path, verify_ok = create_env_and_setup(
            conda_exe,
            install_dir=install_path if repo_path is None else os.path.dirname(repo_path),
            repo_path=repo_path,
        )

        if final_repo_path:
            if not verify_ok:
                print(
                    "\n⚠️  Package verification reported one or more import failures — "
                    "see messages above; you may need to reinstall dependencies."
                )
            if platform.system() == "Windows":
                print(f"   You can now run MAPseq_Wizard.exe from: {final_repo_path}")
            print_post_installation_instructions(final_repo_path)
            batch_outcome = prompt_and_run_sample_batch(conda_exe, final_repo_path)
            if batch_outcome is False:
                print(
                    "\n❌ Sample batch did not complete successfully. "
                    "Inspect processing_*.log in the repository directory above."
                )
            elif batch_outcome is True:
                print("\n✅ Setup and sample batch completed successfully.")
            else:
                print("\n✅ Setup completed successfully (sample batch was skipped).")
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
