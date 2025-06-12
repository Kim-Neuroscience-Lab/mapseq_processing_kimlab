import os
import subprocess
import platform
import shutil
import sys

GIT_URL = "https://github.com/Kim-Neuroscience-Lab/mapseq_processing_kimlab.git"
ENV_NAME = "mapseq_processing"

def prompt_install_path(default_path):
    print(f"\n📁 Default Miniconda install location: {default_path}")
    custom_path = input("Enter custom install path (or press Enter to use default): ").strip()
    return custom_path if custom_path else default_path

def install_miniconda(install_path):
    url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    installer = "Miniconda3.exe"

    print("🔍 Downloading Miniconda...")
    subprocess.run(["curl", "-L", "-o", installer, url], check=True)

    print(f"🔧 Installing Miniconda to: {install_path}")
    subprocess.run([
        installer,
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=1",
        "/S",
        f"/D={install_path}"
    ], check=True)

def conda(cmd, conda_exe):
    subprocess.run([conda_exe] + cmd, check=True)

def create_env_and_setup(conda_exe, install_dir):
    print(f"\n📦 Creating environment '{ENV_NAME}'...")
    conda(["create", "-y", "-n", ENV_NAME, "python=3.9", "pip"], conda_exe)

    print("🔁 Adding channels: conda-forge, bioconda")
    conda(["config", "--add", "channels", "conda-forge"], conda_exe)
    conda(["config", "--add", "channels", "bioconda"], conda_exe)

    print("🐙 Cloning project repository...")
    git_dir = os.path.join(install_dir, "mapseq_processing_kimlab")
    if not os.path.exists(git_dir):
        subprocess.run(["git", "clone", GIT_URL], cwd=install_dir, check=True)
    else:
        print("📂 Repo already cloned.")

    requirements_path = os.path.join(git_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("📄 Installing dependencies from cloned requirements.txt...")
        subprocess.run([
            conda_exe, "run", "-n", ENV_NAME, "pip", "install", "-r", requirements_path
        ], check=True)
    else:
        print(f"⚠️ No requirements.txt found in {git_dir}")

def main():
    try:
        if platform.system() != "Windows":
            print("❌ This setup wizard is for Windows only.")
            input("Press Enter to exit...")
            return

        default_path = os.path.expanduser("~\\Miniconda3")
        install_path = prompt_install_path(default_path)

        if not os.path.isdir(install_path):
            os.makedirs(install_path, exist_ok=True)

        conda_exe = os.path.join(install_path, "Scripts", "conda.exe")

        if not os.path.exists(conda_exe):
            print("\n❗ Conda not found. Installing Miniconda...")
            install_miniconda(install_path)
        else:
            print("✅ Conda already installed.")

        if not os.path.exists(conda_exe):
            raise FileNotFoundError(f"conda.exe not found at {conda_exe}")

        create_env_and_setup(conda_exe, install_path)
        print("\n✅ All steps completed. You can now run the analysis wizard!")

    except subprocess.CalledProcessError as e:
        print(f"\n🚨 Subprocess failed: {e}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {e}")

    input("\n📝 Press Enter to exit...")

if __name__ == "__main__":
    main()
