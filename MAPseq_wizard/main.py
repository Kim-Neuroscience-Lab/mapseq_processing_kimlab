"""
Main entry point for MAPseq Pipeline Wizard
"""

import sys
import customtkinter as ctk
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from MAPseq_wizard.ui.wizard_window import WizardWindow


def main():
    """Launch the wizard"""
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")  # or "light"
    ctk.set_default_color_theme("blue")
    
    # Create and run wizard
    app = WizardWindow()
    app.mainloop()


if __name__ == "__main__":
    main()


