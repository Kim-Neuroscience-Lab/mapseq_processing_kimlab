"""
Step 1: Project Setup UI Component
"""

import customtkinter as ctk
from pathlib import Path
import tkinter.filedialog as filedialog
from ..utils.path_utils import get_repo_root


class ProjectSetupStep(ctk.CTkFrame):
    """Project setup step of the wizard"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Title
        title = ctk.CTkLabel(self, text="Project Setup", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Project name
        ctk.CTkLabel(self, text="Project Name:").pack(anchor="w", padx=20, pady=(10, 5))
        self.project_name_entry = ctk.CTkEntry(self, width=400)
        self.project_name_entry.pack(padx=20, pady=(0, 10), fill="x")
        
        # Repository root
        ctk.CTkLabel(self, text="Repository Root Directory:").pack(anchor="w", padx=20, pady=(10, 5))
        repo_frame = ctk.CTkFrame(self)
        repo_frame.pack(padx=20, pady=(0, 10), fill="x")
        self.repo_root_entry = ctk.CTkEntry(repo_frame, width=350)
        self.repo_root_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(repo_frame, text="Browse", width=100, command=self.browse_repo_root).pack(side="right")
        
        # Base output directory
        ctk.CTkLabel(self, text="Base Output Directory (02_output):").pack(anchor="w", padx=20, pady=(10, 5))
        output_frame = ctk.CTkFrame(self)
        output_frame.pack(padx=20, pady=(0, 10), fill="x")
        self.output_dir_entry = ctk.CTkEntry(output_frame, width=350)
        self.output_dir_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", width=100, command=self.browse_output_dir).pack(side="right")
    
    def browse_repo_root(self):
        """Browse for repository root directory"""
        directory = filedialog.askdirectory(title="Select Repository Root Directory")
        if directory:
            self.repo_root_entry.delete(0, "end")
            self.repo_root_entry.insert(0, directory)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Base Output Directory")
        if directory:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, directory)
    
    def load_config(self):
        """Load configuration values"""
        project_name = self.config.get('project.name', 'My MAPseq Analysis')
        repo_root = self.config.get('project.repo_root', str(get_repo_root()))
        output_dir = self.config.get('project.base_output_dir', str(get_repo_root() / '02_output'))
        
        self.project_name_entry.insert(0, project_name)
        self.repo_root_entry.insert(0, repo_root)
        self.output_dir_entry.insert(0, output_dir)
    
    def save_config(self):
        """Save configuration values"""
        self.config.set('project.name', self.project_name_entry.get())
        self.config.set('project.repo_root', self.repo_root_entry.get())
        self.config.set('project.base_output_dir', self.output_dir_entry.get())
    
    def validate(self):
        """Validate inputs"""
        if not self.project_name_entry.get().strip():
            return False, "Project name is required"
        if not self.repo_root_entry.get().strip():
            return False, "Repository root directory is required"
        if not Path(self.repo_root_entry.get()).exists():
            return False, "Repository root directory does not exist"
        return True, None


