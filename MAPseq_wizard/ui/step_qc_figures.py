"""
Step 5: Quality Control & Figures Configuration UI Component
"""

import customtkinter as ctk
from tkinter import filedialog


class QCFiguresStep(ctk.CTkFrame):
    """Quality control and figure generation configuration step"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Set up the UI components"""
        title = ctk.CTkLabel(self, text="Quality Control & Figures", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Quality Control section
        qc_frame = ctk.CTkFrame(self)
        qc_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(qc_frame, text="Quality Control", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        self.qc_enabled = ctk.BooleanVar()
        ctk.CTkCheckBox(qc_frame, text="Enable Quality Control Checks", variable=self.qc_enabled).pack(anchor="w", padx=10, pady=5)
        
        # Figure Generation section
        fig_frame = ctk.CTkFrame(self)
        fig_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(fig_frame, text="Figure Generation", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        self.fig_enabled = ctk.BooleanVar()
        ctk.CTkCheckBox(fig_frame, text="Enable Figure Generation", variable=self.fig_enabled, 
                       command=self.toggle_fig_options).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(fig_frame, text="Parameterization:").pack(anchor="w", padx=10, pady=(10, 5))
        self.param_entry = ctk.CTkEntry(fig_frame, width=400)
        self.param_entry.pack(padx=10, pady=(0, 10), fill="x")
        
        ctk.CTkLabel(fig_frame, text="Output Directory:").pack(anchor="w", padx=10, pady=(10, 5))
        output_frame = ctk.CTkFrame(fig_frame)
        output_frame.pack(padx=10, pady=(0, 10), fill="x")
        self.output_dir_entry = ctk.CTkEntry(output_frame, width=350)
        self.output_dir_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", width=100, command=self.browse_output_dir).pack(side="right")
    
    def toggle_fig_options(self):
        """Enable/disable figure options"""
        enabled = self.fig_enabled.get()
        self.param_entry.configure(state="normal" if enabled else "disabled")
        self.output_dir_entry.configure(state="normal" if enabled else "disabled")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Figure Output Directory")
        if directory:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, directory)
    
    def load_config(self):
        """Load configuration"""
        self.qc_enabled.set(self.config.get('quality_control.enabled', True))
        self.fig_enabled.set(self.config.get('figure_generation.enabled', True))
        self.param_entry.insert(0, self.config.get('figure_generation.parameterization', ''))
        self.output_dir_entry.insert(0, self.config.get('figure_generation.output_dir', ''))
        self.toggle_fig_options()
    
    def save_config(self):
        """Save configuration"""
        self.config.set('quality_control.enabled', self.qc_enabled.get())
        self.config.set('figure_generation.enabled', self.fig_enabled.get())
        self.config.set('figure_generation.parameterization', self.param_entry.get())
        self.config.set('figure_generation.output_dir', self.output_dir_entry.get())
    
    def validate(self):
        """Validate inputs"""
        if self.fig_enabled.get():
            if not self.param_entry.get().strip():
                return False, "Parameterization is required for figure generation"
        return True, None


