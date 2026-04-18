"""
Step 4: Helper Scripts Configuration UI Component
"""

import customtkinter as ctk


class HelperScriptsStep(ctk.CTkFrame):
    """Helper scripts configuration step"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.checkboxes = {}
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Set up the UI components"""
        title = ctk.CTkLabel(self, text="Helper Scripts Configuration", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        info = ctk.CTkLabel(self, text="Select which helper scripts to run. Dependencies are handled automatically.", 
                           font=ctk.CTkFont(size=12))
        info.pack(pady=10)
        
        frame = ctk.CTkScrollableFrame(self)
        frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        scripts = [
            ('01', 'Motif Analysis Per Animal'),
            ('02', 'Projection Analysis'),
            ('03', 'Composition'),
            ('04', 'Proportions Over Time Stats'),
            ('05', 'Motif Analysis (required for 06, 07)'),
            ('06', 'All Motif Divergence (requires 05)'),
            ('07', 'Motif Significance Trajectories (requires 05)'),
            ('15', 'Volcano Trajectories (requires 07)'),
            ('08', 'Motif Clustering (requires 07)'),
            ('09', 'Plot Normalized Projection Strength'),
            ('13', 'Aggregate Projection Summaries'),
        ]
        
        for script_id, description in scripts:
            var = ctk.BooleanVar()
            cb = ctk.CTkCheckBox(frame, text=f"{script_id}: {description}", variable=var)
            cb.pack(anchor="w", padx=10, pady=5)
            self.checkboxes[script_id] = var
    
    def load_config(self):
        """Load configuration"""
        enabled = self.config.get('helper_scripts.enabled', [])
        for script_id, var in self.checkboxes.items():
            var.set(script_id in enabled)
    
    def save_config(self):
        """Save configuration"""
        enabled = [script_id for script_id, var in self.checkboxes.items() if var.get()]
        self.config.set('helper_scripts.enabled', enabled)
    
    def validate(self):
        """Validate inputs"""
        return True, None


