"""
Step 6: Review & Generate UI Component
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, scrolledtext, messagebox
from ..command_generator import CommandGenerator


class ReviewStep(ctk.CTkFrame):
    """Review and generate step"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.command_generator = CommandGenerator(config_manager)
        self.setup_ui()
        self.refresh_preview()
    
    def setup_ui(self):
        """Set up the UI components"""
        title = ctk.CTkLabel(self, text="Review & Generate", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Configuration summary
        summary_frame = ctk.CTkFrame(self)
        summary_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(summary_frame, text="Configuration Summary", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=10, width=80)
        self.summary_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Command preview
        preview_frame = ctk.CTkFrame(self)
        preview_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(preview_frame, text="Generated Commands Preview", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.command_text = scrolledtext.ScrolledText(preview_frame, height=15, width=80, font=("Courier", 10))
        self.command_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkButton(button_frame, text="Save Configuration", command=self.save_config_file).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Generate Commands File", command=self.generate_commands).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Refresh Preview", command=self.refresh_preview).pack(side="left", padx=5)
    
    def refresh_preview(self):
        """Refresh configuration summary and command preview"""
        # Update summary
        self.summary_text.delete(1.0, "end")
        summary = self._generate_summary()
        self.summary_text.insert(1.0, summary)
        
        # Update command preview
        self.command_text.delete(1.0, "end")
        commands = self._generate_command_preview()
        self.command_text.insert(1.0, commands)
    
    def _generate_summary(self):
        """Generate configuration summary text"""
        lines = []
        lines.append(f"Project: {self.config.get('project.name', 'N/A')}")
        lines.append(f"Repository Root: {self.config.get('project.repo_root', 'N/A')}")
        lines.append(f"Output Directory: {self.config.get('project.base_output_dir', 'N/A')}")
        lines.append("")
        
        lines.append("Preprocessing:")
        lines.append(f"  Input: {self.config.get('preprocessing.input_dir', 'N/A')}")
        lines.append(f"  Output: {self.config.get('preprocessing.output_dir', 'N/A')}")
        lines.append("")
        
        params = self.config.get('main_processing.parameterizations', [])
        lines.append(f"Parameterizations: {len(params)}")
        for param in params:
            lines.append(f"  - {param['name']}")
        lines.append("")
        
        age_groups = self.config.get('main_processing.age_groups', {})
        total_samples = sum(len(age_data.get('samples', [])) for age_data in age_groups.values())
        lines.append(f"Total Samples: {total_samples}")
        for age_group, age_data in age_groups.items():
            samples = age_data.get('samples', [])
            if samples:
                lines.append(f"  {age_group}: {len(samples)} samples")
        lines.append("")
        
        enabled_scripts = self.config.get('helper_scripts.enabled', [])
        lines.append(f"Helper Scripts: {len(enabled_scripts)} enabled")
        lines.append(f"  {', '.join(enabled_scripts)}")
        
        return "\n".join(lines)
    
    def _generate_command_preview(self):
        """Generate command preview (first 50 lines)"""
        # Create temporary file to generate commands
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = Path(f.name)
        
        try:
            self.command_generator.generate_all_commands(temp_path)
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                preview = ''.join(lines[:50])
                if len(lines) > 50:
                    preview += f"\n... ({len(lines) - 50} more lines) ..."
                return preview
        except Exception as e:
            return f"Error generating preview: {e}"
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def save_config_file(self):
        """Save configuration to YAML file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config.save(Path(filename))
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving configuration: {e}")
    
    def generate_commands(self):
        """Generate commands file"""
        filename = filedialog.asksaveasfilename(
            title="Save Commands File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.command_generator.generate_all_commands(Path(filename))
                messagebox.showinfo("Success", f"Commands file generated: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error generating commands: {e}")
    
    def validate(self):
        """Validate configuration"""
        is_valid, errors = self.config.validate()
        if not is_valid:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            return False, error_msg
        return True, None


