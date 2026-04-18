"""
Main wizard window controller
Manages step navigation and coordination
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import messagebox
from ..config_manager import ConfigManager
from .step_project_setup import ProjectSetupStep
from .step_preprocessing import PreprocessingStep
from .step_main_processing import MainProcessingStep
from .step_helper_scripts import HelperScriptsStep
from .step_qc_figures import QCFiguresStep
from .step_review import ReviewStep
from .step_execute import ExecuteStep


class WizardWindow(ctk.CTk):
    """Main wizard window"""
    
    STEPS = [
        ("Project Setup", ProjectSetupStep),
        ("Preprocessing", PreprocessingStep),
        ("Main Processing", MainProcessingStep),
        ("Helper Scripts", HelperScriptsStep),
        ("QC & Figures", QCFiguresStep),
        ("Review & Generate", ReviewStep),
        ("Execute", ExecuteStep),
    ]
    
    def __init__(self):
        super().__init__()
        
        self.title("MAPseq Pipeline Wizard")
        self.geometry("900x700")
        
        # Bring window to front
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.focus_force()
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        self.config_manager.load()
        
        # Current step
        self.current_step_index = 0
        self.steps = []
        self.step_widgets = []
        
        self.setup_ui()
        self.show_step(0)
    
    def setup_ui(self):
        """Set up the main UI"""
        # Header
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=10)
        
        title = ctk.CTkLabel(header, text="MAPseq Pipeline Wizard", font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(pady=10)
        
        # Step indicator
        self.step_indicator = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=14))
        self.step_indicator.pack(pady=5)
        
        # Main content area
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(self)
        nav_frame.pack(fill="x", padx=10, pady=10)
        
        self.prev_button = ctk.CTkButton(nav_frame, text="← Previous", command=self.prev_step, width=120, state="disabled")
        self.prev_button.pack(side="left", padx=5)
        
        self.next_button = ctk.CTkButton(nav_frame, text="Next →", command=self.next_step, width=120)
        self.next_button.pack(side="right", padx=5)
        
        self.cancel_button = ctk.CTkButton(nav_frame, text="Cancel", command=self.on_cancel, width=120, fg_color="gray")
        self.cancel_button.pack(side="right", padx=5)
    
    def show_step(self, step_index):
        """Show a specific step"""
        # Validate and save current step if moving forward (BEFORE destroying widgets)
        if step_index > self.current_step_index and self.current_step_index < len(self.step_widgets):
            current_widget = self.step_widgets[self.current_step_index]
            if hasattr(current_widget, 'save_config'):
                try:
                    current_widget.save_config()
                except Exception as e:
                    # Widget might be destroyed, try to get values directly
                    pass
            if hasattr(current_widget, 'validate'):
                try:
                    is_valid, error = current_widget.validate()
                    if not is_valid:
                        messagebox.showerror("Validation Error", error or "Please fix the errors before proceeding")
                        return
                except Exception:
                    # Widget might be destroyed, skip validation
                    pass
        
        # Clear current step (hide instead of destroy to preserve widget state)
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()
        
        self.current_step_index = step_index
        
        # Create step widget if needed
        if step_index >= len(self.step_widgets):
            step_name, step_class = self.STEPS[step_index]
            widget = step_class(self.content_frame, self.config_manager)
            widget.pack(fill="both", expand=True, padx=20, pady=20)
            self.step_widgets.append(widget)
        else:
            widget = self.step_widgets[step_index]
            widget.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Update step indicator
        step_name, _ = self.STEPS[step_index]
        self.step_indicator.configure(text=f"Step {step_index + 1} of {len(self.STEPS)}: {step_name}")
        
        # Update navigation buttons
        self.prev_button.configure(state="normal" if step_index > 0 else "disabled")
        
        if step_index == len(self.STEPS) - 1:
            self.next_button.configure(text="Finish", state="normal")
        else:
            self.next_button.configure(text="Next →", state="normal")
    
    def prev_step(self):
        """Go to previous step"""
        if self.current_step_index > 0:
            # Save current step (widgets still exist at this point)
            if self.current_step_index < len(self.step_widgets):
                current_widget = self.step_widgets[self.current_step_index]
                if hasattr(current_widget, 'save_config'):
                    try:
                        current_widget.save_config()
                    except Exception:
                        pass  # Ignore errors when going back
            
            self.show_step(self.current_step_index - 1)
    
    def next_step(self):
        """Go to next step"""
        # Validate and save current step (widgets still exist at this point)
        if self.current_step_index < len(self.step_widgets):
            current_widget = self.step_widgets[self.current_step_index]
            if hasattr(current_widget, 'save_config'):
                try:
                    current_widget.save_config()
                except Exception as e:
                    messagebox.showerror("Error", f"Error saving configuration: {e}")
                    return
            if hasattr(current_widget, 'validate'):
                try:
                    is_valid, error = current_widget.validate()
                    if not is_valid:
                        messagebox.showerror("Validation Error", error or "Please fix the errors before proceeding")
                        return
                except Exception as e:
                    messagebox.showerror("Error", f"Validation error: {e}")
                    return
        
        if self.current_step_index < len(self.STEPS) - 1:
            self.show_step(self.current_step_index + 1)
        else:
            # Finish - save configuration
            try:
                config_path = Path(self.config_manager.get('project.base_output_dir', '.')) / 'wizard_config.yaml'
                self.config_manager.save(config_path)
                messagebox.showinfo("Success", f"Configuration saved to {config_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving configuration: {e}")
    
    def on_cancel(self):
        """Handle cancel"""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel? Unsaved changes will be lost."):
            self.destroy()


