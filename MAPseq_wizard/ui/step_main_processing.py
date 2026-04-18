"""
Step 3: Main Processing Configuration UI Component
"""

import customtkinter as ctk
from tkinter import ttk
from ..utils.validation import validate_sample_name, validate_labels, validate_numeric


class MainProcessingStep(ctk.CTkFrame):
    """Main processing configuration step"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Set up the UI components"""
        title = ctk.CTkLabel(self, text="Main Processing Configuration", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Parameterizations tab
        self.param_tab = self.notebook.add("Parameterizations")
        self.setup_parameterizations_tab()
        
        # Samples tab
        self.samples_tab = self.notebook.add("Samples")
        self.setup_samples_tab()
    
    def setup_parameterizations_tab(self):
        """Set up parameterizations management"""
        # Add parameterization button
        ctk.CTkButton(self.param_tab, text="Add Parameterization", command=self.add_parameterization).pack(pady=10)
        
        # Parameterizations list
        frame = ctk.CTkFrame(self.param_tab)
        frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        columns = ('Name', 'i', 'r', 't', 'u', 'Force')
        self.param_table = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.param_table.heading(col, text=col)
            self.param_table.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.param_table.yview)
        self.param_table.configure(yscrollcommand=scrollbar.set)
        
        self.param_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        btn_frame = ctk.CTkFrame(self.param_tab)
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="Edit", command=self.edit_parameterization).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Delete", command=self.delete_parameterization).pack(side="left", padx=5)
    
    def setup_samples_tab(self):
        """Set up samples management"""
        # Age group selector
        age_frame = ctk.CTkFrame(self.samples_tab)
        age_frame.pack(padx=10, pady=10, fill="x")
        ctk.CTkLabel(age_frame, text="Age Group:").pack(side="left", padx=10)
        self.age_group_combo = ctk.CTkComboBox(age_frame, values=['p3', 'p12', 'p20', 'p60'], command=self.on_age_selected)
        self.age_group_combo.pack(side="left", padx=10)
        self.age_group_combo.set('p3')
        
        ctk.CTkButton(age_frame, text="Add Sample", command=self.add_sample).pack(side="right", padx=10)
        
        # Samples table
        frame = ctk.CTkFrame(self.samples_tab)
        frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        columns = ('Name', 'Data File', 'Labels')
        self.samples_table = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.samples_table.heading(col, text=col)
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.samples_table.yview)
        self.samples_table.configure(yscrollcommand=scrollbar.set)
        
        self.samples_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        btn_frame = ctk.CTkFrame(self.samples_tab)
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="Edit", command=self.edit_sample).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Delete", command=self.delete_sample).pack(side="left", padx=5)
    
    def add_parameterization(self):
        """Add a new parameterization"""
        self.show_param_dialog()
    
    def edit_parameterization(self):
        """Edit selected parameterization"""
        selection = self.param_table.selection()
        if not selection:
            return
        # Get parameterization data and show dialog
        self.show_param_dialog()
    
    def delete_parameterization(self):
        """Delete selected parameterization"""
        selection = self.param_table.selection()
        if selection:
            self.param_table.delete(selection[0])
    
    def show_param_dialog(self):
        """Show parameterization dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Parameterization")
        dialog.geometry("400x300")
        
        ctk.CTkLabel(dialog, text="Name:").pack(pady=5)
        name_entry = ctk.CTkEntry(dialog, width=300)
        name_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Injection UMI Min (i):").pack(pady=5)
        i_entry = ctk.CTkEntry(dialog, width=300)
        i_entry.insert(0, "1")
        i_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Min Body-to-Target Ratio (r):").pack(pady=5)
        r_entry = ctk.CTkEntry(dialog, width=300)
        r_entry.insert(0, "1")
        r_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Min Target Count (t):").pack(pady=5)
        t_entry = ctk.CTkEntry(dialog, width=300)
        t_entry.insert(0, "1")
        t_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Target UMI Min (u):").pack(pady=5)
        u_entry = ctk.CTkEntry(dialog, width=300)
        u_entry.insert(0, "2")
        u_entry.pack(pady=5)
        
        force_check = ctk.CTkCheckBox(dialog, text="Force User Threshold")
        force_check.pack(pady=5)
        
        def save():
            # Validate and save
            self.config.add_parameterization(
                name_entry.get(),
                float(i_entry.get()),
                float(r_entry.get()),
                float(t_entry.get()),
                float(u_entry.get()),
                force_check.get()
            )
            self.refresh_parameterizations()
            dialog.destroy()
        
        ctk.CTkButton(dialog, text="Save", command=save).pack(pady=10)
    
    def add_sample(self):
        """Add a new sample"""
        self.show_sample_dialog()
    
    def edit_sample(self):
        """Edit selected sample"""
        selection = self.samples_table.selection()
        if not selection:
            return
        self.show_sample_dialog()
    
    def delete_sample(self):
        """Delete selected sample"""
        selection = self.samples_table.selection()
        if selection:
            self.samples_table.delete(selection[0])
    
    def show_sample_dialog(self):
        """Show sample dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Sample")
        dialog.geometry("500x250")
        
        ctk.CTkLabel(dialog, text="Sample Name:").pack(pady=5)
        name_entry = ctk.CTkEntry(dialog, width=400)
        name_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Data File:").pack(pady=5)
        file_frame = ctk.CTkFrame(dialog)
        file_frame.pack(pady=5, fill="x", padx=50)
        file_entry = ctk.CTkEntry(file_frame, width=300)
        file_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(file_frame, text="Browse", width=100, command=lambda: self.browse_file(file_entry)).pack(side="right")
        
        ctk.CTkLabel(dialog, text="Labels (comma-separated):").pack(pady=5)
        labels_entry = ctk.CTkEntry(dialog, width=400)
        labels_entry.pack(pady=5)
        
        def save():
            age_group = self.age_group_combo.get()
            self.config.add_sample(
                age_group,
                name_entry.get(),
                file_entry.get(),
                labels_entry.get()
            )
            self.refresh_samples()
            dialog.destroy()
        
        ctk.CTkButton(dialog, text="Save", command=save).pack(pady=10)
    
    def browse_file(self, entry):
        """Browse for data file"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(title="Select Data File", filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")])
        if filename:
            entry.delete(0, "end")
            entry.insert(0, filename)
    
    def on_age_selected(self, age_group):
        """Handle age group selection"""
        self.refresh_samples()
    
    def refresh_parameterizations(self):
        """Refresh parameterizations table"""
        for item in self.param_table.get_children():
            self.param_table.delete(item)
        
        params = self.config.get('main_processing.parameterizations', [])
        for param in params:
            self.param_table.insert('', 'end', values=(
                param['name'],
                param['injection_umi_min'],
                param['min_body_to_target_ratio'],
                param['min_target_count'],
                param['target_umi_min'],
                'Yes' if param.get('force_user_threshold') else 'No'
            ))
    
    def refresh_samples(self):
        """Refresh samples table"""
        for item in self.samples_table.get_children():
            self.samples_table.delete(item)
        
        age_group = self.age_group_combo.get()
        age_groups = self.config.get('main_processing.age_groups', {})
        samples = age_groups.get(age_group, {}).get('samples', [])
        
        for sample in samples:
            self.samples_table.insert('', 'end', values=(
                sample['name'],
                sample['data_file'],
                sample['labels']
            ))
    
    def load_config(self):
        """Load configuration"""
        self.refresh_parameterizations()
        self.refresh_samples()
    
    def save_config(self):
        """Save configuration"""
        # Configuration is saved through add/edit methods
        pass
    
    def validate(self):
        """Validate inputs"""
        params = self.config.get('main_processing.parameterizations', [])
        if not params:
            return False, "At least one parameterization is required"
        
        age_groups = self.config.get('main_processing.age_groups', {})
        has_samples = False
        for age_data in age_groups.values():
            if age_data.get('samples'):
                has_samples = True
                break
        
        if not has_samples:
            return False, "At least one sample is required"
        
        return True, None


