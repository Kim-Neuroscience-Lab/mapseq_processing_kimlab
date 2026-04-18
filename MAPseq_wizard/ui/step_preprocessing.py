"""
Step 2: Preprocessing Configuration UI Component
Includes column mapping interface
"""

import customtkinter as ctk
from pathlib import Path
import tkinter.filedialog as filedialog
from tkinter import ttk, messagebox
from ..utils.header_extractor import extract_headers_from_directory
from ..utils.validation import validate_directory


class PreprocessingStep(ctk.CTkFrame):
    """Preprocessing configuration step with column mapping"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.cohorts = {}  # {cohort_name: {input_dir, output_dir, threshold, headers, mappings, negative_columns}}
        self.current_cohort = None
        self.headers = {}
        self.mappings = {}  # {filename: {original_col: standardized_col}}
        self.negative_columns = {}  # {filename: neg_column_name}
        self.current_file = None
        self.mapping_history = []  # For undo/redo
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Title
        title = ctk.CTkLabel(self, text="Preprocessing Configuration", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Cohort management section
        cohort_frame = ctk.CTkFrame(self)
        cohort_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(cohort_frame, text="Cohorts", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # Cohort selector and management
        cohort_control_frame = ctk.CTkFrame(cohort_frame)
        cohort_control_frame.pack(padx=10, pady=5, fill="x")
        
        ctk.CTkLabel(cohort_control_frame, text="Cohort:").pack(side="left", padx=10)
        self.cohort_selector = ctk.CTkComboBox(cohort_control_frame, values=[], command=self.on_cohort_selected, width=200)
        self.cohort_selector.pack(side="left", padx=10)
        
        ctk.CTkButton(cohort_control_frame, text="Add Cohort", width=120, command=self.add_cohort).pack(side="left", padx=5)
        ctk.CTkButton(cohort_control_frame, text="Delete Cohort", width=120, command=self.delete_cohort).pack(side="left", padx=5)
        
        # Input/Output directories for selected cohort
        dir_frame = ctk.CTkFrame(cohort_frame)
        dir_frame.pack(padx=10, pady=10, fill="x")
        
        # Input directory
        ctk.CTkLabel(dir_frame, text="Input Directory (containing .tsv files for this cohort):").pack(anchor="w", padx=10, pady=(10, 5))
        input_frame = ctk.CTkFrame(dir_frame)
        input_frame.pack(padx=10, pady=(0, 10), fill="x")
        self.input_dir_entry = ctk.CTkEntry(input_frame, width=350)
        self.input_dir_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(input_frame, text="Browse", width=100, command=self.browse_input_dir).pack(side="right")
        ctk.CTkButton(input_frame, text="Load Headers", width=100, command=self.load_headers).pack(side="right", padx=(0, 10))
        
        # Output directory
        ctk.CTkLabel(dir_frame, text="Output Directory (for this cohort):").pack(anchor="w", padx=10, pady=(10, 5))
        output_frame = ctk.CTkFrame(dir_frame)
        output_frame.pack(padx=10, pady=(0, 10), fill="x")
        self.output_dir_entry = ctk.CTkEntry(output_frame, width=350)
        self.output_dir_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", width=100, command=self.browse_output_dir).pack(side="right")
        
        # Fallback threshold
        ctk.CTkLabel(dir_frame, text="Fallback Threshold:").pack(anchor="w", padx=10, pady=(10, 5))
        self.threshold_entry = ctk.CTkEntry(dir_frame, width=100)
        self.threshold_entry.insert(0, "2.0")
        self.threshold_entry.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Column mapping section
        mapping_frame = ctk.CTkFrame(self)
        mapping_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(mapping_frame, text="Column Mapping", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        # File selector
        file_frame = ctk.CTkFrame(mapping_frame)
        file_frame.pack(padx=10, pady=5, fill="x")
        ctk.CTkLabel(file_frame, text="File:").pack(side="left", padx=10)
        self.file_selector = ctk.CTkComboBox(file_frame, values=[], command=self.on_file_selected, width=300)
        self.file_selector.pack(side="left", padx=10, fill="x", expand=True)
        
        # Mapping table
        table_frame = ctk.CTkFrame(mapping_frame)
        table_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Create treeview for mapping table
        columns = ('Original', 'Standardized')
        self.mapping_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        self.mapping_table.heading('Original', text='Original Column Name')
        self.mapping_table.heading('Standardized', text='Standardized Name')
        self.mapping_table.column('Original', width=250)
        self.mapping_table.column('Standardized', width=250)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.mapping_table.yview)
        self.mapping_table.configure(yscrollcommand=scrollbar.set)
        
        self.mapping_table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind double-click to edit
        self.mapping_table.bind('<Double-1>', self.on_cell_edit)
        
        # Standardized name suggestions
        self.standardized_options = ['barcodes', 'neg', 'inj', 'rsp', 'pm', 'am', 'al', 'lm', 'a', 'rl']
        
        # Negative column selector
        neg_frame = ctk.CTkFrame(mapping_frame)
        neg_frame.pack(padx=10, pady=5, fill="x")
        ctk.CTkLabel(neg_frame, text="Negative Control Column:").pack(side="left", padx=10)
        self.neg_column_combo = ctk.CTkComboBox(neg_frame, values=[], width=200)
        self.neg_column_combo.pack(side="left", padx=10)
        
        # Buttons
        button_frame = ctk.CTkFrame(mapping_frame)
        button_frame.pack(padx=10, pady=10, fill="x")
        ctk.CTkButton(button_frame, text="Undo", width=100, command=self.undo_mapping).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Clear All", width=100, command=self.clear_mappings).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Copy to Other Files", width=150, command=self.copy_mappings).pack(side="left", padx=5)
    
    def add_cohort(self):
        """Add a new cohort"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add Cohort")
        dialog.geometry("400x150")
        
        ctk.CTkLabel(dialog, text="Cohort Name:").pack(pady=10)
        name_entry = ctk.CTkEntry(dialog, width=300)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def save_cohort():
            cohort_name = name_entry.get().strip()
            if not cohort_name:
                messagebox.showerror("Error", "Cohort name is required")
                return
            
            if cohort_name in self.cohorts:
                messagebox.showerror("Error", f"Cohort '{cohort_name}' already exists")
                return
            
            # Initialize cohort data
            self.cohorts[cohort_name] = {
                'input_dir': '',
                'output_dir': '',
                'fallback_threshold': 2.0,
                'headers': {},
                'mappings': {},
                'negative_columns': {},
            }
            
            self.update_cohort_selector()
            self.cohort_selector.set(cohort_name)
            self.on_cohort_selected(cohort_name)
            dialog.destroy()
        
        ctk.CTkButton(dialog, text="Add", command=save_cohort).pack(pady=10)
        name_entry.bind('<Return>', lambda e: save_cohort())
    
    def delete_cohort(self):
        """Delete current cohort"""
        if not self.current_cohort:
            messagebox.showwarning("Warning", "No cohort selected")
            return
        
        if messagebox.askyesno("Confirm", f"Delete cohort '{self.current_cohort}'? This will remove all its mappings."):
            del self.cohorts[self.current_cohort]
            self.current_cohort = None
            self.update_cohort_selector()
            if self.cohort_selector.cget("values"):
                self.cohort_selector.set(self.cohort_selector.cget("values")[0])
                self.on_cohort_selected(self.cohort_selector.get())
            else:
                # Clear UI
                self.input_dir_entry.delete(0, "end")
                self.output_dir_entry.delete(0, "end")
                self.threshold_entry.delete(0, "end")
                self.threshold_entry.insert(0, "2.0")
                self.headers = {}
                self.mappings = {}
                self.negative_columns = {}
                self.update_mapping_table()
    
    def update_cohort_selector(self):
        """Update cohort selector dropdown"""
        cohort_names = list(self.cohorts.keys())
        self.cohort_selector.configure(values=cohort_names)
    
    def on_cohort_selected(self, cohort_name):
        """Handle cohort selection"""
        if not cohort_name or cohort_name not in self.cohorts:
            return
        
        # Save current cohort data before switching
        if self.current_cohort and self.current_cohort in self.cohorts:
            self.save_current_cohort_data()
        
        # Load new cohort data
        self.current_cohort = cohort_name
        cohort_data = self.cohorts[cohort_name]
        
        self.input_dir_entry.delete(0, "end")
        self.input_dir_entry.insert(0, cohort_data.get('input_dir', ''))
        
        self.output_dir_entry.delete(0, "end")
        self.output_dir_entry.insert(0, cohort_data.get('output_dir', ''))
        
        self.threshold_entry.delete(0, "end")
        self.threshold_entry.insert(0, str(cohort_data.get('fallback_threshold', 2.0)))
        
        self.headers = cohort_data.get('headers', {}).copy()
        self.mappings = cohort_data.get('mappings', {}).copy()
        self.negative_columns = cohort_data.get('negative_columns', {}).copy()
        
        # Update file selector
        if self.headers:
            self.file_selector.configure(values=list(self.headers.keys()))
            self.file_selector.set(list(self.headers.keys())[0])
            self.on_file_selected(list(self.headers.keys())[0])
        else:
            self.file_selector.configure(values=[])
            self.update_mapping_table()
    
    def save_current_cohort_data(self):
        """Save current cohort's data"""
        if not self.current_cohort or self.current_cohort not in self.cohorts:
            return
        
        self.cohorts[self.current_cohort]['input_dir'] = self.input_dir_entry.get()
        self.cohorts[self.current_cohort]['output_dir'] = self.output_dir_entry.get()
        try:
            self.cohorts[self.current_cohort]['fallback_threshold'] = float(self.threshold_entry.get())
        except ValueError:
            pass
        
        # Save headers, mappings, and negative columns
        self.cohorts[self.current_cohort]['headers'] = self.headers.copy()
        self.cohorts[self.current_cohort]['mappings'] = self.mappings.copy()
        self.cohorts[self.current_cohort]['negative_columns'] = self.negative_columns.copy()
    
    def browse_input_dir(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir_entry.delete(0, "end")
            self.input_dir_entry.insert(0, directory)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_entry.delete(0, "end")
            self.output_dir_entry.insert(0, directory)
    
    def load_headers(self):
        """Load headers from input directory"""
        if not self.current_cohort:
            messagebox.showwarning("Warning", "Please select or create a cohort first")
            return
        
        input_dir = self.input_dir_entry.get()
        if not input_dir:
            messagebox.showwarning("Warning", "Please specify input directory first")
            return
        
        is_valid, error = validate_directory(input_dir)
        if not is_valid:
            messagebox.showerror("Error", error)
            return
        
        try:
            self.headers = extract_headers_from_directory(Path(input_dir))
            if not self.headers:
                messagebox.showinfo("Info", "No TSV files found in input directory")
                return
            
            # Save headers to current cohort
            if self.current_cohort:
                self.cohorts[self.current_cohort]['headers'] = self.headers.copy()
            
            # Update file selector
            self.file_selector.configure(values=list(self.headers.keys()))
            if self.headers:
                self.file_selector.set(list(self.headers.keys())[0])
                self.on_file_selected(list(self.headers.keys())[0])
            
            messagebox.showinfo("Success", f"Loaded headers from {len(self.headers)} file(s)")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading headers: {e}")
    
    def on_file_selected(self, filename):
        """Handle file selection"""
        if not filename or filename not in self.headers:
            return
        
        self.current_file = filename
        self.update_mapping_table()
        self.update_neg_column_combo()
    
    def update_mapping_table(self):
        """Update the mapping table for current file"""
        # Clear table
        for item in self.mapping_table.get_children():
            self.mapping_table.delete(item)
        
        if not self.current_file or self.current_file not in self.headers:
            return
        
        # Get headers for current file
        headers = self.headers[self.current_file]
        
        # Get existing mappings for this file
        file_mappings = self.mappings.get(self.current_file, {})
        
        # Populate table
        for header in headers:
            standardized = file_mappings.get(header, "")
            self.mapping_table.insert('', 'end', values=(header, standardized))
    
    def update_neg_column_combo(self):
        """Update negative column combo box"""
        if not self.current_file:
            return
        
        # Get standardized columns for current file
        file_mappings = self.mappings.get(self.current_file, {})
        standardized_cols = list(set(file_mappings.values()))
        
        self.neg_column_combo.configure(values=standardized_cols)
        
        # Set current value
        if self.current_file in self.negative_columns:
            self.neg_column_combo.set(self.negative_columns[self.current_file])
    
    def on_cell_edit(self, event):
        """Handle cell edit (double-click)"""
        selection = self.mapping_table.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.mapping_table.item(item, 'values')
        original_col = values[0]
        
        # Create dialog for editing
        dialog = ctk.CTkToplevel(self)
        dialog.title("Edit Standardized Name")
        dialog.geometry("400x150")
        
        ctk.CTkLabel(dialog, text=f"Original Column: {original_col}").pack(pady=10)
        entry = ctk.CTkEntry(dialog, width=300)
        entry.pack(pady=10)
        entry.insert(0, values[1] if len(values) > 1 else "")
        
        def save_mapping():
            new_value = entry.get().strip()
            self.save_state()
            if self.current_file:
                if self.current_file not in self.mappings:
                    self.mappings[self.current_file] = {}
                self.mappings[self.current_file][original_col] = new_value
                self.update_mapping_table()
                self.update_neg_column_combo()
            dialog.destroy()
        
        ctk.CTkButton(dialog, text="Save", command=save_mapping).pack(pady=10)
        entry.focus()
        entry.bind('<Return>', lambda e: save_mapping())
    
    def save_state(self):
        """Save current state for undo"""
        state = {
            'mappings': {k: v.copy() for k, v in self.mappings.items()},
            'negative_columns': self.negative_columns.copy(),
        }
        self.mapping_history.append(state)
        # Keep only last 50 states
        if len(self.mapping_history) > 50:
            self.mapping_history.pop(0)
    
    def undo_mapping(self):
        """Undo last mapping change"""
        if not self.mapping_history:
            return
        
        state = self.mapping_history.pop()
        self.mappings = state['mappings']
        self.negative_columns = state['negative_columns']
        self.update_mapping_table()
        self.update_neg_column_combo()
    
    def clear_mappings(self):
        """Clear all mappings for current file"""
        if not self.current_file:
            return
        
        self.save_state()
        if self.current_file in self.mappings:
            del self.mappings[self.current_file]
        if self.current_file in self.negative_columns:
            del self.negative_columns[self.current_file]
        self.update_mapping_table()
        self.update_neg_column_combo()
    
    def copy_mappings(self):
        """Copy mappings from current file to other files"""
        if not self.current_file or self.current_file not in self.mappings:
            messagebox.showinfo("Info", "No mappings to copy")
            return
        
        source_mappings = self.mappings[self.current_file]
        
        # Create dialog to select target files
        dialog = ctk.CTkToplevel(self)
        dialog.title("Copy Mappings")
        dialog.geometry("300x400")
        
        ctk.CTkLabel(dialog, text="Select files to copy to:").pack(pady=10)
        
        checkboxes = {}
        for filename in self.headers.keys():
            if filename != self.current_file:
                var = ctk.BooleanVar()
                cb = ctk.CTkCheckBox(dialog, text=filename, variable=var)
                cb.pack(anchor="w", padx=20, pady=5)
                checkboxes[filename] = var
        
        def copy():
            self.save_state()
            for filename, var in checkboxes.items():
                if var.get():
                    self.mappings[filename] = source_mappings.copy()
            dialog.destroy()
            messagebox.showinfo("Success", "Mappings copied")
        
        ctk.CTkButton(dialog, text="Copy", command=copy).pack(pady=10)
    
    def load_config(self):
        """Load configuration values"""
        # Save current cohort before loading
        if self.current_cohort:
            self.save_current_cohort_data()
        
        # Load cohorts from config
        cohorts_config = self.config.get('preprocessing.cohorts', {})
        
        if cohorts_config:
            # New format: multiple cohorts
            self.cohorts = {}
            for cohort_name, cohort_data in cohorts_config.items():
                self.cohorts[cohort_name] = {
                    'input_dir': cohort_data.get('input_dir', ''),
                    'output_dir': cohort_data.get('output_dir', ''),
                    'fallback_threshold': cohort_data.get('fallback_threshold', 2.0),
                    'headers': cohort_data.get('headers', {}),
                    'mappings': cohort_data.get('column_mappings', {}),
                    'negative_columns': cohort_data.get('negative_columns', {}),
                }
        else:
            # Legacy format: single preprocessing config
            input_dir = self.config.get('preprocessing.input_dir', '')
            output_dir = self.config.get('preprocessing.output_dir', '')
            threshold = self.config.get('preprocessing.fallback_threshold', 2.0)
            column_mappings = self.config.get('preprocessing.column_mappings', {})
            negative_columns = self.config.get('preprocessing.negative_columns', {})
            
            if input_dir:
                # Convert to cohort format
                self.cohorts['Cohort_1'] = {
                    'input_dir': input_dir,
                    'output_dir': output_dir,
                    'fallback_threshold': threshold,
                    'headers': {},
                    'mappings': column_mappings,
                    'negative_columns': negative_columns,
                }
        
        self.update_cohort_selector()
        if self.cohorts:
            first_cohort = list(self.cohorts.keys())[0]
            self.cohort_selector.set(first_cohort)
            self.on_cohort_selected(first_cohort)
    
    def save_config(self):
        """Save configuration values"""
        # Save negative column from combo to current cohort
        if self.current_file and self.neg_column_combo.get():
            self.negative_columns[self.current_file] = self.neg_column_combo.get()
        
        # Save current cohort data (includes saving mappings and headers)
        if self.current_cohort:
            self.save_current_cohort_data()
        
        # Save all cohorts to config
        cohorts_config = {}
        for cohort_name, cohort_data in self.cohorts.items():
            cohorts_config[cohort_name] = {
                'input_dir': cohort_data.get('input_dir', ''),
                'output_dir': cohort_data.get('output_dir', ''),
                'fallback_threshold': cohort_data.get('fallback_threshold', 2.0),
                'headers': cohort_data.get('headers', {}),
                'column_mappings': cohort_data.get('mappings', {}),
                'negative_columns': cohort_data.get('negative_columns', {}),
            }
        
        self.config.set('preprocessing.cohorts', cohorts_config)
        
        # Also save legacy format for backward compatibility (use first cohort)
        if self.cohorts:
            first_cohort = list(self.cohorts.values())[0]
            self.config.set('preprocessing.input_dir', first_cohort['input_dir'])
            self.config.set('preprocessing.output_dir', first_cohort['output_dir'])
            self.config.set('preprocessing.fallback_threshold', first_cohort['fallback_threshold'])
            self.config.set('preprocessing.column_mappings', first_cohort['mappings'])
            self.config.set('preprocessing.negative_columns', first_cohort['negative_columns'])
    
    def validate(self):
        """Validate inputs"""
        if not self.cohorts:
            return False, "At least one cohort is required"
        
        # Validate each cohort
        for cohort_name, cohort_data in self.cohorts.items():
            if not cohort_data.get('input_dir'):
                return False, f"Cohort '{cohort_name}': Input directory is required"
            if not cohort_data.get('output_dir'):
                return False, f"Cohort '{cohort_name}': Output directory is required"
            
            # Check that at least one file has 'barcodes' mapping
            mappings = cohort_data.get('mappings', {})
            has_barcodes = False
            for filename, file_mappings in mappings.items():
                if isinstance(file_mappings, dict) and 'barcodes' in file_mappings.values():
                    has_barcodes = True
                    break
            
            if not has_barcodes and mappings:
                return False, f"Cohort '{cohort_name}': At least one file must have a column mapped to 'barcodes'"
        
        return True, None


