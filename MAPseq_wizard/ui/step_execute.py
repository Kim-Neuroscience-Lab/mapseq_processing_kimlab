"""
Step 7: Execute Pipeline UI Component
"""

import customtkinter as ctk
from tkinter import scrolledtext
from pathlib import Path
from ..pipeline_executor import PipelineExecutor
from ..command_generator import CommandGenerator
import threading


class ExecuteStep(ctk.CTkFrame):
    """Pipeline execution step"""
    
    def __init__(self, parent, config_manager, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config_manager
        self.executor = PipelineExecutor(repo_root=Path(config_manager.get('project.repo_root', '.')))
        self.executor.set_progress_callback(self.on_progress_update)
        self.is_executing = False
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        title = ctk.CTkLabel(self, text="Execute Pipeline", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=20)
        
        # Execution mode
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(padx=20, pady=10, fill="x")
        
        ctk.CTkLabel(mode_frame, text="Execution Mode:").pack(side="left", padx=10)
        self.execution_mode = ctk.CTkComboBox(mode_frame, values=['All Commands', 'Selected Parameterization', 'Selected Age Group'])
        self.execution_mode.set('All Commands')
        self.execution_mode.pack(side="left", padx=10)
        
        # Progress
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(padx=20, pady=10, fill="x")
        
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready to execute")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=500)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        
        # Output log
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(log_frame, text="Execution Log", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=80, font=("Courier", 10))
        self.log_text.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Control buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(padx=20, pady=10, fill="x")
        
        self.start_button = ctk.CTkButton(button_frame, text="Start Execution", command=self.start_execution, width=150)
        self.start_button.pack(side="left", padx=5)
        
        self.pause_button = ctk.CTkButton(button_frame, text="Pause", command=self.pause_execution, width=100, state="disabled")
        self.pause_button.pack(side="left", padx=5)
        
        self.stop_button = ctk.CTkButton(button_frame, text="Stop", command=self.stop_execution, width=100, state="disabled")
        self.stop_button.pack(side="left", padx=5)
    
    def start_execution(self):
        """Start pipeline execution"""
        if self.is_executing:
            return
        
        # Generate commands
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            commands_file = Path(f.name)
        
        try:
            generator = CommandGenerator(self.config)
            generator.generate_all_commands(commands_file)
            
            # Read commands
            with open(commands_file, 'r') as f:
                commands = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            
            # Set log file
            log_file = Path(self.config.get('project.base_output_dir', '.')) / 'execution.log'
            self.executor.set_log_file(log_file)
            
            # Start execution in thread
            self.is_executing = True
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            
            thread = threading.Thread(target=self._execute_thread, args=(commands,))
            thread.daemon = True
            thread.start()
        
        except Exception as e:
            self.log_text.insert("end", f"Error: {e}\n")
            self.is_executing = False
            self.start_button.configure(state="normal")
        finally:
            if commands_file.exists():
                commands_file.unlink()
    
    def _execute_thread(self, commands):
        """Execute commands in background thread"""
        try:
            results = self.executor.execute_commands(commands)
            self.log_text.insert("end", f"\nExecution completed: {results['completed']} succeeded, {results['failed']} failed\n")
        except Exception as e:
            self.log_text.insert("end", f"Execution error: {e}\n")
        finally:
            self.is_executing = False
            self.start_button.configure(state="normal")
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
    
    def pause_execution(self):
        """Pause execution"""
        if self.executor.is_paused:
            self.executor.resume()
            self.pause_button.configure(text="Pause")
        else:
            self.executor.pause()
            self.pause_button.configure(text="Resume")
    
    def stop_execution(self):
        """Stop execution"""
        self.executor.stop()
        self.log_text.insert("end", "Execution stopped by user\n")
    
    def on_progress_update(self, update):
        """Handle progress updates from executor"""
        if 'command_num' in update:
            total = update.get('total', 1)
            current = update.get('command_num', 0)
            self.progress_bar.set(current / total if total > 0 else 0)
            self.progress_label.configure(text=f"Command {current}/{total}: {update.get('status', '')}")
        
        if 'output' in update:
            self.log_text.insert("end", update['output'] + "\n")
            self.log_text.see("end")
        
        if 'command' in update:
            self.log_text.insert("end", f"\n>>> {update['command']}\n")
            self.log_text.see("end")
    
    def validate(self):
        """Validate before execution"""
        return True, None


