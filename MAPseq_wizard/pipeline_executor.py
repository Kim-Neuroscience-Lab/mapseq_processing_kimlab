"""
Pipeline executor for MAPseq Pipeline Wizard
Executes commands with progress tracking and error handling
"""

import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Callable, Optional
from datetime import datetime
from .utils.path_utils import find_conda_executable


class PipelineExecutor:
    """Executes pipeline commands with progress tracking"""
    
    def __init__(self, conda_env: str = "mapseq_processing", repo_root: Optional[Path] = None):
        """
        Initialize PipelineExecutor
        
        Args:
            conda_env: Name of conda environment to use
            repo_root: Repository root directory
        """
        self.conda_env = conda_env
        self.conda_exe = find_conda_executable()
        self.repo_root = repo_root or Path.cwd()
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        self.current_command = None
        self.command_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.progress_callback: Optional[Callable] = None
        self.log_file: Optional[Path] = None
    
    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def set_log_file(self, log_path: Path) -> None:
        """Set log file path"""
        self.log_file = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def execute_commands(self, commands: List[str]) -> Dict[str, any]:
        """
        Execute a list of commands sequentially
        
        Args:
            commands: List of command strings to execute
            
        Returns:
            Dictionary with execution results
        """
        self.is_running = True
        self.is_paused = False
        self.should_stop = False
        
        results = {
            'total': len(commands),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': [],
            'start_time': datetime.now(),
            'end_time': None,
        }
        
        # Open log file if specified
        log_file_handle = None
        if self.log_file:
            log_file_handle = open(self.log_file, 'w', encoding='utf-8')
            log_file_handle.write(f"Pipeline Execution Log\n")
            log_file_handle.write(f"Started: {results['start_time']}\n")
            log_file_handle.write(f"{'='*80}\n\n")
        
        try:
            for i, command in enumerate(commands):
                if self.should_stop:
                    results['skipped'] = len(commands) - i
                    break
                
                # Wait if paused
                while self.is_paused and not self.should_stop:
                    time.sleep(0.1)
                
                if self.should_stop:
                    results['skipped'] = len(commands) - i
                    break
                
                self.current_command = command
                command_num = i + 1
                
                # Update progress
                if self.progress_callback:
                    self.progress_callback({
                        'command_num': command_num,
                        'total': len(commands),
                        'command': command,
                        'status': 'running',
                    })
                
                # Log command
                if log_file_handle:
                    log_file_handle.write(f"Command {command_num}/{len(commands)}: {command}\n")
                    log_file_handle.write(f"{'-'*80}\n")
                    log_file_handle.flush()
                
                # Execute command
                success, error_msg = self._execute_single_command(command, log_file_handle)
                
                if success:
                    results['completed'] += 1
                    if self.progress_callback:
                        self.progress_callback({
                            'command_num': command_num,
                            'total': len(commands),
                            'command': command,
                            'status': 'success',
                        })
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'command_num': command_num,
                        'command': command,
                        'error': error_msg,
                    })
                    if self.progress_callback:
                        self.progress_callback({
                            'command_num': command_num,
                            'total': len(commands),
                            'command': command,
                            'status': 'failed',
                            'error': error_msg,
                        })
        
        finally:
            self.is_running = False
            self.current_command = None
            results['end_time'] = datetime.now()
            
            if log_file_handle:
                log_file_handle.write(f"\n{'='*80}\n")
                log_file_handle.write(f"Finished: {results['end_time']}\n")
                log_file_handle.write(f"Completed: {results['completed']}, Failed: {results['failed']}, Skipped: {results['skipped']}\n")
                log_file_handle.close()
        
        return results
    
    def _execute_single_command(self, command: str, log_file_handle=None) -> tuple[bool, Optional[str]]:
        """
        Execute a single command
        
        Returns:
            (success, error_message)
        """
        try:
            # Parse command to determine if it needs conda
            cmd_parts = command.split()
            
            # Check if it's a Python script that needs conda
            needs_conda = False
            if len(cmd_parts) > 0 and cmd_parts[0] == 'python':
                needs_conda = True
            
            # Build command
            if needs_conda:
                # Use conda run
                full_cmd = [
                    self.conda_exe, "run", "-n", self.conda_env,
                    "python"
                ] + cmd_parts[1:]  # Everything after 'python'
            else:
                # Execute directly (e.g., mkdir, find, cp)
                full_cmd = cmd_parts
            
            # Execute with shell=False for better cross-platform support
            # But some commands (like find with xargs) need shell=True
            use_shell = any(op in command for op in ['|', '&&', '||', 'xargs'])
            
            process = subprocess.Popen(
                full_cmd if not use_shell else command,
                shell=use_shell,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Stream output
            output_lines = []
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    return False, "Execution stopped by user"
                
                line = line.rstrip()
                output_lines.append(line)
                
                # Log output
                if log_file_handle:
                    log_file_handle.write(f"{line}\n")
                    log_file_handle.flush()
                
                # Send to output queue if callback exists
                if self.progress_callback:
                    self.progress_callback({
                        'output': line,
                    })
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                return True, None
            else:
                error_msg = f"Command failed with exit code {return_code}"
                if output_lines:
                    # Include last few lines of output
                    error_msg += f"\nLast output:\n" + "\n".join(output_lines[-5:])
                return False, error_msg
        
        except FileNotFoundError as e:
            return False, f"Command not found: {e}"
        except Exception as e:
            return False, f"Error executing command: {e}"
    
    def pause(self) -> None:
        """Pause execution"""
        self.is_paused = True
    
    def resume(self) -> None:
        """Resume execution"""
        self.is_paused = False
    
    def stop(self) -> None:
        """Stop execution"""
        self.should_stop = True
        self.is_paused = False
    
    def get_status(self) -> Dict[str, any]:
        """Get current execution status"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_command': self.current_command,
        }


