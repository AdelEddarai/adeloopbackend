"""
Shell command execution utilities for Jupyter-like kernel
"""

import subprocess
import shlex
import sys
import os

def execute_shell_command(cmd: str) -> int:
    """
    Execute shell command and capture output
    
    Args:
        cmd: Shell command to execute
        
    Returns:
        Exit code of the command
    """
    try:
        # Handle pip commands specifically
        if cmd.startswith('pip '):
            return execute_pip_command(cmd[4:])  # Remove 'pip ' prefix
        
        # Handle Python commands
        if cmd.startswith('python ') or cmd.startswith('python3 '):
            return execute_python_command(cmd)
        
        # Split command into arguments
        try:
            args = shlex.split(cmd)
        except:
            args = cmd.split()
        
        # Execute command
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        
        # Print stdout if available
        if result.stdout:
            print(result.stdout, end='')
        
        # Print stderr if available
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)
        
        # Return exit code
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Shell command timed out after 30 seconds")
        return -1
    except FileNotFoundError:
        print(f"Command not found: {cmd}")
        return -1
    except Exception as e:
        print(f"Error executing shell command: {e}")
        return -1

def execute_pip_command(cmd: str) -> int:
    """
    Execute pip command with proper handling
    
    Args:
        cmd: Pip command to execute (without 'pip')
        
    Returns:
        Exit code of the command
    """
    try:
        # Construct full pip command
        full_cmd = f"pip {cmd}"
        
        # Split command into arguments
        try:
            args = shlex.split(full_cmd)
        except:
            args = full_cmd.split()
        
        # Execute command
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        
        # Print stdout if available
        if result.stdout:
            print(result.stdout, end='')
        
        # Print stderr if available
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)
        
        # Return exit code
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Pip command timed out after 60 seconds")
        return -1
    except Exception as e:
        print(f"Error executing pip command: {e}")
        return -1

def execute_python_command(cmd: str) -> int:
    """
    Execute Python command with proper handling
    
    Args:
        cmd: Python command to execute
        
    Returns:
        Exit code of the command
    """
    try:
        # Split command into arguments
        try:
            args = shlex.split(cmd)
        except:
            args = cmd.split()
        
        # Execute command
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        
        # Print stdout if available
        if result.stdout:
            print(result.stdout, end='')
        
        # Print stderr if available
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)
        
        # Return exit code
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Python command timed out after 60 seconds")
        return -1
    except Exception as e:
        print(f"Error executing Python command: {e}")
        return -1

def list_directory_contents(path: str = '.') -> list:
    """
    List directory contents with details
    
    Args:
        path: Directory path to list (default: current directory)
        
    Returns:
        List of directory contents
    """
    try:
        import os
        import stat
        
        contents = os.listdir(path)
        result = []
        
        for item in contents:
            try:
                full_path = os.path.join(path, item)
                stat_info = os.stat(full_path)
                size = stat_info.st_size
                if os.path.isdir(full_path):
                    item_type = 'DIR'
                    result.append(f"{item}/")
                else:
                    item_type = 'FILE'
                    result.append(f"{item} ({size} bytes)")
            except:
                result.append(f"{item} (inaccessible)")
        
        return result
    except Exception as e:
        print(f"Error listing directory contents: {e}")
        return []

def get_current_directory() -> str:
    """
    Get current working directory
    
    Returns:
        Current working directory path
    """
    try:
        import os
        return os.getcwd()
    except Exception as e:
        print(f"Error getting current directory: {e}")
        return ""