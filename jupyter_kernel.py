"""
Jupyter-like kernel backend for code execution
Provides a clean interface similar to Jupyter notebooks
"""

import json
import sys
import io
import traceback
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
import warnings
import contextlib
import mimetypes
import tempfile
import os

# Optional imports with fallbacks
try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Custom exception for input handling
class InputRequiredException(Exception):
    def __init__(self, prompt=""):
        self.prompt = prompt
        super().__init__(f"Input required: {prompt}")

class JupyterKernel:
    """
    A Jupyter-like kernel for executing Python code with proper output handling
    """
    
    def __init__(self):
        self.namespace = {}
        self.execution_count = 0
        self.setup_namespace()
    
    def setup_namespace(self):
        """Setup the execution namespace with common imports and utilities"""
        # Initialize display outputs
        self._display_outputs = []

        # Core libraries
        self.namespace.update({
            'pd': pd,
            'np': np,
            'plt': plt,
            'json': json,
            'sys': sys,
            'io': io,
            'warnings': warnings,
            # Add other common imports
            'os': __import__('os'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
            'math': __import__('math'),
            'random': __import__('random'),
            'base64': base64,
            # Streamlit support
            'streamlit': None,  # Will be set by main.py when needed
            'st': None,  # Will be set by main.py when needed
        })

        # Add optional libraries if available
        if requests:
            self.namespace['requests'] = requests
        if Image:
            self.namespace['Image'] = Image

        # Add display functions
        def display(*args):
            """Jupyter-like display function for rich outputs"""
            for arg in args:
                media_output = self.display_media(arg)
                if media_output:
                    self._display_outputs.append(media_output)
                else:
                    # Fallback to string representation
                    print(arg)

        def display_image(image_data, format='PNG'):
            """Display image data"""
            if isinstance(image_data, str) and image_data.startswith(('http', 'data:')):
                self._display_outputs.append(image_data)
            else:
                media_output = self.display_media(image_data)
                if media_output:
                    self._display_outputs.append(media_output)

        def display_video(video_path_or_url):
            """Display video"""
            if isinstance(video_path_or_url, str):
                if video_path_or_url.startswith(('http', 'data:')):
                    self._display_outputs.append(video_path_or_url)
                else:
                    media_output = self.display_media(video_path_or_url)
                    if media_output:
                        self._display_outputs.append(media_output)

        # Add display functions to namespace
        self.namespace.update({
            'display': display,
            'display_image': display_image,
            'display_video': display_video,
        })

        # Setup matplotlib for non-interactive backend
        plt.switch_backend('Agg')

        # Override plt.show() to capture plots automatically
        original_show = plt.show
        def custom_show(*args, **kwargs):
            # Don't actually show (since we're in non-interactive mode)
            # The plots will be captured later by capture_plots()
            pass

        plt.show = custom_show
        self.namespace['plt'].show = custom_show

        # Setup interactive input support
        self._pending_input = None
        self._input_prompt = None

        # Initialize input queue for handling user input
        self._user_input_queue = []

        # Custom input function that handles queued input
        def custom_input(prompt=""):
            # Print the prompt
            if prompt:
                print(prompt, end="", flush=True)

            # Check if we have queued input
            if hasattr(self, '_user_input_queue') and self._user_input_queue:
                user_input = self._user_input_queue.pop(0)
                print(user_input)  # Echo the input
                return user_input

            # If no queued input, signal that we need input
            raise InputRequiredException(prompt)

        # Replace input function
        self.namespace['input'] = custom_input

        # Setup Plotly to capture plots
        self._plotly_figures = []

        # Try to import and setup Plotly (suppress ALL output)
        old_stdout = None
        old_stderr = None
        try:
            # Temporarily redirect both stdout and stderr to suppress ALL Plotly output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # Suppress warnings during import
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import plotly.graph_objects as go
                import plotly.express as px
                import plotly.io as pio

            # Restore stdout and stderr immediately after import
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            import plotly.offline as pyo

            # Set Plotly to not show plots in browser and handle missing ipython
            try:
                pio.renderers.default = "json"
            except Exception:
                # Fallback if ipython is not available
                pio.renderers.default = "browser"

            # Disable auto-opening in browser
            try:
                pyo.init_notebook_mode(connected=False)
            except Exception:
                # Handle case where ipython is not available
                pass

            # Store original show method
            self._original_plotly_show = go.Figure.show
            kernel_instance = self  # Capture reference to kernel instance

            def custom_plotly_show(fig_self, *args, **kwargs):
                try:
                    # Generate HTML output that can be displayed directly
                    fig_html = fig_self.to_html(
                        include_plotlyjs='inline',
                        div_id=f"plotly-div-{len(kernel_instance._plotly_figures)}",
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                            'responsive': True
                        }
                    )
                    kernel_instance._plotly_figures.append(fig_html)

                    # Debug info
                    title = fig_self.layout.title.text if fig_self.layout.title else 'Untitled Plot'
                    print(f"📊 Plotly figure captured as HTML: {title}")

                except Exception as e:
                    print(f"❌ Failed to capture Plotly figure: {str(e)}")
                    pass
                # Don't actually show in browser
                pass

            # Apply the override
            go.Figure.show = custom_plotly_show

            # Add Plotly to namespace
            self.namespace.update({
                'go': go,
                'px': px,
                'pio': pio,
                'plotly': __import__('plotly')
            })

            # Plotly setup completed silently

        except ImportError:
            # Add placeholder functions so code doesn't break
            self.namespace.update({
                'go': None,
                'px': None,
                'pio': None,
                'plotly': None
            })
        finally:
            # Always restore stdout and stderr if they were redirected
            if old_stdout is not None:
                sys.stdout = old_stdout
            if old_stderr is not None:
                sys.stderr = old_stderr
        
    def capture_plots(self) -> List[str]:
        """Capture all matplotlib plots as base64 images"""
        plots = []

        # Get all figure numbers
        fig_nums = plt.get_fignums()

        for fig_num in fig_nums:
            try:
                fig = plt.figure(fig_num)

                # Save figure to buffer
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)

                # Encode as base64
                image_data = buffer.getvalue()
                if len(image_data) > 0:
                    encoded_image = base64.b64encode(image_data).decode()
                    plots.append(f"data:image/png;base64,{encoded_image}")

                buffer.close()
            except Exception as e:
                print(f"Error capturing plot {fig_num}: {e}")

        # Close all figures
        plt.close('all')

        return plots

    def capture_plotly_figures(self) -> List[str]:
        """Capture all Plotly figures as HTML for direct display"""
        plotly_plots = []

        if hasattr(self, '_plotly_figures'):
            for fig_html in self._plotly_figures:
                try:
                    # The figure is already HTML, just add it to the plots
                    if isinstance(fig_html, str) and '<div' in fig_html:
                        plotly_plots.append(fig_html)
                        print(f"✅ Plotly HTML figure ready for frontend")
                    else:
                        print(f"⚠️ Plotly figure is not valid HTML: {type(fig_html)}")

                except Exception as e:
                    print(f"❌ Error processing Plotly figure: {str(e)}")
                    pass

            # Clear the figures for next execution
            self._plotly_figures = []

        return plotly_plots

    def provide_input_and_continue(self, user_input: str, original_code: str):
        """Provide user input and continue execution"""
        # Store the user input in a way that the input() function can access it
        self._user_input_queue = [user_input]

        # Redefine input function to use the provided input
        def input_with_provided_value(prompt=""):
            if self._user_input_queue:
                value = self._user_input_queue.pop(0)
                print(f"{prompt}{value}")
                return value
            return ""

        # Temporarily replace input function
        original_input = self.namespace['input']
        self.namespace['input'] = input_with_provided_value

        try:
            # Re-execute the code with the provided input
            result = self.execute_code(original_code)
            return result
        finally:
            # Restore original input function
            self.namespace['input'] = original_input

    def display_media(self, data: Any) -> Optional[str]:
        """Handle display of various media types (images, videos, etc.)"""
        try:
            # Handle PIL Images (if PIL is available)
            if Image and hasattr(data, 'save') and hasattr(data, 'format'):
                buffer = BytesIO()
                format_type = data.format or 'PNG'
                data.save(buffer, format=format_type)
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue()).decode()
                mime_type = f"image/{format_type.lower()}"
                return f"data:{mime_type};base64,{encoded}"

            # Handle numpy arrays as images (if PIL is available)
            elif Image and isinstance(data, np.ndarray) and len(data.shape) in [2, 3]:
                # Convert numpy array to PIL Image
                if len(data.shape) == 2:  # Grayscale
                    img = Image.fromarray((data * 255).astype(np.uint8), mode='L')
                elif data.shape[2] == 3:  # RGB
                    img = Image.fromarray((data * 255).astype(np.uint8), mode='RGB')
                elif data.shape[2] == 4:  # RGBA
                    img = Image.fromarray((data * 255).astype(np.uint8), mode='RGBA')
                else:
                    return None

                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{encoded}"

            # Handle file paths
            elif isinstance(data, str):
                if os.path.isfile(data):
                    mime_type, _ = mimetypes.guess_type(data)
                    if mime_type and mime_type.startswith(('image/', 'video/', 'audio/')):
                        with open(data, 'rb') as f:
                            encoded = base64.b64encode(f.read()).decode()
                            return f"data:{mime_type};base64,{encoded}"

                # Handle URLs (if requests is available)
                elif requests and data.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(data, timeout=10)
                        if response.status_code == 200:
                            content_type = response.headers.get('content-type', '')
                            if content_type.startswith(('image/', 'video/', 'audio/')):
                                encoded = base64.b64encode(response.content).decode()
                                return f"data:{content_type};base64,{encoded}"
                    except:
                        pass

            return None
        except Exception as e:
            print(f"Error displaying media: {e}")
            return None

    def capture_display_calls(self) -> List[str]:
        """Capture any display() calls or rich outputs"""
        # This will be populated by display() function calls
        return getattr(self, '_display_outputs', [])
    
    def format_dataframe_html(self, df: pd.DataFrame, max_rows: int = 100) -> str:
        """Format DataFrame as HTML table"""
        if len(df) > max_rows:
            # Show first and last rows with truncation indicator
            top_rows = df.head(max_rows // 2)
            bottom_rows = df.tail(max_rows // 2)
            
            html_parts = []
            html_parts.append(top_rows.to_html(classes='dataframe', table_id='dataframe'))
            html_parts.append(f'<div class="truncation-indicator">... {len(df) - max_rows} more rows ...</div>')
            html_parts.append(bottom_rows.to_html(classes='dataframe', table_id='dataframe'))
            
            return ''.join(html_parts)
        else:
            return df.to_html(classes='dataframe', table_id='dataframe')
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to handle Streamlit commands and other issues"""
        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip or modify Streamlit commands
            if (line_stripped.startswith('st.') or
                'streamlit' in line_stripped.lower() or
                line_stripped.startswith('import streamlit') or
                line_stripped.startswith('from streamlit') or
                '.st.' in line_stripped or
                'st(' in line_stripped):
                # Comment out Streamlit commands
                processed_lines.append(f"# {line}  # Streamlit command - not supported in this context")
                continue

            # Handle _repr_html_ calls that might cause issues
            if '_repr_html_()' in line_stripped:
                processed_lines.append(f"# {line}  # _repr_html_ call - handled separately")
                continue

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    def execute_code(self, code: str, datasets: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute Python code and return Jupyter-like results
        
        Returns:
        {
            'execution_count': int,
            'status': 'ok' | 'error',
            'stdout': str,
            'stderr': str,
            'result': Any,  # Last expression result
            'plots': List[str],  # Base64 encoded images
            'html_outputs': List[str],  # HTML outputs (tables, etc.)
            'error': Optional[Dict],  # Error information
            'variables': Dict[str, Any],  # Updated namespace variables
        }
        """
        
        self.execution_count += 1

        # Preprocess code to handle Streamlit and other issues
        processed_code = self._preprocess_code(code)
        
        # Add datasets to namespace
        if datasets:
            for i, dataset in enumerate(datasets):
                if dataset and 'data' in dataset:
                    df_data = pd.DataFrame(dataset['data'])
                    dataset_name = dataset.get('name', f'Dataset {i+1}')
                    
                    # Create safe variable name
                    safe_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
                    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                    if not safe_name or safe_name[0].isdigit():
                        safe_name = f'dataset_{safe_name}' if safe_name else f'dataset_{i+1}'
                    
                    # Store datasets with multiple names for flexibility
                    self.namespace[safe_name] = df_data
                    self.namespace[f'df{i+1}'] = df_data
                    
                    if i == 0:
                        self.namespace['df'] = df_data
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = None
        error_info = None
        html_outputs = []
        
        try:
            # Redirect stdout and stderr
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Execute the entire code block as-is (like Jupyter)
                # This handles print statements, loops, functions, etc. naturally
                exec(code, self.namespace)

                # Try to get the result of the last expression if it exists
                code_lines = [line.strip() for line in code.strip().split('\n') if line.strip()]
                if code_lines:
                    last_line = code_lines[-1]

                    # Check if the last line is an expression (not a statement)
                    try:
                        # Compile to see if it's an expression
                        compile(last_line, '<string>', 'eval')
                        # If successful, evaluate it to get the result
                        result = eval(last_line, self.namespace)

                        # Handle special result types
                        if isinstance(result, pd.DataFrame):
                            html_outputs.append(self.format_dataframe_html(result))
                        elif hasattr(result, '_repr_html_'):
                            try:
                                html_outputs.append(result._repr_html_())
                            except Exception as e:
                                # Handle Streamlit objects that don't work in this context
                                if 'streamlit' in str(type(result)).lower() or '_repr_html_' in str(e):
                                    # Skip Streamlit objects - they need to run in Streamlit context
                                    pass
                                else:
                                    html_outputs.append(f"<div>Error rendering HTML: {str(e)}</div>")
                        # Don't show None results
                        elif result is None:
                            result = None

                    except (SyntaxError, TypeError):
                        # Last line is a statement, not an expression
                        result = None

        except InputRequiredException as e:
            # Handle input request
            stdout_text = stdout_capture.getvalue()
            return {
                'execution_count': self.execution_count,
                'status': 'input_required',
                'stdout': stdout_text,
                'stderr': '',
                'input_prompt': e.prompt,
                'needs_input': True,
                'plots': [],
                'html_outputs': [],
                'error': None,
                'variables': {},
                'data': None,
                'media_count': 0,
                'plot_count': 0,
                'display_count': 0,
                'plotly_count': 0
            }

        except Exception as e:
            error_info = {
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': traceback.format_exc().split('\n')
            }
        
        # Capture plots and display outputs
        plots = self.capture_plots()
        display_outputs = self.capture_display_calls()
        plotly_figures = self.capture_plotly_figures()

        # Combine all media outputs including Plotly figures
        all_media = plots + display_outputs + plotly_figures

        # Clear display outputs for next execution
        self._display_outputs = []

        # Handle Plotly figures first - set as result if we have them
        plotly_result = None
        if plotly_figures:
            try:
                # Take the first Plotly figure and parse it
                import json
                plotly_data = json.loads(plotly_figures[0])
                # Check if it's already wrapped
                if 'application/vnd.plotly.v1+json' in plotly_data:
                    plotly_result = plotly_data
                else:
                    plotly_result = {
                        'application/vnd.plotly.v1+json': plotly_data
                    }
                # Plotly result prepared silently
            except Exception:
                # Silently ignore Plotly parsing errors
                pass

        # Get stdout and stderr
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        # Combine stdout with any expression results for complete output
        combined_output = stdout_text
        if result is not None and not plotly_result:
            # Add the result to the output if it's not already printed
            result_str = ""
            if isinstance(result, (str, int, float, bool)):
                result_str = str(result)
            elif isinstance(result, (list, dict, tuple)):
                result_str = repr(result)
            else:
                result_str = str(result)

            # Only add if it's not already in stdout
            if result_str and result_str not in stdout_text:
                combined_output += f"\n{result_str}" if combined_output else result_str
        
        # Extract user-defined variables (exclude built-ins and modules)
        user_variables = {}
        for name, value in self.namespace.items():
            if (not name.startswith('_') and 
                not callable(value) and 
                not hasattr(value, '__module__') and
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random']):
                try:
                    # Only include serializable variables
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        user_variables[name] = value
                    elif isinstance(value, pd.DataFrame):
                        user_variables[name] = f"DataFrame({value.shape[0]} rows, {value.shape[1]} cols)"
                    elif isinstance(value, np.ndarray):
                        user_variables[name] = f"Array{value.shape}"
                    else:
                        user_variables[name] = str(type(value).__name__)
                except:
                    user_variables[name] = "Object"
        


        return {
            'execution_count': self.execution_count,
            'status': 'error' if error_info else 'ok',
            'stdout': combined_output,  # Use combined output for better display
            'stderr': stderr_text,
            'result': plotly_result if plotly_result else result,
            'plots': all_media,  # Include both plots and display outputs
            'html_outputs': html_outputs,
            'error': error_info,
            'variables': user_variables,
            'data': result.to_dict('records') if isinstance(result, pd.DataFrame) else None,
            'media_count': len(all_media),
            'plot_count': len(plots),
            'display_count': len(display_outputs),
            'plotly_count': len(plotly_figures)
        }
    
    def get_namespace_info(self) -> Dict[str, Any]:
        """Get information about current namespace variables"""
        variables = {}
        for name, value in self.namespace.items():
            if not name.startswith('_') and not callable(value):
                variables[name] = {
                    'type': type(value).__name__,
                    'value': str(value) if len(str(value)) < 100 else f"{str(value)[:100]}...",
                    'shape': getattr(value, 'shape', None)
                }
        return variables
    
    def reset_namespace(self):
        """Reset the execution namespace"""
        self.namespace.clear()
        self.execution_count = 0
        self.setup_namespace()

# Global kernel instance
_kernel_instance = None

def get_kernel() -> JupyterKernel:
    """Get or create the global kernel instance"""
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = JupyterKernel()
    return _kernel_instance

def reset_kernel():
    """Reset the global kernel instance"""
    global _kernel_instance
    if _kernel_instance:
        _kernel_instance.reset_namespace()
    else:
        _kernel_instance = JupyterKernel()
