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

# Import shell command utilities
from services.kernel.shell_commands import execute_shell_command

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
        self.execution_history = []  # Track execution history
        self.variables_history = {}  # Track variable changes
        # Create temp directory for CSV files
        self.temp_dir = tempfile.mkdtemp(prefix="jupyter_kernel_")
        # Execution state for input handling
        self._execution_state = None
        self._pending_code = None
        self._accumulated_output = ""
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
            # Jupyter magic commands
            'get_ipython': self.get_ipython_mock,
            # Streamlit support
            'streamlit': None,  # Will be set by main.py when needed
            'st': None,  # Will be set by main.py when needed
        })

        # Add magic command functions
        self._add_magic_commands()

        # Add MediaPipe and OpenCV support
        try:
            import cv2
            self.namespace['cv2'] = cv2
        except ImportError:
            pass

        try:
            import mediapipe as mp
            self.namespace['mp'] = mp
        except ImportError:
            pass

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

        # Add real-time image streaming support
        def stream_image(image_data):
            """Stream image data for real-time display in plots section"""
            if isinstance(image_data, np.ndarray):
                try:
                    from PIL import Image as PILImage
                    import io

                    # Handle different image formats
                    if len(image_data.shape) == 3:
                        if image_data.shape[2] == 3:  # Likely BGR from OpenCV, convert to RGB
                            # OpenCV uses BGR by default, so convert to RGB for proper display
                            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                            img = PILImage.fromarray(rgb_image.astype('uint8'), 'RGB')
                        elif image_data.shape[2] == 4:  # RGBA
                            img = PILImage.fromarray(image_data.astype('uint8'), 'RGBA')
                        else:
                            img = PILImage.fromarray(image_data.astype('uint8'))
                    else:  # Grayscale
                        img = PILImage.fromarray(image_data.astype('uint8'), 'L')

                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    encoded_image = base64.b64encode(buffer.getvalue()).decode()

                    # Add to plots instead of display outputs so it shows in plots section
                    if not hasattr(self, '_streaming_plots'):
                        self._streaming_plots = []
                    self._streaming_plots.append(f"data:image/png;base64,{encoded_image}")

                    print(f"✅ Computer vision image streamed to plots section (shape: {image_data.shape})")

                except Exception as e:
                    print(f"❌ Error streaming image: {e}")
                    import traceback
                    traceback.print_exc()

        # Enhanced cv2.imshow replacement for web display
        def web_imshow(window_name, image):
            """Replace cv2.imshow to display images in plots section"""
            try:
                stream_image(image)
                print(f"📷 Displaying computer vision frame: {window_name}")
                return True
            except Exception as e:
                print(f"❌ Error displaying {window_name}: {e}")
                return False

        # Enhanced video processing with progress feedback
        def process_video_with_progress(video_path, max_frames=30, skip_frames=1):
            """Process video with progress feedback and frame limiting"""
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"❌ Error: Could not open video {video_path}")
                    return False

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0

                print(f"🎥 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

                # Limit frames for web display
                frames_to_process = min(max_frames, total_frames)
                frame_interval = max(1, total_frames // frames_to_process) if frames_to_process < total_frames else 1

                print(f"📊 Processing {frames_to_process} frames (every {frame_interval} frames)")

                frame_count = 0
                processed_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Skip frames based on interval
                    if frame_count % frame_interval == 0 and processed_count < max_frames:
                        # Process and display frame
                        web_imshow(f"Frame {processed_count + 1}", frame)
                        processed_count += 1

                        # Progress feedback
                        if processed_count % 5 == 0:
                            progress = (processed_count / frames_to_process) * 100
                            print(f"🔄 Progress: {progress:.1f}% ({processed_count}/{frames_to_process} frames)")

                    frame_count += 1

                cap.release()
                print(f"✅ Video processing complete! Processed {processed_count} frames")
                return True

            except Exception as e:
                print(f"❌ Error processing video: {e}")
                return False

        # Enhanced cv2.waitKey replacement (no-op for web)
        def web_waitkey(delay=0):
            """Replace cv2.waitKey - no-op for web environment"""
            return 0

        # Add camera control functions
        def stop_camera():
            """Stop all camera captures"""
            if '_active_cameras' in self.namespace:
                for cap in self.namespace['_active_cameras']:
                    try:
                        cap.release()
                    except:
                        pass
                self.namespace['_active_cameras'] = []
            print("📷 All cameras stopped")

        def pause_camera():
            """Pause camera capture"""
            if '_camera_paused' not in self.namespace:
                self.namespace['_camera_paused'] = False
            self.namespace['_camera_paused'] = not self.namespace['_camera_paused']
            status = "paused" if self.namespace['_camera_paused'] else "resumed"
            print(f"📷 Camera {status}")
            return self.namespace['_camera_paused']

        # Enhanced VideoCapture wrapper
        original_VideoCapture = None
        if 'cv2' in self.namespace:
            original_VideoCapture = self.namespace['cv2'].VideoCapture
            kernel_namespace = self.namespace  # Capture reference to namespace

            class WebVideoCapture:
                def __init__(self, source):
                    self.cap = original_VideoCapture(source)
                    if not hasattr(kernel_namespace, '_active_cameras'):
                        kernel_namespace['_active_cameras'] = []
                    kernel_namespace['_active_cameras'].append(self.cap)

                def read(self):
                    if hasattr(kernel_namespace, '_camera_paused') and kernel_namespace.get('_camera_paused', False):
                        return False, None
                    return self.cap.read()

                def release(self):
                    return self.cap.release()

                def __getattr__(self, name):
                    return getattr(self.cap, name)

            self.namespace['cv2'].VideoCapture = WebVideoCapture
            self.namespace['cv2'].imshow = web_imshow
            self.namespace['cv2'].waitKey = web_waitkey
            self.namespace['cv2'].destroyAllWindows = lambda: None  # No-op for web

        # Add control functions to namespace
        self.namespace.update({
            'stop_camera': stop_camera,
            'pause_camera': pause_camera,
            'resume_camera': lambda: pause_camera() if self.namespace.get('_camera_paused', False) else None,
            'stream_image': stream_image,
            'process_video_with_progress': process_video_with_progress,
            'web_imshow': web_imshow,
        })

        # Add display functions to namespace
        self.namespace.update({
            'display': display,
            'display_image': display_image,
            'display_video': display_video,
            'stream_image': stream_image,
            '_execute_shell_command': execute_shell_command
        })

        # Setup matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
        # Configure matplotlib for better plot quality
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 10

        # Override plt.show() to capture plots automatically
        original_show = plt.show
        def custom_show(*args, **kwargs):
            # Don't actually show (since we're in non-interactive mode)
            # The plots will be captured later by capture_plots()
            # Just ensure the current figure is properly finalized
            if plt.get_fignums():
                # Suppress ALL warnings during layout adjustment
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    warnings.filterwarnings('ignore', category=UserWarning)
                    try:
                        # Try tight layout with minimal padding to avoid conflicts
                        current_fig = plt.gcf()
                        current_fig.tight_layout(pad=0.3)
                    except Exception:
                        # If tight_layout fails, use subplots_adjust as fallback
                        try:
                            current_fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
                        except Exception:
                            pass  # Ignore all layout errors
                # Force immediate capture for better reliability
                self._force_plot_capture = True

        plt.show = custom_show
        self.namespace['plt'].show = custom_show

        # Setup interactive input support
        self._pending_input = None
        self._input_prompt = None

        # Initialize input queue for handling user input
        self._user_input_queue = []

        # Enhanced input function with better feedback and persistent state
        def custom_input(prompt=""):
            # Store the prompt in accumulated output
            if not hasattr(self, '_accumulated_output'):
                self._accumulated_output = ""
                
            # Print the prompt with proper formatting
            if prompt:
                print(prompt, end="", flush=True)
                # Add prompt to accumulated output (but don't duplicate if already there)
                if not self._accumulated_output.endswith(prompt):
                    self._accumulated_output += prompt
            else:
                print("Enter input: ", end="", flush=True)
                if not self._accumulated_output.endswith("Enter input: "):
                    self._accumulated_output += "Enter input: "

            # Check if we have queued input
            if hasattr(self, '_user_input_queue') and self._user_input_queue:
                user_input = self._user_input_queue.pop(0)
                print(user_input)  # Echo the input like Jupyter
                # Add the input to accumulated output
                self._accumulated_output += user_input + "\n"
                return user_input

            # If no queued input, signal that we need input with the prompt
            raise InputRequiredException(prompt or "Enter input: ")

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
                    # Generate HTML output that can be displayed directly - Jupyter-style responsive
                    # Use CDN instead of inline for better performance and compatibility
                    fig_html = fig_self.to_html(
                        include_plotlyjs='cdn',
                        div_id=f"plotly-div-{len(kernel_instance._plotly_figures)}",
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                            'responsive': True,
                            'autosizable': True,
                            'fillFrame': False,
                            'frameMargins': 0
                        }
                    )

                    # Add custom CSS and JS to make it truly responsive like Jupyter
                    responsive_html = f"""
                    <style>
                        .plotly-graph-div {{
                            width: 100% !important;
                            height: auto !important;
                            overflow: visible !important;
                        }}
                        .js-plotly-plot {{
                            width: 100% !important;
                            height: auto !important;
                            overflow: visible !important;
                        }}
                        .plot-container {{
                            width: 100% !important;
                            height: auto !important;
                            overflow: visible !important;
                        }}
                    </style>
                    {fig_html}
                    <script>
                        // Ensure plot is responsive after loading
                        setTimeout(function() {{
                            var plotDiv = document.getElementById('plotly-div-{len(kernel_instance._plotly_figures)}');
                            if (plotDiv && window.Plotly) {{
                                window.Plotly.Plots.resize(plotDiv);
                            }}
                        }}, 100);
                    </script>
                    """
                    fig_html = responsive_html
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

        # Setup Altair to capture charts
        self._altair_charts = []
        try:
            import altair as alt

            # Store original renderer
            self._original_altair_renderer = alt.renderers.active

            # Create custom renderer that captures chart specs
            kernel_instance = self

            def altair_capture_renderer(spec, **kwargs):
                """Custom Altair renderer that captures Vega-Lite specs"""
                try:
                    # Store the Vega-Lite spec
                    kernel_instance._altair_charts.append(spec)
                    print(f"📊 Altair chart captured as Vega-Lite spec")
                except Exception as e:
                    print(f"❌ Failed to capture Altair chart: {str(e)}")
                return {}  # Return empty dict to satisfy renderer interface

            # Register and enable our custom renderer
            alt.renderers.register('capture', altair_capture_renderer)
            alt.renderers.enable('capture')

            # Add altair to namespace
            self.namespace['alt'] = alt

        except ImportError:
            # Altair not installed, add placeholder
            self.namespace['alt'] = None

    def capture_plots(self) -> List[str]:
        """Capture all matplotlib plots as base64 images with enhanced reliability"""
        plots = []

        # Get all figure numbers
        fig_nums = plt.get_fignums()

        for fig_num in fig_nums:
            try:
                fig = plt.figure(fig_num)

                # Check if figure has any content (axes with data)
                if not fig.get_axes():
                    continue  # Skip empty figures

                # Check if any axes have content
                has_content = False
                for ax in fig.get_axes():
                    if (ax.lines or ax.patches or ax.collections or
                        ax.images or ax.texts or len(ax.get_children()) > 2):  # More than just spines
                        has_content = True
                        break

                if not has_content:
                    continue  # Skip figures without content

                # Ensure tight layout before saving with comprehensive warning suppression
                try:
                    # Suppress ALL matplotlib warnings including tight layout
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        warnings.filterwarnings('ignore', category=UserWarning)
                        warnings.filterwarnings('ignore', message='Tight layout not applied')
                        warnings.filterwarnings('ignore', message='.*cannot be made large enough.*')
                        # Try tight layout with safe parameters
                        fig.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.2)
                except Exception:
                    # If tight_layout fails completely, try subplots_adjust as fallback
                    try:
                        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
                    except Exception:
                        pass  # Ignore all layout errors

                # Save figure to buffer with high quality
                buffer = BytesIO()
                fig.savefig(
                    buffer, 
                    format='png', 
                    dpi=150, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1
                )
                buffer.seek(0)

                # Encode as base64
                image_data = buffer.getvalue()
                if len(image_data) > 0:
                    encoded_image = base64.b64encode(image_data).decode()
                    plots.append(encoded_image)
                    print(f"✅ Matplotlib plot captured (figure {fig_num})")

                buffer.close()
            except Exception as e:
                print(f"❌ Error capturing plot {fig_num}: {e}")

        # Add streaming plots (from computer vision, etc.)
        if hasattr(self, '_streaming_plots'):
            for streaming_plot in self._streaming_plots:
                if streaming_plot.startswith('data:image/png;base64,'):
                    # Extract just the base64 part
                    base64_data = streaming_plot.split(',', 1)[1]
                    plots.append(base64_data)
                else:
                    plots.append(streaming_plot)
            self._streaming_plots = []  # Clear after capturing

        # Close all figures to prevent accumulation
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

    def capture_altair_charts(self) -> List[dict]:
        """Capture all Altair/Vega-Lite charts as JSON specs"""
        altair_charts = []

        if hasattr(self, '_altair_charts'):
            for chart_spec in self._altair_charts:
                try:
                    # Altair charts are already JSON specs
                    if isinstance(chart_spec, dict):
                        altair_charts.append(chart_spec)
                        print(f"✅ Altair chart captured as Vega-Lite spec")
                    elif isinstance(chart_spec, str):
                        # Try to parse if it's a JSON string
                        import json
                        spec = json.loads(chart_spec)
                        altair_charts.append(spec)
                        print(f"✅ Altair chart parsed from JSON string")
                except Exception as e:
                    print(f"❌ Error processing Altair chart: {str(e)}")
                    pass

            # Clear the charts for next execution
            self._altair_charts = []

        return altair_charts

    def provide_input_and_continue(self, user_input: str, original_code: str):
        """Provide user input and continue execution like Jupyter - FIXED for multiple inputs"""
        print(f"✅ Received input: '{user_input}'")
        
        # Initialize input queue if not exists
        if not hasattr(self, '_user_input_queue'):
            self._user_input_queue = []
        
        # Add the input to our queue
        self._user_input_queue.append(user_input)
        
        # Store original code for continuation
        self._pending_code = original_code
        
        print(f"🔄 Continuing execution with input: {user_input}")
        
        try:
            # Execute the code - input() calls will consume from the queue
            result = self.execute_code(original_code)
            
            # Check if execution completed successfully (no input_required status)
            if result.get('status') != 'input_required':
                print(f"✅ Execution completed successfully")
                # Clear the input queue after successful completion
                self._user_input_queue = []
                self._pending_code = None
                return result
            else:
                # Still needs more input - return the input request
                print(f"⏸️ Execution paused - needs more input: {result.get('input_prompt', 'Enter input')}")
                return result
                        
        except InputRequiredException as input_ex:
            # Code needs more input
            print(f"⏸️ Execution paused - needs input: {input_ex.prompt}")
            return {
                'execution_count': self.execution_count,
                'status': 'input_required',
                'stdout': getattr(self, '_accumulated_output', ''),
                'stderr': '',
                'input_prompt': input_ex.prompt,
                'needs_input': True,
                'plots': [],
                'html_outputs': [],
                'error': None,
                'variables': {},
                'data': None
            }
        except Exception as e:
            print(f"❌ Error during input continuation: {e}")
            return {
                'execution_count': self.execution_count,
                'status': 'error',
                'stdout': getattr(self, '_accumulated_output', ''),
                'stderr': str(e),
                'plots': [],
                'html_outputs': [],
                'error': {
                    'ename': type(e).__name__,
                    'evalue': str(e),
                    'traceback': [str(e)]
                },
                'variables': {},
                'data': None
            }

    def display_media(self, data: Any) -> Optional[str]:
        """Handle display of various media types (images, videos, etc.) with enhanced rich output support"""
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

            # Handle matplotlib figures
            elif hasattr(data, 'savefig') and hasattr(data, '__module__') and 'matplotlib' in data.__module__:
                buffer = BytesIO()
                data.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{encoded}"

            # Handle Plotly figures
            elif hasattr(data, 'to_html') and hasattr(data, '__module__') and 'plotly' in data.__module__:
                try:
                    html = data.to_html(include_plotlyjs='cdn')
                    return html
                except:
                    pass

            # Handle file paths
            elif isinstance(data, str):
                if os.path.isfile(data):
                    mime_type, _ = mimetypes.guess_type(data)
                    if mime_type and mime_type.startswith(('image/', 'video/', 'audio/')):
                        with open(data, 'rb') as f:
                            encoded = base64.b64encode(f.read()).decode()
                            return f"data:{mime_type};base64,{encoded}"
                    elif mime_type and mime_type.startswith('text/'):
                        with open(data, 'r') as f:
                            content = f.read()
                            return f"<pre>{content}</pre>"

                # Handle URLs (if requests is available)
                elif requests and data.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(data, timeout=10)
                        if response.status_code == 200:
                            content_type = response.headers.get('content-type', '')
                            if content_type.startswith(('image/', 'video/', 'audio/')):
                                encoded = base64.b64encode(response.content).decode()
                                return f"data:{content_type};base64,{encoded}"
                            elif content_type.startswith('text/'):
                                return f"<pre>{response.text}</pre>"
                    except:
                        pass

            # Handle HTML content
            elif isinstance(data, str) and data.startswith('<') and data.endswith('>'):
                # Simple HTML detection
                if '<html' in data.lower() or '<div' in data.lower() or '<p' in data.lower():
                    return data

            return None
        except Exception as e:
            print(f"Error displaying media: {e}")
            return None

    def capture_display_calls(self) -> List[str]:
        """Capture any display() calls or rich outputs"""
        # This will be populated by display() function calls
        return getattr(self, '_display_outputs', [])

    def get_ipython_mock(self):
        """Mock IPython interface for magic commands"""
        return self

    def _add_magic_commands(self):
        """Add Jupyter-like magic commands with enhanced functionality"""
        def magic_ls(line=''):
            """List directory contents with details"""
            try:
                import os
                import stat
                contents = os.listdir('.')
                print(f"{'Name':<30} {'Type':<10} {'Size':<10}")
                print("-" * 50)
                for item in contents:
                    try:
                        stat_info = os.stat(item)
                        size = stat_info.st_size
                        if os.path.isdir(item):
                            item_type = 'DIR'
                            print(f"{item+'/':<30} {item_type:<10} {'<DIR>':<10}")
                        else:
                            item_type = 'FILE'
                            print(f"{item:<30} {item_type:<10} {size:<10}")
                    except:
                        print(f"{item:<30} {'UNKNOWN':<10} {'?':<10}")
                return contents
            except Exception as e:
                print(f"Error: {e}")
                return []

        def magic_pwd(line=''):
            """Print working directory"""
            import os
            pwd = os.getcwd()
            print(f"Current directory: {pwd}")
            return pwd

        def magic_history(line=''):
            """Show execution history"""
            if not self.execution_history:
                print("No execution history.")
                return []
            
            print("Execution History:")
            print(f"{'#':<5} {'Code':<50}")
            print("-" * 55)
            for entry in self.execution_history:
                code_preview = entry['code'][:45] + '...' if len(entry['code']) > 45 else entry['code']
                print(f"[{entry['execution_count']:<3}] {code_preview}")
            return self.execution_history

        def magic_who(line=''):
            """List variable names"""
            # Define built-in modules and functions to exclude
            exclude_list = ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 
                          'os', 'datetime', 're', 'math', 'random', 'base64',
                          'get_ipython', 'display', 'display_image', 'display_video',
                          'ls', 'pwd', 'history', 'who', 'whos', 'reset', 'matplotlib',
                          'time', 'requests', 'Image', 'exec', '__builtins__',
                          'ls', 'pwd', 'history', 'who', 'whos', 'reset', 'matplotlib',
                          'time', 'install', 'load', 'cd', 'lsmagic']
            
            user_vars = [name for name in self.namespace.keys() 
                        if not name.startswith('_') and 
                        name not in exclude_list and
                        not callable(self.namespace[name])]
            
            if user_vars:
                print("Variable names:")
                for i, var in enumerate(user_vars, 1):
                    print(f"  {i}. {var}")
            else:
                print("No variables defined.")
            return user_vars

        def magic_whos(line=''):
            """List variables with details"""
            # Define built-in modules and functions to exclude
            exclude_list = ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings',
                          'os', 'datetime', 're', 'math', 'random', 'base64',
                          'get_ipython', 'display', 'display_image', 'display_video',
                          'ls', 'pwd', 'history', 'who', 'whos', 'reset', 'matplotlib',
                          'time', 'requests', 'Image', 'exec', '__builtins__',
                          'ls', 'pwd', 'history', 'who', 'whos', 'reset', 'matplotlib',
                          'time', 'install', 'load', 'cd', 'lsmagic']
            
            user_vars = [name for name in self.namespace.keys() 
                        if not name.startswith('_') and 
                        name not in exclude_list and
                        not callable(self.namespace[name])]
            
            if user_vars:
                print(f"{'Variable':<20} {'Type':<15} {'Size/Value':<20}")
                print("-" * 55)
                for var in user_vars:
                    value = self.namespace[var]
                    var_type = type(value).__name__
                    try:
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            size = len(value)
                            size_str = f"{size} items"
                        else:
                            size_str = str(value)[:20]
                    except:
                        size_str = "<unknown>"
                    print(f"{var:<20} {var_type:<15} {size_str:<20}")
            else:
                print("No variables defined.")
            return user_vars

        def magic_reset(line=''):
            """Reset namespace"""
            # Keep important modules
            keep_vars = ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings',
                        'os', 'datetime', 're', 'math', 'random', 'base64',
                        'get_ipython', 'display', 'display_image', 'display_video',
                        'ls', 'pwd', 'history', 'who', 'whos', 'reset', 'matplotlib',
                        'time', 'requests', 'Image']
            
            # Store values to keep
            keep_values = {name: self.namespace[name] for name in keep_vars if name in self.namespace}
            
            # Clear namespace
            self.namespace.clear()
            
            # Restore keep values
            self.namespace.update(keep_values)
            
            # Reset execution count
            self.execution_count = 0
            self.execution_history = []
            
            print("Namespace reset.")

        def magic_time(line=''):
            """Time execution of a statement"""
            if not line:
                print("Usage: %time statement")
                return
            
            import time as time_module
            start_time = time_module.perf_counter()
            try:
                __builtins__['exec'](line, self.namespace)
                end_time = time_module.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"CPU times: total: {execution_time:.2f} ms")
            except Exception as e:
                end_time = time_module.perf_counter()
                execution_time = (end_time - start_time) * 1000
                print(f"CPU times: total: {execution_time:.2f} ms")
                print(f"Error: {e}")

        def magic_matplotlib(line=''):
            """Set matplotlib backend"""
            try:
                import matplotlib
                if line:
                    matplotlib.use(line)
                    print(f"Matplotlib backend set to: {line}")
                else:
                    current_backend = matplotlib.get_backend()
                    print(f"Current matplotlib backend: {current_backend}")
            except Exception as e:
                print(f"Error setting matplotlib backend: {e}")

        def magic_install(line=''):
            """Install Python packages using pip"""
            if not line:
                print("Usage: %install package_name")
                return
            
            try:
                from services.kernel.shell_commands import execute_pip_command
                result = execute_pip_command(f"install {line}")
                if result == 0:
                    print(f"Successfully installed {line}")
                else:
                    print(f"Failed to install {line}")
            except Exception as e:
                print(f"Error installing package: {e}")

        def magic_load(line=''):
            """Load Python extensions or modules"""
            if not line:
                print("Usage: %load module_name")
                return
            
            try:
                # Try to import the module
                module = __import__(line)
                self.namespace[line] = module
                print(f"Loaded module: {line}")
            except ImportError:
                print(f"Module not found: {line}")
            except Exception as e:
                print(f"Error loading module: {e}")

        def magic_cd(line=''):
            """Change directory"""
            if not line:
                print("Usage: %cd directory_path")
                return
            
            try:
                import os
                os.chdir(line)
                print(f"Changed directory to: {os.getcwd()}")
            except Exception as e:
                print(f"Error changing directory: {e}")

        def magic_ls_magic(line=''):
            """List magic commands"""
            magic_commands = ['who', 'whos', 'history', 'ls', 'pwd', 'time', 'matplotlib', 'install', 'load', 'cd', 'reset']
            print("Available magic commands:")
            for cmd in sorted(magic_commands):
                print(f"  %{cmd}")

        # Add magic commands to namespace
        self.namespace.update({
            'ls': magic_ls,
            'pwd': magic_pwd,
            'history': magic_history,
            'who': magic_who,
            'whos': magic_whos,
            'reset': magic_reset,
            'time': magic_time,
            'matplotlib': magic_matplotlib,
            'install': magic_install,
            'load': magic_load,
            'cd': magic_cd,
            'lsmagic': magic_ls_magic
        })

    def run_line_magic(self, magic_name, line=''):
        """Run a line magic command"""
        # Make sure exec is available as a built-in
        if 'exec' not in self.namespace and 'exec' in __builtins__:
            self.namespace['exec'] = __builtins__['exec']
        
        if magic_name in self.namespace:
            try:
                return self.namespace[magic_name](line)
            except Exception as e:
                print(f"Error in magic command {magic_name}: {e}")
                return None
        else:
            print(f"Unknown magic command: {magic_name}")
            return None

    def format_dataframe_html(self, df: pd.DataFrame, max_rows: int = 100) -> str:
        """Format DataFrame as HTML table with enhanced styling"""
        if len(df) > max_rows:
            # Show first and last rows with truncation indicator
            top_rows = df.head(max_rows // 2)
            bottom_rows = df.tail(max_rows // 2)

            html_parts = []
            html_parts.append(top_rows.to_html(classes='dataframe table table-striped', table_id='dataframe', escape=False))
            html_parts.append(f'<div class="truncation-indicator alert alert-info">... {len(df) - max_rows} more rows ...</div>')
            html_parts.append(bottom_rows.to_html(classes='dataframe table table-striped', table_id='dataframe', escape=False))

            return ''.join(html_parts)
        else:
            return df.to_html(classes='dataframe table table-striped', table_id='dataframe', escape=False)

    def _clean_dataframe_for_json(self, df):
        """
        Clean DataFrame to ensure JSON serialization compatibility
        Handles Excel-specific issues like merged cells, formulas, and special values
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Make a copy to avoid modifying original data
            cleaned_df = df.copy()
            
            # Convert range objects in object columns to lists
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    # Check if any values in this column are range objects
                    if cleaned_df[col].apply(lambda x: isinstance(x, range)).any():
                        # Convert range objects to lists
                        cleaned_df[col] = cleaned_df[col].apply(lambda x: list(x) if isinstance(x, range) else x)
            
            # Replace infinite values with NaN first
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
            
            # Handle different data types
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # For numeric columns
                    if cleaned_df[col].notna().any():
                        # Use median for numeric data, but handle edge cases
                        try:
                            median_val = cleaned_df[col].median()
                            if pd.isna(median_val) or np.isinf(median_val):
                                median_val = 0  # Fallback to 0 if median is also problematic
                            cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        except:
                            # If median calculation fails, fill with 0
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                    else:
                        # All NaN column, fill with 0
                        cleaned_df[col] = 0
                        
                    # Convert to standard Python types to avoid numpy serialization issues
                    cleaned_df[col] = cleaned_df[col].astype(float)
                    
                elif cleaned_df[col].dtype == 'object':
                    # For object columns (strings, mixed types)
                    cleaned_df[col] = cleaned_df[col].fillna('')
                    # Convert to string and handle any remaining problematic values
                    cleaned_df[col] = cleaned_df[col].astype(str)
                    
                elif cleaned_df[col].dtype == 'bool':
                    # For boolean columns
                    cleaned_df[col] = cleaned_df[col].fillna(False)
                    
            # Final check: ensure no NaN or infinite values remain
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                mask = pd.isna(cleaned_df[col]) | np.isinf(cleaned_df[col])
                if mask.any():
                    cleaned_df.loc[mask, col] = 0
                    
            return cleaned_df
            
        except Exception as e:
            self.namespace['print'](f"Warning: Data cleaning failed: {e}")
            # Return original dataframe if cleaning fails
            return df

    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to handle Streamlit commands and other issues"""
        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Handle magic commands
            if line_stripped.startswith('%'):
                # Extract magic command
                magic_parts = line_stripped[1:].split(' ', 1)
                magic_name = magic_parts[0]
                magic_args = magic_parts[1] if len(magic_parts) > 1 else ''
                
                # Replace with function call
                processed_lines.append(f"get_ipython().run_line_magic('{magic_name}', '{magic_args}')")
                continue

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

            # Handle shell commands with output capture
            if line_stripped.startswith('!'):
                # Extract shell command
                shell_cmd = line_stripped[1:]
                # Use subprocess for better output capture
                processed_lines.append(f"_execute_shell_command('{shell_cmd}')")
                continue

            # Handle IPython-style shell commands (%%bash, %%sh, etc.)
            if line_stripped.startswith('%%'):
                # For now, comment out cell magic commands
                processed_lines.append(f"# {line}  # Cell magic command - not supported in this context")
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
        
        # Add to execution history
        self.execution_history.append({
            'execution_count': self.execution_count,
            'code': code,
            'timestamp': str(pd.Timestamp.now())
        })

        # Store current working directory
        old_cwd = os.getcwd()
        
        # Change to temp directory so pd.read_csv() can find files
        os.chdir(self.temp_dir)

        # Preprocess code to handle Streamlit and other issues
        processed_code = self._preprocess_code(code)

        # Add datasets to namespace
        if datasets:
            for i, dataset in enumerate(datasets):
                try:
                    if dataset and 'data' in dataset:
                        df_data = pd.DataFrame(dataset['data'])
                        dataset_name = dataset.get('name', f'Dataset {i+1}')

                        # Create safe variable name
                        safe_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
                        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                        if not safe_name or safe_name[0].isdigit():
                            safe_name = f'dataset_{safe_name}' if safe_name else f'dataset_{i+1}'

                        # Store datasets with multiple names for flexibility
                        # Use safe assignment to prevent namespace corruption
                        if hasattr(self.namespace, '__setitem__'):
                            self.namespace[safe_name] = df_data
                            self.namespace[f'df{i+1}'] = df_data

                            if i == 0:
                                self.namespace['df'] = df_data
                        else:
                            print(f"⚠️ Warning: Cannot assign dataset to namespace - namespace not accessible")
                            continue

                        # Create virtual CSV files so pd.read_csv() works
                        csv_filename = f"{dataset_name}.csv" if dataset_name else f"dataset{i+1}.csv"
                        csv_path = os.path.join(self.temp_dir, csv_filename)
                        df_data.to_csv(csv_path, index=False)

                        # Also create numbered CSV files
                        numbered_csv = os.path.join(self.temp_dir, f"dataset{i+1}.csv")
                        df_data.to_csv(numbered_csv, index=False)
                        
                except (TypeError, ValueError, KeyError, AttributeError) as e:
                    print(f"⚠️ Error processing dataset {i}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Unexpected error processing dataset {i}: {e}")
                    continue

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error_info = None
        html_outputs = []

        try:
            # Redirect stdout and stderr with SIMPLE timeout protection
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # Ensure namespace is in a safe state before execution
                if not hasattr(self.namespace, 'items') or not hasattr(self.namespace, '__setitem__'):
                    raise RuntimeError("Namespace is corrupted - resetting")

                # Execute the entire code block as-is (like Jupyter)
                # NOTE: Removed signal timeout as it blocks on Windows
                exec(processed_code, self.namespace)

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

                        # Handle special result types with enhanced rich output
                        if isinstance(result, pd.DataFrame):
                            html_outputs.append(self.format_dataframe_html(result))
                        elif hasattr(result, '_repr_html_'):
                            try:
                                html_repr = result._repr_html_()
                                if html_repr:
                                    html_outputs.append(html_repr)
                            except Exception as e:
                                # Handle Streamlit objects that don't work in this context
                                if 'streamlit' in str(type(result)).lower() or '_repr_html_' in str(e):
                                    # Skip Streamlit objects - they need to run in Streamlit context
                                    pass
                                else:
                                    html_outputs.append(f"<div class='alert alert-warning'>Error rendering HTML: {str(e)}</div>")
                        elif hasattr(result, 'to_html'):
                            # Handle objects with to_html method (like Plotly figures)
                            try:
                                html_outputs.append(result.to_html())
                            except:
                                pass
                        # Handle matplotlib figures
                        elif hasattr(result, 'savefig') and hasattr(result, '__module__') and 'matplotlib' in result.__module__:
                            media_output = self.display_media(result)
                            if media_output:
                                html_outputs.append(f'<img src="{media_output}" alt="Matplotlib Figure" class="img-fluid" />')
                        # Handle other rich outputs
                        elif result is not None:
                            # Try to convert to string representation
                            try:
                                str_result = str(result)
                                if str_result and not str_result.startswith('<'):
                                    # Simple string output
                                    pass
                                elif str_result.startswith('<'):
                                    # HTML-like output
                                    html_outputs.append(str_result)
                            except:
                                pass

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
            # Check if this is a namespace corruption error
            error_str = str(e).lower()
            if ('dictionary update sequence' in error_str or 
                'vars() argument must have __dict__' in error_str or
                'namespace' in error_str):
                print(f"⚠️ Namespace corruption detected: {e}")
                print("🔧 Attempting to recover namespace...")
                
                try:
                    # Reset and rebuild namespace
                    self.setup_namespace()
                    
                    # Re-add datasets if they were provided
                    if datasets:
                        for i, dataset in enumerate(datasets):
                            try:
                                if dataset and 'data' in dataset:
                                    df_data = pd.DataFrame(dataset['data'])
                                    self.namespace[f'df{i+1}'] = df_data
                                    if i == 0:
                                        self.namespace['df'] = df_data
                            except Exception as recovery_error:
                                print(f"⚠️ Could not recover dataset {i}: {recovery_error}")
                    
                    print("✅ Namespace recovered, but code execution failed")
                except Exception as recovery_error:
                    print(f"❌ Namespace recovery failed: {recovery_error}")
                    
            # Format error information like Jupyter notebook
            error_info = {
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': traceback.format_exc().split('\n')
            }
            
            # Print error to stderr for better visibility
            print(f"❌ Python Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
            print(f"   Traceback: {traceback.format_exc()}", file=sys.stderr)
            
            # Flush any remaining output even in case of error
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass
        finally:
            # Restore working directory
            os.chdir(old_cwd)

        # Capture plots and display outputs separately
        matplotlib_plots = self.capture_plots()
        display_outputs = self.capture_display_calls()
        plotly_figures = self.capture_plotly_figures()

        # Clear display outputs for next execution
        self._display_outputs = []

        # Handle Plotly figures first - set as result if we have them
        plotly_result = None
        if plotly_figures:
            try:
                # Plotly figures are already HTML strings, not JSON
                # Create a simple result object for Plotly HTML figures
                plotly_result = {
                    'type': 'plotly_html',
                    'html': plotly_figures[0],  # Use the first figure HTML
                    'count': len(plotly_figures)
                }
                # Plotly result prepared silently
            except Exception:
                # Silently ignore Plotly parsing errors
                pass

        # Get stdout and stderr
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        # Combine with accumulated output if it exists (for progressive input display)
        if hasattr(self, '_accumulated_output') and self._accumulated_output:
            # Prepend accumulated output to show input progression
            stdout_text = self._accumulated_output + stdout_text
            # Clear accumulated output if execution completed successfully
            if not error_info:
                self._accumulated_output = ""

        # Enhanced DataFrame detection and handling
        table_result = None
        dataframe_data = None

        # Check if result is a DataFrame
        if result is not None and isinstance(result, pd.DataFrame):
            # Clean the DataFrame before converting to dict
            cleaned_result = self._clean_dataframe_for_json(result)
            table_result = {
                'type': 'table',
                'data': cleaned_result.to_dict('records'),
                'columns': [{'key': col, 'label': col} for col in cleaned_result.columns],
                'title': 'DataFrame Result'
            }
            dataframe_data = cleaned_result.to_dict('records')

        # Also check for DataFrames in namespace variables (like result = df)
        dataframe_variables = []
        try:
            # Safely iterate over namespace items with proper error handling
            namespace_items = list(self.namespace.items()) if hasattr(self.namespace, 'items') else []
            for name, value in namespace_items:
                try:
                    # Additional safety checks for the value
                    if (value is not None and 
                        hasattr(value, '__class__') and
                        isinstance(value, pd.DataFrame) and
                        not name.startswith('_') and
                        name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings']):
                        # Clean the DataFrame before converting to dict
                        cleaned_df = self._clean_dataframe_for_json(value)
                        dataframe_variables.append({
                            'name': name,
                            'data': cleaned_df.to_dict('records'),
                            'columns': [{'key': col, 'label': col} for col in cleaned_df.columns],
                            'shape': cleaned_df.shape
                        })
                except Exception as var_error:
                    print(f"⚠️ Error processing variable {name}: {var_error}")
                    continue
        except Exception as e:
            print(f"⚠️ Error checking namespace variables: {e}")

        # Track variable changes for history
        user_variables = {}
        overwritten_variables = []  # Track variables that were overwritten

        # Get previous variables to detect overwrites
        previous_variables = {}
        if self.execution_count > 0 and self.execution_count - 1 in self.variables_history:
            previous_variables = self.variables_history[self.execution_count - 1]

        for name, value in self.namespace.items():
            if (not name.startswith('_') and
                not callable(value) and
                not hasattr(value, '__module__') and
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random', 'base64',
                           'get_ipython', 'display', 'display_image', 'display_video', 'ls', 'pwd', 'history', 'who', 'whos', 'reset']):
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

                    # Check if this variable existed before (overwrite detection)
                    if name in previous_variables:
                        overwritten_variables.append({
                            'name': name,
                            'previous_value': previous_variables[name],
                            'new_value': user_variables[name]
                        })
                except:
                    user_variables[name] = "Object"

        # Store variables in history
        self.variables_history[self.execution_count] = user_variables

        # Return comprehensive execution results
        return {
            'execution_count': self.execution_count,
            'status': 'error' if error_info else 'ok',
            'stdout': stdout_text,
            'stderr': stderr_text,
            'result': result,
            'plots': matplotlib_plots,
            'plotly_figures': plotly_figures,
            'plotly_result': plotly_result,
            'html_outputs': html_outputs,
            'table_result': table_result,
            'dataframe_data': dataframe_data,
            'dataframe_variables': dataframe_variables,
            'display_outputs': display_outputs,
            'error': error_info,
            'variables': user_variables,  # Include current variables
            'execution_history': self.execution_history,  # Include execution history
            'variable_history': self.variables_history,  # Include variable history
            'overwritten_variables': overwritten_variables,  # Include overwritten variables
            'data': None,
            'media_count': len(display_outputs),
            'plot_count': len(matplotlib_plots),
            'display_count': len(display_outputs),
            'plotly_count': len(plotly_figures)
        }

    def cleanup(self):
        """Clean up kernel resources"""
        try:
            # Clean up temp directory
            if hasattr(self, 'temp_dir') and self.temp_dir:
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                    print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not clean up temp directory {self.temp_dir}: {e}")
            
            # Close any open matplotlib figures
            try:
                plt.close('all')
            except:
                pass
                
            # Clear namespace
            self.namespace.clear()
            
            print("🧹 Kernel resources cleaned up")
        except Exception as e:
            print(f"❌ Error during kernel cleanup: {e}")


# Global kernel instance management
_kernel_instance = None


def get_kernel():
    """
    Get or create the global kernel instance
    
    Returns:
        JupyterKernel: The active kernel instance
    """
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = JupyterKernel()
        print("🔧 Created new Jupyter kernel instance")
    return _kernel_instance


def reset_kernel():
    """
    Reset the kernel by creating a new instance
    
    Returns:
        JupyterKernel: New kernel instance
    """
    global _kernel_instance
    if _kernel_instance:
        try:
            _kernel_instance.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during kernel cleanup: {e}")
    
    _kernel_instance = JupyterKernel()
    print("🔄 Reset Jupyter kernel instance")
    return _kernel_instance