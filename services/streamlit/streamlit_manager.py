"""
Streamlit Application Management Module

This module handles the creation, management, and cleanup of Streamlit applications.
It provides functionality for:
- Creating Streamlit apps from Python code
- Managing app lifecycle (start, stop, cleanup)
- Handling both local and cloud deployment scenarios
- Port management for multiple concurrent apps
"""

import os
import tempfile
import subprocess
import shutil
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Global state for Streamlit app management
streamlit_apps = {}
streamlit_port_counter = 5151  # Start from port 5151
streamlit_code_storage = {}  # Store Streamlit code by app_id


def get_next_streamlit_port() -> int:
    """Get the next available port for Streamlit app"""
    global streamlit_port_counter
    port = streamlit_port_counter
    streamlit_port_counter += 1
    return port


def clean_streamlit_code(code: str) -> str:
    """Clean and prepare Streamlit code for execution"""
    # Clean the code to handle DeltaGenerator and other Streamlit objects
    cleaned_code = code.replace('DeltaGenerator()', '').strip()
    if not cleaned_code:
        cleaned_code = """
import streamlit as st
import pandas as pd
import numpy as np

st.title("Generated Streamlit App")
st.write("Your Streamlit app is ready!")

# Add some sample content
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
st.line_chart(data)
"""
    return cleaned_code


def is_cloud_environment() -> bool:
    """Check if running on a cloud platform"""
    return bool(os.getenv('RENDER_SERVICE_NAME') or 
               os.getenv('RAILWAY_STATIC_URL') or 
               os.getenv('HEROKU_APP_NAME'))


def create_streamlit_app(code: str, app_id: str) -> Dict[str, Any]:
    """
    Create and run a Streamlit app from Python code
    
    Args:
        code: Python code containing Streamlit app
        app_id: Unique identifier for the app
        
    Returns:
        Dictionary containing app information and URLs
    """
    try:
        # Clean the code
        cleaned_code = clean_streamlit_code(code)
        
        if is_cloud_environment():
            return _create_cloud_streamlit_app(cleaned_code, app_id)
        else:
            return _create_local_streamlit_app(cleaned_code, app_id)
            
    except Exception as e:
        logger.error(f"Error creating Streamlit app: {str(e)}")
        return {
            'type': 'error',
            'message': f'Failed to create Streamlit app: {str(e)}'
        }


def _create_cloud_streamlit_app(code: str, app_id: str) -> Dict[str, Any]:
    """Create Streamlit app for cloud deployment"""
    # Store the code for the cloud endpoint
    streamlit_code_storage[app_id] = code
    
    # For cloud platforms, create a simple URL endpoint
    cloud_url = f"https://flopbackend.onrender.com/streamlit/{app_id}"
    return {
        'type': 'streamlit_url',
        'app_id': app_id,
        'title': f'Streamlit App {app_id}',
        'status': 'url_ready',
        'url': cloud_url,
        'open_url': cloud_url,
        'code': code,
        'message': f'Streamlit app URL generated: {cloud_url}',
        'host_type': 'cloud_url'
    }


def _create_local_streamlit_app(code: str, app_id: str) -> Dict[str, Any]:
    """Create Streamlit app for local development"""
    # Create temporary directory for the app
    temp_dir = tempfile.mkdtemp(prefix=f"streamlit_app_{app_id}_")
    app_file = os.path.join(temp_dir, "streamlit_app.py")
    
    # Write the cleaned code to a file
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Get next sequential port
    port = get_next_streamlit_port()
    
    # Start Streamlit app
    cmd = [
        "streamlit", "run", app_file,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=temp_dir
    )
    
    app_url = f"http://localhost:{port}"
    
    # Store app info
    app_info = {
        'app_id': app_id,
        'port': port,
        'process': process,
        'temp_dir': temp_dir,
        'app_file': app_file,
        'url': app_url,
        'status': 'starting'
    }
    
    streamlit_apps[app_id] = app_info
    
    # Wait a moment for the app to start
    import time
    time.sleep(2)
    app_info['status'] = 'running'
    
    return {
        'type': 'streamlit_url',
        'app_id': app_id,
        'url': app_info['url'],
        'open_url': app_info['url'],
        'title': f'Streamlit App {app_id}',
        'status': 'running',
        'host_type': 'local',
        'code': code
    }


def cleanup_streamlit_app(app_id: str) -> Dict[str, str]:
    """Stop and cleanup a Streamlit app"""
    try:
        if app_id in streamlit_apps:
            app_info = streamlit_apps[app_id]

            # Stop the process
            if app_info['process']:
                app_info['process'].terminate()
                app_info['process'].wait()

            # Clean up temp directory
            if os.path.exists(app_info['temp_dir']):
                shutil.rmtree(app_info['temp_dir'])

            # Remove from tracking
            del streamlit_apps[app_id]

            return {"status": "success", "message": f"Streamlit app {app_id} stopped"}
        else:
            return {"status": "error", "message": "App not found"}

    except Exception as e:
        logger.error(f"Error stopping Streamlit app: {str(e)}")
        return {"status": "error", "message": str(e)}


def get_streamlit_code(app_id: str) -> Optional[str]:
    """Get stored Streamlit code for an app"""
    return streamlit_code_storage.get(app_id)


def serve_streamlit_app_page(app_id: str) -> str:
    """Generate HTML page for Streamlit app"""
    # Get the stored code for this app
    if app_id in streamlit_code_storage:
        streamlit_code = streamlit_code_storage[app_id]
    else:
        # Default Streamlit app
        streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np

st.title("Generated Streamlit App")
st.write("Your Streamlit app is ready!")

# Add some sample content
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})
st.line_chart(data)
"""
    
    # Return HTML page with the code and instructions
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streamlit App {app_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f2f6; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ color: #ff4b4b; font-size: 24px; margin-bottom: 20px; }}
            .code {{ background: #f8f9fa; padding: 20px; border-radius: 5px; font-family: monospace; margin: 20px 0; white-space: pre-wrap; border: 1px solid #e9ecef; }}
            .button {{ background: #ff4b4b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            .instructions {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">🎈 Streamlit App {app_id}</div>
            
            <div class="instructions">
                <h3>📋 How to run this Streamlit app:</h3>
                <ol>
                    <li>Copy the code below</li>
                    <li>Save it as <code>app.py</code> on your local machine</li>
                    <li>Run: <code>streamlit run app.py</code></li>
                    <li>Or deploy to <a href="https://share.streamlit.io/" target="_blank">Streamlit Cloud</a></li>
                </ol>
            </div>
            
            <h3>📄 Streamlit Code:</h3>
            <div class="code">{streamlit_code}</div>
            
            <button class="button" onclick="copyCode()">📋 Copy Code</button>
            <button class="button" onclick="window.open('https://share.streamlit.io/', '_blank')">🚀 Deploy to Streamlit Cloud</button>
        </div>
        
        <script>
            function copyCode() {{
                const code = `{streamlit_code}`;
                navigator.clipboard.writeText(code).then(() => {{
                    alert('Code copied to clipboard!');
                }});
            }}
        </script>
    </body>
    </html>
    """
