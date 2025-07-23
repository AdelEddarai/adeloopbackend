"""
Clean and simplified FastAPI backend for HRatlas
Uses Jupyter kernel for Python execution with Streamlit support
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import aiohttp
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import jwt
import traceback
import math
from fastapi.responses import HTMLResponse
import tempfile
import subprocess
import threading
import socket
import random
import time
from pathlib import Path

# Track server start time
start_time = time.time()

# Import our Jupyter kernel
from jupyter_kernel import get_kernel, reset_kernel

# Load environment variables
load_dotenv()
NEXTJS_API_URL = os.getenv('NEXTJS_API_URL', 'http://localhost:3000')
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY')

# Setup logging - reduced verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Custom JSON encoder for special float values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            elif math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# FastAPI app
app = FastAPI(title="HRatlas Backend", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    code: str
    datasets: Optional[List[Dict[str, Any]]] = []
    language: Optional[str] = "python"
    datasetId: Optional[str] = None
    datasetIds: Optional[List[str]] = []
    variableContext: Optional[Dict[str, Any]] = {}
    user_input: Optional[str] = None  # For interactive input handling

class SQLQueryRequest(BaseModel):
    query: str
    datasets: Optional[List[Dict[str, Any]]] = []

class JSQueryRequest(BaseModel):
    code: str
    datasets: Optional[List[Dict[str, Any]]] = []

# Streamlit app management
streamlit_apps = {}
streamlit_port_counter = 5151  # Start from port 5151

def get_next_streamlit_port():
    """Get the next available port for Streamlit app"""
    global streamlit_port_counter
    port = streamlit_port_counter
    streamlit_port_counter += 1
    return port

def create_streamlit_app(code: str, app_id: str) -> Dict[str, Any]:
    """Create and run a Streamlit app from Python code"""
    try:
        # Check if we're running on a cloud platform
        is_cloud = bool(os.getenv('RENDER_SERVICE_NAME') or
                       os.getenv('RAILWAY_STATIC_URL') or
                       os.getenv('HEROKU_APP_NAME'))

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

        if is_cloud:
            # For cloud platforms, create a simple URL endpoint
            cloud_url = f"https://flopbackend.onrender.com/streamlit/{app_id}"
            return {
                'type': 'streamlit_url',
                'app_id': app_id,
                'title': f'Streamlit App {app_id}',
                'status': 'url_ready',
                'url': cloud_url,
                'open_url': cloud_url,
                'code': cleaned_code,
                'message': f'Streamlit app URL generated: {cloud_url}',
                'host_type': 'cloud_url'
            }

        # For local development, proceed with normal Streamlit app creation
        # Create temporary directory for the app
        temp_dir = tempfile.mkdtemp(prefix=f"streamlit_app_{app_id}_")
        app_file = os.path.join(temp_dir, "streamlit_app.py")

        # Write the cleaned code to a file
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_code)

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
            'code': cleaned_code
        }

    except Exception as e:
        logger.error(f"Error creating Streamlit app: {str(e)}")
        return {
            'type': 'error',
            'message': f'Failed to create Streamlit app: {str(e)}'
        }

# Helper functions
async def get_dataset_from_nextjs(dataset_id: str, auth_token: str) -> Optional[pd.DataFrame]:
    """Fetch dataset from Next.js API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{NEXTJS_API_URL}/api/datasets/{dataset_id}",
                headers={"Authorization": auth_token}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return pd.DataFrame(data.get('data', []))
                else:
                    logger.warning(f"Failed to fetch dataset {dataset_id}: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching dataset {dataset_id}: {str(e)}")
        return None

def verify_jwt_token(auth_token: str) -> str:
    """Verify JWT token and return user ID"""
    # For development, allow requests without auth if CLERK_SECRET_KEY is not set
    if not CLERK_SECRET_KEY:
        logger.warning("CLERK_SECRET_KEY not set, skipping authentication")
        return "dev_user"

    if not auth_token:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    try:
        token = auth_token.replace("Bearer ", "")
        decoded_token = jwt.decode(token, CLERK_SECRET_KEY, algorithms=["RS256"])
        user_id = decoded_token.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
        return user_id
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "HRatlas Backend API", "version": "2.0.0", "status": "running"}

async def handle_package_installation(code: str):
    """Handle pip install commands like Google Colab"""
    import subprocess
    import sys

    # Extract the pip command
    pip_command = code.strip()
    if pip_command.startswith('!'):
        pip_command = pip_command[1:]  # Remove the ! prefix

    try:
        # Run pip install
        result = subprocess.run(
            [sys.executable, "-m"] + pip_command.split(),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"

        if result.returncode == 0:
            output += f"\n✅ Successfully installed packages!"
        else:
            output += f"\n❌ Installation failed with return code {result.returncode}"

        return {
            'status': 'ok' if result.returncode == 0 else 'error',
            'output': output,
            'plots': [],
            'data': [],
            'variables': {},
            'variableTypes': {},
            'outputType': 'package_install'
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'output': '❌ Package installation timed out (2 minutes)',
            'plots': [],
            'data': []
        }
    except Exception as e:
        return {
            'status': 'error',
            'output': f'❌ Package installation failed: {str(e)}',
            'plots': [],
            'data': []
        }

@app.post("/api/execute-jupyter")
async def execute_jupyter(request: Request, query_request: QueryRequest):
    """
    Jupyter-like code execution endpoint using the JupyterKernel
    """
    try:

        code = query_request.code
        datasets = query_request.datasets or []

        # Handle user input if provided in the request
        user_input = getattr(query_request, 'user_input', None)

        # Get the Jupyter kernel instance
        kernel = get_kernel()

        # Handle user input if provided
        if user_input is not None:
            # Store user input for the next input() call
            if not hasattr(kernel, '_user_input_queue'):
                kernel._user_input_queue = []
            kernel._user_input_queue.append(user_input)
            logger.debug(f"Received user input: {user_input}")

        # Check for package installation commands (like Google Colab)
        if code.strip().startswith('!pip ') or code.strip().startswith('pip '):
            return await handle_package_installation(code)
        
        # Add Streamlit support to kernel namespace
        def run_streamlit_app():
            """Create and run a Streamlit app from current code"""
            app_id = f"cell_{hash(code) % 10000}"
            return create_streamlit_app(code, app_id)
        
        # Add Streamlit functions to kernel namespace
        kernel.namespace.update({
            'run_streamlit_app': run_streamlit_app,
            'create_streamlit_app': lambda code_str, app_name="app": create_streamlit_app(code_str, app_name),
            'streamlit': __import__('streamlit'),
            'st': __import__('streamlit'),
        })
        
        # Prepare datasets for the kernel
        dataset_list = []
        for dataset in datasets:
            if dataset and 'data' in dataset:
                dataset_list.append({
                    'name': dataset.get('name', f'Dataset {len(dataset_list) + 1}'),
                    'data': dataset['data']
                })

        # Execute code using the Jupyter kernel
        result = kernel.execute_code(code, dataset_list)
        
        # Check if code creates a Streamlit app
        streamlit_result = None
        if 'streamlit' in code.lower() or 'st.' in code:
            try:
                app_id = f"auto_{hash(code) % 10000}"
                streamlit_result = create_streamlit_app(code, app_id)
            except Exception as e:
                logger.warning(f"Failed to auto-create Streamlit app: {e}")
        
        # Transform result to match expected format
        response_data = {
            'data': result.get('data', []),
            'output': result.get('stdout', ''),
            'plots': result.get('plots', []),
            'variables': result.get('variables', {}),
            'variableTypes': {name: type(val).__name__ for name, val in result.get('variables', {}).items()},
            'outputType': 'jupyter',
            'execution_count': result.get('execution_count', 0),
            'html_outputs': result.get('html_outputs', []),
            'media_count': result.get('media_count', 0),
            'plot_count': result.get('plot_count', 0),
            'display_count': result.get('display_count', 0)
        }


        
        # Add Streamlit app result if created
        if streamlit_result:
            if streamlit_result.get('type') in ['streamlit_url', 'streamlit_app']:
                # Add URL to output text so user can see it
                streamlit_url = streamlit_result.get('url', '')
                if streamlit_url:
                    streamlit_output = f"\n🎈 Streamlit app created!\nURL: {streamlit_url}\nCopy this URL and open it in your browser."
                    if 'output' in response_data:
                        response_data['output'] += streamlit_output
                    else:
                        response_data['output'] = streamlit_output.strip()
            response_data['result'] = streamlit_result
        
        # Handle errors
        if result.get('status') == 'error' and result.get('error'):
            error_info = result['error']
            response_data['error'] = f"{error_info['ename']}: {error_info['evalue']}"
            response_data['errorDetails'] = {
                'message': error_info['evalue'],
                'code': error_info['ename'],
                'stack': '\n'.join(error_info['traceback'])
            }
        
        # Handle stderr
        if result.get('stderr'):
            if 'output' in response_data:
                response_data['output'] += f"\n{result['stderr']}"
            else:
                response_data['output'] = result['stderr']
        
        # Handle result value - prioritize DataFrame conversion for table display
        if result.get('result') is not None:
            if isinstance(result['result'], pd.DataFrame):
                # Convert DataFrame to records for table display
                response_data['data'] = result['result'].to_dict('records')
                response_data['result'] = f"DataFrame with {len(result['result'])} rows and {len(result['result'].columns)} columns"
                # Also store column info for better table display
                response_data['columns'] = list(result['result'].columns)
                response_data['outputType'] = 'dataframe'
            else:
                response_data['result'] = result['result']

        # Handle input requests
        if result.get('needs_input'):
            response_data['needs_input'] = True
            response_data['input_prompt'] = result.get('input_prompt', 'Enter input:')

        return response_data

    except Exception as e:
        logger.error(f"Error in execute_jupyter: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'errorDetails': {
                'message': str(e),
                'code': 'ExecutionError',
                'stack': traceback.format_exc()
            },
            'data': [],
            'output': '',
            'plots': []
        }

@app.post("/api/execute")
async def execute_legacy(request: Request, query_request: QueryRequest):
    """
    Legacy endpoint that redirects to the new Jupyter kernel
    Maintains compatibility with existing frontend code
    """
    # Just call the new Jupyter endpoint
    return await execute_jupyter(request, query_request)

@app.post("/api/reset-kernel")
async def reset_jupyter_kernel(request: Request):
    """Reset the Jupyter kernel namespace"""
    try:

        # Reset the kernel
        reset_kernel()

        return {"status": "success", "message": "Kernel reset successfully"}

    except Exception as e:
        logger.error(f"Error resetting kernel: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/server/status")
async def get_server_status():
    """Get Python server status and monitoring information"""
    try:
        import psutil
        import sys
        import pkg_resources

        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get Python version and process info
        process = psutil.Process()

        # Get installed packages
        installed_packages = []
        try:
            for dist in pkg_resources.working_set:
                installed_packages.append({
                    'name': dist.project_name,
                    'version': dist.version,
                    'location': dist.location
                })
        except Exception as e:
            logger.warning(f"Failed to get packages: {e}")

        # Sort packages by name
        installed_packages.sort(key=lambda x: x['name'].lower())

        # Get kernel info
        kernel = get_kernel()
        namespace_vars = len([k for k in kernel.namespace.keys() if not k.startswith('_')])

        return {
            'status': 'running',
            'python_version': sys.version,
            'server_uptime': time.time() - start_time if 'start_time' in globals() else 0,
            'cpu_usage': cpu_percent,
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'used': disk.used,
                'percent': (disk.used / disk.total) * 100
            },
            'process': {
                'pid': process.pid,
                'memory_info': process.memory_info()._asdict(),
                'cpu_percent': process.cpu_percent(),
                'create_time': process.create_time()
            },
            'kernel': {
                'execution_count': kernel.execution_count,
                'namespace_variables': namespace_vars
            },
            'packages': {
                'total_count': len(installed_packages),
                'packages': installed_packages
            }
        }

    except Exception as e:
        logger.error(f"Error getting server status: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'python_version': sys.version,
            'packages': {'total_count': 0, 'packages': []}
        }

@app.post("/api/python/continue")
async def continue_with_input(request: Request):
    """Continue Python execution with user input"""
    try:
        data = await request.json()
        user_input = data.get('input', '')
        original_code = data.get('code', '')

        # Get the kernel instance
        kernel = get_kernel()

        # Continue execution with the provided input
        result = kernel.provide_input_and_continue(user_input, original_code)

        # Process the result same as regular execution
        response_data = {
            'status': result.get('status', 'ok'),
            'output': result.get('stdout', ''),
            'plots': result.get('plots', []),
            'result': result.get('result'),
            'data': result.get('data'),
            'execution_time': None
        }

        if result.get('stderr'):
            if response_data['output']:
                response_data['output'] += f"\n{result['stderr']}"
            else:
                response_data['output'] = result['stderr']

        # Handle input requests (in case there are more)
        if result.get('needs_input'):
            response_data['needs_input'] = True
            response_data['input_prompt'] = result.get('input_prompt', 'Enter input:')

        return response_data

    except Exception as e:
        logger.error(f"Error continuing execution: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Execution failed: {str(e)}"}
        )

@app.get("/streamlit/{app_id}")
async def serve_streamlit_app(app_id: str):
    """Serve a simple Streamlit app URL for cloud deployment"""
    try:
        # Create a simple HTML page that shows the Streamlit app info
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Streamlit App {app_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f2f6; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ color: #ff4b4b; font-size: 24px; margin-bottom: 20px; }}
                .url {{ background: #f0f2f6; padding: 15px; border-radius: 5px; font-family: monospace; margin: 10px 0; }}
                .button {{ background: #ff4b4b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">🎈 Streamlit App {app_id}</div>
                <p>Your Streamlit app URL is ready!</p>
                <div class="url">https://flopbackend.onrender.com/streamlit/{app_id}</div>
                <p>Copy this URL and open it in your browser to access your Streamlit app.</p>
                <button class="button" onclick="navigator.clipboard.writeText('https://flopbackend.onrender.com/streamlit/{app_id}')">Copy URL</button>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving Streamlit app: {str(e)}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.post("/api/cleanup-streamlit")
async def cleanup_streamlit_app(request: Request):
    """Stop and cleanup a Streamlit app"""
    try:
        data = await request.json()
        app_id = data.get('app_id')

        if app_id in streamlit_apps:
            app_info = streamlit_apps[app_id]

            # Stop the process
            if app_info['process']:
                app_info['process'].terminate()
                app_info['process'].wait()

            # Clean up temp directory
            import shutil
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

@app.post("/api/query")
async def execute_sql_query(request: Request, query_request: SQLQueryRequest):
    """Execute SQL queries using alasql"""
    try:

        # For now, return a simple response
        # TODO: Implement SQL execution with alasql
        return {
            'data': [],
            'message': 'SQL execution not implemented in simplified version'
        }

    except Exception as e:
        logger.error(f"Error in SQL execution: {str(e)}")
        return {
            'error': str(e),
            'data': []
        }

@app.post("/api/execute-js")
async def execute_javascript(request: Request, query_request: JSQueryRequest):
    """Execute JavaScript code"""
    try:

        # For now, return a simple response
        # TODO: Implement JavaScript execution
        return {
            'data': [],
            'message': 'JavaScript execution not implemented in simplified version'
        }

    except Exception as e:
        logger.error(f"Error in JavaScript execution: {str(e)}")
        return {
            'error': str(e),
            'data': []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
