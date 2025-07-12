from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Add seaborn for better plots
import json
import io
import os
import aiohttp  # Add this import
from contextlib import redirect_stdout
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import jwt
import time
import base64  # For encoding plots
from io import BytesIO
import traceback
import random  # For random Streamlit app selection
import tempfile
import subprocess
import threading
import socket
from pathlib import Path
import math  # For handling special float values

# Load environment variables
load_dotenv()
NEXTJS_API_URL = os.getenv('http://localhost:3000', 'https://loopflow.vercel.app')
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY')

# Log environment setup
# logger.info(f"Using Next.js API URL: {NEXTJS_API_URL}")
# if not CLERK_SECRET_KEY:
#     logger.warning("CLERK_SECRET_KEY not set. Authentication may not work properly.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle special float values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            elif math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
        return super().default(obj)

# Function to safely convert to JSON
def safe_json_dumps(obj):
    """Convert object to JSON string, handling special float values."""
    return json.dumps(obj, cls=CustomJSONEncoder)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://loopflow.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables for Streamlit app management
streamlit_processes = {}
streamlit_ports = {}

def find_free_port():
    """Find a free port for Streamlit app"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def get_host_url():
    """Get the appropriate host URL based on environment"""
    # Check if we're in a cloud environment
    cloud_host = os.getenv('STREAMLIT_HOST')  # Can be set in cloud deployment
    if cloud_host:
        return cloud_host

    # Check for common cloud environment variables
    if os.getenv('VERCEL_URL'):
        return f"https://{os.getenv('VERCEL_URL')}"
    elif os.getenv('HEROKU_APP_NAME'):
        return f"https://{os.getenv('HEROKU_APP_NAME')}.herokuapp.com"
    elif os.getenv('RAILWAY_STATIC_URL'):
        return os.getenv('RAILWAY_STATIC_URL')
    elif os.getenv('RENDER_EXTERNAL_URL'):
        return os.getenv('RENDER_EXTERNAL_URL')

    # Default to localhost for local development
    return "http://localhost"

def create_streamlit_app(code: str, app_id: str) -> dict:
    """Create and run a temporary Streamlit app from user code"""
    try:
        # Create a temporary directory for the app
        temp_dir = tempfile.mkdtemp(prefix=f"streamlit_app_{app_id}_")
        app_file = Path(temp_dir) / "app.py"

        # Write the user's code to the app file
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Find a free port
        port = find_free_port()

        # Start Streamlit app in a subprocess
        cmd = [
            "streamlit", "run", str(app_file),
            "--server.port", str(port),
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

        # Store process and port info
        streamlit_processes[app_id] = {
            'process': process,
            'temp_dir': temp_dir,
            'app_file': str(app_file),
            'port': port
        }
        streamlit_ports[port] = app_id

        # Give Streamlit a moment to start
        import time
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            host_url = get_host_url()
            app_url = f"{host_url}:{port}" if host_url.startswith('http://localhost') else f"{host_url}/streamlit-{port}"
            embed_url = f"{app_url}?embed=true&embed_options=hide_loading_screen"

            return {
                'type': 'streamlit_app',
                'embed_url': embed_url,
                'open_url': app_url,
                'title': f'Live Streamlit App',
                'app_id': app_id,
                'port': port,
                'status': 'running',
                'host_type': 'cloud' if not host_url.startswith('http://localhost') else 'local'
            }
        else:
            # Process failed to start
            stdout, stderr = process.communicate()
            error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
            return {
                'type': 'error',
                'message': f'Failed to start Streamlit app: {error_msg}',
                'status': 'failed'
            }

    except Exception as e:
        return {
            'type': 'error',
            'message': f'Error creating Streamlit app: {str(e)}',
            'status': 'failed'
        }

def cleanup_streamlit_app(app_id: str):
    """Clean up a Streamlit app process and temporary files"""
    if app_id in streamlit_processes:
        app_info = streamlit_processes[app_id]

        # Terminate the process
        try:
            app_info['process'].terminate()
            app_info['process'].wait(timeout=5)
        except:
            try:
                app_info['process'].kill()
            except:
                pass

        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(app_info['temp_dir'])
        except:
            pass

        # Remove from tracking
        port = app_info['port']
        if port in streamlit_ports:
            del streamlit_ports[port]
        del streamlit_processes[app_id]

class PlotManager:
    @staticmethod
    def get_plot_as_base64(dpi=100, format='png'):
        """Convert the current matplotlib plot to a base64 encoded string."""
        buffer = BytesIO()
        plt.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        plt.close('all')  # Clear all plots

        # Return the base64 encoded string with the proper MIME type prefix
        encoded_image = base64.b64encode(image_data).decode()
        # Return with data URL format for direct embedding in HTML
        return f"data:image/{format};base64,{encoded_image}"

    @staticmethod
    def get_multiple_plots(figures, dpi=100, format='png'):
        """Convert multiple matplotlib figures to base64 encoded strings."""
        encoded_plots = []
        for fig in figures:
            buffer = BytesIO()
            fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            buffer.close()
            # Encode with proper data URL format
            encoded_image = base64.b64encode(image_data).decode()
            encoded_plots.append(f"data:image/{format};base64,{encoded_image}")
            plt.close(fig)
        return encoded_plots

    @staticmethod
    def setup_default_style(theme='light'):
        """Set up the default plotting style with theme support."""
        # Reset any previous styles
        plt.rcdefaults()

        if theme == 'dark':
            # Dark theme setup
            plt.style.use('dark_background')
            sns.set_theme(style="darkgrid")
            sns.set_context("notebook", font_scale=1.1)
            sns.set_palette("bright")
            plt.rcParams.update({
                'figure.figsize': (10, 6),
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.color': 'white',
                'axes.labelcolor': 'white',
                'axes.edgecolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'
            })
        else:
            # Light theme setup (default)
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
            sns.set_context("notebook", font_scale=1.1)
            sns.set_palette("husl")
            plt.rcParams.update({
                'figure.figsize': (10, 6),
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10
            })

    @staticmethod
    def create_subplots(rows, cols, figsize=(15, 10)):
        """Create a figure with multiple subplots."""
        return plt.subplots(rows, cols, figsize=figsize)

class DataAnalyzer:
    @staticmethod
    def analyze_numeric(df: pd.DataFrame, column: str) -> Dict:
        """Analyze a numeric column with detailed statistics."""
        numeric_data = pd.to_numeric(df[column], errors='coerce')
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        percentile_values = {f'percentile_{int(p*100)}': numeric_data.quantile(p) for p in percentiles}

        return {
            'mean': numeric_data.mean(),
            'median': numeric_data.median(),
            'std': numeric_data.std(),
            'var': numeric_data.var(),
            'min': numeric_data.min(),
            'max': numeric_data.max(),
            'range': numeric_data.max() - numeric_data.min(),
            'count': numeric_data.count(),
            'null_count': numeric_data.isnull().sum(),
            'null_percentage': (numeric_data.isnull().sum() / len(df)) * 100,
            'skew': numeric_data.skew(),
            'kurtosis': numeric_data.kurtosis(),
            **percentile_values,
            'iqr': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
            'mode': numeric_data.mode().iloc[0] if not numeric_data.mode().empty else None,
            'is_normal': abs(numeric_data.skew()) < 0.5 and abs(numeric_data.kurtosis()) < 0.5
        }

    @staticmethod
    def analyze_categorical(df: pd.DataFrame, column: str) -> Dict:
        """Analyze a categorical column with frequency distribution."""
        value_counts = df[column].value_counts()
        value_percentages = df[column].value_counts(normalize=True) * 100

        return {
            'unique_values': df[column].nunique(),
            'most_common': value_counts.head(10).to_dict(),
            'least_common': value_counts.tail(5).to_dict(),
            'frequencies_pct': value_percentages.head(10).to_dict(),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
            'entropy': -(value_percentages/100 * np.log2(value_percentages/100)).sum() if df[column].nunique() > 1 else 0
        }

    @staticmethod
    def analyze_datetime(df: pd.DataFrame, column: str) -> Dict:
        """Analyze a datetime column with temporal patterns."""
        try:
            datetime_data = pd.to_datetime(df[column], errors='coerce')
            return {
                'min_date': datetime_data.min(),
                'max_date': datetime_data.max(),
                'range_days': (datetime_data.max() - datetime_data.min()).days,
                'null_count': datetime_data.isnull().sum(),
                'null_percentage': (datetime_data.isnull().sum() / len(df)) * 100,
                'weekday_distribution': datetime_data.dt.day_name().value_counts().to_dict(),
                'month_distribution': datetime_data.dt.month_name().value_counts().to_dict(),
                'year_distribution': datetime_data.dt.year.value_counts().to_dict(),
                'is_monotonic': datetime_data.is_monotonic_increasing or datetime_data.is_monotonic_decreasing
            }
        except:
            return {'error': 'Failed to analyze datetime column'}

    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict:
        """Analyze the entire dataset with comprehensive statistics."""
        # Basic dataset info
        basic_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum(),
        }

        # Missing values analysis
        missing_values = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        }

        # Column type analysis
        dtypes = df.dtypes.astype(str).to_dict()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()

        # Correlation analysis for numeric columns
        correlation = {}
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            # Get top 10 highest correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            # Take top 10 or all if less than 10
            top_corr = corr_pairs[:min(10, len(corr_pairs))]
            correlation = {f'{pair[0]}_{pair[1]}': pair[2] for pair in top_corr}

        return {
            'basic_info': basic_info,
            'missing_values': missing_values,
            'dtypes': dtypes,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'datetime_columns': datetime_columns,
            'boolean_columns': boolean_columns,
            'top_correlations': correlation
        }

    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, method='iqr') -> Dict:
        """Detect outliers in a numeric column using various methods."""
        numeric_data = pd.to_numeric(df[column], errors='coerce').dropna()

        if method == 'iqr':
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]
            return {
                'method': 'IQR',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(numeric_data)) * 100,
                'outlier_indices': outliers.index.tolist()[:100]  # Limit to first 100 for performance
            }
        elif method == 'zscore':
            z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
            outliers = numeric_data[abs(z_scores) > 3]
            return {
                'method': 'Z-Score',
                'threshold': 3,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(numeric_data)) * 100,
                'outlier_indices': outliers.index.tolist()[:100]  # Limit to first 100 for performance
            }
        else:
            return {'error': f'Unsupported outlier detection method: {method}'}

class QueryRequest(BaseModel):
    query: str
    language: str
    datasetId: str = ""  # Keep for backward compatibility, make optional
    datasetIds: Optional[List[str]] = None  # New field for multiple datasets
    datasets: Optional[List[Dict[str, Any]]] = None  # New field for dataset data

async def get_auth_token() -> str:
    # Create a JWT token that Clerk will accept
    now = int(time.time())
    payload = {
        "iss": "clerk",
        "sub": "python-backend",
        "iat": now,
        "exp": now + 3600,  # Token expires in 1 hour
        "role": "service-account"
    }
    return jwt.encode(payload, CLERK_SECRET_KEY, algorithm="HS256")

async def get_dataset_from_nextjs(dataset_id: str, auth_token: str) -> Optional[pd.DataFrame]:
    try:
        if not NEXTJS_API_URL:
            # logger.error("NEXTJS_API_URL is not set. Cannot fetch dataset.")
            raise ValueError("NEXTJS_API_URL environment variable is not set")

        nextjs_url = f"{NEXTJS_API_URL}/api/datasets"
        # logger.info(f"Fetching dataset from: {nextjs_url}")

        headers = {
            "Authorization": auth_token,
            "Content-Type": "application/json"
        }
        # logger.info(f"Using headers: {headers}")

        # For debugging - create a mock dataset if needed
        if os.getenv('USE_MOCK_DATA', 'false').lower() == 'true':
            # logger.info("Using mock dataset instead of API call")
            mock_data = {
                'id': dataset_id,
                'name': 'Sample Dataset',
                'age': [25, 30, 35, 40, 45],
                'salary': [50000, 60000, 70000, 80000, 90000],
                'department': ['HR', 'Engineering', 'Sales', 'Marketing', 'Finance']
            }
            return pd.DataFrame(mock_data)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    nextjs_url,
                    params={'datasetId': dataset_id},
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    # logger.info(f"Response status: {response.status}")
                    # logger.info(f"Response text: {response_text[:200]}...")  # Log first 200 chars

                    if response.status == 401:
                        raise HTTPException(status_code=401, detail="Authentication failed with Next.js API")

                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail=f"API Error: {response_text}")

                    try:
                        data = json.loads(response_text)
                        if data.get('success') and data.get('datasetInfo'):
                            df = pd.DataFrame(data['datasetInfo']['data'])
                            # logger.info(f"Successfully loaded dataset: {df.shape}")
                            return df
                    except json.JSONDecodeError as e:
                        # logger.error(f"JSON decode error: {e}")
                        raise HTTPException(status_code=500, detail="Invalid JSON response from API")

                    raise HTTPException(status_code=404, detail="Dataset not found or invalid format")
        except aiohttp.ClientError as e:
            # logger.error(f"HTTP client error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Next.js API: {str(e)}")
    except Exception as e:
        # logger.error(f"Error in get_dataset_from_nextjs: {str(e)}")
        # Create a sample dataset as fallback
        # logger.info("Creating fallback sample dataset")
        try:
            # Check if we can load a sample dataset from a file
            sample_file = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
            if os.path.exists(sample_file):
                # logger.info(f"Loading sample data from {sample_file}")
                return pd.read_csv(sample_file)
        except Exception as e:
            logger.warning(f"Could not load sample file: {e}")

        # Create a more comprehensive sample dataset with common HR columns
        sample_data = {
            'id': list(range(1, 21)),
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy',
                    'Kevin', 'Linda', 'Mike', 'Nancy', 'Oscar', 'Patty', 'Quincy', 'Rachel', 'Steve', 'Tina'],
            'age': [25, 30, 35, 40, 45, 28, 33, 38, 43, 48, 27, 32, 37, 42, 47, 29, 34, 39, 44, 49],
            'salary': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000,
                      52000, 62000, 72000, 82000, 92000, 57000, 67000, 77000, 87000, 97000],
            'department': ['HR', 'Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Engineering',
                          'Sales', 'Marketing', 'Finance', 'HR', 'Engineering', 'Sales', 'Marketing',
                          'Finance', 'HR', 'Engineering', 'Sales', 'Marketing', 'Finance'],
            'years_of_service': [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
            'performance_score': [3.5, 4.0, 4.5, 3.0, 3.5, 4.0, 4.5, 3.0, 3.5, 4.0, 4.5, 3.0, 3.5, 4.0, 4.5, 3.0, 3.5, 4.0, 4.5, 3.0],
            'hire_date': ['2020-01-15', '2019-03-20', '2018-05-10', '2017-07-05', '2016-09-25',
                         '2020-02-15', '2019-04-20', '2018-06-10', '2017-08-05', '2016-10-25',
                         '2020-03-15', '2019-05-20', '2018-07-10', '2017-09-05', '2016-11-25',
                         '2020-04-15', '2019-06-20', '2018-08-10', '2017-10-05', '2016-12-25'],
            'gender': ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
            'location': ['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle', 'New York', 'San Francisco',
                        'Chicago', 'Boston', 'Seattle', 'New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle',
                        'New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'],
            'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor',
                         'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
            'bonus': [2000, 3000, 4000, 1500, 2500, 3500, 1800, 2800, 3800, 1900,
                     2900, 3900, 1700, 2700, 3700, 2100, 3100, 4100, 1600, 2600]
        }

        # Create the DataFrame
        sample_df = pd.DataFrame(sample_data)

        # Add some calculated columns
        sample_df['total_compensation'] = sample_df['salary'] + sample_df['bonus']
        sample_df['salary_per_year'] = sample_df['salary'] / sample_df['years_of_service']

        # Convert hire_date to datetime
        sample_df['hire_date'] = pd.to_datetime(sample_df['hire_date'])

        # Add some missing values to make it more realistic
        for col in ['bonus', 'performance_score', 'years_of_service']:
            mask = sample_df.sample(frac=0.1).index
            sample_df.loc[mask, col] = None

        logger.info(f"Created sample dataset with shape {sample_df.shape}")
        return sample_df

@app.post("/api/execute")
async def execute_query(request: Request, query_request: QueryRequest):
    try:
        auth_token = request.headers.get("authorization")
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization header is required")

        if query_request.language == "python":
            # Handle multiple datasets - use data passed directly from frontend
            datasets = {}
            dataset_paths = {}
            
            # Check if datasets are passed directly in the request
            if hasattr(query_request, 'datasets') and query_request.datasets:
                # Use datasets passed directly from frontend
                for i, dataset_info in enumerate(query_request.datasets):
                    if dataset_info and 'data' in dataset_info:
                        df_data = pd.DataFrame(dataset_info['data'])
                        dataset_id = dataset_info.get('id', f'dataset_{i+1}')
                        dataset_name = dataset_info.get('name', f'Dataset {i+1}')

                        # Create safe variable name from dataset name
                        safe_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
                        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                        if not safe_name or safe_name[0].isdigit():
                            safe_name = f'dataset_{safe_name}' if safe_name else f'dataset_{i+1}'

                        # Store datasets with actual names and numbered fallbacks
                        datasets[safe_name] = df_data
                        datasets[f'df{i+1}'] = df_data  # Keep numbered for compatibility

                        # Create virtual file paths using actual names
                        dataset_paths[f'{safe_name}.csv'] = df_data
                        dataset_paths[f'dataset{i+1}.csv'] = df_data  # Keep numbered for compatibility
                        dataset_paths[f'{dataset_id}.csv'] = df_data

                        if i == 0:
                            datasets['df'] = df_data
                            datasets[safe_name + '_main'] = df_data  # Alternative main reference
                            dataset_paths['dataset.csv'] = df_data
            else:
                # Fallback: try to load from Next.js API (old method)
                dataset_ids = query_request.datasetIds if query_request.datasetIds else [query_request.datasetId] if query_request.datasetId else []
                
                for i, dataset_id in enumerate(dataset_ids):
                    if dataset_id and dataset_id != 'no-dataset':  # Skip empty or placeholder dataset IDs
                        try:
                            df_data = await get_dataset_from_nextjs(dataset_id, auth_token)
                            if df_data is not None:
                                # Store datasets with numbered keys
                                datasets[f'df{i+1}'] = df_data
                                
                                # Create virtual file paths that users can use with pd.read_csv()
                                dataset_paths[f'dataset{i+1}.csv'] = df_data
                                dataset_paths[f'{dataset_id}.csv'] = df_data
                                
                                if i == 0:
                                    datasets['df'] = df_data
                                    dataset_paths['dataset.csv'] = df_data
                        except Exception as e:
                            print(f"Warning: Could not load dataset {dataset_id}: {str(e)}")
                            # Continue without this dataset

            # Initialize plot style
            PlotManager.setup_default_style()

            # Capture stdout and prepare namespace
            output = io.StringIO()
            plots: List[str] = []

            # Get variable context from request if provided
            variable_context = {}
            if hasattr(query_request, 'variableContext') and query_request.variableContext:
                variable_context = query_request.variableContext
                logger.info(f"Received variable context with {len(variable_context)} variables")

            # Prepare the execution environment with enhanced capabilities
            local_ns = {
                # Core libraries
                'pd': pd,
                **datasets,  # Add all datasets to namespace
                'np': np,
                'plt': plt,
                'sns': sns,
                'os': os,
                'sys': __import__('sys'),
                'io': io,
                'json': json,
                'datetime': __import__('datetime'),
                're': __import__('re'),

                # Computer Vision libraries (only available ones)
                'PIL': __import__('PIL'),
                'Image': __import__('PIL.Image', fromlist=['Image']).Image,
                'ImageDraw': __import__('PIL.ImageDraw', fromlist=['ImageDraw']).ImageDraw,
                'ImageFont': __import__('PIL.ImageFont', fromlist=['ImageFont']).ImageFont,
                'requests': __import__('requests'),
                'base64': __import__('base64'),
                'urllib': __import__('urllib'),

                # Streamlit (for interactive apps)
                'streamlit': __import__('streamlit'),
                'st': __import__('streamlit'),

                # Variable context from previous cells
                **variable_context,

                # Utilities
                'describe': lambda x: pd.DataFrame(x.describe()),
                'get_plot': lambda: PlotManager.get_plot_as_base64(),  # Make it a function call
                'analyzer': DataAnalyzer(),
                'available_columns': list(datasets.get('df', pd.DataFrame()).columns) if datasets else [],

                # Image and URL utilities
                'show_image': lambda url: plots.append(url) if url.startswith(('http', 'data:')) else None,

                # Streamlit Community Cloud URL generators
                'generate_streamlit_url': lambda username="streamlit", repo="demo-uber-nyc-pickups", branch="main", app_file="streamlit_app": f"https://{username}-{repo}-{branch}-{app_file.replace('.py', '').replace('_', '')}.streamlit.app",
                'create_demo_url': lambda app_name="30days": f"https://{app_name}.streamlit.app",  # Popular demo apps
                'get_gallery_app': lambda app_name="30days": f"https://{app_name}.streamlit.app",  # Gallery apps

                # Popular working Streamlit apps for demo
                'get_random_streamlit_app': lambda: random.choice([
                    "https://30days.streamlit.app",  # 30 Days of Streamlit
                    "https://streamlit-example-app-calculating-user-growth.streamlit.app",  # User Growth Calculator
                    "https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/main/streamlit_app.py",  # Uber NYC Pickups
                    "https://share.streamlit.io/streamlit/demo-self-driving/main/streamlit_app.py",  # Self Driving Demo
                    "https://streamlit-example-app-iris-eda.streamlit.app",  # Iris EDA
                ]),

                # Create embeddable Streamlit URL with proper embed parameters
                'create_streamlit_embed': lambda url: f"{url}?embed=true&embed_options=hide_loading_screen&embed_options=show_toolbar" if not "embed=true" in url else url,

                # Helper to create a complete Streamlit app result with both embed and open URLs
                'streamlit_app': lambda url_or_name="30days": {
                    'embed_url': f"https://{url_or_name}.streamlit.app?embed=true&embed_options=hide_loading_screen" if not url_or_name.startswith('http') else f"{url_or_name}?embed=true&embed_options=hide_loading_screen",
                    'open_url': f"https://{url_or_name}.streamlit.app" if not url_or_name.startswith('http') else url_or_name.split('?')[0],
                    'type': 'streamlit_app'
                },

                # Function to create and run live Streamlit app from current code
                'run_streamlit_app': lambda: create_streamlit_app(code, f"cell_{hash(code) % 10000}"),
                'available_datasets': list(dataset_paths.keys()) if dataset_paths else [],
                
                # Debug function to show available datasets
                'show_datasets': lambda: print(f"📊 Available datasets: {list(dataset_paths.keys()) if dataset_paths else 'None selected'}"),

                # Package management helpers
                'pip_list': lambda: __import__('subprocess').check_output([__import__('sys').executable, '-m', 'pip', 'list']).decode('utf-8'),
                'pip_install': lambda pkg: __import__('subprocess').check_call([__import__('sys').executable, '-m', 'pip', 'install', pkg]),
                'pip_show': lambda pkg: __import__('subprocess').check_output([__import__('sys').executable, '-m', 'pip', 'show', pkg]).decode('utf-8'),

                # Data analysis helpers
                'correlation_matrix': lambda df: df.select_dtypes(include=['number']).corr(),
                'summary_stats': lambda df: pd.DataFrame({
                    'dtypes': df.dtypes,
                    'nunique': df.nunique(),
                    'missing': df.isnull().sum(),
                    'missing_pct': df.isnull().sum() / len(df) * 100,
                }),
                
                # Multiple dataset helpers
                'list_datasets': lambda: [name for name in locals() if name.startswith('df') and isinstance(locals()[name], pd.DataFrame)],
                'compare_datasets': lambda df1, df2: {
                    'df1_shape': df1.shape,
                    'df2_shape': df2.shape,
                    'common_columns': list(set(df1.columns) & set(df2.columns)),
                    'df1_only_columns': list(set(df1.columns) - set(df2.columns)),
                    'df2_only_columns': list(set(df2.columns) - set(df1.columns))
                },
                'merge_datasets': lambda df1, df2, on=None, how='inner': pd.merge(df1, df2, on=on, how=how) if on else pd.merge(df1, df2, how=how),
                'concat_datasets': lambda *dfs, **kwargs: pd.concat(dfs, **kwargs),
                
                # Dataset information
                'dataset_count': len(datasets),
                'dataset_names': list(datasets.keys()),
                'ui_datasets_loaded': len(datasets) > 0,
                'plot_histogram': lambda col, bins=10: (plt.figure(figsize=(10, 6)), plt.hist(df[col].dropna(), bins=bins), plt.title(f'Histogram of {col}'), plt.xlabel(col), plt.ylabel('Frequency'), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_boxplot': lambda col: (plt.figure(figsize=(10, 6)), plt.boxplot(df[col].dropna()), plt.title(f'Boxplot of {col}'), plt.ylabel(col), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_scatter': lambda x, y: (plt.figure(figsize=(10, 6)), plt.scatter(df[x], df[y]), plt.title(f'Scatter plot of {x} vs {y}'), plt.xlabel(x), plt.ylabel(y), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_correlation': lambda: (plt.figure(figsize=(12, 10)), sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm'), plt.title('Correlation Matrix'), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_pairplot': lambda cols=None: (sns.pairplot(df[cols] if cols else df.select_dtypes(include=['number'])), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_countplot': lambda col: (plt.figure(figsize=(10, 6)), sns.countplot(y=col, data=df), plt.title(f'Count of {col}'), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_barplot': lambda x, y: (plt.figure(figsize=(10, 6)), sns.barplot(x=x, y=y, data=df), plt.title(f'Bar plot of {x} vs {y}'), plt.xticks(rotation=45), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_lineplot': lambda x, y: (plt.figure(figsize=(10, 6)), sns.lineplot(x=x, y=y, data=df), plt.title(f'Line plot of {x} vs {y}'), plt.xticks(rotation=45), plt.tight_layout(), PlotManager.get_plot_as_base64()),
                'plot_distribution': lambda col: (plt.figure(figsize=(10, 6)), sns.histplot(df[col].dropna(), kde=True), plt.title(f'Distribution of {col}'), plt.tight_layout(), PlotManager.get_plot_as_base64()),
            }

            # Add import handling capability
            def custom_import(module_name):
                try:
                    return __import__(module_name)
                except ImportError:
                    # Try to install the module automatically
                    try:
                        print(f"Attempting to install {module_name}...")
                        import subprocess
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])
                        print(f"Successfully installed {module_name}")
                        return __import__(module_name)
                    except Exception as e:
                        print(f"Failed to install {module_name}: {str(e)}")
                        raise

            # Create custom pandas read functions that work with selected datasets
            def custom_read_csv(filepath_or_buffer, **kwargs):
                """Custom pd.read_csv that can use selected datasets as virtual files"""
                if isinstance(filepath_or_buffer, str):
                    # Check if it's one of our virtual dataset paths (exact match)
                    if filepath_or_buffer in dataset_paths:
                        print(f"✓ Found exact match for '{filepath_or_buffer}'")
                        return dataset_paths[filepath_or_buffer].copy()
                    
                    # Check if the filename (without extension) matches any dataset name
                    filename_base = filepath_or_buffer.replace('.csv', '').replace('.xlsx', '').replace('.xls', '').lower()
                    
                    # Look for datasets by name (case-insensitive partial matching)
                    for path, data in dataset_paths.items():
                        path_base = path.replace('.csv', '').lower()
                        if filename_base == path_base or filename_base in path_base or path_base in filename_base:
                            print(f"✓ Found partial match: '{filepath_or_buffer}' → '{path}'")
                            return data.copy()
                
                # Otherwise, try to read the actual file
                try:
                    return pd.read_csv(filepath_or_buffer, **kwargs)
                except Exception as e:
                    # If file not found, provide helpful error message
                    if dataset_paths:
                        available = sorted(list(dataset_paths.keys()))
                        error_msg = f"""
❌ File '{filepath_or_buffer}' not found.

📊 Available datasets from your selection:
{chr(10).join(f"   • {path}" for path in available)}

💡 Usage examples:
   df1 = pd.read_csv('{available[0] if available else "dataset1.csv"}')
   df2 = pd.read_csv('{available[1] if len(available) > 1 else "dataset2.csv"}')

🔍 Make sure the filename matches exactly (case-insensitive).
"""
                        raise FileNotFoundError(error_msg.strip())
                    else:
                        raise FileNotFoundError(f"""
❌ File '{filepath_or_buffer}' not found.

📊 No datasets selected from UI.

💡 To fix this:
   1. Select datasets using the database icon in the cell
   2. Or use actual file paths: pd.read_csv('/path/to/your/file.csv')
""".strip())

            # Add these to the namespace
            local_ns['import_module'] = custom_import
            
            # Override pandas read functions with our custom ones
            local_ns['pd'].read_csv = custom_read_csv

            # Also add them directly to the namespace
            local_ns['read_csv'] = custom_read_csv

            # Override file saving functions to prevent saving in backend directory
            def safe_save_warning(*args, **kwargs):
                print("⚠️  File saving is disabled in hosted environment. Use URLs or data URIs instead.")
                return None

            # Override common save functions
            local_ns['plt'].savefig = safe_save_warning
            if 'cv2' in local_ns:
                local_ns['cv2'].imwrite = safe_save_warning

            # Execute the code with Jupyter-like behavior and real-time output
            with redirect_stdout(output):
                try:
                    # Split the code into cells like Jupyter
                    code_cells = query_request.query.split('# %%')
                    if len(code_cells) == 1:  # No cell markers, treat as single cell
                        code_cells = [query_request.query]

                    # Initialize cell_results for all execution paths
                    cell_results = []

                    # Check if this is Streamlit code
                    full_code = query_request.query
                    is_streamlit_code = ('import streamlit' in full_code or
                                       'from streamlit' in full_code or
                                       'st.' in full_code)

                    if is_streamlit_code:
                        # Create and run a live Streamlit app
                        app_id = f"cell_{hash(full_code) % 10000}"
                        print("🎈 Detected Streamlit code - Creating live app...")
                        print(f"📝 Code preview:\n{full_code[:200]}{'...' if len(full_code) > 200 else ''}")

                        streamlit_result = create_streamlit_app(full_code, app_id)

                        if streamlit_result['type'] == 'streamlit_app':
                            plots.append(json.dumps(streamlit_result))
                            print(f"✅ Live Streamlit app created successfully!")
                            print(f"🌐 App URL: {streamlit_result['open_url']}")
                            print(f"📱 App ID: {app_id}")
                            print(f"🔗 Embed URL: {streamlit_result['embed_url']}")
                            print(f"📊 App Status: Running")
                            print(f"🏠 Host Type: {streamlit_result.get('host_type', 'local')}")
                            print(f"🎯 Interactive features: Enabled")
                            print(f"💾 Save to dashboard: Available")
                            print(f"")
                            print(f"🎈 Your Streamlit app is now live and interactive!")
                            print(f"   You can interact with widgets, upload files, and see real-time updates.")
                            print(f"   The app will continue running until you stop it or restart the server.")
                            if streamlit_result.get('host_type') == 'cloud':
                                print(f"   🌐 Running in cloud mode - URLs are configured for your deployment environment.")

                            # Set a result for Streamlit apps - this will show in output
                            streamlit_info = {
                                'type': 'streamlit_app_info',
                                'url': streamlit_result['open_url'],
                                'embed_url': streamlit_result['embed_url'],
                                'app_id': app_id,
                                'status': 'running',
                                'message': f"Streamlit app is running at {streamlit_result['open_url']}",
                                'features': ['Interactive widgets', 'File uploads', 'Real-time updates', 'Data visualization'],
                                'actions': ['Open in new tab', 'Save to dashboard', 'Stop app']
                            }
                            cell_results.append(streamlit_info)

                            # Also add to local namespace so it can be accessed
                            local_ns['streamlit_app_info'] = streamlit_info
                            local_ns['app_url'] = streamlit_result['open_url']
                            local_ns['result'] = streamlit_info

                        else:
                            print(f"❌ Failed to create Streamlit app: {streamlit_result['message']}")
                            error_info = {
                                'type': 'streamlit_error',
                                'message': streamlit_result['message'],
                                'status': 'failed'
                            }
                            cell_results.append(error_info)
                            local_ns['result'] = error_info
                    else:
                        # Execute each cell normally
                        for i, cell in enumerate(code_cells):
                            if not cell.strip():
                                continue

                            print(f"Executing cell {i+1}..." if len(code_cells) > 1 else "Executing code...")

                            # Execute the cell
                            exec(cell, {}, local_ns)

                            # Check for image URLs or Streamlit URLs in result
                            if 'result' in local_ns:
                                result_val = local_ns.get('result')

                                # Handle Streamlit app objects (dict with embed_url, open_url, type)
                                if isinstance(result_val, dict) and result_val.get('type') == 'streamlit_app':
                                    # Create a special Streamlit app plot entry
                                    streamlit_data = {
                                        'type': 'streamlit_app',
                                        'embed_url': result_val.get('embed_url'),
                                        'open_url': result_val.get('open_url'),
                                        'title': f"Streamlit App"
                                    }
                                    plots.append(json.dumps(streamlit_data))
                                    print(f"Streamlit app captured: {result_val.get('open_url')}")

                            elif isinstance(result_val, str):
                                # Handle image URLs
                                if result_val.startswith(('http', 'https')) and any(ext in result_val.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                                    plots.append(result_val)
                                    print(f"Image URL captured: {result_val}")
                                # Handle Streamlit URLs
                                elif 'streamlit' in result_val.lower() or result_val.startswith('http') and ('8501' in result_val or '.streamlit.app' in result_val):
                                    # Convert to proper Streamlit app object
                                    embed_url = result_val if '?embed=true' in result_val else f"{result_val}?embed=true&embed_options=hide_loading_screen"
                                    open_url = result_val.split('?')[0]  # Remove query params for open URL

                                    streamlit_data = {
                                        'type': 'streamlit_app',
                                        'embed_url': embed_url,
                                        'open_url': open_url,
                                        'title': f"Streamlit App"
                                    }
                                    plots.append(json.dumps(streamlit_data))
                                    print(f"Streamlit URL captured: {open_url}")
                                # Handle data URLs
                                elif result_val.startswith('data:'):
                                    plots.append(result_val)
                                    print(f"Data URL captured")

                        # Flush output for real-time display
                        output.flush()

                        # Capture any plots created in this cell
                        if plt.get_fignums():
                            try:
                                # Make sure plots have proper layout before capturing
                                plt.tight_layout()
                                # Get the plot with proper data URL format
                                plot_data = PlotManager.get_plot_as_base64()
                                plots.append(plot_data)
                                print(f"Plot captured successfully: {len(plots)} plots total")
                                # Check if result is already a plot
                                result_is_plot = False
                                if 'result' in local_ns:
                                    result_val = local_ns.get('result')
                                    if isinstance(result_val, str) and result_val.startswith('data:image'):
                                        result_is_plot = True
                                        print("Result is already a plot image")

                                # If result is not a plot, set it to the plot we just captured
                                if not result_is_plot and 'result' not in local_ns:
                                    print("Setting result to the captured plot")
                                    local_ns['result'] = plot_data
                            except Exception as plot_error:
                                print(f"Error capturing plot: {str(plot_error)}")
                            # plt.close('all') is already called in get_plot_as_base64

                            # Get the result from this cell
                            cell_result = local_ns.get('result')
                            if cell_result is not None:
                                cell_results.append(cell_result)

                    # Use the last cell's result as the final result
                    result = cell_results[-1] if cell_results else None

                    # Check if result is a plot image
                    result_is_plot = False
                    if result is not None and isinstance(result, str) and (result.startswith('data:image') or 'base64,' in result):
                        result_is_plot = True
                        # Make sure this plot is in the plots array
                        if result not in plots:
                            plots.append(result)
                            print("Added result plot to plots array")

                    # Extract variables from the execution namespace for persistence
                    extracted_variables = {}
                    variable_types = {}
                    for var_name, var_value in local_ns.items():
                        # Skip built-in functions, modules, and special variables
                        if (not var_name.startswith('_') and
                            var_name not in ['pd', 'np', 'plt', 'sns', 'os', 'sys', 'io', 'json', 'datetime', 're',
                                           'cv2', 'mediapipe', 'cvzone', 'PIL', 'Image', 'ImageDraw', 'ImageFont',
                                           'imageio', 'skimage', 'streamlit', 'st', 'describe', 'get_plot', 'analyzer',
                                           'available_columns', 'available_datasets', 'import_module', 'read_csv'] and
                            not callable(var_value) and
                            not hasattr(var_value, '__module__')):

                            try:
                                # Try to serialize the variable to check if it's JSON-serializable
                                if isinstance(var_value, (str, int, float, bool, list, dict)):
                                    extracted_variables[var_name] = var_value
                                    variable_types[var_name] = type(var_value).__name__
                                elif isinstance(var_value, pd.DataFrame):
                                    # Store DataFrame info but not the actual data (too large)
                                    variable_types[var_name] = 'DataFrame'
                                    extracted_variables[var_name] = f"DataFrame({var_value.shape[0]}x{var_value.shape[1]})"
                                elif isinstance(var_value, np.ndarray):
                                    variable_types[var_name] = 'ndarray'
                                    extracted_variables[var_name] = f"ndarray{var_value.shape}"
                                else:
                                    variable_types[var_name] = type(var_value).__name__
                                    extracted_variables[var_name] = str(var_value)[:100]  # Truncate long strings
                            except Exception:
                                # Skip variables that can't be serialized
                                pass

                    # Format the response
                    response_data = {
                        "data": None,
                        "output": output.getvalue(),
                        "plots": plots,
                        "error": None,
                        "isSuccess": True,
                        "variables": extracted_variables,
                        "variableTypes": variable_types
                    }

                    if result is not None:
                        try:
                            if isinstance(result, pd.DataFrame):
                                # Replace NaN, Infinity values before converting to dict
                                df_copy = result.copy()
                                # Replace NaN with None (null in JSON)
                                df_copy = df_copy.replace({float('nan'): None})
                                # Replace infinity with very large numbers
                                df_copy = df_copy.replace({float('inf'): 1.0e+308, float('-inf'): -1.0e+308})
                                response_data["data"] = df_copy.to_dict('records')
                            elif isinstance(result, (pd.Series, np.ndarray)):
                                # Convert to list and handle special values
                                values = []
                                for v in result:
                                    if isinstance(v, float):
                                        if math.isnan(v):
                                            values.append({"value": None})
                                        elif math.isinf(v):
                                            values.append({"value": "Infinity" if v > 0 else "-Infinity"})
                                        else:
                                            values.append({"value": v})
                                    else:
                                        values.append({"value": str(v)})
                                response_data["data"] = values
                            elif isinstance(result, dict):
                                # Handle special values in dict
                                cleaned_dict = {}
                                for k, v in result.items():
                                    if isinstance(v, float):
                                        if math.isnan(v):
                                            cleaned_dict[k] = None
                                        elif math.isinf(v):
                                            cleaned_dict[k] = "Infinity" if v > 0 else "-Infinity"
                                        else:
                                            cleaned_dict[k] = v
                                    else:
                                        cleaned_dict[k] = v
                                response_data["data"] = [cleaned_dict]
                            elif result_is_plot:
                                # If result is a plot, don't include it in data to avoid duplication
                                response_data["data"] = [{"result": "[Plot Image]"}]
                            else:
                                response_data["data"] = [{"result": str(result)}]
                        except Exception as e:
                            print(f"Error formatting result: {str(e)}")
                            # Fallback to string representation
                            response_data["data"] = [{"result": str(result)}]
                            response_data["error"] = f"Warning: Could not properly format result: {str(e)}"

                    # Add the result to the response
                    response_data["result"] = result

                    # Use the custom JSON encoder for the response
                    # We don't return this directly, but convert it to a dict that FastAPI will serialize
                    try:
                        # Test if the response can be serialized
                        safe_json_dumps(response_data)
                        return response_data
                    except Exception as json_error:
                        print(f"JSON serialization error: {str(json_error)}")
                        # If serialization fails, return a simplified response
                        return {
                            "data": [{"result": "Data contains values that cannot be serialized to JSON"}],
                            "output": output.getvalue(),
                            "plots": plots,
                            "result": str(result) if result is not None else None,
                            "error": f"Warning: Result contains values that cannot be serialized to JSON: {str(json_error)}",
                            "isSuccess": True
                        }

                except Exception as e:
                    error_msg = f"Python Error: {str(e)}\n{traceback.format_exc()}"
                    print(f"Execution error: {error_msg}")
                    # logger.error(error_msg)
                    error_response = {
                        "data": None,
                        "output": output.getvalue(),
                        "plots": plots,
                        "result": None,
                        "error": error_msg,
                        "isSuccess": False
                    }

                    # Use the custom JSON encoder for the error response
                    try:
                        # Test if the response can be serialized
                        safe_json_dumps(error_response)
                        return error_response
                    except Exception as json_error:
                        print(f"JSON serialization error in error response: {str(json_error)}")
                        # If serialization fails, return a simplified response
                        return {
                            "data": None,
                            "output": str(output.getvalue()),
                            "plots": [],
                            "error": f"Error executing code: {str(e)}. Additionally, the response could not be serialized to JSON.",
                            "isSuccess": False
                        }

        else:
            raise HTTPException(status_code=400, detail="Unsupported language")

    except Exception as e:
        # logger.error(f"Execution error: {str(e)}")
        error_response = {
            "error": str(e),
            "isSuccess": False
        }

        # Use the custom JSON encoder for the error response
        try:
            # Test if the response can be serialized
            safe_json_dumps(error_response)
            return error_response
        except Exception as json_error:
            print(f"JSON serialization error in top-level error handler: {str(json_error)}")
            # If serialization fails, return a simplified response
            return {
                "error": f"Error: {str(e)}. Additionally, the response could not be serialized to JSON.",
                "isSuccess": False
            }

# Move root endpoint to top for easier testing
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

@app.post("/api/cleanup-streamlit")
async def cleanup_streamlit(request: Request):
    """Clean up a Streamlit app"""
    try:
        body = await request.json()
        app_id = body.get('app_id')

        if not app_id:
            raise HTTPException(status_code=400, detail="app_id is required")

        cleanup_streamlit_app(app_id)
        return {"message": f"Streamlit app {app_id} cleaned up successfully"}

    except Exception as e:
        logger.error(f"Error cleaning up Streamlit app: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/streamlit-status")
async def streamlit_status():
    """Get status of all running Streamlit apps"""
    try:
        status = {}
        for app_id, app_info in streamlit_processes.items():
            process = app_info['process']
            status[app_id] = {
                'port': app_info['port'],
                'running': process.poll() is None,
                'url': f"http://localhost:{app_info['port']}"
            }
        return {"apps": status}

    except Exception as e:
        logger.error(f"Error getting Streamlit status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(pd.Timestamp.now())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

