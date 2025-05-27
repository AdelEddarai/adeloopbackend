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
    datasetId: str

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
            df = await get_dataset_from_nextjs(query_request.datasetId, auth_token)

            if df is None:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Initialize plot style
            PlotManager.setup_default_style()

            # Capture stdout and prepare namespace
            output = io.StringIO()
            plots: List[str] = []

            # Prepare the execution environment with enhanced capabilities
            local_ns = {
                # Core libraries
                'pd': pd,
                'df': df,
                'np': np,
                'plt': plt,
                'sns': sns,
                'os': os,
                'sys': __import__('sys'),
                'io': io,
                'json': json,
                'datetime': __import__('datetime'),
                're': __import__('re'),

                # Utilities
                'describe': lambda x: pd.DataFrame(x.describe()),
                'get_plot': lambda: PlotManager.get_plot_as_base64(),  # Make it a function call
                'analyzer': DataAnalyzer(),
                'available_columns': list(df.columns),

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

            # Add file reading capability (safely)
            def safe_read_file(file_path, mode='r'):
                """Safely read a file with various formats supported"""
                import os
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Check file extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.csv':
                    return pd.read_csv(file_path)
                elif ext == '.xlsx' or ext == '.xls':
                    return pd.read_excel(file_path)
                elif ext == '.json':
                    return pd.read_json(file_path)
                elif ext == '.parquet':
                    return pd.read_parquet(file_path)
                elif ext == '.pickle' or ext == '.pkl':
                    return pd.read_pickle(file_path)
                else:
                    # Default to text file
                    with open(file_path, mode) as f:
                        return f.read()

            # Add these to the namespace
            local_ns['import_module'] = custom_import
            local_ns['read_file'] = safe_read_file

            # Execute the code with Jupyter-like behavior
            with redirect_stdout(output):
                try:
                    # Split the code into cells like Jupyter
                    code_cells = query_request.query.split('# %%')
                    if len(code_cells) == 1:  # No cell markers, treat as single cell
                        code_cells = [query_request.query]

                    # Execute each cell
                    cell_results = []
                    for i, cell in enumerate(code_cells):
                        if not cell.strip():
                            continue

                        print(f"Executing cell {i+1}..." if len(code_cells) > 1 else "Executing code...")

                        # Execute the cell
                        exec(cell, {}, local_ns)

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

                    # Format the response
                    response_data = {
                        "data": None,
                        "output": output.getvalue(),
                        "plots": plots,
                        "error": None,
                        "isSuccess": True
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(pd.Timestamp.now())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
