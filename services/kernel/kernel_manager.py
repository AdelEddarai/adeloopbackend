"""
Jupyter Kernel Management Module

This module handles the lifecycle and management of Jupyter kernel instances.
It provides functionality for:
- Creating and managing kernel instances
- Handling kernel state and persistence
- Managing variable contexts across executions
- Cleanup and resource management
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from services.kernel.jupyter_kernel import get_kernel as _get_kernel, reset_kernel as _reset_kernel

logger = logging.getLogger(__name__)


def get_kernel():
    """
    Get or create the global Jupyter kernel instance
    Uses the existing kernel management from jupyter_kernel module

    Returns:
        JupyterKernel: The active kernel instance
    """
    return _get_kernel()


def reset_kernel():
    """
    Reset the kernel by creating a new instance
    Uses the existing reset functionality from jupyter_kernel module

    Returns:
        JupyterKernel: New kernel instance
    """
    return _reset_kernel()


def execute_code_with_kernel(
    code: str, 
    datasets: list = None, 
    variable_context: Dict[str, Any] = None,
    user_input: str = None,
    all_source_data: list = None
) -> Dict[str, Any]:
    """
    Execute code using the managed kernel instance
    
    Args:
        code: Python code to execute
        datasets: List of datasets to make available
        variable_context: Variables to inject into execution context
        user_input: User input for interactive execution
        all_source_data: All source data from pipeline nodes for advanced access
        
    Returns:
        Dictionary containing execution results
    """
    kernel = get_kernel()
    
    try:
        import os
        
        # Store original working directory
        original_cwd = os.getcwd()
        
        # Handle user input if provided
        if user_input is not None:
            logger.debug(f"Received user input: {user_input}")
            # Use the provide_input_and_continue method for better input handling
            result = kernel.provide_input_and_continue(user_input, code)
            logger.debug(f"Input continuation completed with status: {result.get('status', 'unknown')}")
            return result

        # Prepare datasets if provided
        if datasets:
            _prepare_datasets_for_kernel(kernel, datasets)
        
        # Prepare all source data if provided (for pipeline nodes)
        if all_source_data:
            _prepare_all_source_data_for_kernel(kernel, all_source_data)
        
        # Inject variable context if provided
        if variable_context:
            _inject_variable_context(kernel, variable_context)
        
        # Ensure we're in the kernel temp directory for CSV access
        if hasattr(kernel, 'temp_dir') and os.path.exists(kernel.temp_dir):
            os.chdir(kernel.temp_dir)
            logger.debug(f"Changed working directory to: {kernel.temp_dir}")
        
        # Execute the code
        result = kernel.execute_code(code, datasets)
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        # Ensure both matplotlib and Plotly plots are captured
        if 'plots' not in result:
            result['plots'] = []
        if 'plotly_figures' not in result:
            result['plotly_figures'] = []
            
        # Capture any additional plots that might have been generated
        additional_plots = kernel.capture_plots()
        if additional_plots:
            result['plots'].extend(additional_plots)
            
        additional_plotly = kernel.capture_plotly_figures()
        if additional_plotly:
            result['plotly_figures'].extend(additional_plotly)
        
        # Add execution history and variable history to result
        if hasattr(kernel, 'execution_history'):
            result['execution_history'] = kernel.execution_history
        if hasattr(kernel, 'variables_history'):
            result['variable_history'] = kernel.variables_history
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing code with kernel: {e}")
        # Restore original working directory on error
        try:
            import os
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
        except:
            pass
            
        kernel = get_kernel()  # Get kernel for error handling
        return {
            'status': 'error',
            'error': str(e),
            'stdout': '',
            'stderr': str(e),
            'plots': [],
            'execution_count': getattr(kernel, 'execution_count', 0),
            'execution_history': getattr(kernel, 'execution_history', []),
            'variable_history': getattr(kernel, 'variables_history', {})
        }


def _prepare_datasets_for_kernel(kernel, datasets: list):
    """
    Prepare and inject datasets into the kernel namespace
    Handles both local datasets and external database tables
    
    Args:
        kernel: The kernel instance
        datasets: List of dataset dictionaries
    """
    try:
        import pandas as pd
        import os
        
        # Change working directory to kernel temp directory so pd.read_csv() works
        original_cwd = os.getcwd()
        os.chdir(kernel.temp_dir)
        
        # Store original directory in kernel namespace for reference
        kernel.namespace['_original_cwd'] = original_cwd
        kernel.namespace['_kernel_temp_dir'] = kernel.temp_dir
        
        for i, dataset in enumerate(datasets):
            if not dataset:
                continue
                
            # Check if this is an external table (has externalMetadata but no data)
            is_external = dataset.get('isExternal', False)
            has_data = 'data' in dataset and dataset['data']
            external_metadata = dataset.get('externalMetadata', {})
            
            # If it's an external table without data, fetch it automatically
            if is_external and not has_data and external_metadata:
                logger.info(f"🔄 Auto-fetching external table: {dataset.get('name')}")
                try:
                    df_data = _fetch_external_table_data(external_metadata)
                    if df_data is not None and not df_data.empty:
                        dataset['data'] = df_data.to_dict('records')
                        has_data = True
                        logger.info(f"✅ Fetched {len(df_data)} rows from external table: {dataset.get('name')}")
                    else:
                        logger.warning(f"⚠️ External table {dataset.get('name')} returned no data")
                        continue
                except Exception as fetch_error:
                    logger.error(f"❌ Failed to fetch external table {dataset.get('name')}: {fetch_error}")
                    # Create an empty DataFrame as fallback
                    df_data = pd.DataFrame()
                    logger.warning(f"Using empty DataFrame for {dataset.get('name')}")
            
            # Now process the dataset (either local or fetched external)
            if has_data:
                df_data = pd.DataFrame(dataset['data'])
                dataset_name = dataset.get('name', f'Dataset {i+1}')

                # Create safe variable name
                safe_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
                safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                if not safe_name or safe_name[0].isdigit():
                    safe_name = f'dataset_{safe_name}' if safe_name else f'dataset_{i+1}'

                # Store datasets with multiple names for flexibility
                kernel.namespace[safe_name] = df_data
                kernel.namespace[f'df{i+1}'] = df_data

                if i == 0:
                    kernel.namespace['df'] = df_data

                # Create virtual CSV files so pd.read_csv() works
                # Use the original dataset name (with .csv extension)
                csv_filename = f"{dataset_name}.csv" if dataset_name else f"dataset{i+1}.csv"
                csv_path = os.path.join(kernel.temp_dir, csv_filename)
                df_data.to_csv(csv_path, index=False)

                # Also create numbered CSV files
                numbered_csv = os.path.join(kernel.temp_dir, f"dataset{i+1}.csv")
                df_data.to_csv(numbered_csv, index=False)
                
                # Create a version with safe name
                safe_csv = os.path.join(kernel.temp_dir, f"{safe_name}.csv")
                df_data.to_csv(safe_csv, index=False)
                
                source_type = "external" if is_external else "local"
                logger.debug(f"Prepared {source_type} dataset: {safe_name} with shape {df_data.shape}")
                logger.debug(f"Created CSV files: {csv_filename}, {safe_name}.csv, dataset{i+1}.csv")
        
        # Create a helper variable to show available datasets
        if datasets:
            dataset_info = []
            for i, dataset in enumerate(datasets):
                if dataset and 'data' in dataset:
                    dataset_name = dataset.get('name', f'Dataset {i+1}')
                    safe_name = dataset_name.lower().replace(' ', '_').replace('-', '_')
                    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                    if not safe_name or safe_name[0].isdigit():
                        safe_name = f'dataset_{safe_name}' if safe_name else f'dataset_{i+1}'
                    
                    is_external = dataset.get('isExternal', False)
                    dataset_info.append({
                        'name': dataset_name,
                        'variable': safe_name,
                        'csv_file': f"{dataset_name}.csv",
                        'shape': (len(dataset['data']), len(dataset['data'][0]) if dataset['data'] else 0),
                        'type': 'external' if is_external else 'local'
                    })
            
            kernel.namespace['_available_datasets'] = dataset_info
            logger.info(f"Prepared {len(datasets)} datasets. Available as: {[d['variable'] for d in dataset_info]}")
                
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")


def _fetch_external_table_data(external_metadata: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Fetch data from an external database table
    
    Args:
        external_metadata: Dictionary containing connection details and table name
        
    Returns:
        pandas DataFrame with the table data, or None if fetch fails
    """
    try:
        import pandas as pd
        
        db_type = external_metadata.get('type', '').lower()
        host = external_metadata.get('host')
        port = external_metadata.get('port')
        database = external_metadata.get('database')
        table = external_metadata.get('table')
        username = external_metadata.get('username')
        password = external_metadata.get('password')
        
        if not all([host, database, table]):
            logger.error(f"Missing required connection details: host={host}, database={database}, table={table}")
            return None
        
        logger.info(f"Connecting to {db_type} database: {host}:{port}/{database}")
        
        # PostgreSQL
        if db_type == 'postgresql':
            try:
                import psycopg2
                conn_string = f"host='{host}' port='{port}' dbname='{database}'"
                if username:
                    conn_string += f" user='{username}'"
                if password:
                    conn_string += f" password='{password}'"
                    
                conn = psycopg2.connect(conn_string)
                query = f'SELECT * FROM "{table}" LIMIT 10000'
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            except ImportError:
                logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
                return None
        
        # MySQL
        elif db_type == 'mysql':
            try:
                import pymysql
                conn = pymysql.connect(
                    host=host,
                    port=int(port) if port else 3306,
                    user=username or 'root',
                    password=password or '',
                    database=database
                )
                query = f'SELECT * FROM `{table}` LIMIT 10000'
                df = pd.read_sql(query, conn)
                conn.close()
                return df
            except ImportError:
                logger.error("pymysql not installed. Install with: pip install pymysql")
                return None
        
        # MongoDB
        elif db_type == 'mongodb':
            try:
                from pymongo import MongoClient
                conn_string = f"mongodb://"
                if username and password:
                    conn_string += f"{username}:{password}@"
                conn_string += f"{host}:{port}/{database}"
                
                client = MongoClient(conn_string)
                db = client[database]
                collection = db[table]
                data = list(collection.find().limit(10000))
                
                # Convert MongoDB documents to DataFrame
                if data:
                    # Remove _id field if present
                    for doc in data:
                        if '_id' in doc:
                            doc['_id'] = str(doc['_id'])
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame()
                    
                client.close()
                return df
            except ImportError:
                logger.error("pymongo not installed. Install with: pip install pymongo")
                return None
        
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching external table data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _prepare_all_source_data_for_kernel(kernel, all_source_data: list):
    """
    Prepare and inject all source data from pipeline nodes into the kernel namespace
    
    Args:
        kernel: The kernel instance
        all_source_data: List of source data dictionaries from pipeline nodes
    """
    try:
        import pandas as pd
        import os
        
        # Create a special variable to hold all source data
        source_data_dict = {}
        
        for i, source in enumerate(all_source_data):
            if source and 'data' in source:
                df_data = pd.DataFrame(source['data'])
                source_name = source.get('name', f'Source {i+1}')
                
                # Create safe variable name
                safe_name = source_name.lower().replace(' ', '_').replace('-', '_')
                safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
                if not safe_name or safe_name[0].isdigit():
                    safe_name = f'source_{safe_name}' if safe_name else f'source_{i+1}'
                
                # Store in kernel namespace
                kernel.namespace[safe_name] = df_data
                
                # Store in source data dictionary for easy access
                source_data_dict[safe_name] = df_data
                source_data_dict[f'source_{i+1}'] = df_data
                
                logger.debug(f"Prepared source data: {safe_name} with shape {df_data.shape}")
        
        # Make all source data available as a special variable
        kernel.namespace['sourceNodesData'] = source_data_dict
        kernel.namespace['all_sources'] = source_data_dict
        
        logger.debug(f"Prepared {len(source_data_dict)} source datasets for pipeline node")
                
    except Exception as e:
        logger.error(f"Error preparing all source data: {e}")


def _inject_variable_context(kernel, variable_context: Dict[str, Any]):
    """
    Inject variable context into the kernel namespace
    
    Args:
        kernel: The kernel instance
        variable_context: Dictionary of variables to inject
    """
    try:
        for name, value in variable_context.items():
            # Ensure variable name is safe
            if isinstance(name, str) and name.isidentifier() and not name.startswith('_'):
                kernel.namespace[name] = value
                logger.debug(f"Injected variable: {name} = {type(value).__name__}")
            else:
                logger.warning(f"Skipped invalid variable name: {name}")
                
    except Exception as e:
        logger.error(f"Error injecting variable context: {e}")


def get_kernel_variables() -> Dict[str, Any]:
    """
    Get current kernel variables (user-defined only)
    
    Returns:
        Dictionary of user-defined variables
    """
    try:
        kernel = get_kernel()
        user_variables = {}
        
        for name, value in kernel.namespace.items():
            if (not name.startswith('_') and 
                not callable(value) and 
                not hasattr(value, '__module__') and
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random',
                           'get_ipython', 'display', 'display_image', 'display_video', 'ls', 'pwd', 'history', 'who', 'whos', 'reset']):
                try:
                    # Only include serializable variables
                    import pandas as pd
                    import numpy as np
                    
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
        
        return user_variables
        
    except Exception as e:
        logger.error(f"Error getting kernel variables: {e}")
        return {}


def clear_kernel_variables():
    """Clear all user-defined variables from the kernel"""
    try:
        kernel = get_kernel()
        
        # Get list of user-defined variable names
        user_vars = []
        for name in kernel.namespace.keys():
            if (not name.startswith('_') and 
                not callable(kernel.namespace[name]) and 
                not hasattr(kernel.namespace[name], '__module__') and
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random',
                           'get_ipython', 'display', 'display_image', 'display_video', 'ls', 'pwd', 'history', 'who', 'whos', 'reset']):
                user_vars.append(name)
        
        # Remove user-defined variables
        for name in user_vars:
            del kernel.namespace[name]
            
        logger.info(f"Cleared {len(user_vars)} user-defined variables")
        
    except Exception as e:
        logger.error(f"Error clearing kernel variables: {e}")


def get_kernel_status() -> Dict[str, Any]:
    """
    Get current kernel status and information
    
    Returns:
        Dictionary containing kernel status information
    """
    try:
        kernel = get_kernel()
        variables = get_kernel_variables()
        
        return {
            'status': 'active',
            'execution_count': getattr(kernel, 'execution_count', 0),
            'variables_count': len(variables),
            'variables': variables,  # Include current variables
            'execution_history': getattr(kernel, 'execution_history', []),
            'has_temp_dir': hasattr(kernel, 'temp_dir'),
            'temp_dir': getattr(kernel, 'temp_dir', None)
        }
        
    except Exception as e:
        logger.error(f"Error getting kernel status: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }