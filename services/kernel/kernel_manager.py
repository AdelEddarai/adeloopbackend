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
        
        # Execute the code
        result = kernel.execute_code(code, datasets)
        
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
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing code with kernel: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'stdout': '',
            'stderr': str(e),
            'plots': [],
            'execution_count': getattr(kernel, 'execution_count', 0)
        }


def _prepare_datasets_for_kernel(kernel, datasets: list):
    """
    Prepare and inject datasets into the kernel namespace
    
    Args:
        kernel: The kernel instance
        datasets: List of dataset dictionaries
    """
    try:
        import pandas as pd
        import os
        
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
                kernel.namespace[safe_name] = df_data
                kernel.namespace[f'df{i+1}'] = df_data

                if i == 0:
                    kernel.namespace['df'] = df_data

                # Create virtual CSV files so pd.read_csv() works
                csv_filename = f"{dataset_name}.csv" if dataset_name else f"dataset{i+1}.csv"
                csv_path = os.path.join(kernel.temp_dir, csv_filename)
                df_data.to_csv(csv_path, index=False)

                # Also create numbered CSV files
                numbered_csv = os.path.join(kernel.temp_dir, f"dataset{i+1}.csv")
                df_data.to_csv(numbered_csv, index=False)
                
                logger.debug(f"Prepared dataset: {safe_name} with shape {df_data.shape}")
                
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")


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
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random']):
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
                name not in ['pd', 'np', 'plt', 'json', 'sys', 'io', 'warnings', 'os', 'datetime', 're', 'math', 'random']):
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
            'has_temp_dir': hasattr(kernel, 'temp_dir'),
            'temp_dir': getattr(kernel, 'temp_dir', None)
        }
        
    except Exception as e:
        logger.error(f"Error getting kernel status: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }