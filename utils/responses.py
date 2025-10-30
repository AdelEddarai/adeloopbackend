"""
API Response Utilities Module

This module provides utilities for formatting and processing API responses.
It handles:
- Standardizing response formats across different endpoints
- Processing execution results into consistent structures
- Handling error responses and logging
- Data type conversions and serialization
"""

import json
import math
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special data types"""
    
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
            # Check if array contains objects that need special handling
            if obj.dtype == object:
                # Convert to list first to handle special objects like range
                obj_list = obj.tolist()
                # Process the list to handle any remaining special objects
                return self._clean_object_list(obj_list)
            else:
                # Clean numeric numpy arrays of NaN/infinity values
                try:
                    cleaned_array = np.where(np.isfinite(obj), obj, 0)
                    return cleaned_array.tolist()
                except (TypeError, ValueError):
                    # If isfinite fails, convert to list and handle element by element
                    obj_list = obj.tolist()
                    return self._clean_object_list(obj_list)
        elif isinstance(obj, pd.DataFrame):
            # Clean DataFrame before converting to dict
            cleaned_df = self._clean_dataframe_for_serialization(obj)
            return cleaned_df.to_dict('records')
        elif isinstance(obj, pd.Series):
            # Clean Series before converting to list
            cleaned_series = obj.replace([np.inf, -np.inf], np.nan).fillna(0)
            return cleaned_series.tolist()
        elif isinstance(obj, range):
            # Convert range objects to lists
            return list(obj)
        elif hasattr(obj, '__module__') and obj.__module__:
            # Handle modules and module objects
            return {
                "type": "module",
                "name": getattr(obj, '__name__', str(obj)),
                "module": obj.__module__
            }
        elif callable(obj):
            # Handle functions and callable objects
            return {
                "type": "function",
                "name": getattr(obj, '__name__', str(obj)),
                "repr": str(obj)
            }
        elif hasattr(obj, '__dict__'):
            # Handle custom objects with attributes
            try:
                return {
                    "type": type(obj).__name__,
                    "repr": str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
                }
            except:
                return {"type": type(obj).__name__, "repr": "<object>"}
        return super().default(obj)
    
    def _clean_object_list(self, obj_list):
        """Clean a list that may contain special objects"""
        cleaned_list = []
        for item in obj_list:
            if isinstance(item, range):
                # Convert range objects to lists
                cleaned_list.append(list(item))
            elif isinstance(item, (list, tuple)):
                # Recursively clean nested lists/tuples
                cleaned_list.append(self._clean_object_list(list(item)))
            elif isinstance(item, float) and math.isnan(item):
                # Replace NaN with None
                cleaned_list.append(None)
            elif isinstance(item, float) and math.isinf(item):
                # Replace infinity with None
                cleaned_list.append(None)
            else:
                # Keep the item as is
                cleaned_list.append(item)
        return cleaned_list
    
    def _clean_dataframe_for_serialization(self, df):
        """Clean DataFrame to ensure JSON serialization compatibility"""
        try:
            # Make a copy to avoid modifying original data
            cleaned_df = df.copy()
            
            # Replace infinite values with NaN first
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
            
            # Handle different data types
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # For numeric columns, fill with 0 or median
                    if cleaned_df[col].notna().any():
                        try:
                            median_val = cleaned_df[col].median()
                            if pd.isna(median_val) or np.isinf(median_val):
                                median_val = 0
                            cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        except:
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # For non-numeric columns, fill with empty string
                    cleaned_df[col] = cleaned_df[col].fillna('')
            
            # Final check: ensure no NaN or infinite values remain
            for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                mask = pd.isna(cleaned_df[col]) | np.isinf(cleaned_df[col])
                if mask.any():
                    cleaned_df.loc[mask, col] = 0
                    
            return cleaned_df
            
        except Exception:
            # Return original dataframe if cleaning fails
            return df


def format_execution_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format execution result into standardized API response
    
    Args:
        result: Raw execution result from kernel
        
    Returns:
        Formatted response dictionary
    """
    try:
        # Combine matplotlib and Plotly plots into a single plots array
        all_plots = []
        
        # Add matplotlib plots (base64 images)
        matplotlib_plots = result.get('plots', [])
        for plot in matplotlib_plots:
            if isinstance(plot, str) and plot.strip():
                # Ensure proper data URL format
                if not plot.startswith('data:'):
                    plot = f"data:image/png;base64,{plot}"
                all_plots.append(plot)
        
        # Add Plotly plots (HTML) - these should be rendered directly
        plotly_plots = result.get('plotly_figures', [])
        for plot in plotly_plots:
            if isinstance(plot, str) and plot.strip():
                # Ensure Plotly HTML is properly formatted
                if '<div' in plot or 'plotly' in plot.lower():
                    all_plots.append(plot)

        # Create the formatted response with unified plots array
        formatted_response = {
            'status': result.get('status', 'ok'),
            'output': result.get('stdout', ''),
            'error': result.get('stderr', '') or (result.get('error', {}).get('evalue', '') if result.get('error') else ''),
            'plots': all_plots,  # Unified plots array containing both matplotlib and plotly
            'execution_count': result.get('execution_count', 0),
            
            # Include structured data for tables
            'data': result.get('data'),
            'result': result.get('result'),
            
            # Include metadata
            'plot_count': len(matplotlib_plots),
            'plotly_count': len(plotly_plots),
            'total_plots': len(all_plots)
        }
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error formatting execution response: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'output': '',
            'plots': [],
            'execution_count': 0
        }



def process_result_data(result_data: Any) -> Dict[str, Any]:
    """
    Process and format result data based on its type
    
    Args:
        result_data: The result data to process
        
    Returns:
        Dictionary with processed data and metadata
    """
    try:
        response = {'result': result_data}

        # Handle DataFrame results
        if isinstance(result_data, pd.DataFrame):
            # Clean DataFrame of NaN and Infinity values before serialization
            cleaned_df = result_data.copy()
            
            # Replace infinite values with NaN, then fill NaN with appropriate values
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults based on column type
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # For numeric columns, fill with 0 or median
                    if cleaned_df[col].notna().any():
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # For non-numeric columns, fill with empty string
                    cleaned_df[col] = cleaned_df[col].fillna('')
            
            response.update({
                'data': cleaned_df.to_dict('records'),
                'columns': [{'name': col, 'type': str(cleaned_df[col].dtype)} for col in cleaned_df.columns],
                'outputType': 'dataframe',
                'result': {
                    'type': 'table',
                    'data': cleaned_df.to_dict('records'),
                    'columns': list(cleaned_df.columns)
                }
            })

        # Handle list/array results
        elif isinstance(result_data, (list, np.ndarray)):
            if isinstance(result_data, np.ndarray):
                # Clean numpy arrays of NaN/Infinity
                cleaned_array = np.where(np.isfinite(result_data), result_data, 0)
                result_data = cleaned_array.tolist()
            else:
                # Clean list of any NaN/Infinity values
                cleaned_list = []
                for item in result_data:
                    if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
                        cleaned_list.append(0)  # Replace with 0
                    else:
                        cleaned_list.append(item)
                result_data = cleaned_list
            
            response.update({
                'data': result_data,
                'outputType': 'array'
            })

        # Handle dictionary results
        elif isinstance(result_data, dict):
            response.update({
                'data': [result_data],
                'outputType': 'object'
            })

        # Handle scalar results
        else:
            response.update({
                'data': [{'value': result_data}],
                'outputType': 'scalar'
            })

        return response

    except Exception as e:
        logger.error(f"Error processing result data: {str(e)}")
        return {
            'result': str(result_data),
            'data': [],
            'outputType': 'error',
            'error': str(e)
        }


def create_error_response(
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_message: Main error message
        error_details: Additional error details
        status_code: HTTP status code
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        'status': 'error',
        'error': error_message,
        'data': [],
        'output': '',
        'plots': []
    }
    
    if error_details:
        response['errorDetails'] = error_details
    
    return response


def create_success_response(
    data: List[Dict] = None,
    output: str = '',
    plots: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        output: Text output
        plots: List of plot images
        **kwargs: Additional response fields
        
    Returns:
        Standardized success response dictionary
    """
    response = {
        'status': 'success',
        'data': data or [],
        'output': output,
        'plots': plots or []
    }
    
    response.update(kwargs)
    return response


def format_streamlit_response(app_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format Streamlit app creation response
    
    Args:
        app_info: Streamlit app information
        
    Returns:
        Formatted Streamlit response
    """
    try:
        return {
            'status': 'success',
            'data': [app_info],
            'output': app_info.get('message', 'Streamlit app created successfully'),
            'plots': [],
            'streamlit': app_info
        }
        
    except Exception as e:
        logger.error(f"Error formatting Streamlit response: {str(e)}")
        return create_error_response(f"Streamlit response formatting failed: {str(e)}")


def safe_json_serialize(data: Any) -> str:
    """
    Safely serialize data to JSON using custom encoder
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False)
    except Exception as e:
        logger.error(f"JSON serialization error: {str(e)}")
        return json.dumps({'error': f'Serialization failed: {str(e)}'})
