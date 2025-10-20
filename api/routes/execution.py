"""
Code Execution API Routes

This module contains all API endpoints related to code execution:
- Python code execution via Jupyter kernel
- JavaScript execution (placeholder)
- Interactive input handling
- Package installation
"""

import logging
import traceback
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from services.kernel.kernel_manager import execute_code_with_kernel, get_kernel
from models.requests import QueryRequest, JSQueryRequest, ContinueExecutionRequest, PackageInstallRequest
from utils.responses import format_execution_response, create_error_response, create_success_response

logger = logging.getLogger(__name__)

# Create router for execution endpoints
router = APIRouter(prefix="/api", tags=["execution"])


@router.post("/execute-pipeline")
async def execute_pipeline_code(request: Request):
    """
    Pipeline code execution endpoint for DataPipeline component
    Supports variable sharing between blocks
    """
    try:
        data = await request.json()
        language = data.get('language', 'python')
        code = data.get('code', '')
        input_data = data.get('input_data', [])
        input_headers = data.get('input_headers', [])
        all_source_data = data.get('all_source_data', [])
        variables = data.get('variables', {})  # Get shared variables from previous blocks

        logger.info(f"Executing pipeline code: {language} with {len(input_data)} rows and {len(variables)} variables")

        # Execute code using kernel manager with variable context
        result = execute_code_with_kernel(
            code=code,
            datasets=[{
                'data': input_data,
                'headers': input_headers,
                'name': 'input_data'
            }] if input_data else [],
            all_source_data=all_source_data,
            variable_context=variables  # Pass variables to kernel
        )

        # Format response using response utilities
        response_data = format_execution_response(result)
        
        # Log plot information for debugging
        if result.get('plots'):
            logger.info(f"📊 Generated {len(result['plots'])} matplotlib plots")
        if result.get('plotly_figures'):
            logger.info(f"📈 Generated {len(result['plotly_figures'])} Plotly figures")
        
        # Return response_data directly - FastAPI will handle JSON serialization with our custom encoder
        return response_data

    except Exception as e:
        logger.error(f"Error in execute_pipeline_code: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(
            error_message=str(e),
            error_details={
                'message': str(e),
                'code': 'ExecutionError',
                'stack': traceback.format_exc()
            }
        )


@router.post("/execute-jupyter")
async def execute_jupyter(request: Request, query_request: QueryRequest):
    """
    Jupyter-like code execution endpoint using the JupyterKernel
    """
    try:
        code = query_request.code
        datasets = query_request.datasets or []
        user_input = getattr(query_request, 'user_input', None)

        # Execute code using kernel manager
        result = execute_code_with_kernel(
            code=code,
            datasets=datasets,
            variable_context=query_request.variableContext,
            user_input=user_input
        )

        # Format response using response utilities
        response_data = format_execution_response(result)
        
        # Log plot information for debugging
        if result.get('plots'):
            logger.info(f"📊 Generated {len(result['plots'])} matplotlib plots")
        if result.get('plotly_figures'):
            logger.info(f"📈 Generated {len(result['plotly_figures'])} Plotly figures")
        
        # Return response_data directly - FastAPI will handle JSON serialization with our custom encoder
        return response_data

    except Exception as e:
        logger.error(f"Error in execute_jupyter: {str(e)}")
        logger.error(traceback.format_exc())
        return create_error_response(
            error_message=str(e),
            error_details={
                'message': str(e),
                'code': 'ExecutionError',
                'stack': traceback.format_exc()
            }
        )


@router.post("/execute")
async def execute_legacy(request: Request, query_request: QueryRequest):
    """
    Legacy endpoint that redirects to the new Jupyter kernel
    Maintains compatibility with existing frontend code
    """
    # Just call the new Jupyter endpoint
    return await execute_jupyter(request, query_request)


@router.post("/python/continue")
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

        # Format response using response utilities
        response_data = format_execution_response(result)
        
        return response_data

    except Exception as e:
        logger.error(f"Error continuing execution: {str(e)}")
        return create_error_response(f"Execution failed: {str(e)}")


@router.post("/python/reset")
async def reset_python_kernel():
    """Reset the Python kernel"""
    try:
        from services.kernel.kernel_manager import reset_kernel
        reset_kernel()
        
        return create_success_response(
            data=[],
            output="Python kernel reset successfully"
        )

    except Exception as e:
        logger.error(f"Error resetting kernel: {str(e)}")
        return create_error_response(f"Kernel reset failed: {str(e)}")


@router.get("/python/status")
async def get_python_kernel_status():
    """Get Python kernel status"""
    try:
        from services.kernel.kernel_manager import get_kernel_status
        status = get_kernel_status()
        
        return create_success_response(
            data=[status],
            output="Kernel status retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting kernel status: {str(e)}")
        return create_error_response(f"Failed to get kernel status: {str(e)}")


@router.post("/python/install")
async def install_package(request: Request, package_request: PackageInstallRequest):
    """Install a Python package"""
    try:
        import subprocess
        import sys
        
        package_name = package_request.package_name
        version = package_request.version
        
        # Construct pip install command
        if version:
            install_cmd = [sys.executable, "-m", "pip", "install", f"{package_name}=={version}"]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", package_name]
        
        # Run pip install with timeout
        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout
        )
        
        if result.returncode == 0:
            return create_success_response(
                data=[],
                output=f"✅ Package '{package_name}' installed successfully\n{result.stdout}"
            )
        else:
            return create_error_response(
                error_message=f"Package installation failed: {result.stderr}",
                error_details={
                    'package': package_name,
                    'version': version,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            )

    except subprocess.TimeoutExpired:
        return create_error_response(
            error_message="Package installation timed out (2 minutes)",
            error_details={'package': package_name, 'timeout': 120}
        )
    except Exception as e:
        logger.error(f"Error installing package: {str(e)}")
        return create_error_response(f"Package installation failed: {str(e)}")


@router.post("/execute-js")
async def execute_javascript(request: Request, query_request: JSQueryRequest):
    """Execute JavaScript code (placeholder for future implementation)"""
    try:
        # For now, return a simple response
        # TODO: Implement JavaScript execution
        return create_success_response(
            data=[],
            output="JavaScript execution not implemented in simplified version"
        )

    except Exception as e:
        logger.error(f"Error in JavaScript execution: {str(e)}")
        return create_error_response(str(e))


@router.get("/python/variables")
async def get_kernel_variables():
    """Get current kernel variables"""
    try:
        from services.kernel.kernel_manager import get_kernel_variables
        variables = get_kernel_variables()
        
        return create_success_response(
            data=[variables],
            output=f"Retrieved {len(variables)} variables"
        )

    except Exception as e:
        logger.error(f"Error getting kernel variables: {str(e)}")
        return create_error_response(f"Failed to get variables: {str(e)}")


@router.post("/python/clear-variables")
async def clear_kernel_variables():
    """Clear all user-defined variables from the kernel"""
    try:
        from services.kernel.kernel_manager import clear_kernel_variables
        clear_kernel_variables()
        
        return create_success_response(
            data=[],
            output="All user-defined variables cleared successfully"
        )

    except Exception as e:
        logger.error(f"Error clearing kernel variables: {str(e)}")
        return create_error_response(f"Failed to clear variables: {str(e)}")