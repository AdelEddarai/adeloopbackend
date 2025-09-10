"""
Streamlit Application API Routes

This module contains all API endpoints related to Streamlit applications:
- Creating and managing Streamlit apps
- Serving Streamlit app pages
- App lifecycle management
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from services.streamlit.streamlit_manager import (
    create_streamlit_app,
    cleanup_streamlit_app,
    serve_streamlit_app_page,
    get_streamlit_code
)
from models.requests import StreamlitRequest
from utils.responses import format_streamlit_response, create_error_response, create_success_response

logger = logging.getLogger(__name__)

# Create router for streamlit endpoints
router = APIRouter(prefix="/api", tags=["streamlit"])


@router.post("/streamlit/create")
async def create_streamlit_application(request: Request, streamlit_request: StreamlitRequest):
    """Create a new Streamlit application"""
    try:
        code = streamlit_request.code
        app_id = streamlit_request.app_id or f"app_{int(time.time())}"
        
        # Create the Streamlit app
        app_info = create_streamlit_app(code, app_id)
        
        # Format response using response utilities
        response_data = format_streamlit_response(app_info)
        
        return response_data

    except Exception as e:
        logger.error(f"Error creating Streamlit app: {str(e)}")
        return create_error_response(f"Failed to create Streamlit app: {str(e)}")


@router.delete("/streamlit/{app_id}")
async def stop_streamlit_app(app_id: str):
    """Stop and cleanup a Streamlit application"""
    try:
        result = cleanup_streamlit_app(app_id)
        
        if result.get('status') == 'success':
            return create_success_response(
                data=[result],
                output=result.get('message', 'Streamlit app stopped successfully')
            )
        else:
            return create_error_response(
                error_message=result.get('message', 'Failed to stop Streamlit app'),
                error_details=result
            )

    except Exception as e:
        logger.error(f"Error stopping Streamlit app: {str(e)}")
        return create_error_response(f"Failed to stop Streamlit app: {str(e)}")


@router.get("/streamlit/{app_id}/status")
async def get_streamlit_app_status(app_id: str):
    """Get status of a Streamlit application"""
    try:
        from services.streamlit.streamlit_manager import streamlit_apps
        
        if app_id in streamlit_apps:
            app_info = streamlit_apps[app_id]
            status_data = {
                'app_id': app_id,
                'status': app_info.get('status', 'unknown'),
                'url': app_info.get('url'),
                'port': app_info.get('port'),
                'created_at': app_info.get('created_at')
            }
            
            return create_success_response(
                data=[status_data],
                output=f"Streamlit app {app_id} status retrieved"
            )
        else:
            return create_error_response(
                error_message=f"Streamlit app {app_id} not found",
                error_details={'app_id': app_id, 'available_apps': list(streamlit_apps.keys())}
            )

    except Exception as e:
        logger.error(f"Error getting Streamlit app status: {str(e)}")
        return create_error_response(f"Failed to get app status: {str(e)}")


@router.get("/streamlit/apps")
async def list_streamlit_apps():
    """List all active Streamlit applications"""
    try:
        from services.streamlit.streamlit_manager import streamlit_apps
        
        apps_list = []
        for app_id, app_info in streamlit_apps.items():
            apps_list.append({
                'app_id': app_id,
                'status': app_info.get('status', 'unknown'),
                'url': app_info.get('url'),
                'port': app_info.get('port'),
                'created_at': app_info.get('created_at')
            })
        
        return create_success_response(
            data=apps_list,
            output=f"Found {len(apps_list)} active Streamlit apps"
        )

    except Exception as e:
        logger.error(f"Error listing Streamlit apps: {str(e)}")
        return create_error_response(f"Failed to list apps: {str(e)}")


# Non-API routes for serving Streamlit app pages
streamlit_router = APIRouter(tags=["streamlit-pages"])


@streamlit_router.get("/streamlit/{app_id}", response_class=HTMLResponse)
async def serve_streamlit_app(app_id: str):
    """Serve the Streamlit app page with code and instructions"""
    try:
        html_content = serve_streamlit_app_page(app_id)
        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error serving Streamlit app page: {str(e)}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Streamlit App {app_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f2f6; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .error {{ color: #ff4b4b; font-size: 18px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="error">Error Loading Streamlit App</h1>
                <p>Failed to load Streamlit app "{app_id}"</p>
                <p>Error: {str(e)}</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@streamlit_router.get("/streamlit/{app_id}/code")
async def get_streamlit_app_code(app_id: str):
    """Get the source code for a Streamlit app"""
    try:
        code = get_streamlit_code(app_id)
        
        if code:
            return create_success_response(
                data=[{'app_id': app_id, 'code': code}],
                output=f"Retrieved code for Streamlit app {app_id}"
            )
        else:
            return create_error_response(
                error_message=f"No code found for Streamlit app {app_id}",
                error_details={'app_id': app_id}
            )

    except Exception as e:
        logger.error(f"Error getting Streamlit app code: {str(e)}")
        return create_error_response(f"Failed to get app code: {str(e)}")


@streamlit_router.post("/streamlit/{app_id}/restart")
async def restart_streamlit_app(app_id: str):
    """Restart a Streamlit application"""
    try:
        # Get the existing code
        code = get_streamlit_code(app_id)
        
        if not code:
            return create_error_response(
                error_message=f"Cannot restart app {app_id}: no code found",
                error_details={'app_id': app_id}
            )
        
        # Stop the existing app
        cleanup_result = cleanup_streamlit_app(app_id)
        
        # Create a new app with the same code
        app_info = create_streamlit_app(code, app_id)
        
        # Format response
        response_data = format_streamlit_response(app_info)
        response_data['restarted'] = True
        response_data['cleanup_result'] = cleanup_result
        
        return response_data

    except Exception as e:
        logger.error(f"Error restarting Streamlit app: {str(e)}")
        return create_error_response(f"Failed to restart Streamlit app: {str(e)}")


# Import time for app_id generation
import time
