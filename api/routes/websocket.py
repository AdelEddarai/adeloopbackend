"""
WebSocket API Routes for real-time code execution

This module handles WebSocket connections for:
- Real-time Python code execution
- Interactive input/output handling
- Kernel management
- Progress updates
"""

import json
import logging
import traceback
from typing import Dict, Any
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from services.kernel.kernel_manager import get_kernel, execute_code_with_kernel, reset_kernel
from services.kernel.jupyter_kernel import InputRequiredException
from utils.responses import CustomJSONEncoder  # Add this import

logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time code execution"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "message": "Connected to Python kernel"
    }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
    
    kernel = get_kernel()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received WebSocket message: {message.get('type', 'unknown')}")
            
            # Handle different message types
            if message["type"] == "execute":
                await handle_code_execution(websocket, message, kernel)
            elif message["type"] == "provide_input":
                await handle_input_provision(websocket, message, kernel)
            elif message["type"] == "cancel_execution":
                await handle_execution_cancellation(websocket, kernel)
            elif message["type"] == "reset_kernel":
                await handle_kernel_reset(websocket, kernel)
            else:
                logger.warning(f"Unknown WebSocket message type: {message['type']}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": f"Unknown message type: {message['type']}"
                }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        logger.error(traceback.format_exc())
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": f"Server error: {str(e)}"
            }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
    finally:
        logger.info("WebSocket connection closed")

async def handle_code_execution(websocket: WebSocket, message: Dict[str, Any], kernel):
    """Handle code execution requests"""
    try:
        code = message.get("code", "")
        cell_id = message.get("cell_id", "unknown")
        language = message.get("language", "python")
        datasets = message.get("datasets", [])
        
        logger.info(f"Executing {language} code in cell {cell_id}")
        
        # Send execution start notification
        await websocket.send_text(json.dumps({
            "type": "execution_start",
            "cell_id": cell_id
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
        # Execute code using kernel manager
        result = execute_code_with_kernel(
            code=code,
            datasets=datasets
        )
        
        # Send result back to client
        response = {
            "type": "execution_result",
            "cell_id": cell_id,
            "status": result.get("status", "ok"),
            "output": result.get("stdout", ""),
            "error": result.get("stderr", "") or (result.get("error", {}).get("evalue", "") if result.get("error") else ""),
            "plots": result.get("plots", []),
            "data": result.get("data"),
            "result": result.get("result"),
            "variables": result.get("variables", {}),
            "execution_count": result.get("execution_count", 0)
        }
        
        # Include error details if present
        if result.get("error"):
            response["error_details"] = {
                "message": result["error"].get("evalue", "") if isinstance(result["error"], dict) else result["error"],
                "traceback": result["error"].get("traceback", []) if isinstance(result["error"], dict) else []
            }
        
        await websocket.send_text(json.dumps(response, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except InputRequiredException as e:
        # Handle input request
        logger.info(f"Input required: {e.prompt}")
        await websocket.send_text(json.dumps({
            "type": "input_request",
            "prompt": e.prompt,
            "original_code": code
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e),
            "error_details": {
                "message": str(e),
                "traceback": traceback.format_exc().split('\n')
            }
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder

async def handle_input_provision(websocket: WebSocket, message: Dict[str, Any], kernel):
    """Handle input provision for interactive execution"""
    try:
        user_input = message.get("input", "")
        original_code = message.get("original_code", "")
        
        logger.info(f"Providing input: {user_input}")
        
        # Continue execution with the provided input
        result = kernel.provide_input_and_continue(user_input, original_code)
        
        # Send result back to client
        response = {
            "type": "execution_result",
            "status": result.get("status", "ok"),
            "output": result.get("stdout", ""),
            "error": result.get("stderr", "") or (result.get("error", {}).get("evalue", "") if result.get("error") else ""),
            "plots": result.get("plots", []),
            "data": result.get("data"),
            "result": result.get("result"),
            "variables": result.get("variables", {}),
            "execution_count": result.get("execution_count", 0)
        }
        
        # Include error details if present
        if result.get("error"):
            response["error_details"] = {
                "message": result["error"].get("evalue", "") if isinstance(result["error"], dict) else result["error"],
                "traceback": result["error"].get("traceback", []) if isinstance(result["error"], dict) else []
            }
        
        await websocket.send_text(json.dumps(response, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except Exception as e:
        logger.error(f"Input provision error: {str(e)}")
        logger.error(traceback.format_exc())
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e),
            "error_details": {
                "message": str(e),
                "traceback": traceback.format_exc().split('\n')
            }
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder

async def handle_execution_cancellation(websocket: WebSocket, kernel):
    """Handle execution cancellation"""
    try:
        # TODO: Implement proper execution cancellation
        logger.info("Execution cancellation requested")
        
        await websocket.send_text(json.dumps({
            "type": "execution_cancelled",
            "message": "Execution cancelled"
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except Exception as e:
        logger.error(f"Cancellation error: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e)
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder

async def handle_kernel_reset(websocket: WebSocket, kernel):
    """Handle kernel reset"""
    try:
        logger.info("Resetting kernel")
        reset_kernel()
        
        await websocket.send_text(json.dumps({
            "type": "kernel_reset",
            "message": "Kernel reset successfully"
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except Exception as e:
        logger.error(f"Kernel reset error: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e)
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
