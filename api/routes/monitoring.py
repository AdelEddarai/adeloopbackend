"""
Server Monitoring API Routes

This module contains all API endpoints related to server monitoring:
- Health checks and status reporting
- System resource monitoring
- Package information
- Performance metrics
- WebSocket monitoring for real-time updates
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from services.monitoring.server_monitoring import (
    get_server_status,
    get_installed_packages,
    run_health_check,
    get_system_info,
    get_cpu_info,
    get_memory_info,
    get_disk_info,
    get_network_info,
    get_process_info
)
from models.requests import HealthCheckRequest
from utils.responses import create_success_response, create_error_response
from config.settings import SERVER_START_TIME

logger = logging.getLogger(__name__)

# Create router for monitoring endpoints
router = APIRouter(tags=["monitoring"])

# Active monitoring WebSocket connections
active_monitoring_connections: Dict[str, WebSocket] = {}


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        uptime = time.time() - SERVER_START_TIME
        health_data = run_health_check()
        
        return {
            'status': 'healthy',
            'uptime': uptime,
            'timestamp': time.time(),
            'message': 'HRatlas backend is running',
            'health': health_data
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }


@router.post("/health/detailed")
async def detailed_health_check(health_request: HealthCheckRequest):
    """Detailed health check with optional package information"""
    try:
        health_data = run_health_check()
        
        response_data = {
            'status': health_data.get('status', 'unknown'),
            'timestamp': health_data.get('timestamp'),
            'uptime': time.time() - SERVER_START_TIME,
            'checks': health_data.get('checks', {}),
            'server_info': {
                'start_time': SERVER_START_TIME,
                'current_time': time.time()
            }
        }
        
        if health_request.include_detailed_metrics:
            response_data['system'] = get_system_info()
            response_data['cpu'] = get_cpu_info()
            response_data['memory'] = get_memory_info()
            response_data['disk'] = get_disk_info()
            response_data['network'] = get_network_info()
            response_data['process'] = get_process_info()
        
        if health_request.include_packages:
            response_data['packages'] = get_installed_packages()
        
        return create_success_response(
            data=[response_data],
            output="Detailed health check completed"
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        return create_error_response(f"Health check failed: {str(e)}")


@router.get("/status")
async def server_status():
    """Get comprehensive server status"""
    try:
        status_data = get_server_status()
        
        return create_success_response(
            data=[status_data],
            output="Server status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting server status: {str(e)}")
        return create_error_response(f"Failed to get server status: {str(e)}")


@router.get("/system")
async def system_info():
    """Get system information"""
    try:
        system_data = get_system_info()
        
        return create_success_response(
            data=[system_data],
            output="System information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return create_error_response(f"Failed to get system info: {str(e)}")


@router.get("/cpu")
async def cpu_info():
    """Get CPU information and usage"""
    try:
        cpu_data = get_cpu_info()
        
        return create_success_response(
            data=[cpu_data],
            output="CPU information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting CPU info: {str(e)}")
        return create_error_response(f"Failed to get CPU info: {str(e)}")


@router.get("/memory")
async def memory_info():
    """Get memory information and usage"""
    try:
        memory_data = get_memory_info()
        
        return create_success_response(
            data=[memory_data],
            output="Memory information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting memory info: {str(e)}")
        return create_error_response(f"Failed to get memory info: {str(e)}")


@router.get("/disk")
async def disk_info():
    """Get disk information and usage"""
    try:
        disk_data = get_disk_info()
        
        return create_success_response(
            data=[disk_data],
            output="Disk information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting disk info: {str(e)}")
        return create_error_response(f"Failed to get disk info: {str(e)}")


@router.get("/network")
async def network_info():
    """Get network information"""
    try:
        network_data = get_network_info()
        
        return create_success_response(
            data=[network_data],
            output="Network information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting network info: {str(e)}")
        return create_error_response(f"Failed to get network info: {str(e)}")


@router.get("/process")
async def process_info():
    """Get current process information"""
    try:
        process_data = get_process_info()
        
        return create_success_response(
            data=[process_data],
            output="Process information retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting process info: {str(e)}")
        return create_error_response(f"Failed to get process info: {str(e)}")


@router.get("/packages")
async def installed_packages():
    """Get list of installed Python packages"""
    try:
        packages = get_installed_packages()
        
        return create_success_response(
            data=packages,
            output=f"Retrieved {len(packages)} installed packages"
        )
        
    except Exception as e:
        logger.error(f"Error getting packages: {str(e)}")
        return create_error_response(f"Failed to get packages: {str(e)}")


@router.get("/uptime")
async def server_uptime():
    """Get server uptime information"""
    try:
        uptime_seconds = time.time() - SERVER_START_TIME
        uptime_data = {
            'uptime_seconds': uptime_seconds,
            'uptime_minutes': uptime_seconds / 60,
            'uptime_hours': uptime_seconds / 3600,
            'uptime_days': uptime_seconds / 86400,
            'start_time': SERVER_START_TIME,
            'current_time': time.time(),
            'formatted_uptime': _format_uptime(uptime_seconds)
        }
        
        return create_success_response(
            data=[uptime_data],
            output=f"Server uptime: {uptime_data['formatted_uptime']}"
        )
        
    except Exception as e:
        logger.error(f"Error getting uptime: {str(e)}")
        return create_error_response(f"Failed to get uptime: {str(e)}")


def _format_uptime(seconds: float) -> str:
    """Format uptime in a human-readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


@router.websocket("/ws/monitoring/{client_id}")
async def monitoring_websocket(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time server monitoring

    Best practices implementation:
    - Push-based updates (no polling)
    - Heartbeat/ping-pong mechanism
    - Automatic cleanup on disconnect
    - Error handling and recovery
    """
    await websocket.accept()
    active_monitoring_connections[client_id] = websocket
    logger.info(f"[MonitoringWS] Client {client_id} connected. Active connections: {len(active_monitoring_connections)}")

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to monitoring WebSocket",
            "client_id": client_id
        })

        # Main message loop
        while True:
            try:
                # Wait for client messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                message = json.loads(data)

                message_type = message.get("type", "")

                if message_type == "ping":
                    # Respond to heartbeat
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })

                elif message_type == "request_monitoring_data":
                    # Send current server status
                    try:
                        logger.info(f"[MonitoringWS] Collecting server status for client {client_id}")
                        status_data = get_server_status()

                        # Transform data to match frontend expectations
                        monitoring_data = {
                            "status": "running",
                            "python_version": status_data.get("system", {}).get("python_version", "Unknown"),
                            "server_uptime": time.time() - SERVER_START_TIME,
                            "cpu_usage": status_data.get("cpu", {}).get("usage_percent", 0),
                            "memory": {
                                "total": status_data.get("memory", {}).get("virtual", {}).get("total", 0),
                                "available": status_data.get("memory", {}).get("virtual", {}).get("available", 0),
                                "percent": status_data.get("memory", {}).get("virtual", {}).get("percent", 0),
                                "used": status_data.get("memory", {}).get("virtual", {}).get("used", 0)
                            },
                            "disk": {
                                "total": status_data.get("disk", {}).get("usage", {}).get("total", 0),
                                "free": status_data.get("disk", {}).get("usage", {}).get("free", 0),
                                "used": status_data.get("disk", {}).get("usage", {}).get("used", 0),
                                "percent": status_data.get("disk", {}).get("usage", {}).get("percent", 0)
                            },
                            "process": {
                                "pid": status_data.get("process", {}).get("pid", 0),
                                "memory_info": status_data.get("process", {}).get("memory_info", {}),
                                "cpu_percent": status_data.get("process", {}).get("cpu_percent", 0),
                                "create_time": status_data.get("process", {}).get("create_time", time.time())
                            },
                            "kernel": {
                                "execution_count": 0,
                                "namespace_variables": 0
                            }
                        }

                        await websocket.send_json({
                            "type": "monitoring",
                            "data": monitoring_data,
                            "timestamp": time.time()
                        })

                        logger.info(f"[MonitoringWS] Sent monitoring data to client {client_id}")

                    except Exception as e:
                        logger.error(f"[MonitoringWS] Error getting server status: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Failed to get server status: {str(e)}"
                        })

                else:
                    logger.warning(f"[MonitoringWS] Unknown message type: {message_type}")

            except asyncio.TimeoutError:
                # No message received in 60 seconds, send heartbeat
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": time.time()
                })

            except json.JSONDecodeError as e:
                logger.error(f"[MonitoringWS] Invalid JSON: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format"
                })

    except WebSocketDisconnect:
        logger.info(f"[MonitoringWS] Client {client_id} disconnected normally")

    except Exception as e:
        logger.error(f"[MonitoringWS] Error for client {client_id}: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

    finally:
        # Clean up connection
        if client_id in active_monitoring_connections:
            del active_monitoring_connections[client_id]
        logger.info(f"[MonitoringWS] Client {client_id} cleaned up. Active connections: {len(active_monitoring_connections)}")
