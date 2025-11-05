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

# ZeroMQ imports (optional)
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None
    zmq.asyncio = None

from services.kernel.kernel_manager import get_kernel, execute_code_with_kernel, reset_kernel
from services.kernel.jupyter_kernel import InputRequiredException
from services.kernel.redis_manager import redis_manager
from utils.responses import CustomJSONEncoder  # Add this import
from config.settings import ENABLE_REDIS

logger = logging.getLogger(__name__)

# ZeroMQ context and sockets (initialized at startup)
zmq_ctx = None
router_socket = None  # ROUTER socket for worker communication
sub_socket = None      # SUB socket for receiving streaming output

# Client registry (maps client WebSocket to kernel sessions)
client_registry = {}

# In-memory fallback for kernel tracking when Redis is not available
kernel_worker_map = {}  # kernel_id -> worker_id
worker_kernel_map = {}  # worker_id -> set of kernel_ids

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
    
    # Initialize client session
    kernel = get_kernel()
    kernel_sessions = {}  # Track kernel sessions for this client
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"Received WebSocket message: {message.get('type', 'unknown')}")
            
            # Handle different message types
            if message["type"] == "execute":
                # Check if we should use ZeroMQ streaming mode
                # Only use streaming if ZeroMQ is available AND Redis is enabled
                if message.get("streaming", False) and ZMQ_AVAILABLE and ENABLE_REDIS:
                    await handle_code_execution_streaming(websocket, message)
                else:
                    # Legacy mode - preserve existing functionality
                    # Also fallback to legacy mode when Redis is disabled
                    await handle_code_execution(websocket, message, kernel)
            elif message["type"] == "provide_input":
                await handle_input_provision(websocket, message, kernel)
            elif message["type"] == "cancel_execution":
                await handle_execution_cancellation(websocket, kernel)
            elif message["type"] == "reset_kernel":
                await handle_kernel_reset(websocket, kernel)
            elif message["type"] == "start_kernel_session":
                # Start a new kernel session for ZeroMQ mode
                # Only allow kernel sessions when Redis is enabled
                if ENABLE_REDIS:
                    await handle_start_kernel_session(websocket, message)
                else:
                    # Send error response when Redis is disabled
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Kernel sessions require Redis to be enabled"
                    }, cls=CustomJSONEncoder))
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
        # Clean up client registry
        unregister_client(websocket)
        logger.info("WebSocket connection closed")

async def handle_code_execution(websocket: WebSocket, message: Dict[str, Any], kernel):
    """Handle code execution requests with real-time streaming"""
    try:
        code = message.get("code", "")
        cell_id = message.get("cell_id", "unknown")
        language = message.get("language", "python")
        datasets = message.get("datasets", [])
        is_streaming_request = message.get("streaming", False)

        logger.info(f"Executing {language} code in cell {cell_id}")

        # Send execution start notification
        await websocket.send_text(json.dumps({
            "type": "execution_start",
            "cell_id": cell_id
        }, cls=CustomJSONEncoder))

        # Prepare datasets before execution
        if datasets:
            from services.kernel.kernel_manager import _prepare_datasets_for_kernel
            _prepare_datasets_for_kernel(kernel, datasets)
            logger.info(f"Prepared {len(datasets)} datasets for execution")

        # Use modern streaming executor for enterprise-grade real-time output
        from services.kernel.streaming_executor import StreamingExecutor
        executor = StreamingExecutor(kernel)

        # Check if streaming is requested
        use_realtime = message.get('streaming', False)

        if use_realtime:
            # Use real-time streaming for better performance than Jupyter/Databricks
            async for chunk in executor.execute_realtime_streaming(code, cell_id):
                await websocket.send_text(json.dumps(chunk, cls=CustomJSONEncoder))
        else:
            # Use standard streaming
            async for chunk in executor.execute_streaming(code, cell_id):
                await websocket.send_text(json.dumps(chunk, cls=CustomJSONEncoder))

    except InputRequiredException as e:
        # Handle input request
        logger.info(f"Input required: {e.prompt}")
        await websocket.send_text(json.dumps({
            "type": "input_request",
            "prompt": e.prompt,
            "original_code": code
        }, cls=CustomJSONEncoder))

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
        }, cls=CustomJSONEncoder))


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

async def handle_code_execution_streaming(websocket: WebSocket, message: Dict[str, Any]):
    """Handle code execution requests with ZeroMQ streaming"""
    try:
        code = message.get("code", "")
        cell_id = message.get("cell_id", "unknown")
        language = message.get("language", "python")
        datasets = message.get("datasets", [])
        kernel_id = message.get("kernel_id", f"kernel_{int(asyncio.get_event_loop().time() * 1000000)}")
        
        logger.info(f"Executing {language} code in cell {cell_id} with streaming (kernel: {kernel_id})")
        
        # Send execution start notification
        await websocket.send_text(json.dumps({
            "type": "execution_start",
            "cell_id": cell_id,
            "kernel_id": kernel_id
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
        # Send execute request to worker
        msg_id = await send_execute_request_to_worker(kernel_id, code, datasets)
        
        # Register this kernel session
        register_kernel_session(kernel_id, "", websocket)  # Worker ID will be set when worker claims kernel
        
        logger.info(f"📤 Execute request sent (msg_id: {msg_id})")
        
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

async def handle_start_kernel_session(websocket: WebSocket, message: Dict[str, Any]):
    """Handle kernel session start requests"""
    try:
        kernel_id = message.get("kernel_id")
        if not kernel_id:
            # Generate new kernel ID
            kernel_id = f"kernel_{int(asyncio.get_event_loop().time() * 1000000)}"
        
        logger.info(f"Starting kernel session: {kernel_id}")
        
        # Send start kernel request to worker
        start_request = {
            "type": "start_kernel",
            "kernel_id": kernel_id
        }
        
        # Send to any available worker
        await router_socket.send_multipart([b"", json.dumps(start_request).encode()])
        
        # Register kernel session
        register_kernel_session(kernel_id, "", websocket)
        
        # Send response to client
        await websocket.send_text(json.dumps({
            "type": "kernel_session_started",
            "kernel_id": kernel_id
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder
        
    except Exception as e:
        logger.error(f"Kernel session start error: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e)
        }, cls=CustomJSONEncoder))  # Use CustomJSONEncoder

# ZeroMQ Integration Functions

def init_zmq_sockets():
    """Initialize ZeroMQ sockets for worker communication"""
    global zmq_ctx, router_socket, sub_socket
    
    if not ZMQ_AVAILABLE:
        logger.warning("⚠️ ZeroMQ not available, skipping initialization")
        return
    
    try:
        # Create ZeroMQ context
        zmq_ctx = zmq.asyncio.Context()
        
        # Create ROUTER socket for worker communication
        router_socket = zmq_ctx.socket(zmq.ROUTER)
        router_socket.bind("tcp://*:5555")  # Workers connect here
        
        # Create SUB socket for receiving streaming output
        sub_socket = zmq_ctx.socket(zmq.SUB)
        sub_socket.bind("tcp://*:5556")  # Workers publish here
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        logger.info("✅ ZeroMQ sockets initialized")
        logger.info("   ROUTER: tcp://*:5555")
        logger.info("   SUB: tcp://*:5556")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ZeroMQ sockets: {e}")
        raise

def close_zmq_sockets():
    """Close ZeroMQ sockets and context"""
    global zmq_ctx, router_socket, sub_socket
    
    if not ZMQ_AVAILABLE:
        return
    
    try:
        if router_socket:
            router_socket.close()
        if sub_socket:
            sub_socket.close()
        if zmq_ctx:
            zmq_ctx.term()
        logger.info("🧹 ZeroMQ sockets closed")
    except Exception as e:
        logger.error(f"❌ Error closing ZeroMQ sockets: {e}")

def register_kernel_session(kernel_id: str, worker_id: str, websocket: WebSocket):
    """Register a kernel session with its worker and client"""
    global client_registry, kernel_worker_map, worker_kernel_map
    
    try:
        # Register kernel to worker mapping in Redis or fallback to in-memory
        if redis_manager.use_redis and ENABLE_REDIS:
            import asyncio
            asyncio.create_task(redis_manager.register_kernel(kernel_id, worker_id, str(id(websocket))))
        else:
            # Use in-memory fallback
            kernel_worker_map[kernel_id] = worker_id
            if worker_id not in worker_kernel_map:
                worker_kernel_map[worker_id] = set()
            worker_kernel_map[worker_id].add(kernel_id)
        
        # Register client to kernel mapping
        if websocket not in client_registry:
            client_registry[websocket] = []
        client_registry[websocket].append(kernel_id)
        
        logger.info(f"📝 Registered kernel {kernel_id} with worker {worker_id}")
    except Exception as e:
        logger.error(f"❌ Failed to register kernel session: {e}")

def unregister_client(websocket: WebSocket):
    """Unregister a client and its kernel sessions"""
    global client_registry
    
    if websocket in client_registry:
        kernel_ids = client_registry.pop(websocket, [])
        logger.info(f"🗑️ Unregistered client with kernels: {kernel_ids}")

async def get_worker_for_kernel(kernel_id: str) -> str:
    """Get the worker ID for a kernel session"""
    try:
        if redis_manager.use_redis and ENABLE_REDIS:
            worker_id = await redis_manager.get_worker_for_kernel(kernel_id)
            return worker_id or ""
        else:
            # Use in-memory fallback
            return kernel_worker_map.get(kernel_id, "")
    except Exception as e:
        logger.error(f"❌ Failed to get worker for kernel {kernel_id}: {e}")
        return ""

async def forward_streaming_output():
    """Background task to forward streaming output from workers to clients"""
    global sub_socket, client_registry
    
    if not ZMQ_AVAILABLE or not sub_socket:
        logger.warning("⚠️ ZeroMQ not available, skipping streaming output forwarder")
        return
    
    logger.info("🚀 Starting streaming output forwarder...")
    
    try:
        while True:
            # Receive streaming message from worker
            message = await sub_socket.recv_json()
            
            # Extract message details
            msg_type = message.get("type", "")
            kernel_id = message.get("kernel_id", "")
            
            # Forward to appropriate clients
            if msg_type == "stream_output":
                # Find clients interested in this kernel
                clients_to_notify = []
                for websocket, kernel_ids in client_registry.items():
                    if kernel_id in kernel_ids and websocket.client_state == WebSocketState.CONNECTED:
                        clients_to_notify.append(websocket)
                
                # Forward to all interested clients
                for websocket in clients_to_notify:
                    try:
                        await websocket.send_text(json.dumps(message, cls=CustomJSONEncoder))
                    except Exception as e:
                        logger.error(f"❌ Error forwarding to client: {e}")
                        
            # Small delay to prevent busy looping
            await asyncio.sleep(0.001)
            
    except Exception as e:
        logger.error(f"❌ Error in streaming forwarder: {e}")
        logger.error(traceback.format_exc())

async def send_execute_request_to_worker(kernel_id: str, code: str, datasets: list = None):
    """Send execute request to appropriate worker via ZeroMQ"""
    global router_socket
    
    if not ZMQ_AVAILABLE or not router_socket:
        # Fallback to direct execution without ZeroMQ
        logger.warning("⚠️ ZeroMQ not available, falling back to direct execution")
        raise Exception("ZeroMQ not available")
    
    try:
        # Generate unique message ID
        msg_id = f"msg_{int(asyncio.get_event_loop().time() * 1000000)}"
        
        # Prepare execute request
        execute_request = {
            "type": "execute_request",
            "kernel_id": kernel_id,
            "msg_id": msg_id,
            "code": code,
            "datasets": datasets or []
        }
        
        # Get worker for this kernel from Redis or fallback
        worker_id = await get_worker_for_kernel(kernel_id)
        
        if worker_id:
            # Route to specific worker
            logger.info(f"📤 Routing execute request to worker {worker_id} for kernel {kernel_id}")
            await router_socket.send_multipart([
                worker_id.encode(), 
                b"", 
                json.dumps(execute_request).encode()
            ])
        else:
            # Send to any available worker (load balancing)
            logger.info(f"📤 Sending execute request to any worker for kernel {kernel_id}")
            await router_socket.send_multipart([b"", json.dumps(execute_request).encode()])
            
        return msg_id
        
    except Exception as e:
        logger.error(f"❌ Error sending execute request: {e}")
        raise