"""
ZeroMQ-based kernel worker for distributed code execution with streaming support

This module implements a kernel worker that:
- Connects to the FastAPI gateway via DEALER socket
- Publishes streaming output via PUB socket
- Maintains persistent namespace per kernel session
- Handles code execution requests from the gateway
"""

import zmq
import zmq.asyncio
import asyncio
import json
import sys
import io
import uuid
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
import tempfile
import os
import warnings

# Import the existing JupyterKernel for code execution
from services.kernel.jupyter_kernel import JupyterKernel, InputRequiredException

class KernelWorker:
    """
    A ZeroMQ-based kernel worker that executes Python code and streams output
    """
    
    def __init__(self, gateway_router_addr: str = "tcp://localhost:5555", 
                 gateway_pub_addr: str = "tcp://localhost:5556",
                 worker_id: str = None):
        """
        Initialize the kernel worker
        
        Args:
            gateway_router_addr: Address of the gateway's ROUTER socket
            gateway_pub_addr: Address of the gateway's PUB socket for streaming
            worker_id: Unique identifier for this worker
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.gateway_router_addr = gateway_router_addr
        self.gateway_pub_addr = gateway_pub_addr
        
        # Initialize ZeroMQ context
        self.ctx = zmq.asyncio.Context()
        
        # DEALER socket for communication with gateway (requests/responses)
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.worker_id)
        
        # PUB socket for streaming output to gateway
        self.pub = self.ctx.socket(zmq.PUB)
        
        # Kernel namespace storage (kernel_id -> namespace)
        self.namespace_store = {}
        
        # Active kernels (kernel_id -> JupyterKernel instance)
        self.active_kernels = {}
        
        print(f"🔧 Kernel worker {self.worker_id} initialized")
    
    async def connect(self):
        """Connect to the gateway sockets"""
        try:
            self.dealer.connect(self.gateway_router_addr)
            self.pub.connect(self.gateway_pub_addr)
            print(f"✅ Kernel worker {self.worker_id} connected to gateway")
            print(f"   ROUTER: {self.gateway_router_addr}")
            print(f"   PUB: {self.gateway_pub_addr}")
        except Exception as e:
            print(f"❌ Failed to connect to gateway: {e}")
            raise
    
    async def start_kernel(self, kernel_id: str) -> Dict[str, Any]:
        """
        Start a new kernel session
        
        Args:
            kernel_id: Unique identifier for the kernel session
            
        Returns:
            Dictionary with kernel startup information
        """
        try:
            # Create new kernel instance
            kernel = JupyterKernel()
            self.active_kernels[kernel_id] = kernel
            self.namespace_store[kernel_id] = kernel.namespace
            
            print(f"✅ Kernel {kernel_id} started by worker {self.worker_id}")
            
            return {
                "type": "kernel_started",
                "kernel_id": kernel_id,
                "worker_id": self.worker_id,
                "status": "ok"
            }
        except Exception as e:
            print(f"❌ Failed to start kernel {kernel_id}: {e}")
            return {
                "type": "kernel_start_error",
                "kernel_id": kernel_id,
                "worker_id": self.worker_id,
                "error": str(e),
                "status": "error"
            }
    
    async def execute_code_streaming(self, kernel_id: str, code: str, msg_id: str, 
                                   datasets: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute code with real-time streaming output
        
        Args:
            kernel_id: Kernel session identifier
            code: Python code to execute
            msg_id: Message identifier for this execution
            datasets: Optional datasets to make available to the code
            
        Returns:
            Dictionary with execution result
        """
        # Get or create kernel
        if kernel_id not in self.active_kernels:
            kernel_result = await self.start_kernel(kernel_id)
            if kernel_result.get("status") != "ok":
                return kernel_result
            kernel = self.active_kernels[kernel_id]
        else:
            kernel = self.active_kernels[kernel_id]
        
        try:
            # Store original stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create custom stream handler for real-time output
            class StreamingOutput:
                def __init__(self, worker, kernel_id, msg_id):
                    self.worker = worker
                    self.kernel_id = kernel_id
                    self.msg_id = msg_id
                    self.buffer = ""
                    self.chunk_id = 0
                
                def write(self, data):
                    self.buffer += data
                    # Send data in chunks as lines
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Keep incomplete line
                    for line in lines[:-1]:
                        if line.strip():
                            asyncio.create_task(self.worker.stream_output(
                                self.kernel_id, self.msg_id, line + '\n', self.chunk_id))
                            self.chunk_id += 1
                
                def flush(self):
                    # Send any remaining buffered data
                    if self.buffer:
                        asyncio.create_task(self.worker.stream_output(
                            self.kernel_id, self.msg_id, self.buffer, self.chunk_id))
                        self.chunk_id += 1
                        self.buffer = ""
            
            # Replace stdout/stderr with streaming handler
            streaming_handler = StreamingOutput(self, kernel_id, msg_id)
            sys.stdout = streaming_handler
            sys.stderr = streaming_handler
            
            # Execute code using existing kernel method
            result = kernel.execute_code(code, datasets)
            
            # Flush any remaining output
            streaming_handler.flush()
            
            # Send final execution result
            await self.stream_output(kernel_id, msg_id, "", -1, is_last=True, result=result)
            
            return result
            
        except InputRequiredException as e:
            # Handle interactive input requests
            stdout_text = getattr(sys.stdout, 'buffer', '') if hasattr(sys.stdout, 'buffer') else ''
            
            # Flush any remaining output
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Send input request
            await self.stream_output(kernel_id, msg_id, "", -1, is_last=True, 
                                   input_required=True, input_prompt=e.prompt)
            
            return {
                'status': 'input_required',
                'input_prompt': e.prompt,
                'stdout': stdout_text
            }
            
        except Exception as e:
            # Handle execution errors
            error_info = {
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': traceback.format_exc().split('\n')
            }
            
            # Flush any remaining output
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Send error result
            await self.stream_output(kernel_id, msg_id, "", -1, is_last=True, error=error_info)
            
            return {
                'status': 'error',
                'error': error_info,
                'stdout': '',
                'stderr': str(e)
            }
            
        finally:
            # Always restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    async def stream_output(self, kernel_id: str, msg_id: str, content: str, 
                           chunk_id: int, is_last: bool = False, 
                           result: Dict[str, Any] = None,
                           input_required: bool = False,
                           input_prompt: str = "",
                           error: Dict[str, Any] = None):
        """
        Stream output chunk to the gateway
        
        Args:
            kernel_id: Kernel session identifier
            msg_id: Message identifier
            content: Output content chunk
            chunk_id: Chunk sequence number
            is_last: Whether this is the final chunk
            result: Final execution result (for last chunk)
            input_required: Whether input is required (for last chunk)
            input_prompt: Input prompt (for input requests)
            error: Error information (for error chunks)
        """
        try:
            # Prepare streaming message
            message = {
                "type": "stream_output",
                "kernel_id": kernel_id,
                "msg_id": msg_id,
                "chunk_id": chunk_id,
                "content": content,
                "is_last": is_last,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Add result/error/input info for final chunk
            if is_last:
                if result:
                    message["result"] = result
                if input_required:
                    message["input_required"] = True
                    message["input_prompt"] = input_prompt
                if error:
                    message["error"] = error
            
            # Send via PUB socket
            await self.pub.send_json(message)
            
            # Small delay to allow event loop to process
            await asyncio.sleep(0.001)
            
        except Exception as e:
            print(f"❌ Error streaming output: {e}")
    
    async def handle_messages(self):
        """
        Main message handling loop for the worker
        """
        print(f"🚀 Kernel worker {self.worker_id} listening for messages...")
        
        try:
            while True:
                # Receive message from gateway
                parts = await self.dealer.recv_multipart()
                
                # Extract message (DEALER framing: [empty][payload] or [payload])
                raw_msg = parts[-1]  # Last part is the actual message
                msg = json.loads(raw_msg.decode())
                
                msg_type = msg.get("type")
                kernel_id = msg.get("kernel_id")
                msg_id = msg.get("msg_id")
                
                print(f"📥 Worker {self.worker_id} received {msg_type} for kernel {kernel_id}")
                
                # Handle different message types
                if msg_type == "start_kernel":
                    # Start a new kernel session
                    result = await self.start_kernel(kernel_id)
                    await self.dealer.send_json(result)
                    
                elif msg_type == "execute_request":
                    # Execute code in kernel
                    code = msg.get("code", "")
                    datasets = msg.get("datasets", [])
                    
                    # Execute with streaming
                    await self.execute_code_streaming(kernel_id, code, msg_id, datasets)
                    
                elif msg_type == "shutdown_worker":
                    # Gracefully shutdown worker
                    print(f"🛑 Worker {self.worker_id} shutting down...")
                    break
                    
                else:
                    # Unknown message type
                    error_response = {
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}",
                        "kernel_id": kernel_id,
                        "msg_id": msg_id
                    }
                    await self.dealer.send_json(error_response)
                    
        except Exception as e:
            print(f"❌ Error in message handling loop: {e}")
            traceback.print_exc()
        finally:
            # Cleanup
            self.dealer.close()
            self.pub.close()
            self.ctx.term()
            print(f"🧹 Worker {self.worker_id} cleaned up")

# Standalone worker execution
async def main():
    """Main entry point for running the kernel worker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZeroMQ Kernel Worker")
    parser.add_argument("--router", default="tcp://localhost:5555", 
                       help="Gateway ROUTER socket address")
    parser.add_argument("--pub", default="tcp://localhost:5556", 
                       help="Gateway PUB socket address")
    parser.add_argument("--id", help="Worker ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Create and start worker
    worker = KernelWorker(
        gateway_router_addr=args.router,
        gateway_pub_addr=args.pub,
        worker_id=args.id
    )
    
    # Connect to gateway
    await worker.connect()
    
    # Notify gateway that worker is ready
    ready_msg = {
        "type": "worker_ready",
        "worker_id": worker.worker_id
    }
    await worker.dealer.send_json(ready_msg)
    
    # Start message handling loop
    await worker.handle_messages()

if __name__ == "__main__":
    asyncio.run(main())