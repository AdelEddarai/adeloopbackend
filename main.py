"""
HRatlas Backend API Server (Refactored)

This is the main FastAPI server that provides:
- Python code execution via Jupyter kernel
- SQL query execution
- Dataset management
- Streamlit app creation and management
- Server monitoring and health checks

The server uses modular components for better maintainability.
"""

import logging
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utils.responses import CustomJSONEncoder
import json
import asyncio

# Import Redis manager
from services.kernel.redis_manager import redis_manager

# Import configuration
from config.settings import APP_TITLE, APP_VERSION, APP_DESCRIPTION, LOG_LEVEL, ENABLE_REDIS, SERVER_START_TIME

# Import middleware setup
from api.middleware import setup_all_middleware

# Import route modules
from api.routes.execution import router as execution_router
from api.routes.monitoring import router as monitoring_router
from api.routes.streamlit import router as streamlit_router, streamlit_router
from api.routes.websocket import router as websocket_router, init_zmq_sockets, close_zmq_sockets, forward_streaming_output

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


# Custom JSON response class that handles NaN/Infinity values
class CustomJSONResponse(JSONResponse):
    """Custom JSON response that uses our CustomJSONEncoder"""
    
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(',', ':'),
        ).encode('utf-8')


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with custom JSON response class
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        default_response_class=CustomJSONResponse  # Use our custom JSON encoder
    )

    # Setup all middleware
    setup_all_middleware(app)

    # Include API routers
    app.include_router(execution_router)
    app.include_router(monitoring_router)
    app.include_router(streamlit_router)
    app.include_router(websocket_router)  # Add WebSocket routes

    # Include non-API routers (for serving pages)
    app.include_router(streamlit_router)

    logger.info(f"FastAPI application created: {APP_TITLE} v{APP_VERSION}")
    return app


# Create the FastAPI application
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("🚀 HRatlas Backend starting up...")
    logger.info(f"📊 Application: {APP_TITLE} v{APP_VERSION}")
    logger.info(f"📊 Redis enabled: {ENABLE_REDIS}")

    # DO NOT initialize heavy services on startup - causes blocking
    # Services will be initialized on first use (lazy loading)
    
    # Initialize Redis manager (will fallback to in-memory if Redis not available)
    try:
        await redis_manager.connect()
        if redis_manager.use_redis and ENABLE_REDIS:
            logger.info("✅ Redis manager initialized")
        else:
            logger.info("⚠️ Redis disabled or not available, using in-memory storage")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Redis: {e}")
    
    # Initialize ZeroMQ sockets for kernel workers (if available)
    try:
        init_zmq_sockets()
        # Start background task for streaming output forwarding
        asyncio.create_task(forward_streaming_output())
        logger.info("✅ ZeroMQ integration initialized")
    except Exception as e:
        logger.warning(f"⚠️ ZeroMQ not available: {e}")
        # Continue without ZeroMQ - fallback to direct execution
    
    logger.info("🎉 HRatlas Backend startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("🛑 HRatlas Backend shutting down...")

    # Cleanup services if needed
    try:
        # Cleanup kernel resources
        from services.kernel.jupyter_kernel import _kernel_instance
        if _kernel_instance:
            _kernel_instance.cleanup()
            logger.info("✅ Kernel resources cleaned up")

        # Cleanup Streamlit apps
        from services.streamlit.streamlit_manager import streamlit_apps
        for app_id in list(streamlit_apps.keys()):
            from services.streamlit.streamlit_manager import cleanup_streamlit_app
            cleanup_streamlit_app(app_id)
        logger.info("✅ Streamlit apps cleaned up")

        # Close Redis connection
        try:
            await redis_manager.disconnect()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis: {e}")
        
        # Close ZeroMQ sockets
        close_zmq_sockets()
        logger.info("✅ ZeroMQ sockets closed")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")

    logger.info("👋 HRatlas Backend shutdown complete!")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HRatlas Backend API",
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Fast health check endpoint - NO BLOCKING OPERATIONS"""
    return {
        "status": "healthy",
        "service": "HRatlas Backend API",
        "version": APP_VERSION
    }

# Quick ping endpoint for connectivity test
@app.get("/ping")
async def ping():
    """Ultra-fast ping endpoint"""
    return {"status": "pong", "timestamp": time.time()}

# Backend status endpoint for frontend monitoring
@app.get("/backend-status")
async def backend_status():
    """Quick backend status check for frontend monitoring"""
    return {
        "status": "online",
        "service": "HRatlas Backend API",
        "version": APP_VERSION,
        "uptime": time.time() - SERVER_START_TIME,
        "timestamp": time.time()
    }

# WebSocket test endpoint
@app.get("/ws-test")
async def websocket_test():
    """Test endpoint to verify WebSocket setup"""
    return {
        "websocket_available": True,
        "endpoint": "/ws/execute/{client_id}",
        "status": "ready"
    }



if __name__ == "__main__":
    import uvicorn
    from config.settings import HOST, PORT

    logger.info(f"🚀 Starting HRatlas Backend server on {HOST}:{PORT}")
    
    # Use uvicorn with better configuration for stability
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        log_level="warning",  # Reduce log noise
        access_log=False,     # Disable access logs for performance
        workers=1             # Single worker to avoid resource conflicts
    )