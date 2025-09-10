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
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utils.responses import CustomJSONEncoder
import json

# Import configuration
from config.settings import APP_TITLE, APP_VERSION, APP_DESCRIPTION, LOG_LEVEL

# Import middleware setup
from api.middleware import setup_all_middleware

# Import route modules
from api.routes.execution import router as execution_router
from api.routes.monitoring import router as monitoring_router
from api.routes.streamlit import router as streamlit_router, streamlit_router

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

    # Initialize services if needed
    try:
        # Test kernel initialization
        from services.kernel.kernel_manager import get_kernel
        kernel = get_kernel()
        logger.info("✅ Jupyter kernel initialized successfully")

        # Test monitoring services
        from services.monitoring.server_monitoring import get_system_info
        system_info = get_system_info()
        logger.info(f"✅ Monitoring services initialized - System: {system_info.get('system', 'Unknown')}")

    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")

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
    """Health check endpoint for frontend to verify backend availability"""
    try:
        # Test kernel availability
        from services.kernel.kernel_manager import get_kernel
        kernel = get_kernel()
        
        return {
            "status": "healthy",
            "service": "HRatlas Backend API",
            "version": APP_VERSION,
            "kernel_status": "available",
            "timestamp": None  # Can add datetime.now() if needed
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "HRatlas Backend API", 
            "version": APP_VERSION,
            "kernel_status": "unavailable",
            "error": str(e)
        }



if __name__ == "__main__":
    import uvicorn
    from config.settings import HOST, PORT

    logger.info(f"🚀 Starting HRatlas Backend server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
