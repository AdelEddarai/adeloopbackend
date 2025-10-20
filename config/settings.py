"""
Configuration Settings Module

This module centralizes all configuration settings for the HRatlas backend.
It handles environment variables, default values, and application settings.
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server Configuration
NEXTJS_API_URL = os.getenv('NEXTJS_API_URL', 'http://localhost:3000')
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))

# Redis Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
ENABLE_REDIS = os.getenv('ENABLE_REDIS', 'false').lower() == 'true'  # Disable Redis by default

# Server Tracking
SERVER_START_TIME = time.time()

# CORS Configuration
CORS_ORIGINS = ["*"]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING')

# Streamlit Configuration
STREAMLIT_START_PORT = 5151
STREAMLIT_CLOUD_BASE_URL = "https://flopbackend.onrender.com"

# Application Metadata
APP_TITLE = "HRatlas Backend"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = """
HRatlas Backend API Server

This is the main FastAPI server that provides:
- Python code execution via Jupyter kernel
- SQL query execution  
- Dataset management
- Streamlit app creation and management
- Server monitoring and health checks
"""

# Cloud Environment Detection
def is_cloud_environment() -> bool:
    """Check if running on a cloud platform"""
    return bool(
        os.getenv('RENDER_SERVICE_NAME') or 
        os.getenv('RAILWAY_STATIC_URL') or 
        os.getenv('HEROKU_APP_NAME')
    )

# Environment Info
ENVIRONMENT = {
    'is_cloud': is_cloud_environment(),
    'render_service': os.getenv('RENDER_SERVICE_NAME'),
    'railway_url': os.getenv('RAILWAY_STATIC_URL'),
    'heroku_app': os.getenv('HEROKU_APP_NAME')
}
