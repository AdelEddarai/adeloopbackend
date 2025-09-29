"""
API Middleware Module

This module contains middleware configurations for the FastAPI application:
- CORS middleware
- Authentication middleware (if needed)
- Request/Response logging
- Error handling middleware
"""

import logging
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.responses import CustomJSONEncoder
import json

from config.settings import CORS_ORIGINS, CORS_CREDENTIALS, CORS_METHODS, CORS_HEADERS

# Create custom JSON response class in middleware to avoid circular imports
class CustomJSONResponse(JSONResponse):
    """Custom JSON response that uses our CustomJSONEncoder"""
    
    def render(self, content) -> bytes:
        # Pre-process content to clean any DataFrames before encoding
        cleaned_content = self._clean_content_for_json(content)
        return json.dumps(
            cleaned_content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(',', ':'),
        ).encode('utf-8')
    
    def _clean_content_for_json(self, obj):
        """Recursively clean content to ensure JSON compatibility"""
        import pandas as pd
        import numpy as np
        
        if isinstance(obj, pd.DataFrame):
            # Use the same cleaning logic as in our CustomJSONEncoder
            try:
                cleaned_df = obj.copy()
                cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
                
                for col in cleaned_df.columns:
                    if cleaned_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
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
                        cleaned_df[col] = cleaned_df[col].fillna('')
                
                # Final check: ensure no NaN or infinite values remain
                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                    mask = pd.isna(cleaned_df[col]) | np.isinf(cleaned_df[col])
                    if mask.any():
                        cleaned_df.loc[mask, col] = 0
                        
                return cleaned_df.to_dict('records')
            except Exception:
                # If cleaning fails, return empty list to avoid JSON errors
                return []
        
        elif isinstance(obj, dict):
            # Recursively clean dictionary values
            return {key: self._clean_content_for_json(value) for key, value in obj.items()}
        
        elif isinstance(obj, list):
            # Recursively clean list items
            return [self._clean_content_for_json(item) for item in obj]
        
        elif isinstance(obj, (int, float)):
            # Handle individual numeric values
            if np.isnan(obj) or np.isinf(obj):
                return 0
            return obj
        
        else:
            # Return other types as-is
            return obj

logger = logging.getLogger(__name__)


def setup_cors_middleware(app: FastAPI):
    """
    Setup CORS middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=CORS_CREDENTIALS,
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
    )
    logger.info("CORS middleware configured")


def setup_logging_middleware(app: FastAPI):
    """
    Setup LIGHTWEIGHT request/response logging middleware
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable):
        """Log incoming requests and responses - MINIMAL"""
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Only log errors and important endpoints
            if response.status_code >= 400 or request.url.path in ["/health", "/ping"]:
                process_time = time.time() - start_time
                logger.info(
                    f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
                )
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s"
            )
            
            # Return error response using our custom JSON encoder
            return CustomJSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e)
                }
            )


def setup_error_handling_middleware(app: FastAPI):
    """
    Setup global error handling middleware
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions globally"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        
        return CustomJSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "path": str(request.url.path),
                "method": request.method,
                "type": type(exc).__name__
            }
        )
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Handle 404 errors"""
        return CustomJSONResponse(
            status_code=404,
            content={
                "error": "Not found",
                "message": f"The requested resource was not found",
                "path": str(request.url.path),
                "method": request.method
            }
        )


def setup_security_middleware(app: FastAPI):
    """
    Setup security-related middleware
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable):
        """Add security headers to responses"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


def setup_rate_limiting_middleware(app: FastAPI):
    """
    Setup basic rate limiting middleware (simple implementation)
    
    Args:
        app: FastAPI application instance
    """
    # Simple in-memory rate limiting (for production, use Redis or similar)
    request_counts = {}
    
    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Callable):
        """Basic rate limiting based on client IP"""
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        request_counts[client_ip] = [
            timestamp for timestamp in request_counts.get(client_ip, [])
            if timestamp > cutoff_time
        ]
        
        # Check rate limit (max 1000 requests per minute per IP - INCREASED)
        # Exclude health check endpoints from rate limiting
        if request.url.path not in ["/health", "/ping", "/", "/ws-test"]:
            if len(request_counts.get(client_ip, [])) >= 1000:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return CustomJSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": 60
                    }
                )
        
        # Add current request timestamp
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


def setup_all_middleware(app: FastAPI):
    """
    Setup all middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    # Setup middleware in order (last added is executed first)
    setup_cors_middleware(app)
    setup_security_middleware(app)
    setup_rate_limiting_middleware(app)
    setup_error_handling_middleware(app)
    setup_logging_middleware(app)
    
    logger.info("All middleware configured successfully")
