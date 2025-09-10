"""
Request Models Module

This module defines all Pydantic models for API request validation.
It includes models for:
- Python code execution requests
- SQL query requests  
- JavaScript execution requests
- Dataset and variable context handling
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class QueryRequest(BaseModel):
    """Model for Python code execution requests"""
    code: str
    datasets: Optional[List[Dict[str, Any]]] = []
    language: Optional[str] = "python"
    datasetId: Optional[str] = None
    datasetIds: Optional[List[str]] = []
    variableContext: Optional[Dict[str, Any]] = {}
    user_input: Optional[str] = None  # For interactive input handling


class SQLQueryRequest(BaseModel):
    """Model for SQL query execution requests"""
    query: str
    datasets: Optional[List[Dict[str, Any]]] = []


class JSQueryRequest(BaseModel):
    """Model for JavaScript execution requests"""
    code: str
    datasets: Optional[List[Dict[str, Any]]] = []


class StreamlitRequest(BaseModel):
    """Model for Streamlit app creation requests"""
    code: str
    app_id: Optional[str] = None


class HealthCheckRequest(BaseModel):
    """Model for health check requests"""
    include_packages: Optional[bool] = False
    include_detailed_metrics: Optional[bool] = True


class ContinueExecutionRequest(BaseModel):
    """Model for continuing Python execution with user input"""
    input: str
    code: Optional[str] = ""


class PackageInstallRequest(BaseModel):
    """Model for package installation requests"""
    package_name: str
    version: Optional[str] = None
