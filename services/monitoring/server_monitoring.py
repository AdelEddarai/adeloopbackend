"""
Server Monitoring and Health Check Module

This module provides server monitoring capabilities including:
- System resource monitoring (CPU, memory, disk)
- Health checks and status reporting
- Package and dependency information
- Performance metrics and diagnostics
"""

import psutil
import sys
import os
import platform
import subprocess
import pkg_resources
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information
    
    Returns:
        Dictionary containing system details
    """
    try:
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'hostname': platform.node(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU usage and information
    
    Returns:
        Dictionary containing CPU metrics
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        return {
            'usage_percent': cpu_percent,
            'count_logical': cpu_count,
            'count_physical': psutil.cpu_count(logical=False),
            'frequency': {
                'current': cpu_freq.current if cpu_freq else None,
                'min': cpu_freq.min if cpu_freq else None,
                'max': cpu_freq.max if cpu_freq else None
            } if cpu_freq else None,
            'per_cpu': psutil.cpu_percent(percpu=True)
        }
    except Exception as e:
        logger.error(f"Error getting CPU info: {e}")
        return {'error': str(e)}


def get_memory_info() -> Dict[str, Any]:
    """
    Get memory usage information
    
    Returns:
        Dictionary containing memory metrics
    """
    try:
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return {
            'virtual': {
                'total': virtual_memory.total,
                'available': virtual_memory.available,
                'used': virtual_memory.used,
                'percent': virtual_memory.percent,
                'free': virtual_memory.free
            },
            'swap': {
                'total': swap_memory.total,
                'used': swap_memory.used,
                'free': swap_memory.free,
                'percent': swap_memory.percent
            }
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {'error': str(e)}


def get_disk_info() -> Dict[str, Any]:
    """
    Get disk usage information
    
    Returns:
        Dictionary containing disk metrics
    """
    try:
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        return {
            'usage': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100
            },
            'io': {
                'read_count': disk_io.read_count if disk_io else None,
                'write_count': disk_io.write_count if disk_io else None,
                'read_bytes': disk_io.read_bytes if disk_io else None,
                'write_bytes': disk_io.write_bytes if disk_io else None
            } if disk_io else None
        }
    except Exception as e:
        logger.error(f"Error getting disk info: {e}")
        return {'error': str(e)}


def get_network_info() -> Dict[str, Any]:
    """
    Get network interface information
    
    Returns:
        Dictionary containing network metrics
    """
    try:
        network_io = psutil.net_io_counters()
        network_connections = len(psutil.net_connections())
        
        return {
            'io': {
                'bytes_sent': network_io.bytes_sent if network_io else None,
                'bytes_recv': network_io.bytes_recv if network_io else None,
                'packets_sent': network_io.packets_sent if network_io else None,
                'packets_recv': network_io.packets_recv if network_io else None
            } if network_io else None,
            'connections': network_connections
        }
    except Exception as e:
        logger.error(f"Error getting network info: {e}")
        return {'error': str(e)}


def get_process_info() -> Dict[str, Any]:
    """
    Get current process information
    
    Returns:
        Dictionary containing process metrics
    """
    try:
        current_process = psutil.Process()
        
        return {
            'pid': current_process.pid,
            'name': current_process.name(),
            'status': current_process.status(),
            'cpu_percent': current_process.cpu_percent(),
            'memory_info': {
                'rss': current_process.memory_info().rss,
                'vms': current_process.memory_info().vms
            },
            'memory_percent': current_process.memory_percent(),
            'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat(),
            'num_threads': current_process.num_threads(),
            'open_files': len(current_process.open_files()) if hasattr(current_process, 'open_files') else None
        }
    except Exception as e:
        logger.error(f"Error getting process info: {e}")
        return {'error': str(e)}


def get_environment_variables() -> Dict[str, str]:
    """
    Get relevant environment variables (filtered for security)
    
    Returns:
        Dictionary of safe environment variables
    """
    try:
        # Only include safe environment variables
        safe_vars = [
            'PATH', 'PYTHONPATH', 'HOME', 'USER', 'SHELL',
            'LANG', 'LC_ALL', 'TZ', 'TERM',
            'NODE_ENV', 'PORT', 'HOST',
            'RENDER_SERVICE_NAME', 'RAILWAY_STATIC_URL', 'HEROKU_APP_NAME'
        ]
        
        env_vars = {}
        for var in safe_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    except Exception as e:
        logger.error(f"Error getting environment variables: {e}")
        return {'error': str(e)}


def get_installed_packages() -> List[Dict[str, str]]:
    """
    Get list of installed Python packages
    
    Returns:
        List of dictionaries containing package information
    """
    try:
        packages = []
        for package in pkg_resources.working_set:
            packages.append({
                'name': package.project_name,
                'version': package.version,
                'location': package.location
            })
        
        # Sort by package name
        packages.sort(key=lambda x: x['name'].lower())
        return packages
    except Exception as e:
        logger.error(f"Error getting installed packages: {e}")
        return [{'error': str(e)}]


def run_health_check() -> Dict[str, Any]:
    """
    Run comprehensive health check
    
    Returns:
        Dictionary containing health status and metrics
    """
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': datetime.now() - datetime.fromtimestamp(psutil.boot_time()),
            'checks': {}
        }
        
        # CPU check
        cpu_usage = psutil.cpu_percent(interval=1)
        health_status['checks']['cpu'] = {
            'status': 'warning' if cpu_usage > 80 else 'ok',
            'usage_percent': cpu_usage
        }
        
        # Memory check
        memory = psutil.virtual_memory()
        health_status['checks']['memory'] = {
            'status': 'warning' if memory.percent > 85 else 'ok',
            'usage_percent': memory.percent
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        health_status['checks']['disk'] = {
            'status': 'warning' if disk_percent > 90 else 'ok',
            'usage_percent': disk_percent
        }
        
        # Overall status
        warning_checks = [check for check in health_status['checks'].values() 
                         if check['status'] == 'warning']
        if warning_checks:
            health_status['status'] = 'warning'
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error running health check: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_server_status() -> Dict[str, Any]:
    """
    Get comprehensive server status information
    
    Returns:
        Dictionary containing all server metrics and status
    """
    try:
        return {
            'timestamp': datetime.now().isoformat(),
            'system': get_system_info(),
            'cpu': get_cpu_info(),
            'memory': get_memory_info(),
            'disk': get_disk_info(),
            'network': get_network_info(),
            'process': get_process_info(),
            'health': run_health_check(),
            'environment': get_environment_variables(),
            'packages_count': len(get_installed_packages())
        }
        
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
