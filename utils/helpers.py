"""
General Helper Utilities Module

This module contains general utility functions that are used across the application:
- Data validation and sanitization
- String manipulation utilities
- File and path utilities
- Common data transformations
"""

import os
import re
import hashlib
import tempfile
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem use
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'untitled'
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def create_safe_variable_name(name: str) -> str:
    """
    Create a safe Python variable name from a string
    
    Args:
        name: The string to convert
        
    Returns:
        Safe variable name
    """
    # Convert to lowercase and replace spaces/hyphens with underscores
    safe_name = name.lower().replace(' ', '_').replace('-', '_')
    
    # Remove non-alphanumeric characters except underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', safe_name)
    
    # Ensure it doesn't start with a number
    if safe_name and safe_name[0].isdigit():
        safe_name = f'var_{safe_name}'
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = 'variable'
    
    return safe_name


def generate_unique_id(prefix: str = '', length: int = 8) -> str:
    """
    Generate a unique ID with optional prefix
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part
        
    Returns:
        Unique ID string
    """
    import time
    import random
    import string
    
    # Use timestamp and random string for uniqueness
    timestamp = str(int(time.time()))
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    else:
        return f"{timestamp}_{random_part}"


def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of the file hash
    """
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False


def get_file_size_human_readable(size_bytes: int) -> str:
    """
    Convert file size in bytes to human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human readable size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length
    
    Args:
        text: The string to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def is_valid_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL format is valid
    """
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text
    
    Args:
        text: Text containing HTML tags
        
    Returns:
        Text with HTML tags removed
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def extract_numbers_from_string(text: str) -> List[float]:
    """
    Extract all numbers from a string
    
    Args:
        text: String to extract numbers from
        
    Returns:
        List of numbers found in the string
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]


def create_temp_file(content: str, suffix: str = '.txt', prefix: str = 'temp_') -> str:
    """
    Create a temporary file with content
    
    Args:
        content: Content to write to the file
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Path to the created temporary file
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, prefix=prefix, delete=False) as f:
            f.write(content)
            return f.name
    except Exception as e:
        logger.error(f"Error creating temporary file: {e}")
        return ""


def batch_process(items: List[Any], batch_size: int = 100):
    """
    Process items in batches
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
