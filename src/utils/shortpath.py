"""
Safe path handling for Windows long-path limitations.
Ensures paths stay under 240 chars total while remaining stable and collision-free.
"""

import os
import hashlib
import re
import sys
from pathlib import Path
from typing import Union

# Constants
MAX_SEGMENT_LEN = 80
MAX_TOTAL_LEN = 240
HASH_SUFFIX_LEN = 8
WINDOWS_LONGPATH_PREFIX = r"\\?\\"


def _slug_segment(segment: str, max_len: int = MAX_SEGMENT_LEN) -> str:
    """
    Slugify a path segment to be filesystem-safe and bounded in length.
    
    Args:
        segment: Original segment name
        max_len: Maximum allowed length
    
    Returns:
        Safe, shortened segment with stable hash suffix if needed
    """
    # Remove/replace problematic characters
    # Keep: alphanumeric, underscore, hyphen, dot
    safe = re.sub(r'[^\w\-.]', '_', segment)
    
    # Collapse multiple underscores/dots
    safe = re.sub(r'_+', '_', safe)
    safe = re.sub(r'\.+', '.', safe)
    
    # Strip leading/trailing dots and underscores
    safe = safe.strip('._')
    
    # If within limit, return as-is
    if len(safe) <= max_len:
        return safe
    
    # Need to shorten - use stable hash suffix
    # Take first part + hash of full segment for collision resistance
    hash_suffix = hashlib.sha1(segment.encode('utf-8')).hexdigest()[:HASH_SUFFIX_LEN]
    
    # Reserve space for underscore and hash
    available = max_len - HASH_SUFFIX_LEN - 1
    
    # Take as much of the safe string as possible
    shortened = safe[:available]
    
    return f"{shortened}_{hash_suffix}"


def safe_path(root: Union[str, Path], *parts: str) -> str:
    """
    Construct a safe filesystem path that respects Windows length limits.
    
    Args:
        root: Root directory
        *parts: Path components to join
    
    Returns:
        Safe absolute path, with \\\\?\\ prefix on Windows if needed
    """
    # Convert root to Path
    root_path = Path(root).resolve()
    
    # Slug each part
    safe_parts = [_slug_segment(part) for part in parts if part]
    
    # Build path
    full_path = root_path / Path(*safe_parts)
    
    # Convert to string
    path_str = str(full_path)
    
    # Check total length
    if len(path_str) > MAX_TOTAL_LEN:
        # More aggressive shortening needed
        # Shorten middle segments more aggressively
        aggressive_parts = []
        for part in safe_parts:
            if len(part) > 40:
                # Very aggressive shortening for middle segments
                hash_suffix = hashlib.sha1(part.encode('utf-8')).hexdigest()[:HASH_SUFFIX_LEN]
                aggressive_parts.append(f"{part[:31]}_{hash_suffix}")
            else:
                aggressive_parts.append(part)
        
        full_path = root_path / Path(*aggressive_parts)
        path_str = str(full_path)
    
    # On Windows, prefix with \\?\ if path is long
    if sys.platform == 'win32' and len(path_str) > 200:
        # Only if not already prefixed
        if not path_str.startswith(WINDOWS_LONGPATH_PREFIX):
            # Convert to absolute and add prefix
            abs_path = os.path.abspath(path_str)
            path_str = WINDOWS_LONGPATH_PREFIX + abs_path
    
    return path_str


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, handling Windows long paths.
    
    Args:
        path: Directory path (possibly with \\\\?\\ prefix)
    
    Returns:
        The same path after ensuring it exists
    """
    # Handle Windows long path prefix
    create_path = path
    if path.startswith(WINDOWS_LONGPATH_PREFIX):
        # Remove prefix for os.makedirs on some Python versions
        # But try with prefix first
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except (OSError, FileNotFoundError):
            # Try without prefix
            create_path = path[len(WINDOWS_LONGPATH_PREFIX):]
    
    os.makedirs(create_path, exist_ok=True)
    return path


def open_safe(path: str, mode: str = 'r', **kwargs):
    """
    Open file with Windows long path support.
    
    Args:
        path: File path (possibly with \\\\?\\ prefix)
        mode: File mode
        **kwargs: Additional arguments for open()
    
    Returns:
        File handle
    """
    # Try with the path as-is first
    try:
        return open(path, mode, **kwargs)
    except (OSError, FileNotFoundError):
        # If it has Windows prefix, try without
        if path.startswith(WINDOWS_LONGPATH_PREFIX):
            return open(path[len(WINDOWS_LONGPATH_PREFIX):], mode, **kwargs)
        # Otherwise, try adding it
        elif sys.platform == 'win32':
            abs_path = os.path.abspath(path)
            return open(WINDOWS_LONGPATH_PREFIX + abs_path, mode, **kwargs)
        else:
            raise


def get_safe_basename(path: str) -> str:
    """
    Get basename from a path, handling Windows long path prefix.
    
    Args:
        path: Path (possibly with \\\\?\\ prefix)
    
    Returns:
        Basename
    """
    if path.startswith(WINDOWS_LONGPATH_PREFIX):
        path = path[len(WINDOWS_LONGPATH_PREFIX):]
    return os.path.basename(path)
