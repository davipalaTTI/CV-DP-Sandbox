import json
import yaml
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Union, Callable
import psutil
from contextlib import contextmanager

# ======================== FILE I/O UTILITIES ========================

def safe_read_json(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Safely read JSON file with error handling

    Args:
        filepath: Path to JSON file

    Returns:
        Parsed JSON data or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        logging.getLogger(__name__).error(f"Failed to read JSON file {filepath}: {e}")
        return None


def safe_write_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely write JSON file with error handling

    Args:
        data: Data to write
        filepath: Output file path
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except (PermissionError, OSError) as e:
        logging.getLogger(__name__).error(f"Failed to write JSON file {filepath}: {e}")
        return False


def safe_read_yaml(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    Safely read YAML file with error handling

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML data or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError, PermissionError) as e:
        logging.getLogger(__name__).error(f"Failed to read YAML file {filepath}: {e}")
        return None


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """
    Get file size in megabytes

    Args:
        filepath: Path to file

    Returns:
        File size in MB
    """
    try:
        size_bytes = Path(filepath).stat().st_size
        return size_bytes / (1024 * 1024)
    except (FileNotFoundError, PermissionError):
        return 0.0


def cleanup_old_files(directory: Union[str, Path],
                      pattern: str = "*",
                      max_age_hours: float = 24,
                      max_files: Optional[int] = None) -> int:
    """
    Clean up old files in directory

    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_hours: Maximum file age in hours
        max_files: Maximum number of files to keep (newest)

    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    files = list(directory.glob(pattern))
    if not files:
        return 0

    current_time = time.time()
    deleted_count = 0

    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for i, file_path in enumerate(files):
        should_delete = False

        # Check age
        if max_age_hours is not None:
            file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
            if file_age_hours > max_age_hours:
                should_delete = True

        # Check count limit
        if max_files is not None and i >= max_files:
            should_delete = True

        if should_delete:
            try:
                file_path.unlink()
                deleted_count += 1
            except PermissionError:
                pass

    return deleted_count

# ======================== STRING UTILITIES ========================

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename


def generate_unique_filename(base_path: Union[str, Path],
                             extension: str = "",
                             suffix_format: str = "_{:03d}") -> Path:
    """
    Generate unique filename by adding numeric suffix if needed

    Args:
        base_path: Base file path without extension
        extension: File extension (with or without dot)
        suffix_format: Format string for numeric suffix

    Returns:
        Unique file path
    """
    # Ensure extension starts with dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension

    base_path = Path(base_path)

    # Try the base filename first
    if extension:
        candidate = base_path.with_suffix(extension)
    else:
        candidate = base_path

    if not candidate.exists():
        return candidate

    # Add numeric suffix until we find a unique name
    counter = 1
    while True:
        if extension:
            stem_with_suffix = base_path.stem + suffix_format.format(counter)
            candidate = base_path.with_name(stem_with_suffix + extension)
        else:
            candidate = base_path.with_name(base_path.name + suffix_format.format(counter))

        if not candidate.exists():
            return candidate

        counter += 1

        # Safety check to prevent infinite loop
        if counter > 9999:
            timestamp = int(time.time())
            if extension:
                stem_with_timestamp = f"{base_path.stem}_{timestamp}"
                candidate = base_path.with_name(stem_with_timestamp + extension)
            else:
                candidate = base_path.with_name(f"{base_path.name}_{timestamp}")
            return candidate