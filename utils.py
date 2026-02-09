"""
Utility Functions Module

Provides common utility functions and helper classes including:
- Logging setup and configuration
- Coordinate transformations and geometry calculations
- File I/O operations and path management
- Performance monitoring and timing utilities
- Dependency checking and system validation
- Mathematical operations for computer vision
"""

import logging
import logging.handlers
import sys
import time
import os
import platform
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any, Callable, Iterable
import numpy as np
import cv2
import json
import yaml
from datetime import datetime, timedelta, date
from functools import wraps
import threading
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import importlib
import pkg_resources



@dataclass
class SystemInfo:
    """System information container"""
    platform: str
    python_version: str
    opencv_version: str
    numpy_version: str
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_info: str


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        self.logger.debug(f"{self.name} took {duration:.4f} seconds")

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        return self.end_time - self.start_time


from collections import deque


class MovingAverage:
    """Efficient moving average calculator using deque"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value: float) -> float:
        """Add new value and return current average"""
        if len(self.values) == self.window_size:
            self.sum -= self.values[0]  # Remove oldest from sum

        self.values.append(value)
        self.sum += value
        return self.sum / len(self.values)

    def get_average(self) -> float:
        """Get current average"""
        return self.sum / len(self.values) if self.values else 0.0

    def reset(self):
        """Reset the moving average"""
        self.values.clear()
        self.sum = 0.0


class FPSCounter:
    """Efficient FPS counter with deque-based moving average"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()
        self._sum = 0.0

    def update(self) -> float:
        """Update FPS counter and return current FPS"""
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        if len(self.frame_times) == self.window_size:
            self._sum -= self.frame_times[0]

        self.frame_times.append(frame_time)
        self._sum += frame_time

        if self._sum > 0:
            return len(self.frame_times) / self._sum
        return 0.0

    def get_fps(self) -> float:
        """Get current FPS without updating"""
        if self._sum > 0:
            return len(self.frame_times) / self._sum
        return 0.0


# ======================== LOGGING UTILITIES ========================

# Global reference to queue listener (for cleanup)
_log_queue_listener = None


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    A RotatingFileHandler that gracefully handles file locking issues on Windows.

    When rotation fails (e.g., due to OneDrive sync or another process locking
    the file), it simply continues logging to the current file instead of raising
    an error. Rotation will be attempted again on the next write that exceeds maxBytes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotation_failed = False

    def doRollover(self):
        """Perform rotation, but handle errors gracefully"""
        try:
            # Close the stream before rotation
            if self.stream:
                self.stream.close()
                self.stream = None

            # Attempt rotation
            if self.backupCount > 0:
                # Delete oldest backup if it exists
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                    dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                    if os.path.exists(sfn):
                        try:
                            if os.path.exists(dfn):
                                os.remove(dfn)
                            os.rename(sfn, dfn)
                        except (OSError, PermissionError):
                            pass  # Ignore rotation errors for backups

                # Rotate current to .1
                dfn = self.rotation_filename(f"{self.baseFilename}.1")
                try:
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    if os.path.exists(self.baseFilename):
                        os.rename(self.baseFilename, dfn)
                    self._rotation_failed = False
                except (OSError, PermissionError):
                    # Rotation failed - file is locked (probably OneDrive)
                    # Just continue with the same file
                    self._rotation_failed = True

            # Reopen stream
            if not self.delay:
                self.stream = self._open()

        except Exception:
            # Any other error - just try to keep logging
            self._rotation_failed = True
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    pass

    def shouldRollover(self, record):
        """Check if rollover should occur, but be more lenient after failed rotation"""
        if self._rotation_failed:
            # After a failed rotation, only try again after file grows significantly more
            # This prevents constant rotation attempts
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    return False

            try:
                self.stream.seek(0, 2)  # Seek to end
                # Only retry rotation after file grows another 50% beyond maxBytes
                if self.stream.tell() >= self.maxBytes * 1.5:
                    self._rotation_failed = False  # Reset and try again
                    return True
            except Exception:
                pass
            return False

        return super().shouldRollover(record)


def setup_logging(level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  console_output: bool = True,
                  format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging configuration with thread-safe file handling.

    Uses QueueHandler + QueueListener pattern with a SafeRotatingFileHandler
    to prevent file locking issues on Windows/OneDrive.

    Args:
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        format_string: Custom format string

    Returns:
        Configured logger
    """
    global _log_queue_listener

    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers and stop old listener
    root_logger.handlers.clear()
    if _log_queue_listener:
        _log_queue_listener.stop()
        _log_queue_listener = None

    # Create handlers list for QueueListener
    handlers = []

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler with safe rotation (handles OneDrive/Windows locking)
    if log_file:
        file_handler = SafeRotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            delay=True  # Delay file opening until first write
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Use queue-based logging for thread safety
    if handlers:
        from queue import Queue
        log_queue = Queue(-1)  # Unlimited queue size

        # QueueHandler for the root logger (all threads put logs here)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)

        # QueueListener processes the queue in a single thread (thread-safe file writes)
        _log_queue_listener = logging.handlers.QueueListener(
            log_queue,
            *handlers,
            respect_handler_level=True
        )
        _log_queue_listener.start()

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return root_logger


def stop_logging():
    """Stop the logging queue listener (call on application exit)"""
    global _log_queue_listener
    if _log_queue_listener:
        _log_queue_listener.stop()
        _log_queue_listener = None


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


# ======================== COORDINATE UTILITIES ========================

def normalize_coordinates(points: Union[Tuple[int, int], List[Tuple[int, int]]],
                          frame_size: Tuple[int, int]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Normalize coordinates to 0-1 range

    Args:
        points: Single point or list of points (x, y)
        frame_size: Frame dimensions (width, height)

    Returns:
        Normalized coordinates
    """
    w, h = frame_size

    def normalize_point(point: Tuple[int, int]) -> Tuple[float, float]:
        x, y = point
        return (x / w, y / h)

    if isinstance(points, tuple) and len(points) == 2:
        return normalize_point(points)
    else:
        return [normalize_point(p) for p in points]


def denormalize_coordinates(points: Union[Tuple[float, float], List[Tuple[float, float]]],
                            frame_size: Tuple[int, int]) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Convert normalized coordinates back to pixels

    Args:
        points: Normalized coordinates (0-1 range)
        frame_size: Frame dimensions (width, height)

    Returns:
        Pixel coordinates
    """
    w, h = frame_size

    def denormalize_point(point: Tuple[float, float]) -> Tuple[int, int]:
        norm_x, norm_y = point
        return (int(norm_x * w), int(norm_y * h))

    if isinstance(points, tuple) and len(points) == 2:
        return denormalize_point(points)
    else:
        return [denormalize_point(p) for p in points]


def scale_coordinates(points: Union[Tuple[int, int], List[Tuple[int, int]]],
                      scale_factor: float) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Scale coordinates by a factor

    Args:
        points: Point(s) to scale
        scale_factor: Scaling factor

    Returns:
        Scaled coordinates
    """

    def scale_point(point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        return (int(x * scale_factor), int(y * scale_factor))

    if isinstance(points, tuple) and len(points) == 2:
        return scale_point(points)
    else:
        return [scale_point(p) for p in points]


# ======================== GEOMETRY UTILITIES ========================

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm

    Args:
        point: Point to test (x, y)
        polygon: List of polygon vertices

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                      line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines

    Args:
        line1: First line ((x1, y1), (x2, y2))
        line2: Second line ((x1, y1), (x2, y2))

    Returns:
        Intersection point or None if lines don't intersect
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Lines intersect within segments
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def distance_point_to_line(point: Tuple[float, float],
                           line_start: Tuple[float, float],
                           line_end: Tuple[float, float]) -> float:
    """
    Calculate perpendicular distance from point to line segment

    Args:
        point: Point coordinates
        line_start: Line start point
        line_end: Line end point

    Returns:
        Distance to line
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line vector
    A = x2 - x1
    B = y2 - y1

    # Point vector from line start
    C = x0 - x1
    D = y0 - y1

    dot = C * A + D * B
    len_sq = A * A + B * B

    if len_sq == 0:
        # Line is actually a point
        return np.sqrt(C * C + D * D)

    param = dot / len_sq

    # Find closest point on line segment
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * A
        yy = y1 + param * B

    # Calculate distance
    dx = x0 - xx
    dy = y0 - yy
    return np.sqrt(dx * dx + dy * dy)


def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    Calculate area of polygon using shoelace formula

    Args:
        polygon: List of polygon vertices

    Returns:
        Polygon area
    """
    if len(polygon) < 3:
        return 0.0

    area = 0.0
    n = len(polygon)

    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0


def calculate_polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate centroid of polygon

    Args:
        polygon: List of polygon vertices

    Returns:
        Centroid coordinates
    """
    if not polygon:
        return (0.0, 0.0)

    area = calculate_polygon_area(polygon)
    if area == 0:
        # Degenerate polygon, return average of points
        x_avg = sum(p[0] for p in polygon) / len(polygon)
        y_avg = sum(p[1] for p in polygon) / len(polygon)
        return (x_avg, y_avg)

    cx = 0.0
    cy = 0.0
    n = len(polygon)

    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross

    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


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


# ======================== SYSTEM UTILITIES ========================

def get_system_info() -> SystemInfo:
    """
    Get comprehensive system information

    Returns:
        SystemInfo object
    """
    # GPU detection
    gpu_available = False
    gpu_info = "No GPU detected"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
    except ImportError:
        pass

    # If no CUDA, check for other GPU APIs
    if not gpu_available:
        try:
            # Try OpenCV DNN with GPU backend
            net = cv2.dnn.readNet()  # Empty net for testing
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_available = True
                gpu_info = f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}"
        except:
            pass

    return SystemInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        cpu_count=psutil.cpu_count(),
        memory_gb=psutil.virtual_memory().total / (1024 ** 3),
        gpu_available=gpu_available,
        gpu_info=gpu_info
    )


def check_dependencies() -> List[str]:
    """
    Check for required dependencies

    Returns:
        List of missing dependencies
    """
    required_packages = [
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'ultralytics',
        'psutil',
        'yaml'
    ]

    missing = []

    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    return missing


def get_available_memory_gb() -> float:
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_cpu_usage() -> float:
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=1)


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """
    Get GPU memory usage if available

    Returns:
        Dictionary with GPU memory info or None
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(device) / (1024 ** 3)

            return {
                'total_gb': total,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'free_gb': total - allocated
            }
    except ImportError:
        pass

    return None


# ======================== PERFORMANCE UTILITIES ========================

def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with performance monitoring
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        try:
            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.debug(
                f"Performance: {func.__name__} took {duration:.4f}s, "
                f"memory delta: {memory_delta:+.2f}MB"
            )

            return result

        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
            raise

    return wrapper


@contextmanager
def memory_monitor(operation_name: str = "Operation"):
    """
    Context manager to monitor memory usage

    Args:
        operation_name: Name of the operation being monitored
    """
    logger = logging.getLogger(__name__)
    process = psutil.Process()

    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = start_memory

    try:
        yield
    finally:
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = max(peak_memory, end_memory)

        memory_delta = end_memory - start_memory
        peak_delta = peak_memory - start_memory

        logger.debug(
            f"Memory usage for {operation_name}: "
            f"delta={memory_delta:+.2f}MB, peak_delta={peak_delta:+.2f}MB"
        )


# ======================== VIDEO UTILITIES ========================

def get_video_properties(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive video properties

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {}

    try:
        properties = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration_seconds': 0,
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'file_size_mb': get_file_size_mb(video_path)
        }

        if properties['fps'] > 0:
            properties['duration_seconds'] = properties['frame_count'] / properties['fps']

        return properties

    finally:
        cap.release()


def estimate_processing_time(video_properties: Dict[str, Any],
                             target_fps: float = 30.0) -> Dict[str, float]:
    """
    Estimate processing time for video

    Args:
        video_properties: Video properties from get_video_properties
        target_fps: Target processing FPS

    Returns:
        Time estimates in different units
    """
    if not video_properties or 'frame_count' not in video_properties:
        return {}

    frame_count = video_properties['frame_count']
    estimated_seconds = frame_count / target_fps

    return {
        'seconds': estimated_seconds,
        'minutes': estimated_seconds / 60,
        'hours': estimated_seconds / 3600,
        'frames': frame_count,
        'target_fps': target_fps
    }


# ======================== VALIDATION UTILITIES ========================

def validate_model_file(model_path: Union[str, Path]) -> bool:
    """
    Validate YOLO model file

    Args:
        model_path: Path to model file

    Returns:
        True if valid model file
    """
    model_path = Path(model_path)

    if not model_path.exists():
        return False

    # Check file extension
    valid_extensions = ['.pt', '.onnx', '.engine']
    if model_path.suffix.lower() not in valid_extensions:
        return False

    # Check file size (should be > 1MB for real models)
    if model_path.stat().st_size < 1024 * 1024:
        return False

    return True


def validate_video_file(video_path: Union[str, Path]) -> bool:
    """
    Validate video file

    Args:
        video_path: Path to video file

    Returns:
        True if valid video file
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None

    except Exception:
        return False


def validate_configuration(config_dict: Dict) -> List[str]:
    """
    Validate configuration dictionary

    Args:
        config_dict: Configuration to validate

    Returns:
        List of validation errors
    """
    errors = []

    # Required fields
    required_fields = ['model_path', 'input_source', 'output_folder']
    for field in required_fields:
        if field not in config_dict or not config_dict[field]:
            errors.append(f"Missing required field: {field}")

    # Validate model file
    if 'model_path' in config_dict:
        if not validate_model_file(config_dict['model_path']):
            errors.append("Invalid model file")

    # Validate confidence threshold
    if 'confidence_threshold' in config_dict:
        conf = config_dict['confidence_threshold']
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")

    return errors


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


# ======================== THREADING UTILITIES ========================

class ThreadSafeCounter:
    """Thread-safe counter with lock"""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value"""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value"""
        with self._lock:
            self._value -= amount
            return self._value

    def set(self, value: int) -> int:
        """Set counter value and return new value"""
        with self._lock:
            self._value = value
            return self._value

    def get(self) -> int:
        """Get current counter value"""
        with self._lock:
            return self._value

    def reset(self) -> int:
        """Reset counter to zero and return previous value"""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class RateLimiter:
    """Rate limiter using token bucket algorithm"""

    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """Check if call is allowed under rate limit"""
        with self._lock:
            now = time.time()

            # Remove old calls outside time window
            self.calls = [call_time for call_time in self.calls
                          if now - call_time < self.time_window]

            # Check if under limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True

            return False

    def wait_if_needed(self) -> float:
        """Wait if necessary to respect rate limit, return wait time"""
        with self._lock:
            now = time.time()

            # Remove old calls
            self.calls = [call_time for call_time in self.calls
                          if now - call_time < self.time_window]

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return 0.0

            # Calculate wait time
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)

            if wait_time > 0:
                time.sleep(wait_time)
                return wait_time

            return 0.0


# ======================== DATA STRUCTURES ========================

class CircularBuffer:
    """Fixed-size circular buffer with thread safety"""

    def __init__(self, capacity: int):
        """
        Initialize circular buffer

        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0
        self._lock = threading.Lock()

    def put(self, item: Any) -> Optional[Any]:
        """
        Add item to buffer

        Returns:
            Evicted item if buffer was full, None otherwise
        """
        with self._lock:
            evicted = None
            if self.size == self.capacity:
                evicted = self.buffer[self.head]

            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.capacity

            if self.size < self.capacity:
                self.size += 1

            return evicted

    def get_all(self) -> List[Any]:
        """Get all items in chronological order"""
        with self._lock:
            if self.size == 0:
                return []

            if self.size < self.capacity:
                # Buffer not full yet
                return [self.buffer[i] for i in range(self.size)]
            else:
                # Buffer is full, need to handle wraparound
                items = []
                for i in range(self.capacity):
                    index = (self.head + i) % self.capacity
                    items.append(self.buffer[index])
                return items

    def get_latest(self, n: int) -> List[Any]:
        """Get n most recent items"""
        with self._lock:
            if self.size == 0:
                return []

            n = min(n, self.size)
            items = []

            for i in range(n):
                index = (self.head - 1 - i) % self.capacity
                items.append(self.buffer[index])

            return items

    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self.buffer = [None] * self.capacity
            self.head = 0
            self.size = 0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self.size == 0

    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self._lock:
            return self.size == self.capacity


# ======================== ERROR HANDLING ========================

class ConfigurationError(Exception):
    """Exception for configuration-related errors"""
    pass


class ProcessingError(Exception):
    """Exception for processing-related errors"""
    pass


class ValidationError(Exception):
    """Exception for validation errors"""
    pass


def handle_exception(func: Callable) -> Callable:
    """
    Decorator for comprehensive exception handling

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with exception handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"{func.__module__}.{func.__name__}")

        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.info(f"Function {func.__name__} interrupted by user")
            raise
        except ConfigurationError as e:
            logger.error(f"Configuration error in {func.__name__}: {e}")
            raise
        except ProcessingError as e:
            logger.error(f"Processing error in {func.__name__}: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ProcessingError(f"Unexpected error in {func.__name__}: {e}") from e

    return wrapper


# ======================== DEBUGGING UTILITIES ========================

def debug_frame_info(frame: np.ndarray, name: str = "Frame") -> Dict[str, Any]:
    """
    Get debugging information about a frame

    Args:
        frame: OpenCV frame
        name: Frame identifier

    Returns:
        Frame information dictionary
    """
    if frame is None:
        return {"name": name, "valid": False, "error": "Frame is None"}

    info = {
        "name": name,
        "valid": True,
        "shape": frame.shape,
        "dtype": str(frame.dtype),
        "min_value": float(np.min(frame)),
        "max_value": float(np.max(frame)),
        "mean_value": float(np.mean(frame)),
        "memory_mb": frame.nbytes / (1024 * 1024)
    }

    # Channel information
    if len(frame.shape) == 3:
        info["channels"] = frame.shape[2]
        info["color_space"] = "BGR" if frame.shape[2] == 3 else "RGBA" if frame.shape[2] == 4 else "Unknown"
    else:
        info["channels"] = 1
        info["color_space"] = "Grayscale"

    return info


def save_debug_frame(frame: np.ndarray,
                     filename: str,
                     output_dir: Union[str, Path] = "debug_frames") -> Optional[Path]:
    """
    Save frame for debugging purposes

    Args:
        frame: Frame to save
        filename: Base filename
        output_dir: Output directory

    Returns:
        Path to saved file or None if failed
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{timestamp}_{filename}"

        # Generate unique filename
        filepath = generate_unique_filename(
            output_dir / filename_with_timestamp,
            extension=".png"
        )

        cv2.imwrite(str(filepath), frame)
        return filepath

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save debug frame: {e}")
        return None


def create_debug_overlay(frame: np.ndarray,
                         info_dict: Dict[str, Any],
                         position: str = "top_left") -> np.ndarray:
    """
    Add debug information overlay to frame

    Args:
        frame: Input frame
        info_dict: Information to display
        position: Overlay position ("top_left", "top_right", "bottom_left", "bottom_right")

    Returns:
        Frame with debug overlay
    """
    overlay_frame = frame.copy()
    h, w = overlay_frame.shape[:2]

    # Prepare text lines
    text_lines = []
    for key, value in info_dict.items():
        if isinstance(value, float):
            text_lines.append(f"{key}: {value:.3f}")
        else:
            text_lines.append(f"{key}: {value}")

    # Calculate text dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    max_width = 0
    total_height = 0
    line_heights = []

    for line in text_lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        max_width = max(max_width, text_size[0])
        line_heights.append(text_size[1])
        total_height += text_size[1] + 5  # 5 pixels spacing

    # Calculate overlay position
    margin = 10
    if position == "top_left":
        x_start = margin
        y_start = margin + line_heights[0]
    elif position == "top_right":
        x_start = w - max_width - margin
        y_start = margin + line_heights[0]
    elif position == "bottom_left":
        x_start = margin
        y_start = h - total_height - margin + line_heights[0]
    elif position == "bottom_right":
        x_start = w - max_width - margin
        y_start = h - total_height - margin + line_heights[0]
    else:
        x_start = margin
        y_start = margin + line_heights[0]

    # Draw background rectangle
    bg_x1 = x_start - 5
    bg_y1 = y_start - line_heights[0] - 5
    bg_x2 = x_start + max_width + 5
    bg_y2 = y_start + total_height - line_heights[0] + 5

    cv2.rectangle(overlay_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(overlay_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)

    # Draw text lines
    y_current = y_start
    for i, line in enumerate(text_lines):
        cv2.putText(overlay_frame, line, (x_start, y_current),
                    font, font_scale, (255, 255, 255), thickness)
        y_current += line_heights[i] + 5

    return overlay_frame


# ==== HEATMAP: begin ====
class HeatmapAccumulator:
    """
    Accumulates object presence over a time window using detection boxes/points.
    Produces a single overlay snapshot per interval, then resets.

    Works two ways:
      • call update_*() each frame and let maybe_emit(...) time out and save; or
      • call render_and_save(...) yourself (e.g., on segment rollover) to flush immediately.
    """

    def __init__(
            self,
            frame_size: Tuple[int, int],  # (h, w)
            alpha: float = 0.35,
            colormap: int = cv2.COLORMAP_HOT,
            out_dir: str = "outputs/heatmaps",
            interval_sec: float = 10.0,
            radius_px: int = 10,
            decay: float = 0.0,
            gamma: float = 1.6,  # Make sure this parameter exists
            saturation_boost: float = 1.0
    ) -> None:
        self.h, self.w = frame_size
        self.alpha = float(alpha)
        self.colormap = int(colormap)
        self.interval_sec = float(interval_sec)
        self.radius_px = int(max(0, radius_px))
        self.decay = float(max(0.0, min(1.0, decay)))
        self.gamma = float(max(0.1, gamma))  # Store gamma properly
        self.saturation_boost = float(max(1.0, saturation_boost))
        self.accum = np.zeros((self.h, self.w), dtype=np.float32)
        self.last_emit_t = time.time()

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- core state --------------------

    def reset(self) -> None:
        """Clear the accumulation buffer."""
        self.accum.fill(0.0)

    def _apply_decay(self) -> None:
        """Apply optional exponential-ish decay to gradually fade old heat."""
        if self.decay > 0.0:
            self.accum *= (1.0 - self.decay)

    # -------------------- updates --------------------

    def update_from_boxes(self, boxes_xyxy: Iterable[Tuple[int, int, int, int]], weight: float = 1.5) -> None:
        """
        Add heat from bounding boxes with gaussian-like falloff for smoother blending
        """
        self._apply_decay()
        if not boxes_xyxy:
            return

        w = self.w
        h = self.h
        wgt = float(max(0.0, weight))

        if self.radius_px > 0:
            # Use gaussian-like kernel for smoother overlaps
            for (x1, y1, x2, y2) in boxes_xyxy:
                cx = int(0.5 * (x1 + x2))
                cy = int(0.5 * (y1 + y2))

                if 0 <= cx < w and 0 <= cy < h:
                    # Create gaussian kernel for smoother blending
                    radius = self.radius_px
                    y_min = max(0, cy - radius)
                    y_max = min(h, cy + radius + 1)
                    x_min = max(0, cx - radius)
                    x_max = min(w, cx + radius + 1)

                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                            if dist <= radius:
                                # Gaussian-like falloff
                                intensity = wgt * np.exp(-(dist ** 2) / (2 * (radius / 3) ** 2))
                                self.accum[y, x] += intensity
        else:
            # Fill rectangles with gradient for overlap emphasis
            for (x1, y1, x2, y2) in boxes_xyxy:
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))

                if x2 > x1 and y2 > y1:
                    # Add with slight center weighting
                    box_w = x2 - x1
                    box_h = y2 - y1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            # Distance from center normalized
                            dx = abs(x - cx) / max(box_w / 2, 1)
                            dy = abs(y - cy) / max(box_h / 2, 1)
                            center_weight = 1.0 - 0.3 * max(dx, dy)  # Slight center emphasis
                            self.accum[y, x] += wgt * center_weight

    def update_from_points(self, points_xy: Iterable[Tuple[int, int]], weight: float = 1.0) -> None:
        """
        Optional helper: add heat from a list of points (x, y). Uses radius_px; if radius_px == 0 uses a 1px dot.
        """
        self._apply_decay()
        if not points_xy:
            return

        wgt = float(max(0.0, weight))
        r = max(1, self.radius_px)  # at least 1px dot

        for (x, y) in points_xy:
            if 0 <= x < self.w and 0 <= y < self.h:
                cv2.circle(self.accum, (int(x), int(y)), r, wgt, thickness=-1)

    # -------------------- rendering/saving --------------------

    def _make_colormap(self) -> np.ndarray:
        """Create vibrant heatmap with enhanced contrast"""
        heat = self.accum
        if heat.size == 0:
            return np.zeros((self.h, self.w), dtype=np.uint8)

        vmax = float(heat.max())
        if vmax <= 0:
            return np.zeros_like(heat, dtype=np.uint8)

        # Normalize
        norm = heat / vmax

        # Apply gamma correction for enhanced contrast
        norm = np.power(norm, 1.0 / self.gamma)

        # Apply sigmoid-like curve to enhance mid-tones
        # This makes moderately hot areas more visible
        norm = 1.0 / (1.0 + np.exp(-10 * (norm - 0.3)))

        # Scale to 8-bit
        heat_u8 = (255.0 * norm).astype(np.uint8)

        return heat_u8

    def make_overlay(self, frame_bgr: Optional[np.ndarray]) -> np.ndarray:
        """Blend vibrant heatmap over frame with color enhancement"""
        heat_u8 = self._make_colormap()
        color_map = cv2.applyColorMap(heat_u8, self.colormap)

        # Boost saturation for more vibrant colors
        if self.saturation_boost > 1.0:
            hsv = cv2.cvtColor(color_map, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_boost, 0, 255)
            color_map = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if frame_bgr is None:
            base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            # Resize if needed
            if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w:
                color_map = cv2.resize(color_map, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            base = frame_bgr

        # Use screen blending for more vibrant overlay
        # This brightens overlapping areas more dramatically
        inv_alpha = 1.0 - self.alpha

        # Screen blend mode: 1 - (1-A)*(1-B)
        base_norm = base.astype(np.float32) / 255.0
        color_norm = color_map.astype(np.float32) / 255.0

        screen_blend = 1.0 - (1.0 - base_norm) * (1.0 - color_norm * self.alpha)
        overlay = (screen_blend * 255).astype(np.uint8)

        return overlay

    def render_and_save(
            self,
            frame_bgr: Optional[np.ndarray],
            label: Optional[str] = None,
            when: Optional[float] = None,
            suffix: Optional[str] = None,  # keep this since _flush_heatmap may pass suffix="final"
    ) -> str:
        """
        Immediately render the current heatmap over `frame_bgr` and write a PNG.
        Filename: heatmap_Year_day_month_HHMM_to_HHMM[_{label}][_{suffix}].png
        Uses the previous emit time as the start of the window.
        Resets the accumulator and updates last_emit_t. Returns the file path (str).
        """
        # Normalize end time
        t_now = self._as_timestamp(when)

        # Use last emit as window start; if none, approximate by interval length
        start_t = getattr(self, "last_emit_t", None)
        if start_t is None:
            start_t = t_now - float(getattr(self, "interval_sec", 0) or 0)
        # Normalize start time (in case someone set a datetime here)
        start_t = self._as_timestamp(start_t)

        start_dt = datetime.fromtimestamp(start_t)
        end_dt = datetime.fromtimestamp(t_now)

        overlay_bgr = self.make_overlay(frame_bgr)

        # Build short, Windows-safe filename stem: heatmap_YYYY_DD_MM_HHMM_to_HHMM
        date_part = start_dt.strftime("%Y_%d_%m")
        stem = f"heatmap_{date_part}_{start_dt.strftime('%H%M')}_to_{end_dt.strftime('%H%M')}"

        if label:
            stem += f"_{sanitize_filename(str(label))[:32]}"
        if suffix and str(suffix) != str(label):
            stem += f"_{sanitize_filename(str(suffix))[:16]}"

        out_path = generate_unique_filename(Path(self.out_dir) / stem, extension=".png")

        ok = cv2.imwrite(str(out_path), overlay_bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for '{out_path}'")

        # Store as float timestamp (not datetime) to avoid future type issues
        self.last_emit_t = float(t_now)
        self.reset()

        return str(out_path)

    def maybe_emit(
            self,
            frame_bgr: Optional[np.ndarray],
            t_now: Optional[float] = None,
            label: Optional[str] = None
    ) -> Optional[str]:
        """
        If the interval has elapsed since the last emit, render+save and reset.
        Returns the output file path (str) when it emits, else None.
        """
        t_now = self._as_timestamp(t_now)

        last = getattr(self, "last_emit_t", None)
        if last is None:
            # initialize timer on first run
            self.last_emit_t = float(t_now)
            return None

        last_ts = self._as_timestamp(last)
        if (t_now - last_ts) >= float(getattr(self, "interval_sec", 0) or 0):
            return self.render_and_save(frame_bgr=frame_bgr, label=label, when=t_now)

        return None

    # -------------------- small helpers --------------------

    def _as_timestamp(self, t) -> float:
        if t is None:
            return time.time()
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, datetime):
            return t.timestamp()
        if isinstance(t, date):
            return datetime(t.year, t.month, t.day).timestamp()
        raise TypeError(f"Expected timestamp float/int or datetime/date, got {type(t)}")

    def set_interval_minutes(self, minutes: int) -> None:
        """Convenience: set emission interval in minutes."""
        self.interval_sec = max(1.0, float(minutes) * 60.0)

    def set_colormap(self, colormap: int) -> None:
        """Update the OpenCV colormap enum."""
        self.colormap = int(colormap)

    def _build_filename(self, start_dt, end_dt, suffix: str = "") -> str:
        """
        Return filename stem: heatmap_Year_day_month_HHMM_to_HHMM (+ optional suffix)
        Example: heatmap_2025_19_09_1025_to_1058[_final]
        """
        date_part = start_dt.strftime("%Y_%d_%m")
        t_start = start_dt.strftime("%H%M")
        t_end = end_dt.strftime("%H%M")
        stem = f"heatmap_{date_part}_{t_start}_to_{t_end}"
        if suffix:
            stem += f"_{suffix}"
        return stem


# ======================== CONSTANTS ========================

# Common YOLO class names (COCO dataset)
COCO_CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Default configuration values
DEFAULT_CONFIG = {
    'confidence_threshold': 0.45,
    'device': 'auto',
    'segment_seconds': 60,
    'max_track_age': 30.0,
    'display_width': 1280,
    'display_height': 720,
    'save_video': True,
    'enable_zones': False
}

# File extensions
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']
SUPPORTED_MODEL_EXTENSIONS = ['.pt', '.onnx', '.engine']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

if __name__ == "__main__":
    # Quick tests for utility functions
    print("Testing utility functions...")

    # Test system info
    system_info = get_system_info()
    print(f"System: {system_info.platform}")
    print(f"Python: {system_info.python_version}")
    print(f"OpenCV: {system_info.opencv_version}")
    print(f"GPU Available: {system_info.gpu_available}")

    # Test dependency check
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
    else:
        print("All dependencies available")

    # Test geometry functions
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    test_point = (5, 5)
    print(f"Point {test_point} in polygon: {point_in_polygon(test_point, polygon)}")

    area = calculate_polygon_area(polygon)
    centroid = calculate_polygon_centroid(polygon)
    print(f"Polygon area: {area}, centroid: {centroid}")

    print("Utility tests completed!")