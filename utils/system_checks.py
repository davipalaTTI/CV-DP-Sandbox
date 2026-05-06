import platform
from pathlib import Path

import psutil
import cv2
import importlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np

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