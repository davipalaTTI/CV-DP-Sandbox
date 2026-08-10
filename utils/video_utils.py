import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, Optional

import cv2
import numpy as np

from utils.file_io import get_file_size_mb, generate_unique_filename


# ======================== VIDEO UTILITIES ========================

VIDEO_FILE_PATTERNS = (
    "*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv",
    "*.MP4", "*.AVI", "*.MOV", "*.MKV", "*.WMV", "*.FLV",
)


@dataclass
class SourcePreviewResult:
    """Result of opening a configured source and reading one frame."""

    frame: Optional[np.ndarray] = None
    error: Optional[str] = None


def _input_type_value(config: Any) -> str:
    input_type = getattr(config, "input_type", "")
    return str(getattr(input_type, "value", input_type)).lower()


def _resolve_preview_source(config: Any) -> tuple[Optional[Union[str, int]], Optional[str]]:
    """Resolve a config to one concrete source without exposing stream credentials."""
    if _input_type_value(config) != "folder":
        return getattr(config, "input_source", None), None

    folder_path = Path(config.input_source)
    video_files = []
    for pattern in VIDEO_FILE_PATTERNS:
        video_files.extend(folder_path.glob(pattern))
    video_files.sort(key=lambda path: path.name.casefold())
    if not video_files:
        return None, "No supported video files were found in the selected folder."
    return str(video_files[0]), None


def _open_preview_capture(
    source: Union[str, int],
    input_type: str,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> cv2.VideoCapture:
    """Open a source with backend-level timeouts where OpenCV supports them."""
    if input_type == "rtsp":
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            int(open_timeout_ms),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            int(read_timeout_ms),
        ]
        try:
            return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, params)
        except (cv2.error, TypeError):
            # Older OpenCV builds lack the constructor overload. The caller also
            # enforces an application-level deadline while this runs in a worker.
            return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)

    return cv2.VideoCapture(source)


def load_source_preview(
    config: Any,
    open_timeout_ms: int = 8000,
    read_timeout_ms: int = 8000,
) -> SourcePreviewResult:
    """Read one preview frame from folder, video, camera, or RTSP input."""
    source, error = _resolve_preview_source(config)
    if error:
        return SourcePreviewResult(error=error)
    if source is None or source == "":
        return SourcePreviewResult(error="The selected source is empty.")

    input_type = _input_type_value(config)
    capture = None
    try:
        capture = _open_preview_capture(
            source,
            input_type,
            open_timeout_ms,
            read_timeout_ms,
        )
        if not capture.isOpened():
            return SourcePreviewResult(
                error="The source could not be opened. Check the camera selection, "
                "stream address, credentials, and network connection."
            )

        success, frame = capture.read()
        if not success or frame is None:
            return SourcePreviewResult(
                error="The source opened, but no preview frame could be read."
            )
        return SourcePreviewResult(frame=frame)
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Source preview failed for input type %s: %s", input_type, exc
        )
        return SourcePreviewResult(error=f"OpenCV could not read the source: {exc}")
    finally:
        if capture is not None:
            capture.release()

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
