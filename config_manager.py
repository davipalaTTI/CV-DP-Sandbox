"""
Configuration Management Module

Handles all configuration aspects including:
- Initial user configuration through GUI
- Saving/loading configuration files
- Configuration validation
- Default settings management
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Union
import cv2
from enum import Enum

class InputType(Enum):
    """Enumeration for input types"""
    CAMERA = "camera"
    FOLDER = "folder"
    VIDEO = "video"
    RTSP = "rtsp"


@dataclass(frozen=True)
class DeploymentRequest:
    """Startup selection that delegates processing to a deployment manifest."""

    manifest_path: str

@dataclass
class CountingLine:
    """Configuration for a counting line"""
    name: str
    start_norm: Tuple[float, float]  # Normalized coordinates (0-1)
    end_norm: Tuple[float, float]  # Normalized coordinates (0-1)
    direction: str  # "up", "down", "left", "right"
    classes: List[int]  # Class IDs to count
    enabled: bool = True
    poi_mode: str = "center"

@dataclass
class CountingZone:
    """Configuration for a counting zone"""
    name: str
    points_norm: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    classes: List[int]  # Class IDs to count
    enabled: bool = True
    # max stats
    track_max_concurrent: bool = False  # compute peak occupancy?
    show_peak_overlay: bool = True  # display “peak …” on video overlay?
    poi_mode: str = "center"

@dataclass
class ExclusionZone:
    """Configuration for an exclusion zone"""
    name: str
    points_norm: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    enabled: bool = True

@dataclass
class AppConfig:
    """Main application configuration"""
    # Required fields (no defaults) - MUST come first
    model_path: str
    input_source: Union[str, int]  # Path or camera index
    output_folder: str

    # Optional fields (with defaults) - come after required fields
    confidence_threshold: float = 0.45
    device: str = "auto"
    input_type: InputType = InputType.FOLDER
    is_camera: bool = False
    enable_zones: bool = False
    save_video: bool = True
    segment_seconds: int = 60
    display_width: int = 1280
    display_height: int = 720
    lines_config: List[CountingLine] = field(default_factory=list)
    zones_config: List[CountingZone] = field(default_factory=list)
    exclusion_zones: List[ExclusionZone] = field(default_factory=list)
    allowed_classes: Set[int] = field(default_factory=set)
    tracker_config: Optional[str] = None
    max_track_age: int = 30
    min_track_length: int = 3
    frame_skip: int = 1  # Process every Nth frame (1 = no skip)
    interpolate_tracks: bool = True  # Interpolate positions for skipped frames
    show_live_video: bool = True  # Camera mode: start with full annotated video view enabled by default
    source_name: str = ""  # Stable camera/source identifier used in exports
    camera_stall_timeout_seconds: float = 20.0
    inference_stall_timeout_seconds: float = 120.0
    max_consecutive_detection_errors: int = 30
    performance_log_interval_seconds: float = 30.0
    video_writer_queue_size: int = 8
    video_writer_stall_timeout_seconds: float = 30.0

    # --- Heatmap options ---
    enable_heatmap: bool = False
    heatmap_interval_sec: float = 600.0
    heatmap_alpha: float = 0.35
    heatmap_colormap: str = "hot"
    heatmap_radius_px: int = 10
    heatmap_decay: float = 0.0
    heatmap_out_dir: str = "outputs/heatmaps"
    # ==== HEATMAP CONFIG: end ====

    # --- Speed estimation options ---
    enable_speed: bool = True
    speed_units: str = "pxps"  # "pxps" | "mps" | "kmh" | "mph"
    meters_per_pixel: float = 0.0  # if 0 => stay in px/s
    speed_smooth_window: int = 5
    annotate_speed: bool = True

    # --- Training mode options ---
    training_mode: bool = False
    training_interval_seconds: float = 5.0
    training_output_folder: str = "training_data"
    training_max_captures: int = 0  # 0 = unlimited
    training_auto_stop_hours: float = 2.0  # Auto-stop after N hours
    training_min_confidence: float = 0.5
    training_include_empty: bool = False
    training_augment: bool = False

    # Segment aggregation controls
    segment_split_minutes: int = 60  # Fixed at 60 minutes
    align_segments_to_clock: bool = True  # Always align to hour boundaries

    # Video output settings
    output_resolution: str = "720p"  # Options: "720p", "480p", "1080p", "original"
    playback_speed_multiplier: float = 1.0  # Export FPS multiplier (1.0 = real-time, 4.0 = 4x speed)

    # Parallel processing settings
    max_parallel_videos: int = 1  # Number of videos to process simultaneously (1-4 recommended)

    # Live recording / growing file settings
    wait_for_growing_files: bool = True  # Wait for files that are still being recorded
    growing_file_check_interval: float = 2.0  # Seconds between file size checks
    growing_file_timeout: float = 30.0  # Seconds to wait for file to grow before considering it complete
    folder_idle_timeout: float = 0.0  # Seconds to wait for new files before exiting (0 = wait forever)
    pre_process_stability_seconds: float = 10.0  # Seconds file must be stable before starting processing

    # API Connection
    enable_api_upload: bool = False

def resolve_colormap(name: str) -> int:
    """
    Resolve a colormap name to an OpenCV colormap constant.

    Args:
        name: Name of the colormap (e.g. "hot", "jet", "turbo", "autumn")

    Returns:
        OpenCV colormap constant usable with cv2.applyColorMap
    """
    name = (name or "").strip().lower()
    mapping = {
        "hot": cv2.COLORMAP_HOT,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "autumn": cv2.COLORMAP_AUTUMN,
        "cool": cv2.COLORMAP_COOL,
        "winter": cv2.COLORMAP_WINTER,
        "spring": cv2.COLORMAP_SPRING,
        "summer": cv2.COLORMAP_SUMMER,
        "bone": cv2.COLORMAP_BONE,
        "ocean": cv2.COLORMAP_OCEAN,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "parula": cv2.COLORMAP_PARULA
    }
    # Default fallback
    return mapping.get(name, cv2.COLORMAP_HOT)
