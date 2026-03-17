"""
Configuration Management Module

Handles all configuration aspects including:
- Initial user configuration through GUI
- Saving/loading configuration files
- Configuration validation
- Default settings management
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Set, Union
import logging
import cv2
from enum import Enum
from utils import get_available_axis_cameras


class InputType(Enum):
    """Enumeration for input types"""
    CAMERA = "camera"
    FOLDER = "folder"
    VIDEO = "video"


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


class ConfigManager:
    """Manages application configuration"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._available_cameras = self._detect_cameras()
        # Scan for AXIS IP cameras
        self._network_cameras = get_available_axis_cameras()

    def get_initial_config(self) -> Optional[AppConfig]:
        """
        Get initial configuration from user through GUI

        Returns:
            AppConfig object or None if canceled
        """
        try:
            return self._show_initial_config_dialog()
        except Exception as e:
            self.logger.error(f"Failed to get initial configuration: {e}")
            return None

    def _show_initial_config_dialog(self) -> Optional[AppConfig]:
        """Show the initial configuration dialog"""
        root = tk.Tk()
        root.title("Multi-Line Counter - Configuration")
        root.resizable(True, True)

        # --- HEADLESS / REMOTE DESKTOP SAFETY ---
        screen_h = root.winfo_screenheight()
        # If the screen reports an absurdly large virtual height, force a safe default of 750px
        if screen_h > 2000 or screen_h < 600:
            win_h = 750
        else:
            win_h = min(850, int(screen_h * 0.85))
        root.geometry(f"650x{win_h}")

        # --- REPAIRED SCROLLBAR SETUP ---
        main_container = tk.Frame(root)
        main_container.pack(fill="both", expand=True)

        # Row 0 (Canvas) gets all extra space, Row 1 (Buttons) stays at the bottom
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=0)
        main_container.grid_columnconfigure(0, weight=1)

        # 1. Scrollable Viewport (Canvas)
        canvas = tk.Canvas(main_container, highlightthickness=0, width=600)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame = tk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        # 2. FIXED BUTTONS (Now outside the canvas)
        # These will ALWAYS be at the bottom of the window
        button_frame = tk.Frame(main_container, pady=10, relief="raised", borderwidth=1)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Mouse wheel support
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        root.bind_all("<MouseWheel>", _on_mousewheel)

        # Variables
        config_vars = {
            'model_path': tk.StringVar(value="yolo11s.pt"),
            'input_type': tk.StringVar(value="folder"),
            'input_source': tk.StringVar(),
            'camera_index': tk.StringVar(value="0"),
            'output_folder': tk.StringVar(),
            'enable_zones': tk.BooleanVar(value=False),
            'enable_heatmap': tk.BooleanVar(value=False),
            'save_video': tk.BooleanVar(value=True),
            'confidence': tk.DoubleVar(value=0.45),
            'segment_seconds': tk.IntVar(value=60),
            'device': tk.StringVar(value="auto"),
            'enable_speed': tk.BooleanVar(value=False),
            'speed_units': tk.StringVar(value="pxps"),
            'meters_per_pixel': tk.StringVar(value="0.0"),  # string; we’ll float() it on submit
            'speed_smooth_window': tk.IntVar(value=5),
            'annotate_speed': tk.BooleanVar(value=True),
            'frame_skip': tk.IntVar(value=1),
            'interpolate_tracks': tk.BooleanVar(value=True),
            'max_parallel_videos': tk.IntVar(value=1),  # Number of videos to process in parallel

        }

        training_vars = {
            'training_mode': tk.BooleanVar(value=False),
            'training_interval': tk.StringVar(value="5.0"),
            'training_autostop': tk.StringVar(value="2.0"),
            'training_confidence': tk.StringVar(value="0.5"),
            'training_empty': tk.BooleanVar(value=False),
            'training_augment': tk.BooleanVar(value=False),
        }

        cloud_db_vars = {
            'enable_api_upload': tk.BooleanVar(value=False),
        }

        # --- NEW: Aggregation + alignment vars ---
        agg_var = tk.StringVar(value="15")  # 15, 30, 60
        align_var = tk.BooleanVar(value=True)  # align to :00/:15/:30/:45

        result_config = None

        def browse_model():
            """Browse for model file"""
            filetypes = [
                ("YOLO Models", "*.pt *.onnx *.engine"),
                ("PyTorch Models", "*.pt"),
                ("ONNX Models", "*.onnx"),
                ("TensorRT Models", "*.engine"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=filetypes,
                parent=root
            )
            if filename:
                config_vars['model_path'].set(filename)

        def browse_input():
            """Browse for input folder or video"""
            if config_vars['input_type'].get() == "folder":
                folder = filedialog.askdirectory(
                    title="Select Input Folder",
                    parent=root
                )
                if folder:
                    config_vars['input_source'].set(folder)
            else:  # video file
                filetypes = [
                    ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("All Files", "*.*")
                ]
                filename = filedialog.askopenfilename(
                    title="Select Video File",
                    filetypes=filetypes,
                    parent=root
                )
                if filename:
                    config_vars['input_source'].set(filename)

        def browse_output():
            """Browse for output folder"""
            folder = filedialog.askdirectory(
                title="Select Output Folder",
                parent=root
            )
            if folder:
                config_vars['output_folder'].set(folder)

        def toggle_input_fields():
            """Toggle input fields based on input type"""
            input_type = config_vars['input_type'].get()

            if input_type == "camera":
                input_entry.config(state="disabled")
                input_browse_btn.config(state="disabled")
                camera_combo.config(state="readonly")
            else:
                input_entry.config(state="normal")
                input_browse_btn.config(state="normal")
                camera_combo.config(state="disabled")

        def validate_and_submit():
            """Validate inputs and create configuration"""
            nonlocal result_config

            # Validation
            if not config_vars['model_path'].get():
                messagebox.showerror("Error", "Please select a model file.", parent=root)
                return

            if not Path(config_vars['model_path'].get()).exists():
                messagebox.showerror("Error", "Model file does not exist.", parent=root)
                return

            input_type = config_vars['input_type'].get()

            if input_type == "camera":
                selected = config_vars['camera_index'].get().strip()

                # Check if they selected an Axis camera from the list
                if selected in self._network_cameras:
                    source = self._network_cameras[selected]  # Gets the RTSP URL
                else:
                    # It's a standard webcam
                    try:
                        source = int(selected.split(":", 1)[0].strip())
                    except Exception:
                        messagebox.showerror("Error", "Please select a valid camera.", parent=root)
                        return

                is_camera = True
                input_type_enum = InputType.CAMERA

            else:
                source = config_vars['input_source'].get()
                if not source:
                    messagebox.showerror("Error", f"Please select an input {input_type}.", parent=root)
                    return

                if input_type == "folder" and not Path(source).is_dir():
                    messagebox.showerror("Error", "Input folder does not exist.", parent=root)
                    return
                elif input_type == "video" and not Path(source).is_file():
                    messagebox.showerror("Error", "Input video file does not exist.", parent=root)
                    return

                is_camera = False
                input_type_enum = InputType.FOLDER if input_type == "folder" else InputType.VIDEO

            if not config_vars['output_folder'].get():
                messagebox.showerror("Error", "Please select an output folder.", parent=root)
                return

            # Create output folder if it doesn't exist
            output_path = Path(config_vars['output_folder'].get())
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder: {e}", parent=root)
                return

            # NEW: read aggregation options
            segment_split_minutes = int(agg_var.get())
            align_segments_to_clock = bool(align_var.get())

            # --- Speed validation ---
            try:
                mpp = float(config_vars['meters_per_pixel'].get() or 0.0)
            except ValueError:
                messagebox.showerror("Error", "Meters per pixel must be a number.", parent=root)
                return

            if config_vars['enable_speed'].get() and config_vars['speed_units'].get() in (
            "mps", "kmh", "mph") and mpp <= 0.0:
                messagebox.showerror("Error", "Meters per pixel must be > 0 for mps / kmh / mph.", parent=root)
                return

            # Create configuration
            result_config = AppConfig(
                model_path=config_vars['model_path'].get(),
                input_source=source,
                input_type=input_type_enum,
                is_camera=is_camera,
                output_folder=str(output_path),
                enable_zones=config_vars['enable_zones'].get(),
                save_video=config_vars['save_video'].get(),
                confidence_threshold=config_vars['confidence'].get(),
                segment_seconds=config_vars['segment_seconds'].get(),
                enable_heatmap=bool(config_vars['enable_heatmap'].get()),
                segment_split_minutes=60,  # Always hourly
                align_segments_to_clock=True,  # Always align to clock hours
                device=config_vars['device'].get(),
                # --- NEW: speed params ---
                enable_speed=bool(config_vars['enable_speed'].get()),
                speed_units=str(config_vars['speed_units'].get()),
                meters_per_pixel=mpp,
                speed_smooth_window=int(config_vars['speed_smooth_window'].get()),
                annotate_speed=bool(config_vars['annotate_speed'].get()),
                frame_skip=int(config_vars['frame_skip'].get()),
                interpolate_tracks=bool(config_vars['interpolate_tracks'].get()),
                max_parallel_videos=int(config_vars['max_parallel_videos'].get()),
                # --- NEW: training params ---
                training_mode=training_vars['training_mode'].get(),
                training_interval_seconds=float(training_vars['training_interval'].get() or 5.0),
                training_output_folder=str(Path(config_vars['output_folder'].get()) / "training_data"),
                training_max_captures=0,  # Can be set via advanced config
                training_auto_stop_hours=float(training_vars['training_autostop'].get() or 2.0),
                training_min_confidence=float(training_vars['training_confidence'].get() or 0.5),
                training_include_empty=training_vars['training_empty'].get(),
                training_augment=training_vars['training_augment'].get(),
                # --- API data upload param ---
                enable_api_upload=cloud_db_vars['enable_api_upload'].get(),
            )

            root.quit()

        def cancel():
            """Cancel configuration"""
            root.quit()

        # --- UPDATED LAYOUT (Everything uses scrollable_frame) ---
        row = 0

        # Model selection
        tk.Label(scrollable_frame, text="YOLO Model File:", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )
        model_entry = tk.Entry(scrollable_frame, textvariable=config_vars['model_path'], width=50)
        model_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(scrollable_frame, text="Browse...", command=lambda: browse_model()).grid(
            row=row, column=2, padx=5, pady=5
        )
        row += 1

        # Input type selection
        tk.Label(scrollable_frame, text="Input Type:", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )
        input_frame = tk.Frame(scrollable_frame)
        input_frame.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        tk.Radiobutton(input_frame, text="Folder", variable=config_vars['input_type'],
                       value="folder", command=lambda: toggle_input_fields()).pack(side="left")
        tk.Radiobutton(input_frame, text="Video", variable=config_vars['input_type'],
                       value="video", command=lambda: toggle_input_fields()).pack(side="left")
        tk.Radiobutton(input_frame, text="Camera", variable=config_vars['input_type'],
                       value="camera", command=lambda: toggle_input_fields()).pack(side="left")
        row += 1

        # Input source
        tk.Label(scrollable_frame, text="Input Source:").grid(row=row, column=0, sticky="w", padx=10, pady=5)
        input_entry = tk.Entry(scrollable_frame, textvariable=config_vars['input_source'], width=50)
        input_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        input_browse_btn = tk.Button(scrollable_frame, text="Browse...", command=lambda: browse_input())
        input_browse_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # Camera selection (Directly using tk.ttk)
        tk.Label(scrollable_frame, text="Camera:").grid(row=row, column=0, sticky="w", padx=10, pady=5)

        # Build the unified camera list
        camera_values = []
        for i in range(len(self._available_cameras)):
            camera_values.append(f"{i}: Standard Webcam {i}")

        # Add the dynamically found Axis cameras
        for axis_name in self._network_cameras.keys():
            camera_values.append(axis_name)

        if not camera_values:
            camera_values = ["No cameras found"]

        camera_combo = tk.ttk.Combobox(scrollable_frame, textvariable=config_vars['camera_index'],
                                       values=camera_values, state="disabled", width=47)

        # Auto-select the first Axis camera if one was found, otherwise default to first webcam
        if self._network_cameras:
            config_vars['camera_index'].set(list(self._network_cameras.keys())[0])
        elif self._available_cameras:
            config_vars['camera_index'].set(camera_values[0])

        camera_combo.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        row += 1

        # Output folder
        tk.Label(scrollable_frame, text="Output Folder:", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=10, pady=5
        )
        output_entry = tk.Entry(scrollable_frame, textvariable=config_vars['output_folder'], width=50)
        output_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(scrollable_frame, text="Browse...", command=lambda: browse_output()).grid(
            row=row, column=2, padx=5, pady=5
        )
        row += 1

        # Options Section
        options_frame = tk.LabelFrame(scrollable_frame, text="Options", font=("Arial", 10, "bold"))
        options_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        tk.Checkbutton(options_frame, text="Enable Zones", variable=config_vars['enable_zones']).grid(row=0,
                                                                                                      column=0,
                                                                                                      sticky="w",
                                                                                                      padx=5)
        tk.Checkbutton(options_frame, text="Save Video", variable=config_vars['save_video']).grid(row=0, column=1,
                                                                                                  sticky="w",
                                                                                                  padx=5)
        tk.Checkbutton(options_frame, text="Heatmap", variable=config_vars['enable_heatmap']).grid(row=0, column=2,
                                                                                                   sticky="w",
                                                                                                   padx=5)
        row += 1
        # Advanced settings
        advanced_frame = tk.Frame(options_frame)
        advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        tk.Label(advanced_frame, text="Confidence:").grid(row=0, column=0, sticky="w", padx=5)
        confidence_scale = tk.Scale(advanced_frame, from_=0.1, to=0.9, resolution=0.05,
                                    orient="horizontal", variable=config_vars['confidence'])
        confidence_scale.grid(row=0, column=1, sticky="ew", padx=5)



        tk.Label(advanced_frame, text="Device:").grid(row=1, column=0, sticky="w", padx=5)
        device_combo = ttk.Combobox(advanced_frame, textvariable=config_vars['device'],
                                    values=["auto", "cpu", "cuda", "mps"], width=10)
        device_combo.grid(row=1, column=1, sticky="w", padx=5)

        # --- Speed (optional) ---
        row += 1
        speed_frame = tk.LabelFrame(scrollable_frame, text="Speed (optional)", font=("Arial", 10, "bold"))
        speed_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=6, sticky="ew")

        # Left side: master toggle + draw labels
        tk.Checkbutton(speed_frame, text="Enable speed estimation",
                       variable=config_vars['enable_speed']).grid(row=0, column=0, sticky="w", padx=5, pady=4)
        tk.Checkbutton(speed_frame, text="Draw speed labels",
                       variable=config_vars['annotate_speed']).grid(row=1, column=0, sticky="w", padx=5, pady=2)

        # Right side: units + (optional) scale + smoothing
        ttk.Label(speed_frame, text="Units:").grid(row=0, column=1, sticky="e", padx=5)
        units_combo = ttk.Combobox(speed_frame, textvariable=config_vars['speed_units'],
                                   state="readonly", values=["pxps", "mps", "kmh", "mph"], width=8)
        units_combo.grid(row=0, column=2, sticky="w", padx=5)

        ttk.Label(speed_frame, text="Meters per pixel:").grid(row=1, column=1, sticky="e", padx=5)
        mpp_entry = tk.Entry(speed_frame, textvariable=config_vars['meters_per_pixel'], width=10)
        mpp_entry.grid(row=1, column=2, sticky="w", padx=5)

        ttk.Label(speed_frame, text="Smooth window:").grid(row=2, column=1, sticky="e", padx=5)
        smooth_combo = ttk.Combobox(speed_frame, textvariable=config_vars['speed_smooth_window'],
                                    state="readonly", values=["3", "5", "7", "9"], width=6)
        smooth_combo.grid(row=2, column=2, sticky="w", padx=5)



        # --- Frame Skipping ---
        row += 1
        frame_skip_frame = tk.LabelFrame(scrollable_frame, text="Performance Optimization", font=("Arial", 10, "bold"))
        frame_skip_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=6, sticky="ew")

        # Frame skip dropdown
        tk.Label(frame_skip_frame, text="Process every:").grid(row=0, column=0, sticky="w", padx=5, pady=4)

        frame_skip_var = tk.IntVar(value=1)

        frame_skip_combo = ttk.Combobox(
            frame_skip_frame,
            textvariable=config_vars['frame_skip'],  # Use directly from config_vars
            state="readonly",
            values=["1", "2", "3", "4", "5"],
            width=8
        )

        frame_skip_combo.grid(row=0, column=1, sticky="w", padx=5, pady=4)
        frame_skip_combo.set("1")

        tk.Label(frame_skip_frame, text="frame(s)").grid(row=0, column=2, sticky="w", padx=2, pady=4)

        # Interpolation checkbox
        tk.Checkbutton(
            frame_skip_frame,
            text="Interpolate tracks between skipped frames",
            variable=config_vars['interpolate_tracks']  # Use directly from config_vars
        ).grid(row=0, column=4, columnspan=4, sticky="w", padx=5, pady=2)

        # Parallel videos processing
        tk.Label(frame_skip_frame, text="Parallel videos:").grid(row=1, column=0, sticky="w", padx=5, pady=4)

        parallel_combo = ttk.Combobox(
            frame_skip_frame,
            textvariable=config_vars['max_parallel_videos'],
            state="readonly",
            values=["1", "2", "3", "4"],
            width=8
        )
        parallel_combo.grid(row=1, column=1, sticky="w", padx=5, pady=4)
        parallel_combo.set("1")
        
        tk.Label(frame_skip_frame, text="(higher = faster but uses more GPU memory)").grid(
            row=1, column=2, columnspan=4, sticky="w", padx=5, pady=4)

        # --- Training Mode (optional) ---
        row += 1
        training_frame = tk.LabelFrame(scrollable_frame, text="Training Mode (optional)", font=("Arial", 10, "bold"))
        training_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=6, sticky="ew")

        # Enable training mode checkbox
        tk.Checkbutton(training_frame, text="Enable training data capture",
                       variable=training_vars['training_mode']).grid(row=0, column=0, sticky="w", padx=5, pady=4)

        # Capture interval
        ttk.Label(training_frame, text="Capture interval (seconds):").grid(row=1, column=0, sticky="e", padx=5)
        interval_entry = tk.Entry(training_frame, textvariable=training_vars['training_interval'], width=10)
        interval_entry.grid(row=1, column=1, sticky="w", padx=5)

        # Auto-stop after hours
        ttk.Label(training_frame, text="Auto-stop after (hours):").grid(row=2, column=0, sticky="e", padx=5)
        autostop_entry = tk.Entry(training_frame, textvariable=training_vars['training_autostop'], width=10)
        autostop_entry.grid(row=2, column=1, sticky="w", padx=5)

        # Min confidence
        ttk.Label(training_frame, text="Min confidence:").grid(row=3, column=0, sticky="e", padx=5)
        conf_entry = tk.Entry(training_frame, textvariable=training_vars['training_confidence'], width=10)
        conf_entry.grid(row=3, column=1, sticky="w", padx=5)

        # Include empty frames
        tk.Checkbutton(training_frame, text="Include frames with no detections",
                       variable=training_vars['training_empty']).grid(row=4, column=0, columnspan=2, sticky="w", padx=5,
                                                                      pady=2)

        # Augment captures
        tk.Checkbutton(training_frame, text="Apply augmentation (flip, brightness)",
                       variable=training_vars['training_augment']).grid(row=5, column=0, columnspan=2, sticky="w",
                                                                        padx=5, pady=2)

        # --- API Upload (optional) ---
        row += 1
        cloud_frame = tk.LabelFrame(scrollable_frame, text="API Upload", font=("Arial", 10, "bold"))
        cloud_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=6, sticky="ew")

        # Checkbox to enable API Upload
        tk.Checkbutton(
            cloud_frame,
            text="Enable API Upload",
            variable=cloud_db_vars['enable_api_upload']
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=4)

        row += 1

        # Load config function
        def load_existing_config():
            """Load an existing configuration file"""
            nonlocal result_config
            
            filetypes = [
                ("Config Files", "*.json *.yaml *.yml"),
                ("JSON Files", "*.json"),
                ("YAML Files", "*.yaml *.yml"),
                ("All Files", "*.*")
            ]
            filename = filedialog.askopenfilename(
                title="Select Configuration File",
                filetypes=filetypes,
                parent=root
            )
            if not filename:
                return
            
            try:
                # Load the config
                loaded_config = self.load_config(filename)
                if loaded_config is None:
                    messagebox.showerror("Error", "Failed to load configuration file.", parent=root)
                    return
                
                # Validate essential paths
                errors = []
                if not Path(loaded_config.model_path).exists():
                    errors.append(f"Model file not found: {loaded_config.model_path}")
                
                if not loaded_config.is_camera:
                    if not Path(loaded_config.input_source).exists():
                        errors.append(f"Input source not found: {loaded_config.input_source}")
                
                if errors:
                    error_msg = "Configuration has invalid paths:\n\n" + "\n".join(errors)
                    error_msg += "\n\nWould you like to load it anyway and fix the paths?"
                    if not messagebox.askyesno("Path Validation", error_msg, parent=root):
                        return
                
                # Check if config has lines configured - offer to skip setup
                if loaded_config.lines_config:
                    msg = f"Configuration loaded successfully!\n\n"
                    msg += f"• {len(loaded_config.lines_config)} counting line(s)\n"
                    msg += f"• {len(loaded_config.zones_config)} zone(s)\n"
                    msg += f"• {len(loaded_config.exclusion_zones)} exclusion zone(s)\n\n"
                    msg += "Would you like to use this configuration and skip to processing?\n\n"
                    msg += "(Click 'No' to modify settings first)"
                    
                    if messagebox.askyesno("Configuration Loaded", msg, parent=root):
                        # Use directly - skip GUI setup
                        result_config = loaded_config
                        result_config._skip_gui_setup = True  # Flag to skip interactive setup
                        root.quit()
                        return
                
                # Populate the dialog fields with loaded values
                config_vars['model_path'].set(loaded_config.model_path)
                config_vars['output_folder'].set(loaded_config.output_folder)
                config_vars['enable_zones'].set(loaded_config.enable_zones)
                config_vars['enable_heatmap'].set(loaded_config.enable_heatmap)
                config_vars['save_video'].set(loaded_config.save_video)
                config_vars['confidence'].set(loaded_config.confidence_threshold)
                config_vars['device'].set(loaded_config.device)
                config_vars['enable_speed'].set(loaded_config.enable_speed)
                config_vars['speed_units'].set(loaded_config.speed_units)
                config_vars['meters_per_pixel'].set(str(loaded_config.meters_per_pixel))
                config_vars['speed_smooth_window'].set(loaded_config.speed_smooth_window)
                config_vars['annotate_speed'].set(loaded_config.annotate_speed)
                config_vars['frame_skip'].set(loaded_config.frame_skip)
                config_vars['interpolate_tracks'].set(loaded_config.interpolate_tracks)
                config_vars['max_parallel_videos'].set(loaded_config.max_parallel_videos)
                
                # Set input type and source
                if loaded_config.is_camera:
                    config_vars['input_type'].set("camera")
                    config_vars['camera_index'].set(str(loaded_config.input_source))
                elif loaded_config.input_type == InputType.FOLDER:
                    config_vars['input_type'].set("folder")
                    config_vars['input_source'].set(str(loaded_config.input_source))
                else:
                    config_vars['input_type'].set("video")
                    config_vars['input_source'].set(str(loaded_config.input_source))
                
                # Training mode settings
                training_vars['training_mode'].set(loaded_config.training_mode)
                training_vars['training_interval'].set(str(loaded_config.training_interval_seconds))
                training_vars['training_autostop'].set(str(loaded_config.training_auto_stop_hours))
                training_vars['training_confidence'].set(str(loaded_config.training_min_confidence))
                training_vars['training_empty'].set(loaded_config.training_include_empty)
                training_vars['training_augment'].set(loaded_config.training_augment)
                
                # Cloud database settings
                cloud_db_vars['enable_api_upload'].set(getattr(loaded_config, 'enable_api_upload', False))
                
                # Update field states
                toggle_input_fields()
                
                messagebox.showinfo("Config Loaded", 
                    "Configuration loaded. Review/modify settings and click Continue.", 
                    parent=root)
                
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}", parent=root)

        # 2. FIXED BUTTONS (Stays at the very bottom)
        button_frame = tk.Frame(main_container, pady=10, relief="raised", borderwidth=1)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        # NEW: A sub-frame that centers itself within button_frame
        center_btn_container = tk.Frame(button_frame)
        center_btn_container.pack(expand=True)  # This is the magic line for centering

        # Now pack your buttons into the container instead of the main frame
        tk.Button(center_btn_container, text="Load Config...", command=load_existing_config,
                  width=15, bg="#2196F3", fg="white").pack(side="left", padx=10)

        tk.Button(center_btn_container, text="Cancel", command=cancel,
                  width=15).pack(side="left", padx=10)

        tk.Button(center_btn_container, text="Continue", command=validate_and_submit,
                  width=15, bg="#4CAF50", fg="white").pack(side="left", padx=10)

        scrollable_frame.columnconfigure(1, weight=1)
        toggle_input_fields()

        # so the window is centered correctly once all widgets are drawn.
        self.center_window(root)

        root.mainloop()
        root.destroy()
        return result_config

        # Show effective FPS
        def update_fps_label(*args):
            skip_value = frame_skip_var.get()
            if skip_value > 1:
                fps_text = f"(~{skip_value}x faster, processing every {skip_value} frames)"
            else:
                fps_text = "(No frame skipping)"
            fps_label.config(text=fps_text)

        frame_skip_var.trace('w', update_fps_label)
        fps_label = tk.Label(frame_skip_frame, text="(No frame skipping)", fg="gray")
        fps_label.grid(row=0, column=3, sticky="w", padx=10, pady=4)

        # Interpolation option
        interpolate_var = tk.BooleanVar(value=True)

        tk.Checkbutton(
            frame_skip_frame,
            text="Interpolate tracks between skipped frames",
            variable=interpolate_var
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=2)

        tk.Label(
            frame_skip_frame,
            text="Note: Higher skip values increase speed but may reduce accuracy",
            fg="gray",
            font=("Arial", 8)
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=2)

    def _detect_cameras(self) -> List[int]:
        """Detect available camera devices"""
        available_cameras = []

        # Check up to 10 camera indices
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                cap.release()
            except:
                continue

        return available_cameras

    def save_config(self, config: AppConfig, filepath: Union[str, Path]) -> bool:
        """
        Save configuration to file including lines, zones, and exclusions

        Args:
            config: Configuration to save
            filepath: Path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary with full serialization
            config_dict = self._config_to_dict(config)

            # Ensure all complex objects are properly serialized
            # Convert lines_config
            if hasattr(config, 'lines_config') and config.lines_config:
                config_dict['lines_config'] = [
                    {
                        'name': line.name,
                        'start_norm': line.start_norm,
                        'end_norm': line.end_norm,
                        'direction': line.direction,
                        'classes': line.classes,
                        'enabled': line.enabled,
                        'poi_mode': getattr(line, 'poi_mode', 'center')
                    }
                    for line in config.lines_config
                ]

            # Convert zones_config
            if hasattr(config, 'zones_config') and config.zones_config:
                config_dict['zones_config'] = [
                    {
                        'name': zone.name,
                        'points_norm': zone.points_norm,
                        'classes': zone.classes,
                        'enabled': zone.enabled,
                        'track_max_concurrent': getattr(zone, 'track_max_concurrent', False),
                        'show_peak_overlay': getattr(zone, 'show_peak_overlay', True),
                        'poi_mode': getattr(zone, 'poi_mode', 'center')
                    }
                    for zone in config.zones_config
                ]

            # Convert exclusion_zones
            if hasattr(config, 'exclusion_zones') and config.exclusion_zones:
                config_dict['exclusion_zones'] = [
                    {
                        'name': exc.name,
                        'points_norm': exc.points_norm,
                        'enabled': exc.enabled
                    }
                    for exc in config.exclusion_zones
                ]

            # Determine format from extension
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                with open(filepath, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:  # Default to JSON
                with open(filepath, 'w') as f:
                    json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def load_config(self, filepath: Union[str, Path]) -> Optional[AppConfig]:
        """
        Load configuration from file

        Args:
            filepath: Path to config file

        Returns:
            AppConfig instance or None if failed
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                self.logger.error(f"Config file not found: {filepath}")
                return None

            # Load based on extension
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                with open(filepath, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:  # Default to JSON
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)

            # Convert to AppConfig
            config = self._dict_to_config(config_dict)
            self.logger.info(f"Configuration loaded from {filepath}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return None

    def _config_to_dict(self, config: AppConfig) -> Dict:
        """Convert AppConfig to dictionary for serialization"""
        config_dict = asdict(config)

        # Convert enums and sets to serializable types
        config_dict['input_type'] = config.input_type.value
        config_dict['allowed_classes'] = list(config.allowed_classes)

        return config_dict

    def _dict_to_config(self, config_dict: Dict) -> AppConfig:
        """Convert dictionary to AppConfig"""
        # Handle enum conversion
        if 'input_type' in config_dict:
            config_dict['input_type'] = InputType(config_dict['input_type'])

        # Handle set conversion
        if 'allowed_classes' in config_dict:
            config_dict['allowed_classes'] = set(config_dict['allowed_classes'])

        # Handle lists of dataclasses
        if 'lines_config' in config_dict:
            config_dict['lines_config'] = [
                CountingLine(**line) if isinstance(line, dict) else line
                for line in config_dict['lines_config']
            ]

        if 'zones_config' in config_dict:
            config_dict['zones_config'] = [
                CountingZone(**zone) if isinstance(zone, dict) else zone
                for zone in config_dict['zones_config']
            ]

        if 'exclusion_zones' in config_dict:
            config_dict['exclusion_zones'] = [
                ExclusionZone(**exc) if isinstance(exc, dict) else exc
                for exc in config_dict['exclusion_zones']
            ]

        return AppConfig(**config_dict)

    def validate_config(self, config: AppConfig) -> List[str]:
        """
        Validate configuration and return list of errors

        Args:
            config: Configuration to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check model file
        if not config.model_path or not Path(config.model_path).exists():
            errors.append("Model file does not exist")

        # Check input source
        if config.is_camera:
            if not isinstance(config.input_source, int) or config.input_source < 0:
                errors.append("Invalid camera index")
        else:
            if not config.input_source or not Path(config.input_source).exists():
                errors.append("Input source does not exist")

        # Check output folder
        if not config.output_folder:
            errors.append("Output folder not specified")

        # Check confidence threshold
        if not 0.0 <= config.confidence_threshold <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")

        # Check segment duration
        if config.segment_seconds <= 0:
            errors.append("Segment duration must be positive")

        return errors

    def get_default_config(self) -> AppConfig:
        """Get default configuration"""
        return AppConfig(
            model_path="",
            input_source="",
            output_folder="",
            confidence_threshold=0.45,
            device="auto",
            enable_zones=False,
            save_video=True,
            segment_seconds=60,
            # heatmap defaults …
            enable_heatmap=False,
            heatmap_interval_sec=600.0,
            heatmap_alpha=0.35,
            heatmap_colormap="hot",
            heatmap_radius_px=10,
            heatmap_decay=0.0,
            heatmap_out_dir="outputs/heatmaps",
            # speed defaults
            enable_speed=False,
            speed_units="pxps",
            meters_per_pixel=0.0,
            speed_smooth_window=5,
            annotate_speed=True,
        )

    def center_window(self, window):
        """Center the window relative to the area above the taskbar, safe for headless."""
        window.update_idletasks()

        window_width = window.winfo_width()
        window_height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        # Headless fallback: If the virtual screen is massive, assume a standard 1080p viewer
        if screen_width > 3000 or screen_height > 2000:
            screen_width = 1920
            screen_height = 1080

        usable_height = screen_height * 0.95
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, int((usable_height - window_height) // 2))

        window.geometry(f"+{x}+{y}")

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