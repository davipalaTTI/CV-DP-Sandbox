import copy
import cv2
import math
import time
import signal
import logging
import threading
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from queue import Queue as ThreadQueue
from queue import Empty
import tkinter as tk

from config_manager import AppConfig, resolve_colormap, save_app_config
from core.detection_engine import DetectionEngine
from core.tracking import ObjectCounter, CountingEvent
from core.training_mode import TrainingModeCapture, TrainingConfig
from utils.results_export import ResultsExporter, ExportConfig, get_master_log_writer
from utils.footage_retention import cleanup_live_footage, footage_policy_label

from .stats import ProcessingStats
from .async_io import (
    AsyncCameraReader,
    AsyncVideoWriter,
    ThreadedDetectionEngine,
    put_latest,
)
from .visualizer import Visualizer
from .interaction_handler import InteractionHandler
from ..visualizer_heatmap import HeatmapAccumulator


class CameraRunner:
    """High-Performance orchestrator for Live Webcams and RTSP streams"""

    def __init__(self, config: AppConfig, detection_engine: DetectionEngine):
        self.config = config
        self.detection_engine = detection_engine
        self.logger = logging.getLogger(__name__)
        self.config.save_video = bool(getattr(config, "save_video", False))
        self.footage_retention_days = max(
            0, int(getattr(config, "footage_retention_days", 0))
        )
        self.config.footage_retention_days = self.footage_retention_days
        self.logger.info(
            "Footage recording policy: %s",
            footage_policy_label(self.config.save_video, self.footage_retention_days),
        )

        # Live edit state (for moving lines/zones)
        self.edit_mode = False
        self.dragging = False
        self.drag_target = None
        self.drag_threshold = 12

        # Inline creation state
        self.create_mode = "none"  # "none" | "line" | "zone"
        self.temp_points = []
        self.tk_root = None

        # UI visibility states
        self.show_stats = True
        self.show_controls = True
        self.ui_minimized = False
        self.headless = bool(getattr(config, 'runtime_headless', False))
        self.live_view_enabled = (
            bool(getattr(config, 'show_live_video', True)) and not self.headless
        )
        self.stop_at = getattr(config, 'runtime_stop_at', None)
        self.source_name = str(getattr(config, 'source_name', '') or 'LIVE CAMERA')
        self.runtime_config_path = Path(
            getattr(config, 'runtime_config_path', '')
            or (Path(config.output_folder) / "config.json")
        ).expanduser()
        self.config_persist_lock = threading.Lock()
        self._config_dirty = False
        self.window_index = max(0, int(getattr(config, 'runtime_window_index', 0)))
        self.window_count = max(1, int(getattr(config, 'runtime_window_count', 1)))
        self.camera_stall_timeout = max(
            5.0, float(getattr(config, 'camera_stall_timeout_seconds', 20.0))
        )
        self.inference_stall_timeout = max(
            15.0, float(getattr(config, 'inference_stall_timeout_seconds', 120.0))
        )
        self.performance_log_interval = max(
            5.0, float(getattr(config, 'performance_log_interval_seconds', 30.0))
        )
        self.video_writer_queue_size = max(
            2, min(32, int(getattr(config, 'video_writer_queue_size', 8)))
        )
        self.video_writer_stall_timeout = max(
            10.0,
            float(getattr(config, 'video_writer_stall_timeout_seconds', 30.0)),
        )
        self.max_consecutive_detection_errors = max(
            1, int(getattr(config, 'max_consecutive_detection_errors', 30))
        )
        window_source = self.source_name.strip()[:80] or "LIVE CAMERA"
        self.live_window_name = f"Live Object Counter - {window_source}"
        self.status_window_name = f"Live Counter Status - {window_source}"
        self._live_window_created = False
        self._status_window_created = False
        self._last_status_draw_time = 0.0
        self._status_frame = np.zeros((180, 460, 3), dtype=np.uint8)

        # Frame skipping & Mouse tracking (Required by Visualizer)
        self.frame_skip = config.frame_skip
        self.interpolate_tracks = config.interpolate_tracks
        self.frame_skip_counter = 0
        self.last_detections = []
        self._last_mouse_pos = (0, 0)

        # Pause & Notification States
        self.is_paused = False
        self.notification = None

        # Training mode initialization
        self.training_capture = None
        if getattr(config, 'training_mode', False):
            training_config = TrainingConfig(
                enabled=True,
                capture_interval_seconds=getattr(config, 'training_interval_seconds', 5.0),
                output_folder=getattr(config, 'training_output_folder', 'training_data'),
                include_empty_frames=getattr(config, 'training_include_empty', False),
                max_captures_per_session=getattr(config, 'training_max_captures', 0),
                auto_stop_after_hours=getattr(config, 'training_auto_stop_hours', 2.0),
                min_confidence=getattr(config, 'training_min_confidence', 0.5),
                augment_captures=getattr(config, 'training_augment', False),
                save_metadata=True
            )
            self.training_capture = TrainingModeCapture(
                training_config,
                self.detection_engine.class_names
            )
            self.logger.info("Training mode initialized")

        # Cache config values
        self._enable_speed = bool(getattr(config, "enable_speed", False))
        self._speed_units = str(getattr(config, "speed_units", "pxps"))
        self._annotate_speed = bool(getattr(config, "annotate_speed", True))
        self._enable_heatmap = bool(getattr(config, "enable_heatmap", False))

        # Counter Setup
        frame_size = (config.display_width, config.display_height)
        self.counter = self._create_configured_counter(frame_size)

        # UI & Hardware State
        self.cap = None
        self.video_writer = None
        self.is_running = False
        self.stop_requested = False
        self.stats = ProcessingStats()
        self.video_frame_number = 0
        self.is_live_source = True
        self.video_current_time = datetime.now()

        # Heatmap, Visualizer, Interaction Handlers
        self.heatmap_acc = None
        self.visualizer = Visualizer(self.config)
        self.interaction_handler = InteractionHandler(self)

        # Results exporter
        export_config = ExportConfig(enable_api_upload=self.config.enable_api_upload)
        self.exporter = ResultsExporter(config.output_folder, export_config)

        # Segment tracking
        self.current_segment = 0
        self.segment_start_time = time.time()
        self.current_segment_start_dt = datetime.now()

        # Segment time rules (Required by Visualizer)
        self.segment_split_minutes = 60
        self.align_segments_to_clock = True
        self.segment_duration_seconds = 3600

        self.next_split_dt = self._compute_next_clock_boundary(datetime.now(), 60)

        # Live Master Log Export settings
        self.live_export_interval = 300.0
        self.last_live_export_time = time.time()
        self.last_exported_event_count = 0
        self.export_lock = threading.Lock()
        self.active_zone_events = {}

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

    # --- SETUP & INITIALIZATION ---
    def _create_configured_counter(self, frame_size: Tuple[int, int]) -> ObjectCounter:
        counter = ObjectCounter(
            self.config.lines_config,
            self.config.zones_config,
            frame_size,
            exclusion_zones=getattr(self.config, 'exclusion_zones', []),
            max_track_age=self.config.max_track_age
        )
        counter.configure_speed(
            enable=self._enable_speed,
            units=self._speed_units,
            meters_per_pixel=float(getattr(self.config, "meters_per_pixel", 0.0) or 0.0),
            smooth_window=int(getattr(self.config, "speed_smooth_window", 5) or 5),
            annotate=self._annotate_speed
        )
        return counter

    @staticmethod
    def _is_rtsp_source(source: object) -> bool:
        return isinstance(source, str) and source.strip().lower().startswith(("rtsp://", "rtsps://"))

    def _initialize_camera(self) -> bool:
        try:
            source = self.config.input_source
            is_rtsp = self._is_rtsp_source(source)

            # RTSP streams are still live sources, so they use CameraRunner.
            # CAP_FFMPEG is preferred for RTSP when OpenCV was built with FFmpeg support;
            # local webcams keep the default backend.
            if is_rtsp:
                timeout_ms = max(1000, int(self.camera_stall_timeout * 500))
                params = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    min(timeout_ms, 10000),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    timeout_ms,
                ]
                try:
                    self.cap = cv2.VideoCapture(
                        str(source), cv2.CAP_FFMPEG, params
                    )
                except (cv2.error, TypeError):
                    self.cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                source_type = "RTSP stream" if is_rtsp else "camera"
                self.logger.error("Failed to open %s", source_type)
                return False

            # Width/FPS setters are useful for local webcams, but RTSP stream dimensions
            # are usually controlled by the camera/server and these setters may be ignored.
            if not is_rtsp:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.video_fps = actual_fps if actual_fps > 0 else 30.0
            source_type = "RTSP stream" if is_rtsp else "Camera"
            self.logger.info(f"{source_type} initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

            self.cap = AsyncCameraReader(self.cap)
            return True
        except Exception as e:
            self.logger.error(f"Camera/stream initialization failed: {e}")
            return False

    def _initialize_video_writer(self, filename: Optional[str] = None, frame: Optional[np.ndarray] = None) -> bool:
        if not self.config.save_video:
            self.video_writer = None
            return False
        if not self.cap:
            return False
        try:
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            w -= (w % 2)
            h -= (h % 2)

            fps_raw = self.cap.get(cv2.CAP_PROP_FPS)
            base_fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0
            playback_speed = 2.0
            export_fps = base_fps * playback_speed

            now = datetime.now()
            out_dir = Path(self.config.output_folder) / "live_footage"
            out_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                filename = f"live_camera_{timestamp}.mp4"

            output_path = out_dir / filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(str(output_path), fourcc, export_fps, (w, h))

            if vw.isOpened():
                self.video_writer = AsyncVideoWriter(
                    vw, queue_size=self.video_writer_queue_size
                )
                self._video_writer_size = (w, h)
                self.logger.info(f"Video writer initialized successfully at: {output_path}")
                return True
            else:
                self.logger.error("Failed to initialize video writer.")
                self.video_writer = None
                return False
        except Exception as e:
            self.logger.error(f"Video writer initialization error: {e}")
            self.video_writer = None
            return False

    # --- MAIN PROCESSING LOOP ---
    def run(self) -> Dict:
        self.logger.info("Starting High-Performance camera processing...")

        self._cleanup_expired_footage()

        if not self._initialize_camera():
            raise RuntimeError("Failed to initialize camera")

        # Initialize Zero-Copy Queues
        self.camera_to_engine_queue = ThreadQueue(maxsize=1)
        self.engine_to_analytics_queue = ThreadQueue(maxsize=1)

        # Start the YOLO Background Thread
        self.threaded_engine = ThreadedDetectionEngine(
            engine=self.detection_engine,
            input_queue=self.camera_to_engine_queue,
            output_queue=self.engine_to_analytics_queue,
            max_consecutive_errors=self.max_consecutive_detection_errors,
        )

        if self.config.save_video:
            self._initialize_video_writer()

        self.is_running = True
        self.stats.start_time = time.time()

        first_frame = True
        last_submitted_sequence = -1
        self._input_frames_dropped = 0
        self._performance_last_at = time.monotonic()
        self._performance_last_capture_count = self.cap.frames_read
        self._performance_last_inference_count = 0
        self._performance_last_input_drops = 0
        self._performance_process = psutil.Process()

        try:
            while self.is_running and not self.stop_requested:
                if self.stop_at is not None and datetime.now() >= self.stop_at:
                    self.logger.info(f"Scheduled stop reached: {self.stop_at.isoformat()}")
                    break

                self._check_pipeline_health()

                # Submit each captured frame at most once. If inference is busy,
                # replace its one queued frame with the newest camera frame.
                ret, raw_frame, sequence = self.cap.read_latest(
                    last_submitted_sequence, copy_frame=False
                )
                if ret and raw_frame is not None:
                    if last_submitted_sequence >= 0 and sequence > last_submitted_sequence + 1:
                        self._input_frames_dropped += (
                            sequence - last_submitted_sequence - 1
                        )
                    if put_latest(self.camera_to_engine_queue, raw_frame):
                        self._input_frames_dropped += 1
                    last_submitted_sequence = sequence

                    if first_frame:
                        h, w = raw_frame.shape[:2]

                        # Build the counter/exclusion math from the actual camera frame size.
                        actual_size = (w, h)
                        if self.counter.frame_size != actual_size:
                            self.counter.set_frame_size(actual_size)

                        if getattr(self.config, 'exclusion_zones', None):
                            self.detection_engine.set_exclusion_zones(self.config.exclusion_zones, raw_frame.shape)

                        if self.headless:
                            pass
                        elif self.live_view_enabled:
                            self._ensure_live_window(w, h)
                        else:
                            self._ensure_status_window()

                        first_frame = False

                # 2. Process Detections
                try:
                    processed_frame, detections = self.engine_to_analytics_queue.get_nowait()
                    self.last_frame = processed_frame
                    self.video_frame_number += 1
                    self.video_current_time = datetime.now()

                    events = self.counter.update_counts(detections, timestamp=self.video_current_time)
                    self.stats.total_detections += len(detections)

                    if events:
                        segment_id = f"hour_{self.current_segment_start_dt.hour:02d}"
                        self.exporter.queue_api_events(
                            events,
                            video_source=self.source_name,
                            segment_id=segment_id
                        )

                    # --- NEW: Capture training frames from the live camera! ---
                    if self.training_capture:
                        captured_path = self.training_capture.capture_frame(processed_frame, detections)
                        if captured_path:
                            self.logger.debug(f"Training frame saved: {captured_path}")

                    if self.heatmap_acc is not None and self.video_frame_number % 5 == 0:
                        boxes_xyxy = []
                        for det in detections:
                            x1, y1, x2, y2 = det.bbox
                            boxes_xyxy.append((int(x1), int(y1), int(x2), int(y2)))
                        if boxes_xyxy:
                            self.heatmap_acc.update_from_boxes(boxes_xyxy, weight=1.2)

                    # Only draw the expensive overlays when needed:
                    # - full live view is open, or
                    # - save_video is enabled and we need an annotated recording.
                    display_frame = None
                    if self.live_view_enabled or self.config.save_video:
                        display_frame = self.visualizer.draw_all(self, processed_frame, detections)

                    if self.config.save_video and display_frame is not None:
                        self._save_video_frame(display_frame)

                    if self._should_rollover_segment():
                        self._rollover_segment()

                    self._trigger_live_export(force=False)
                    self._update_stats(len(events) if events else 0)
                    self._log_pipeline_performance()

                    if self.headless:
                        pass
                    elif self.live_view_enabled:
                        self._ensure_live_window(processed_frame.shape[1], processed_frame.shape[0])
                        cv2.imshow(self.live_window_name, display_frame)
                    else:
                        self._ensure_status_window()
                        self._show_status_window()

                except Empty:
                    if self.headless:
                        time.sleep(0.001)
                    elif self.live_view_enabled:
                        self._ensure_live_window()
                    else:
                        self._ensure_status_window()
                        self._show_status_window()

                # 3. Handle UI Inputs
                if not self.headless:
                    key_raw = cv2.waitKey(1)
                    key = -1 if key_raw == -1 else (key_raw & 0xFF)
                    if not self.interaction_handler.handle_keyboard_input(key):
                        break

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        finally:
            self.threaded_engine.stop()
            self._finalize_current_segment()
            self.cleanup()

        return self._get_final_results()

    def _check_pipeline_health(self) -> None:
        if not self.cap.running:
            raise RuntimeError(
                f"Camera stream disconnected: {self.cap.last_error or 'read stopped'}"
            )

        camera_idle = self.cap.seconds_since_last_frame()
        if camera_idle >= self.camera_stall_timeout:
            raise RuntimeError(
                f"Camera produced no new frame for {camera_idle:.1f} seconds"
            )

        fatal_error = self.threaded_engine.fatal_error
        if fatal_error is not None:
            raise RuntimeError(f"Detection worker failed: {fatal_error}") from fatal_error

        inference_started_at = self.threaded_engine.inference_started_at
        if inference_started_at is not None:
            inference_age = time.monotonic() - inference_started_at
            if inference_age >= self.inference_stall_timeout:
                raise RuntimeError(
                    f"Inference did not complete for {inference_age:.1f} seconds"
                )

        if not self.threaded_engine.thread.is_alive() and self.threaded_engine.running:
            raise RuntimeError("Detection worker stopped unexpectedly")

        video_writer = getattr(self, "video_writer", None)
        if video_writer is not None:
            if video_writer.last_error is not None:
                raise RuntimeError(
                    f"Video writer failed: {video_writer.last_error}"
                ) from video_writer.last_error
            write_started_at = video_writer.write_started_at
            if write_started_at is not None:
                write_age = time.monotonic() - write_started_at
                if write_age >= self.video_writer_stall_timeout:
                    raise RuntimeError(
                        f"Video writer blocked for {write_age:.1f} seconds"
                    )

    def _log_pipeline_performance(self) -> None:
        now = time.monotonic()
        elapsed = now - self._performance_last_at
        if elapsed < self.performance_log_interval:
            return

        capture_count = self.cap.frames_read
        inference_count = self.threaded_engine.frames_processed
        capture_fps = (
            capture_count - self._performance_last_capture_count
        ) / elapsed
        inference_fps = (
            inference_count - self._performance_last_inference_count
        ) / elapsed
        new_input_drops = self._input_frames_dropped - self._performance_last_input_drops
        writer_drops = (
            self.video_writer.dropped_frames if self.video_writer is not None else 0
        )
        writer_depth = (
            self.video_writer.frame_queue.qsize() if self.video_writer is not None else 0
        )
        rss_mb = self._performance_process.memory_info().rss / (1024 * 1024)
        self.logger.info(
            "Performance: capture=%.1f FPS, inference=%.1f FPS, analytics=%.1f FPS, "
            "stale_input_drops=%d, writer_queue=%d/%d, writer_drops=%d, rss=%.0f MB",
            capture_fps,
            inference_fps,
            self.stats.fps,
            new_input_drops,
            writer_depth,
            self.video_writer_queue_size,
            writer_drops,
            rss_mb,
        )
        self._performance_last_at = now
        self._performance_last_capture_count = capture_count
        self._performance_last_inference_count = inference_count
        self._performance_last_input_drops = self._input_frames_dropped

    def _ensure_live_window(self, w: Optional[int] = None, h: Optional[int] = None) -> None:
        if self._status_window_created:
            try:
                cv2.destroyWindow(self.status_window_name)
            except cv2.error:
                pass
            self._status_window_created = False

        if not self._live_window_created:
            cv2.namedWindow(self.live_window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.live_window_name, self.interaction_handler.mouse_callback)
            if w and h:
                self.fit_center(self.live_window_name, w, h, frac=0.9)
            self._live_window_created = True

    def _ensure_status_window(self) -> None:
        if self._live_window_created:
            try:
                cv2.destroyWindow(self.live_window_name)
            except cv2.error:
                pass
            self._live_window_created = False

        if not self._status_window_created:
            cv2.namedWindow(self.status_window_name, cv2.WINDOW_NORMAL)
            self.fit_center(self.status_window_name, 460, 180, frac=0.9)
            self._status_window_created = True

    def _show_status_window(self) -> None:
        # Keep the status window cheap: redraw a small static panel at most 4 times/second.
        now = time.time()
        if now - self._last_status_draw_time < 0.25:
            cv2.imshow(self.status_window_name, self._status_frame)
            return

        self._last_status_draw_time = now
        frame = np.zeros_like(self._status_frame)
        lines = [
            "Live camera processing",
            f"View: OFF   Press V to open video",
            f"FPS: {self.stats.fps:.1f}",
            f"Frames: {self.stats.frames_processed}",
            f"Events: {self.stats.total_events}",
            f"Objects: {len(self.counter.object_states)}",
            "ESC: exit   V: toggle view",
        ]

        y = 28
        for i, text in enumerate(lines):
            scale = 0.65 if i == 0 else 0.5
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
            y += 24

        self._status_frame = frame
        cv2.imshow(self.status_window_name, self._status_frame)

    def toggle_live_view(self) -> None:
        self.live_view_enabled = not self.live_view_enabled
        state = "ON" if self.live_view_enabled else "OFF"
        self.logger.info(f"Live video view: {state}")

        if self.live_view_enabled:
            frame = getattr(self, 'last_frame', None)
            if frame is not None:
                h, w = frame.shape[:2]
                self._ensure_live_window(w, h)
        else:
            self.edit_mode = False
            self.dragging = False
            self.drag_target = None
            self.create_mode = "none"
            self.temp_points.clear()
            self._ensure_status_window()

    # --- HELPERS, EXPORTS, & CLEANUP ---
    def _save_video_frame(self, clean_frame: np.ndarray) -> None:
        if not self.config.save_video:
            return
        if self.video_writer is None or not self.video_writer.isOpened():
            self._initialize_video_writer(frame=clean_frame)
            if self.video_writer is None:
                return
        if hasattr(self, '_video_writer_size'):
            h, w = clean_frame.shape[:2]
            if (w, h) != self._video_writer_size:
                clean_frame = cv2.resize(clean_frame, self._video_writer_size)
        self.video_writer.write(clean_frame)

    def _should_rollover_segment(self) -> bool:
        now_dt = datetime.now()
        if self.current_segment_start_dt is None:
            self.current_segment_start_dt = now_dt
            return False
        if self.next_split_dt is None:
            self.next_split_dt = self._compute_next_clock_boundary(now_dt, 60)
        return now_dt >= self.next_split_dt

    def _compute_next_clock_boundary(self, base_dt: datetime, interval_min: int) -> datetime:
        """Round base_dt up to the next clock-aligned boundary `interval_min` minutes
        past the top of the hour. With interval_min=60 at 12:07:30 returns 13:00:00;
        with interval_min=15 at 12:07:30 returns 12:15:00. If base_dt is already exactly
        on a boundary, returns the NEXT boundary (callers compare with >=)."""
        if interval_min <= 0:
            interval_min = 60

        hour_start = base_dt.replace(minute=0, second=0, microsecond=0)
        minutes_into_hour = (
            base_dt.minute
            + base_dt.second / 60.0
            + base_dt.microsecond / 60_000_000.0
        )
        intervals_elapsed = int(minutes_into_hour // interval_min)
        return hour_start + timedelta(minutes=(intervals_elapsed + 1) * interval_min)

    def _rollover_segment(self) -> None:
        current_time = datetime.now()
        self.counter.update_events_with_final_stats(current_time)
        window_start = self.current_segment_start_dt
        window_end = self.next_split_dt if self.next_split_dt else datetime.now()

        self.logger.info(f"Hourly segment complete: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")
        self._export_hourly_segment(window_start, window_end)

        self.current_segment += 1
        self.counter.reset_all_counts()
        self.current_segment_start_dt = window_end
        self.next_split_dt = self._compute_next_clock_boundary(window_end, 60)

        if self.config.save_video:
            self._rotate_hourly_video(window_start, window_end)
            self._cleanup_expired_footage()

    def _cleanup_expired_footage(self) -> None:
        if not self.config.save_video or self.footage_retention_days <= 0:
            return
        result = cleanup_live_footage(
            self.config.output_folder,
            self.footage_retention_days,
            logger=self.logger,
        )
        if result.deleted_files or result.errors:
            self.logger.info(
                "Live footage cleanup: deleted=%d, freed=%.1f MB, errors=%d",
                result.deleted_files,
                result.freed_bytes / (1024 * 1024),
                result.errors,
            )

    def _export_hourly_segment(self, window_start: datetime, window_end: datetime):
        try:
            self._trigger_live_export(force=True)
            counts_dict = self.counter.get_current_counts()
            events_dict = self.counter.get_events_summary(datetime.now())

            events_dict["hour_start"] = window_start.isoformat()
            events_dict["hour_end"] = window_end.isoformat()
            events_dict["hour_of_day"] = window_start.hour
            events_dict["segment_number"] = self.current_segment + 1

            enhanced_stats = self.stats.__dict__.copy() if hasattr(self.stats, '__dict__') else {}
            enhanced_stats['video_time_based'] = False
            segment_id = f"hour_{window_start.hour:02d}"

            original_master_setting = self.exporter.config.enable_master_log
            self.exporter.config.enable_master_log = False
            try:
                self.exporter.export_segment_results(
                    segment_id=segment_id,
                    counts=counts_dict,
                    events=events_dict,
                    stats=enhanced_stats,
                    video_source=self.source_name
                )
            finally:
                self.exporter.config.enable_master_log = original_master_setting
        except Exception as e:
            self.logger.error(f"Failed to export hourly data: {e}")

    def _rotate_hourly_video(self, window_start: datetime, window_end: datetime):
        try:
            if self.video_writer and self.video_writer.isOpened():
                if not self.video_writer.release():
                    raise RuntimeError("Previous video writer could not be stopped")
                self.video_writer = None
            if self.config.save_video:
                date_str = window_end.strftime("%Y%m%d")
                hour_str = window_end.strftime("%H")
                filename = f"live_{date_str}_{hour_str}00.mp4"
                self._initialize_video_writer(filename)
        except Exception as e:
            self.logger.error(f"Failed to rotate hourly video: {e}")
            raise

    def _finalize_current_segment(self) -> None:
        try:
            current_time = datetime.now()
            self.counter.update_events_with_final_stats(current_time)
            counts_dict = self.counter.get_current_counts()
            events_dict = self.counter.get_events_summary(current_time)

            window_start = self.current_segment_start_dt
            window_end = datetime.now()
            events_dict["_window_start"] = window_start.isoformat()
            events_dict["_window_end"] = window_end.isoformat()

            enhanced_stats = self.stats.__dict__.copy() if hasattr(self.stats, '__dict__') else {}

            # Serialize object states
            serialized_states = {}
            for track_id, obj_state in self.counter.object_states.items():
                if hasattr(obj_state, '__dict__'):
                    state_dict = {
                        'track_id': obj_state.track_id,
                        'class_id': obj_state.class_id,
                        'class_name': obj_state.class_name,
                        'current_speed_pxps': obj_state.current_speed_pxps,
                        'avg_speed_pxps': obj_state.avg_speed_pxps,
                        'zone_presence': dict(obj_state.zone_presence),
                        'zone_entry_times': dict(obj_state.zone_entry_times),
                        'zone_dwell_times': dict(obj_state.zone_dwell_times),
                        'last_seen': obj_state.last_seen
                    }
                    serialized_states[track_id] = state_dict

            enhanced_stats['object_states'] = serialized_states
            self._trigger_live_export(force=True)

            original_master_setting = self.exporter.config.enable_master_log
            self.exporter.config.enable_master_log = False
            try:
                self.exporter.export_segment_results(
                    segment_id=self.current_segment,
                    counts=counts_dict,
                    events=events_dict,
                    stats=enhanced_stats,
                    video_source=self.source_name,
                )
            finally:
                self.exporter.config.enable_master_log = original_master_setting

            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
        except Exception as e:
            self.logger.warning(f"Finalize current segment failed: {e}")

    def _update_stats(self, new_events_count: int):
        self.stats.frames_processed += 1
        self.stats.total_events += new_events_count
        self.stats.processing_time = time.time() - self.stats.start_time
        self.stats.update_fps()

    def _get_final_results(self) -> Dict:
        return {
            "stats": self.stats,
            "final_counts": self.counter.get_current_counts(),
            "events_summary": self.counter.get_events_summary(),
            "segments_processed": self.current_segment + 1
        }

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_requested = True
        self.is_running = False

    def pause(self):
        """Pause processing"""
        self.is_paused = True

    def resume(self):
        """Resume processing"""
        self.is_paused = False

    def stop(self):
        """Stop processing"""
        self.stop_requested = True
        self.is_running = False

    def _show_notification(self, text: str, duration: float = 2.0):
        """Show on-screen notification (store for drawing)"""
        self.notification = {
            'text': text,
            'start_time': time.time(),
            'duration': duration
        }

    def cleanup(self):
        try:
            if self._config_dirty:
                self._persist_runtime_config("final cleanup")
            if hasattr(self, 'exporter') and self.exporter is not None:
                self.exporter.shutdown(finalize_shared=True)
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self.logger.info("Camera runner cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    def _persist_runtime_config(self, reason: str) -> bool:
        """Persist completed live geometry edits without exposing a partial file."""
        self._config_dirty = True
        try:
            with self.config_persist_lock:
                saved_path = save_app_config(self.config, self.runtime_config_path)
            self.config.runtime_config_path = str(saved_path)
            self.runtime_config_path = saved_path
            self._config_dirty = False
            self.logger.info("Runtime config saved after %s: %s", reason, saved_path)
            self._show_notification("Configuration saved", duration=1.5)
            return True
        except Exception as exc:
            self.logger.error("Could not save runtime config after %s: %s", reason, exc)
            self._show_notification("Configuration save failed", duration=3.0)
            return False

    def fit_center(self, name, w, h, frac=0.9):
        r = tk.Tk()
        r.withdraw()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        if sw > 3000 or sh > 2000:
            sw, sh = 1920, 1080
        if self.window_count > 1:
            columns = math.ceil(math.sqrt(self.window_count))
            rows = math.ceil(self.window_count / columns)
            cell_width = sw / columns
            cell_height = sh / rows
            slot = self.window_index % self.window_count
            column = slot % columns
            row = slot // columns
            s = min(
                (cell_width * frac) / w,
                (cell_height * frac) / h,
                1.0,
            )
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            x = int(column * cell_width + (cell_width - nw) / 2)
            y = int(row * cell_height + (cell_height - nh) / 2)
        else:
            s = min((sw * frac) / w, (sh * frac) / h, 1.0)
            nw, nh = int(w * s), int(h * s)
            x, y = max(0, (sw - nw) // 2), max(0, (sh - nh) // 2)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, nw, nh)
        cv2.moveWindow(name, x, y)
        r.destroy()

    def _background_append_task(self, new_events: List, segment_id: str, video_name: str):
        if not new_events: return
        if self.export_lock.acquire(False):
            try:
                self.exporter._append_to_master_log(new_events, video_source=video_name, segment_id=segment_id)
            except Exception as e:
                self.logger.error(f"Live master log update failed: {e}")
            finally:
                self.export_lock.release()

    def _trigger_live_export(self, force: bool = False):
        now = time.time()
        if force or (now - self.last_live_export_time >= self.live_export_interval):
            current_time = datetime.now()
            self.counter.update_events_with_final_stats(current_time)
            summary = self.counter.get_events_summary(current_time)
            all_events = summary['events']
            current_total = len(all_events)

            segment_id = f"hour_{self.current_segment_start_dt.hour:02d}"
            video_name = self.source_name

            if current_total > self.last_exported_event_count:
                new_events_slice = all_events[self.last_exported_event_count:]
                for event in new_events_slice:
                    if event.get('zone_name'):
                        key = (event.get('track_id'), event.get('zone_name'))
                        self.active_zone_events[key] = True

                events_to_export = copy.deepcopy(new_events_slice)
                if events_to_export:
                    threading.Thread(target=self._background_append_task,
                                     args=(events_to_export, segment_id, video_name), daemon=True).start()
                self.last_exported_event_count = current_total

            if self.active_zone_events:
                zone_updates = []
                keys_to_remove = []
                for (track_id, zone_name), is_active in list(self.active_zone_events.items()):
                    if not is_active: continue
                    for event in all_events:
                        if event.get('track_id') == track_id and event.get('zone_name') == zone_name:
                            dwell = event.get('dwell_seconds', 0.0)
                            obj_state = self.counter.object_states.get(track_id)
                            still_in_zone = obj_state.zone_presence.get(zone_name, False) if obj_state else False
                            zone_updates.append({'track_id': track_id, 'zone_name': zone_name, 'dwell_seconds': dwell})
                            if not still_in_zone or force:
                                keys_to_remove.append((track_id, zone_name))
                            break

                if zone_updates:
                    def update_task():
                        if self.export_lock.acquire(timeout=5):
                            try:
                                self.exporter.update_zone_dwell_times(zone_updates, video_source=video_name)
                            except Exception as e:
                                self.logger.error(f"Live master log dwell update failed: {e}")
                            finally:
                                self.export_lock.release()

                    threading.Thread(target=update_task, daemon=True).start()

                for key in keys_to_remove:
                    self.active_zone_events.pop(key, None)
            self.last_live_export_time = now
