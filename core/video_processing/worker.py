import os
import cv2
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from config_manager import AppConfig, resolve_colormap
from core.detection_engine import DetectionEngine
from core.tracking import ObjectCounter
from utils.results_export import ResultsExporter, ExportConfig


from .stats import ProcessingStats
from .async_io import AsyncVideoWriter
from ..visualizer_heatmap import HeatmapAccumulator


class VideoWorker:
    """
    Worker for processing a single video. Shares the orchestrator's
    DetectionEngine (thread-safe via internal lock); owns its own counter
    and exporter so per-video state stays isolated.
    """

    def __init__(self, config: 'AppConfig', video_path: Path, worker_id: int,
                 detection_engine: DetectionEngine,
                 queue_size_callback=None,
                 stop_event: Optional[threading.Event] = None):
        self.config = config
        self.video_path = video_path
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"{__name__}.worker_{worker_id}")

        # Callback to get video queue size (for extended stability wait when queue <= 1)
        self.queue_size_callback = queue_size_callback

        # Cooperative cancellation flag set by the orchestrator on shutdown.
        # Polled inside the main frame loop and during growing-file waits.
        self.stop_event = stop_event

        # Shared detection engine owned by the orchestrator. Reused across all
        # videos to avoid leaking TensorRT contexts on each worker construction.
        self.detection_engine = detection_engine

        # Each worker gets its own counter
        frame_size = (config.display_width, config.display_height)
        self.counter = ObjectCounter(
            config.lines_config,
            config.zones_config,
            frame_size,
            exclusion_zones=getattr(config, 'exclusion_zones', []),
            max_track_age=config.max_track_age
        )

        # Apply speed settings
        self.counter.configure_speed(
            enable=bool(getattr(config, "enable_speed", False)),
            units=str(getattr(config, "speed_units", "pxps")),
            meters_per_pixel=float(getattr(config, "meters_per_pixel", 0.0) or 0.0),
            smooth_window=int(getattr(config, "speed_smooth_window", 5) or 5),
            annotate=bool(getattr(config, "annotate_speed", True))
        )

        # Each worker gets its own exporter (with cloud upload settings from AppConfig)
        export_config = ExportConfig(
            enable_api_upload=self.config.enable_api_upload
        )
        self.exporter = ResultsExporter(config.output_folder, export_config)

        # Worker state
        self.cap = None
        self.video_writer = None
        self.stats = ProcessingStats()
        self.frame_skip = config.frame_skip
        self.frame_skip_counter = 0
        self.last_detections = []
        self.interpolate_tracks = config.interpolate_tracks

        # Video timing
        self.video_start_time = None
        self.video_fps = 30.0
        self.video_frame_number = 0
        self.video_current_time = None
        self.is_live_source = False

        # Segment tracking
        self.current_segment = 0
        self.segment_events = []
        self.segment_start_dt = None

        # Live export tracking (update master_log during processing)
        self.live_export_interval = 300.0  # Update Excel every 5 minutes
        self.last_live_export_time = time.time()
        self.last_exported_event_count = 0

        # Track active zone events that need dwell time updates
        # Key: (track_id, zone_name), Value: True if still active (object in zone)
        self.active_zone_events = {}

        # Heatmap (optional)
        self.heatmap_acc = None

    def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep up to `seconds`, returning True early if a stop was requested.
        Lets cancellation interrupt the long waits used by growing-file polling."""
        if self.stop_event is None:
            time.sleep(seconds)
            return False
        return self.stop_event.wait(timeout=seconds)

    def process(self, progress_callback=None) -> Dict:
        """
        Process the video file and return results.

        Args:
            progress_callback: Optional callable(worker_id, frames_done, total_frames, fps, events)
                              for progress updates

        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Worker {self.worker_id} starting: {self.video_path.name}")

        try:
            wait_for_growing = getattr(self.config, 'wait_for_growing_files', True)

            # For growing files: Wait for valid container headers before starting
            # This prevents "EBML header parsing failed" errors on freshly created files
            if wait_for_growing:
                max_header_wait = 60  # Max 60 seconds to wait for valid headers
                header_wait_start = time.time()
                last_size_for_header_check = 0

                while time.time() - header_wait_start < max_header_wait:
                    try:
                        current_size = self.video_path.stat().st_size
                    except:
                        break  # File doesn't exist

                    # Try to open and check if headers are valid
                    test_cap = cv2.VideoCapture(str(self.video_path))
                    if test_cap.isOpened():
                        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = test_cap.get(cv2.CAP_PROP_FPS)
                        # Try to actually read a frame to confirm headers are valid
                        ret, _ = test_cap.read()
                        test_cap.release()

                        if frame_count > 0 and fps > 0 and ret:
                            self.logger.debug \
                                (f"Worker {self.worker_id}: Container valid ({frame_count} frames, {fps} fps)")
                            break
                        else:
                            # Headers not valid yet
                            if current_size > last_size_for_header_check:
                                self.logger.info \
                                    (f"Worker {self.worker_id}: Waiting for valid container headers (size: {current_size}, frames: {frame_count})...")
                                last_size_for_header_check = current_size
                            time.sleep(2.0)
                    else:
                        test_cap.release()
                        if current_size > last_size_for_header_check:
                            self.logger.info \
                                (f"Worker {self.worker_id}: Waiting for file to be readable (size: {current_size})...")
                            last_size_for_header_check = current_size
                        time.sleep(2.0)
                else:
                    self.logger.warning(f"Worker {self.worker_id}: Timed out waiting for valid container headers")

            # Initialize video capture
            if not self._initialize_video():
                return {"error": f"Failed to open video: {self.video_path}", "video_path": str(self.video_path)}

            # Apply exclusion zones
            if hasattr(self.config, 'exclusion_zones') and self.config.exclusion_zones:
                ret, first_frame = self.cap.read()
                if ret:
                    self.detection_engine.set_exclusion_zones(
                        self.config.exclusion_zones,
                        first_frame.shape
                    )
                    # Reset to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Update counter frame size
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.counter.set_frame_size((width, height))

            # Initialize video writer if needed
            if self.config.save_video:
                self._initialize_video_writer()

            # Initialize heatmap if enabled
            if getattr(self.config, "enable_heatmap", False):
                interval_sec = 3600.0  # 1 hour segments
                alpha = float(getattr(self.config, "heatmap_alpha", 0.35))
                cmap_name = getattr(self.config, "heatmap_colormap", "HOT")
                out_dir = str(Path(self.config.output_folder) / "heatmaps")
                radius_px = int(getattr(self.config, "heatmap_radius_px", 10))
                decay = float(getattr(self.config, "heatmap_decay", 0.0))
                gamma = float(getattr(self.config, "heatmap_gamma", 1.6))

                # Create output directory
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                self.heatmap_acc = HeatmapAccumulator(
                    frame_size=(height, width),
                    alpha=alpha,
                    colormap=resolve_colormap(cmap_name),
                    out_dir=out_dir,
                    interval_sec=interval_sec,
                    radius_px=radius_px,
                    decay=decay,
                    gamma=gamma
                )
                self.logger.info(f"[HEATMAP] Worker {self.worker_id} initialized ({width}x{height})")

            # Get total frames (may increase for growing files)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize stats
            self.stats.start_time = time.time()
            self._reset_segment()

            # FPS calculation variables
            fps_update_interval = 30  # Update FPS every N frames
            fps_frame_times = deque(maxlen=fps_update_interval)  # Use deque for O(1) operations
            last_fps_time = time.time()

            # File stability tracking for growing files
            # Use 10s when other videos in queue, 30s when queue is empty (no more work)
            stability_required_seconds_default = 10.0
            stability_required_seconds_extended = 30.0
            last_file_size = self.video_path.stat().st_size
            file_stable_since = time.time()
            # wait_for_growing already defined at start of process()

            # Buffer to stay behind live edge (avoid reading incomplete data)
            # This prevents "File ended prematurely" FFmpeg errors
            live_edge_buffer_frames = int(self.video_fps * 3)  # Stay 3 seconds behind live edge
            consecutive_read_failures = 0

            # Track waiting state for progress display
            waiting_until_recheck = 0.0  # Seconds until next recheck (0 = not waiting)

            # Process frames
            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    self.logger.info(f"Worker {self.worker_id}: stop requested, exiting frame loop")
                    break

                frame_start = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    consecutive_read_failures += 1

                    # Reached end of currently available frames
                    # Check if file is still growing (being recorded)
                    if wait_for_growing:
                        try:
                            current_size = self.video_path.stat().st_size
                        except:
                            break  # File disappeared, exit

                        if current_size > last_file_size:
                            # File is still growing - reset stable timer, reopen and continue
                            self.logger.info \
                                (f"Worker {self.worker_id}: File growing ({last_file_size} -> {current_size}), continuing...")
                            last_file_size = current_size
                            file_stable_since = time.time()

                            # Wait proportionally longer based on consecutive failures
                            # This gives time for more complete data to be written
                            wait_time = min(2.0 + (consecutive_read_failures * 1.0), 5.0)
                            self.logger.debug \
                                (f"Worker {self.worker_id}: Waiting {wait_time:.1f}s for more data to be written...")
                            if self._sleep_or_stop(wait_time):
                                break

                            # Reopen video at current position to read new frames
                            current_pos = self.video_frame_number
                            self.cap.release()
                            self.cap = cv2.VideoCapture(str(self.video_path))
                            if self.cap.isOpened():
                                # Get new frame count
                                new_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                                # Stay behind live edge to avoid reading incomplete container data
                                safe_max_frame = max(0, new_total_frames - live_edge_buffer_frames)

                                # Always seek to current position first
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                                total_frames = new_total_frames

                                # Only continue if there are new frames to process
                                if safe_max_frame > current_pos:
                                    continue  # Try to read more frames
                                else:
                                    # Not enough new frames yet, close and wait more
                                    self.logger.debug \
                                        (f"Worker {self.worker_id}: Caught up to live edge, waiting for buffer ({current_pos}/{safe_max_frame})...")
                                    self.cap.release()
                                    if self._sleep_or_stop(2.0):
                                        break
                                    # Reopen for next iteration
                                    self.cap = cv2.VideoCapture(str(self.video_path))
                                    if self.cap.isOpened():
                                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                                    continue
                            else:
                                break  # Failed to reopen
                        else:
                            # File size unchanged - check if stable long enough
                            # Use extended stability time (30s) if queue has 0 or 1 videos
                            queue_size = self.queue_size_callback() if self.queue_size_callback else 0
                            stability_required_seconds = (
                                stability_required_seconds_extended if queue_size <= 1
                                else stability_required_seconds_default
                            )

                            stable_duration = time.time() - file_stable_since
                            remaining_wait = stability_required_seconds - stable_duration

                            if stable_duration >= stability_required_seconds:
                                # File has been stable for required time, truly finished
                                self.logger.info \
                                    (f"Worker {self.worker_id}: File stable for {stable_duration:.1f}s, finishing")
                                break
                            else:
                                # Wait a bit and try again
                                self.logger.debug \
                                    (f"Worker {self.worker_id}: Caught up, waiting for more content ({stable_duration:.1f}s/{stability_required_seconds}s)...")

                                # Continue live exports while waiting
                                self._trigger_live_export(force=False)

                                # Calculate wait time until next recheck
                                wait_time = min(2.0 + (consecutive_read_failures * 0.5), 5.0)
                                waiting_until_recheck = wait_time

                                # Update progress to show we're caught up with waiting info
                                if progress_callback:
                                    progress_callback(
                                        self.worker_id,
                                        self.stats.frames_processed,
                                        total_frames,  # Show we're at 100% of current content
                                        0,  # 0 FPS since waiting
                                        self.stats.total_events,
                                        waiting_until_recheck,  # Seconds until next recheck
                                        remaining_wait  # Seconds until stability timeout
                                    )

                                if self._sleep_or_stop(wait_time):
                                    break
                                waiting_until_recheck = 0.0  # Reset after sleep

                                # Reopen to check for new frames
                                current_pos = self.video_frame_number
                                self.cap.release()
                                self.cap = cv2.VideoCapture(str(self.video_path))
                                if self.cap.isOpened():
                                    new_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                                    # Stay behind live edge
                                    safe_max_frame = max(0, new_total_frames - live_edge_buffer_frames)

                                    # Always seek to maintain position
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                                    total_frames = new_total_frames

                                    if safe_max_frame <= current_pos:
                                        # Still too close to live edge, wait more
                                        self.cap.release()
                                        if self._sleep_or_stop(1.0):
                                            break
                                        self.cap = cv2.VideoCapture(str(self.video_path))
                                        if self.cap.isOpened():
                                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                                    continue
                                else:
                                    break
                    else:
                        break  # Not waiting for growing files, exit normally

                # Successful read - reset failure counter
                consecutive_read_failures = 0

                self.video_frame_number += 1
                self.video_current_time = self._get_current_timestamp()

                # Process frame with skip logic
                events = self._process_frame(frame)

                # Update stats
                self.stats.frames_processed += 1
                self.stats.total_events += len(events)

                # Track frame time for FPS calculation (deque auto-maintains maxlen)
                frame_time = time.time() - frame_start
                fps_frame_times.append(frame_time)

                # Progress callback with FPS
                if progress_callback and self.stats.frames_processed % fps_update_interval == 0:
                    # Calculate current FPS from recent frames
                    if fps_frame_times:
                        avg_frame_time = sum(fps_frame_times) / len(fps_frame_times)
                        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    else:
                        current_fps = 0

                    progress_callback(
                        self.worker_id,
                        self.stats.frames_processed,
                        total_frames,
                        current_fps,
                        self.stats.total_events,
                        0.0,  # Not waiting - no recheck countdown
                        0.0   # Not waiting - no stability countdown
                    )

                # Trigger live export to master_log (checks interval internally)
                self._trigger_live_export(force=False)

            # Check for 0-frame edge case - file may not have been ready.
            # Skip this retry path if a stop was requested.
            stopped = self.stop_event is not None and self.stop_event.is_set()
            if self.stats.frames_processed == 0 and wait_for_growing and not stopped:
                try:
                    file_size = self.video_path.stat().st_size
                    if file_size > 1024:  # File has content (> 1KB)
                        self.logger.warning \
                            (f"Worker {self.worker_id}: Processed 0 frames from {file_size} byte file - retrying after delay...")
                        self._sleep_or_stop(5.0)  # interruptible; falls through to finalize regardless

                        # Retry by reopening
                        self._cleanup()
                        if self._initialize_video():
                            self.stats.frames_processed = 0
                            self.stats.start_time = time.time()
                            self._reset_segment()

                            # Re-run the main loop (recursive call with retry limit)
                            # For simplicity, just log the issue - the file will be reprocessed
                            # if it's still in the folder on next scan
                            self.logger.warning \
                                (f"Worker {self.worker_id}: Zero frames processed - file may need reprocessing")
                except:
                    pass  # File might have been deleted

            # Finalize
            self._finalize_segment()
            self.stats.processing_time = time.time() - self.stats.start_time
            self.stats.update_fps()

            # Prepare results
            results = {
                "video_path": str(self.video_path),
                "stats": self.stats,
                "final_counts": self.counter.get_current_counts(),
                "events_summary": self.counter.get_events_summary(),
                "worker_id": self.worker_id
            }

            # Export results
            try:
                video_name = self.video_path.stem
                self.exporter.export_video_summary(results, video_name)
            except Exception as e:
                self.logger.error(f"Export failed: {e}")

            self.logger.info(f"Worker {self.worker_id} completed: {self.video_path.name} "
                             f"({self.stats.frames_processed} frames, {self.stats.fps:.1f} FPS)")

            return results

        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} error: {e}")
            return {"error": str(e), "video_path": str(self.video_path)}

        finally:
            self._cleanup()

    def _initialize_video(self) -> bool:
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                return False

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_fps = fps if fps > 0 else 30.0

            # Get file timestamp
            if hasattr(os.stat(self.video_path), 'st_birthtime'):
                file_timestamp = os.stat(self.video_path).st_birthtime
            else:
                file_timestamp = os.path.getmtime(self.video_path)

            self.video_start_time = datetime.fromtimestamp(file_timestamp)
            self.video_current_time = self.video_start_time
            self.video_frame_number = 0

            return True
        except Exception as e:
            self.logger.error(f"Video init failed: {e}")
            return False

    def _initialize_video_writer(self):
        """Initialize video writer for output"""
        if not bool(getattr(self.config, "save_video", False)):
            self.video_writer = None
            return
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            base_fps = self.video_fps if self.video_fps > 0 else 30.0

            # Playback speed multiplier from config (default 1.0 = real-time).
            # Values >1.0 produce sped-up output for review; timestamp overlays will
            # advance at real-time but the video plays faster, so they desync.
            playback_speed = float(getattr(self.config, "playback_speed_multiplier", 1.0) or 1.0)
            if playback_speed <= 0:
                playback_speed = 1.0
            export_fps = base_fps * playback_speed

            output_path = Path(self.config.output_folder) / f"{self.video_path.stem}_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(str(output_path), fourcc, export_fps, (width, height))
            self.video_writer = AsyncVideoWriter(vw)  # <-- Wrap it here!
        except Exception as e:
            self.logger.warning(f"Video writer init failed: {e}")

    def _get_current_timestamp(self) -> datetime:
        """Get current video timestamp"""
        if self.video_start_time and self.video_fps > 0:
            offset = self.video_frame_number / self.video_fps
            return self.video_start_time + timedelta(seconds=offset)
        return datetime.now()

    def _process_frame(self, frame: np.ndarray) -> List:
        """Process a single frame"""
        self.frame_skip_counter += 1
        self.last_frame = frame  # Store for heatmap

        if (self.frame_skip_counter % self.frame_skip) == 0:
            # Full detection
            detections = self.detection_engine.detect_and_track(frame)
            self.last_detections = detections
            self.stats.total_detections += len(detections)

            # Update counts
            events = self.counter.update_counts(detections, timestamp=self.video_current_time)

            # Queue each newly-created event exactly once. The shared API batcher
            # assigns it to the next wall-clock five-minute upload while this
            # worker continues processing frames.
            if events:
                self.exporter.queue_api_events(
                    events,
                    video_source=self.video_path.name,
                    segment_id=f"segment_{self.current_segment}"
                )

            # Update heatmap if enabled
            if self.heatmap_acc is not None and detections:
                boxes_xyxy = [det.bbox for det in detections]
                self.heatmap_acc.update_from_boxes(boxes_xyxy, weight=1.0)

            # Write video if enabled
            if self.video_writer:
                # Draw visualizations
                vis_frame = self._draw_basic_visualization(frame, detections)
                self.video_writer.write(vis_frame)

            # Store events for segment
            self.segment_events.extend(events)

            return events
        else:
            # Skip frame, optionally interpolate
            if self.video_writer and self.last_detections:
                vis_frame = self._draw_basic_visualization(frame, self.last_detections)
                self.video_writer.write(vis_frame)
            return []

    def _draw_basic_visualization(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """Draw basic bounding boxes and counts on frame"""
        vis_frame = frame.copy()

        # 1. Draw the bounding boxes for the objects
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} #{det.track_id}"
            cv2.putText(vis_frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Let the ObjectCounter draw all Lines, Zones, Text, and Arrows!
        vis_frame = self.counter.draw_overlays(vis_frame, show_counts=True)

        return vis_frame

    def _reset_segment(self):
        """Reset segment tracking"""
        self.segment_events = []
        self.segment_start_dt = self.video_current_time or datetime.now()

    def _finalize_segment(self):
        """Finalize and export current segment"""
        # Force final live export before segment finalize
        # This appends any remaining events to master_log with proper tracking
        self._trigger_live_export(force=True)

        if self.segment_events:
            try:
                segment_end_dt = self.video_current_time or datetime.now()

                # Disable master_log export here since _trigger_live_export already handled it
                # (prevents duplicate entries in master_log)
                original_master_setting = self.exporter.config.enable_master_log
                self.exporter.config.enable_master_log = False
                try:
                    self.exporter.export_segment(
                        self.segment_events,
                        self.current_segment,
                        self.segment_start_dt,
                        segment_end_dt,
                        source_name=self.video_path.stem
                    )
                finally:
                    self.exporter.config.enable_master_log = original_master_setting
            except Exception as e:
                self.logger.warning(f"Segment export failed: {e}")

    def _trigger_live_export(self, force: bool = False):
        """
        Check if we should export new events to master_log.
        Called periodically during processing to keep master_log updated in real-time.

        For zone events:
        - New zone events are appended immediately
        - Dwell times are updated periodically while object is in zone
        - Final dwell time is set when object exits or at segment/video end
        """
        now = time.time()

        # Only run if interval passed OR forced (e.g. at end of segment/video)
        if force or (now - self.last_live_export_time >= self.live_export_interval):
            # Update events with current video time before getting summary
            self.counter.update_events_with_final_stats(self.video_current_time)

            # Get all events currently in memory (pass video time for accurate dwell calculation)
            summary = self.counter.get_events_summary(self.video_current_time)
            all_events = summary.get('events', [])
            current_total = len(all_events)

            # Get segment info
            segment_id = f"segment_{self.current_segment}"
            video_name = self.video_path.name if self.video_path else "unknown"

            # STEP 1: Append NEW events (both line and zone)
            if current_total > self.last_exported_event_count:
                new_events_slice = all_events[self.last_exported_event_count:]

                # Track new zone events for future dwell updates
                for event in new_events_slice:
                    if event.get('zone_name'):
                        key = (event.get('track_id'), event.get('zone_name'))
                        self.active_zone_events[key] = True

                # Append all new events to master log
                try:
                    self.exporter._append_to_master_log(
                        new_events_slice,
                        video_source=video_name,
                        segment_id=segment_id
                    )
                except Exception as e:
                    self.logger.warning(f"Live master log append failed: {e}")

                self.last_exported_event_count = current_total

            # STEP 2: Update dwell times for active zone events
            if self.active_zone_events:
                zone_updates = []
                keys_to_remove = []

                for (track_id, zone_name), is_active in self.active_zone_events.items():
                    if not is_active:
                        continue

                    # Find the event in all_events to get current dwell
                    for event in all_events:
                        if event.get('track_id') == track_id and event.get('zone_name') == zone_name:
                            dwell = event.get('dwell_seconds', 0.0)

                            # Check if object is still in zone
                            obj_state = self.counter.object_states.get(track_id)
                            still_in_zone = False
                            if obj_state:
                                still_in_zone = obj_state.zone_presence.get(zone_name, False)

                            zone_updates.append({
                                'track_id': track_id,
                                'zone_name': zone_name,
                                'dwell_seconds': dwell
                            })

                            # If object left zone or force=True (segment end), mark as finalized
                            if not still_in_zone or force:
                                keys_to_remove.append((track_id, zone_name))
                            break

                # Update dwell times in master log
                if zone_updates:
                    try:
                        self.exporter.update_zone_dwell_times(zone_updates, video_source=video_name)
                    except Exception as e:
                        self.logger.warning(f"Live master log dwell update failed: {e}")

                # Remove finalized zone events from tracking
                for key in keys_to_remove:
                    self.active_zone_events.pop(key, None)

            self.last_live_export_time = now

    def _cleanup(self):
        """Release resources"""
        # --- NEW: Safely flush all data to the hard drive before closing! ---
        if hasattr(self, 'exporter') and self.exporter is not None:
            self.exporter.shutdown(finalize_shared=False)

        # Save final heatmap if enabled - one heatmap per video with video timestamps only
        if self.heatmap_acc is not None:
            try:
                last_frame = getattr(self, 'last_frame', None)
                if last_frame is not None:
                    # Use video timestamps only (no fallback to system time)
                    if self.video_start_time and self.video_current_time:
                        start_time = self.video_start_time
                        end_time = self.video_current_time

                        # Set the start time for filename generation
                        self.heatmap_acc.last_emit_t = start_time.timestamp()

                        out_path = self.heatmap_acc.render_and_save(
                            frame_bgr=last_frame,
                            label=self.video_path.stem,
                            when=end_time.timestamp(),
                            suffix="final"
                        )
                        self.logger.info(f"[HEATMAP] Final heatmap saved: {out_path}")
                    else:
                        self.logger.warning("[HEATMAP] Skipped - video timestamps not available")
            except Exception as e:
                self.logger.warning(f"[HEATMAP] Failed to save final heatmap: {e}")

        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
