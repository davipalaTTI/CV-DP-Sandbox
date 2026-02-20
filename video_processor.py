"""
Video Processing Module

Handles the main video processing pipeline including:
- Camera and video file input management
- Frame-by-frame processing with detection and counting
- Segment-based recording and statistics export
- Real-time visualization and user controls
- Performance monitoring and optimization
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from folder_monitor import FolderMonitor
from queue import Queue as ThreadQueue
import cv2
from datetime import datetime, timedelta
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import signal
import sys
from config_manager import AppConfig
from detection_engine import DetectionEngine, Detection
from counter_logic import ObjectCounter, CountingEvent
from results_export import ResultsExporter, ExportConfig
from gui_setup import LinePropertiesDialog, ZonePropertiesDialog
from config_manager import CountingLine, CountingZone
import tkinter as tk
from tkinter import simpledialog
from training_mode import TrainingModeCapture, TrainingConfig
from progress_window import ProgressWindow

import os

# --- Heatmap imports ---
from utils import HeatmapAccumulator  # add if you added the class in utils.py
from config_manager import resolve_colormap  # add if you added the helper in config_manager.py


@dataclass
class ProcessingStats:
    """Processing performance statistics"""
    frames_processed: int = 0
    total_detections: int = 0
    total_events: int = 0
    start_time: float = 0
    processing_time: float = 0
    fps: float = 0
    avg_detection_time: float = 0
    avg_processing_time: float = 0

    def update_fps(self):
        """Update FPS calculation"""
        if self.processing_time > 0:
            self.fps = self.frames_processed / self.processing_time


class VideoWorker:
    """
    Isolated worker for processing a single video.
    Each worker has its own detection engine, counter, and exporter instance
    to avoid shared state issues during concurrent processing.
    """

    def __init__(self, config: 'AppConfig', video_path: Path, worker_id: int, 
                 queue_size_callback=None):
        self.config = config
        self.video_path = video_path
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
        
        # Callback to get video queue size (for extended stability wait when queue <= 1)
        self.queue_size_callback = queue_size_callback

        # Each worker gets its own detection engine
        self.detection_engine = DetectionEngine(
            config.model_path,
            config.confidence_threshold,
            allowed_classes=(config.allowed_classes or None),
            device=config.device
        )

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
            enable_api_upload=self.config.enable_api_upload,
            cloud_db_name=getattr(config, 'cloud_db_name', '') or '',
            cloud_table_name=getattr(config, 'cloud_table_name', '') or '',
            cloud_db_config_path=getattr(config, 'cloud_db_config_path', '') or ''
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
                            self.logger.debug(f"Worker {self.worker_id}: Container valid ({frame_count} frames, {fps} fps)")
                            break
                        else:
                            # Headers not valid yet
                            if current_size > last_size_for_header_check:
                                self.logger.info(f"Worker {self.worker_id}: Waiting for valid container headers (size: {current_size}, frames: {frame_count})...")
                                last_size_for_header_check = current_size
                            time.sleep(2.0)
                    else:
                        test_cap.release()
                        if current_size > last_size_for_header_check:
                            self.logger.info(f"Worker {self.worker_id}: Waiting for file to be readable (size: {current_size})...")
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
                            self.logger.info(f"Worker {self.worker_id}: File growing ({last_file_size} -> {current_size}), continuing...")
                            last_file_size = current_size
                            file_stable_since = time.time()
                            
                            # Wait proportionally longer based on consecutive failures
                            # This gives time for more complete data to be written
                            wait_time = min(2.0 + (consecutive_read_failures * 1.0), 5.0)
                            self.logger.debug(f"Worker {self.worker_id}: Waiting {wait_time:.1f}s for more data to be written...")
                            time.sleep(wait_time)
                            
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
                                    self.logger.debug(f"Worker {self.worker_id}: Caught up to live edge, waiting for buffer ({current_pos}/{safe_max_frame})...")
                                    self.cap.release()
                                    time.sleep(2.0)
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
                                self.logger.info(f"Worker {self.worker_id}: File stable for {stable_duration:.1f}s, finishing")
                                break
                            else:
                                # Wait a bit and try again
                                self.logger.debug(f"Worker {self.worker_id}: Caught up, waiting for more content ({stable_duration:.1f}s/{stability_required_seconds}s)...")
                                
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
                                
                                time.sleep(wait_time)
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
                                        time.sleep(1.0)
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

            # Check for 0-frame edge case - file may not have been ready
            if self.stats.frames_processed == 0 and wait_for_growing:
                try:
                    file_size = self.video_path.stat().st_size
                    if file_size > 1024:  # File has content (> 1KB)
                        self.logger.warning(f"Worker {self.worker_id}: Processed 0 frames from {file_size} byte file - retrying after delay...")
                        time.sleep(5.0)  # Wait for more data
                        
                        # Retry by reopening
                        self._cleanup()
                        if self._initialize_video():
                            self.stats.frames_processed = 0
                            self.stats.start_time = time.time()
                            self._reset_segment()
                            
                            # Re-run the main loop (recursive call with retry limit)
                            # For simplicity, just log the issue - the file will be reprocessed
                            # if it's still in the folder on next scan
                            self.logger.warning(f"Worker {self.worker_id}: Zero frames processed - file may need reprocessing")
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
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_fps

            output_path = Path(self.config.output_folder) / f"{self.video_path.stem}_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
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

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} #{det.track_id}"
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw lines
        for name, line_counter in self.counter.line_counters.items():
            cv2.line(vis_frame, line_counter.start_px, line_counter.end_px, (255, 0, 0), 2)

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


class VideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, config: AppConfig, detection_engine: DetectionEngine):
        self.config = config
        self.detection_engine = detection_engine

        self.logger = logging.getLogger(__name__)

        # Live edit state (for moving lines/zones)
        self.edit_mode = False
        self.dragging = False
        self.drag_target = None
        self.drag_threshold = 12

        # NEW: inline creation state
        self.create_mode = "none"  # "none" | "line" | "zone"
        self.temp_points = []  # collecting clicks while drawing
        self.tk_root = None  # for dialogs on top

        # Cache frequently accessed config values to avoid repeated getattr calls
        self._enable_speed = bool(getattr(config, "enable_speed", False))
        self._speed_units = str(getattr(config, "speed_units", "pxps"))
        self._annotate_speed = bool(getattr(config, "annotate_speed", True))
        self._enable_heatmap = bool(getattr(config, "enable_heatmap", False))

        # Initialize counter with exclusion zones
        frame_size = (config.display_width, config.display_height)
        self.counter = self._create_configured_counter((config.display_width, config.display_height))

        # NOTE: configure_speed is already called inside _create_configured_counter

        # Frame skipping
        self.frame_skip = config.frame_skip
        self.interpolate_tracks = config.interpolate_tracks
        self.frame_skip_counter = 0
        self.last_detections = []  # Store last detections for interpolation
        self._last_mouse_pos = (0, 0)  # Initialize for edit mode drawing

        # UI visibility states
        self.show_stats = True  # Stats panel visibility
        self.show_controls = True  # Instructions panel visibility
        self.ui_minimized = False  # Global minimize state

        # Results exporter (with cloud upload settings from AppConfig)
        export_config = ExportConfig(
            enable_api_upload=self.config.enable_api_upload,
            cloud_db_name=getattr(config, 'cloud_db_name', '') or '',
            cloud_table_name=getattr(config, 'cloud_table_name', '') or '',
            cloud_db_config_path=getattr(config, 'cloud_db_config_path', '') or ''
        )
        self.exporter = ResultsExporter(config.output_folder, export_config)

        # Heatmap state
        self.heatmap_acc = None

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

        # Processing state
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False

        # Fixed hourly segments with clock alignment
        self.segment_split_minutes = 60  # Always hourly
        self.align_segments_to_clock = True  # Always align
        self.segment_duration_seconds = 3600  # 1 hour in seconds

        # Initialize segment tracking
        self.current_segment = 0
        self.segment_start_time = time.time()
        self.current_segment_start_dt = None
        self.next_split_dt = None

        if self.segment_split_minutes > 0 and self.align_segments_to_clock:
            self.next_split_dt = self._compute_next_clock_boundary(datetime.now(), self.segment_split_minutes)
        else:
            # fallback to existing duration-based rollover if you have it
            self.next_split_dt = None

        # Video I/O
        self.cap = None
        self.video_writer = None
        self.frame_queue = Queue(maxsize=10)

        # Statistics
        self.stats = ProcessingStats()

        # Live Master Log Export settings
        self.live_export_interval = 300.0  # Update Excel every 5 minutes
        self.last_live_export_time = time.time()
        self.last_exported_event_count = 0
        self.export_lock = threading.Lock()
        
        # Track active zone events that need dwell time updates
        self.active_zone_events = {}

        # Threading
        self.processing_thread = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self) -> Dict:
        """
        Main processing entry point

        Returns:
            Processing results dictionary
        """
        try:
            if self.config.is_camera:
                return self._process_camera()
            else:
                return self._process_videos()
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return self._get_final_results()
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
        finally:
            self.cleanup()

    def _process_camera(self) -> Dict:
        """Process camera input with real-time display and controls"""
        self.logger.info("Starting camera processing...")

        # Initialize camera
        if not self._initialize_camera():
            raise RuntimeError("Failed to initialize camera")

        # Initialize counter with actual frame size
        self._update_counter_frame_size()

        frame_counter = 0

        # ---- HEATMAP INIT (camera) ----
        if getattr(self.config, "enable_heatmap", False):
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define all variables properly
            interval_sec = float(max(1, int(self.segment_split_minutes) * 60))
            alpha = float(getattr(self.config, "heatmap_alpha", 0.35))
            cmap_name = getattr(self.config, "heatmap_colormap", "HOT")
            out_dir = str(Path(self.config.output_folder) / "heatmaps")
            radius_px = int(getattr(self.config, "heatmap_radius_px", 10))
            decay = float(getattr(self.config, "heatmap_decay", 0.0))
            gamma = float(getattr(self.config, "heatmap_gamma", 1.6))

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
            self.logger.info(f"[HEATMAP] enabled ({width}x{height}), interval={interval_sec}s, alpha={alpha}")
        else:
            self.heatmap_acc = None
        # ---- /HEATMAP INIT ----

        # ... rest of the method ...
        # ---- /HEATMAP INIT ----

        # Initialize video writer if needed
        if self.config.save_video:
            self._initialize_video_writer()

        # Start processing
        self.is_running = True
        self.stats.start_time = time.time()
        self._reset_segment()

        # Create window + mouse callback for live editing
        cv2.namedWindow("Live Object Counter")
        cv2.setMouseCallback("Live Object Counter", self._mouse_callback)

        # Frame Bool for window centering
        first_frame = True
        try:
            while self.is_running and not self.stop_requested:
                # Read frame
                ret, frame = self.cap.read()
                if first_frame == True:
                    h, w = frame.shape[:2]
                    self.fit_center("Live Object Counter", w, h, frac=0.9)
                    first_frame = False

                if not ret:
                    self.logger.warning("Failed to read camera frame")
                    break

                frame_counter += 1

                # Check if we should process this frame
                should_process = (frame_counter % self.frame_skip) == 0

                if should_process:
                    # Process frame normally
                    processed_frame, events = self._process_frame(frame)
                    self.last_frame = processed_frame
                else:
                    # Skip processing but optionally interpolate
                    if self.interpolate_tracks and self.last_detections:
                        self._interpolate_tracks()

                    # Use last processed frame for display
                    processed_frame = self.last_frame if hasattr(self, 'last_frame') else frame
                    events = []

                # Handle segment rollover
                if self._should_rollover_segment():
                    self._rollover_segment()

                # Trigger live export check
                self._trigger_live_export(force=False)

                # Display frame with controls
                display_frame = self._add_live_controls(processed_frame)

                # Display frame with controls
                display_frame = self._add_live_controls(processed_frame)
                cv2.imshow("Live Object Counter", display_frame)

                # Handle keyboard input
                key_raw = cv2.waitKey(1)  # returns -1 when no key
                key = -1 if key_raw == -1 else (key_raw & 0xFF)
                if not self._handle_keyboard_input(key):
                    break

                # Update statistics
                self._update_stats(len(events) if events else 0)

        finally:
            self._finalize_current_segment()

        return self._get_final_results()

    def _create_configured_counter(self, frame_size: Tuple[int, int]) -> ObjectCounter:
        """Create a properly configured ObjectCounter instance.

        Centralizes counter creation to avoid duplicate code and ensure
        consistent configuration across all code paths.
        """
        counter = ObjectCounter(
            self.config.lines_config,
            self.config.zones_config,
            frame_size,
            exclusion_zones=getattr(self.config, 'exclusion_zones', []),
            max_track_age=self.config.max_track_age
        )

        # Apply speed configuration
        counter.configure_speed(
            enable=self._enable_speed,
            units=self._speed_units,
            meters_per_pixel=float(getattr(self.config, "meters_per_pixel", 0.0) or 0.0),
            smooth_window=int(getattr(self.config, "speed_smooth_window", 5) or 5),
            annotate=self._annotate_speed
        )

        return counter

    def _process_frame_with_skip(self, frame: np.ndarray) -> Tuple[np.ndarray, List[CountingEvent]]:
        """Process frame with skip logic"""
        self.frame_skip_counter += 1

        # Check if we should process this frame
        if (self.frame_skip_counter % self.frame_skip) == 0:
            # Process normally - full detection and tracking
            processed_frame, events = self._process_frame(frame)
            self.last_detections = self.detection_engine.last_detections  # Store for interpolation

            return processed_frame, events
        else:
            # Skip detection but still do counting if interpolating
            if self.interpolate_tracks and self.last_detections:
                # Interpolate object positions for line crossing detection
                interpolated_detections = self._interpolate_detections(self.last_detections)

                # Update counters for line crossings BUT mark as interpolated
                # This prevents speed calculation updates while allowing line crossing detection
                events = self.counter.update_counts(
                    interpolated_detections,
                    timestamp=self.video_current_time,
                    skip_speed_update=True  # New parameter to skip speed calculations
                )

                # Draw visualizations with interpolated positions
                processed_frame = self._draw_visualizations(frame.copy(), interpolated_detections)
            else:
                # Just return the frame without processing
                processed_frame = frame.copy()
                events = []

            return processed_frame, events

    def _process_videos(self, max_workers: int = None) -> Dict:
        """
        Process video files from folder with concurrent workers.

        Args:
            max_workers: Maximum number of videos to process simultaneously.
                        If None, uses config.max_parallel_videos (default: 1)
                        Keep this conservative to avoid GPU memory issues.
        """
        # Get max_workers from config if not specified
        if max_workers is None:
            max_workers = getattr(self.config, 'max_parallel_videos', 1)

        # Initialize video queue
        video_queue = ThreadQueue()
        folder_monitor = None

        # Track active and completed work
        pending_futures = {}
        total_results = []
        video_count = 0
        worker_id_counter = 0

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        worker_progress = {}  # worker_id -> {frames, total, fps, events}

        # Determine initial video files
        if self.config.input_type.value == "folder":
            folder_path = Path(self.config.input_source)

            # Setup folder monitoring for new videos (this also scans existing files)
            def on_new_video(video_path: Path):
                self.logger.info(f"Adding new video to queue: {video_path.name}")
                video_queue.put(video_path)

            folder_monitor = FolderMonitor(
                folder_path=str(folder_path),
                callback=on_new_video,
                poll_interval=2.0
            )

            # Get initial files from the monitor (avoids duplicate scanning)
            initial_files = folder_monitor.get_known_files()

            # Add initial files to queue
            for video_file in initial_files:
                video_queue.put(video_file)

            # Start monitoring for NEW files only
            folder_monitor.start()

            self.logger.info(f"Folder monitoring enabled for: {folder_path}")
            self.logger.info(f"Starting with {video_queue.qsize()} existing video(s)")
            self.logger.info(f"Concurrent workers: {max_workers}")
        else:
            # Single file mode
            video_queue.put(Path(self.config.input_source))
            max_workers = 1  # No need for concurrency with single file

        if video_queue.empty():
            raise RuntimeError("No video files found")

        # Create enhanced progress window
        progress = ProgressWindow("Processing Videos")
        progress.set_total_queued(video_queue.qsize())

        # Track overall start time
        overall_start_time = time.time()

        # Thread pool for concurrent processing
        executor = ThreadPoolExecutor(max_workers=max_workers)

        # Callback for workers to get queue size (for extended stability wait when queue <= 1)
        def get_queue_size():
            return video_queue.qsize()

        # Progress callback that workers will call
        def on_worker_progress(worker_id, frames_done, total_frames, fps, events,
                               waiting_until_recheck=0.0, remaining_stability_wait=0.0):
            with progress_lock:
                worker_progress[worker_id] = {
                    'frames': frames_done,
                    'total': total_frames,
                    'fps': fps,
                    'events': events,
                    'waiting_recheck': waiting_until_recheck,
                    'waiting_stability': remaining_stability_wait
                }

        try:
            active_workers = 0

            while True:
                # Submit new jobs if we have capacity and videos waiting
                while active_workers < max_workers and not video_queue.empty():
                    try:
                        video_path = video_queue.get_nowait()
                    except:
                        break

                    if not video_path.exists():
                        self.logger.warning(f"Video not found, skipping: {video_path}")
                        continue

                    video_count += 1
                    worker_id_counter += 1

                    # Create isolated worker for this video
                    # Pass queue_size_callback for extended stability wait when queue <= 1
                    worker = VideoWorker(self.config, video_path, worker_id_counter, 
                                        queue_size_callback=get_queue_size)

                    # Get total frames for registration
                    temp_cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_cap.release()

                    # Register video with progress window
                    progress.register_video(worker_id_counter, video_path.name, total_frames)
                    progress.set_total_queued(video_queue.qsize())

                    # Submit to thread pool with progress callback
                    future = executor.submit(worker.process, on_worker_progress)
                    pending_futures[future] = {
                        'video_path': video_path,
                        'worker_id': worker_id_counter,
                        'start_time': time.time()
                    }
                    active_workers += 1

                    self.logger.info(f"Started worker {worker_id_counter} for: {video_path.name} "
                                     f"(Active: {active_workers}/{max_workers})")

                # Update progress window from worker progress data
                with progress_lock:
                    for wid, data in worker_progress.items():
                        progress.update_video_progress(
                            wid,
                            data['frames'],
                            data['fps'],
                            data['events'],
                            data.get('waiting_recheck', 0.0),
                            data.get('waiting_stability', 0.0)
                        )

                # Check for completed futures
                done_futures = [f for f in pending_futures if f.done()]

                for future in done_futures:
                    info = pending_futures.pop(future)
                    active_workers -= 1
                    worker_id = info['worker_id']

                    try:
                        result = future.result(timeout=1.0)
                        total_results.append(result)

                        elapsed = time.time() - info['start_time']
                        final_fps = result.get('stats', {})
                        if hasattr(final_fps, 'fps'):
                            final_fps = final_fps.fps
                        else:
                            final_fps = 0

                        self.logger.info(f"Worker {worker_id} finished: {info['video_path'].name} "
                                         f"in {elapsed:.1f}s ({final_fps:.1f} avg FPS)")

                        # Mark as complete in progress window
                        progress.complete_video(worker_id, success=True)

                    except Exception as e:
                        self.logger.error(f"Worker {worker_id} failed: {e}")
                        total_results.append({
                            "error": str(e),
                            "video_path": str(info['video_path'])
                        })
                        progress.complete_video(worker_id, success=False)

                    # Clean up worker progress tracking
                    with progress_lock:
                        worker_progress.pop(worker_id, None)

                # Update queue count and elapsed time
                progress.set_total_queued(video_queue.qsize())
                progress.update_elapsed_time(time.time() - overall_start_time)

                # Check if we're done
                if active_workers == 0 and video_queue.empty():
                    if self.config.input_type.value == "folder":
                        # Wait indefinitely for new files (folder monitoring mode)
                        self.logger.info("All current videos processed, waiting for new files... (Ctrl+C to stop)")
                        progress.set_status("Waiting for new files... (Ctrl+C to stop)")
                        while video_queue.empty():
                            time.sleep(2)  # Check every 2 seconds
                        self.logger.info("New video detected, resuming processing...")
                    else:
                        break

                # Small sleep to prevent busy-waiting
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            executor.shutdown(wait=False, cancel_futures=True)

        finally:
            if folder_monitor:
                folder_monitor.stop()
            executor.shutdown(wait=True)
            progress.close()

        return {
            "total_videos": len(total_results),
            "video_results": total_results,
            "combined_stats": self._combine_video_stats(total_results)
        }

    def _interpolate_detections(self, last_detections: List[Detection]) -> List[Detection]:
        """Interpolate detection positions based on motion history - FOR VISUALIZATION ONLY"""
        interpolated = []

        for det in last_detections:
            if det.track_id in self.counter.object_states:
                obj_state = self.counter.object_states[det.track_id]

                # Simple linear interpolation based on velocity
                if len(obj_state.positions) >= 2:
                    # Get last two positions to estimate velocity
                    pos_history = list(obj_state.positions)
                    if len(pos_history) >= 2:
                        last_pos = pos_history[-1]['center']
                        prev_pos = pos_history[-2]['center']

                        # Get time delta for accurate velocity
                        t1 = pos_history[-2]['timestamp']
                        t2 = pos_history[-1]['timestamp']
                        dt = max(1e-6, t2 - t1)

                        # Calculate velocity in pixels per second
                        vx = (last_pos[0] - prev_pos[0]) / dt
                        vy = (last_pos[1] - prev_pos[1]) / dt

                        # Calculate time since last detection (estimated)
                        # Assume frame rate if we have video info
                        if hasattr(self, 'video_fps') and self.video_fps > 0:
                            time_step = (self.frame_skip - (self.frame_skip_counter % self.frame_skip)) / self.video_fps
                        else:
                            # Fallback: use 30 fps assumption
                            time_step = (self.frame_skip - (self.frame_skip_counter % self.frame_skip)) / 30.0

                        # Predict new position based on velocity and time
                        predicted_center = (
                            int(last_pos[0] + vx * time_step),
                            int(last_pos[1] + vy * time_step)
                        )

                        # Create interpolated detection
                        bbox_width = det.bbox[2] - det.bbox[0]
                        bbox_height = det.bbox[3] - det.bbox[1]

                        interpolated_det = Detection(
                            track_id=det.track_id,
                            class_id=det.class_id,
                            class_name=det.class_name,
                            bbox=(
                                predicted_center[0] - bbox_width // 2,
                                predicted_center[1] - bbox_height // 2,
                                predicted_center[0] + bbox_width // 2,
                                predicted_center[1] + bbox_height // 2
                            ),
                            confidence=det.confidence * 0.9,  # Slightly reduce confidence
                            center_point=predicted_center,
                            bottom_point=(predicted_center[0], predicted_center[1] + bbox_height // 2)
                        )
                        interpolated.append(interpolated_det)
                        continue

            # If can't interpolate, use last position
            interpolated.append(det)

        return interpolated

    def _process_single_video(self, video_path: Path, progress: Optional[ProgressWindow] = None) -> Dict:
        """Process a single video file"""

        # Store current video path for exports
        self.current_video_path = video_path

        # Initialize video capture
        if not self._initialize_video_capture(str(video_path)):
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Initialize counter with actual frame size
        self._update_counter_frame_size()

        # ---- HEATMAP INIT (video) ----
        if getattr(self.config, "enable_heatmap", False):
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define all variables properly
            interval_sec = float(max(1, int(self.segment_split_minutes) * 60))
            alpha = float(getattr(self.config, "heatmap_alpha", 0.35))
            cmap_name = getattr(self.config, "heatmap_colormap", "HOT")
            out_dir = str(Path(self.config.output_folder) / "heatmaps")
            radius_px = int(getattr(self.config, "heatmap_radius_px", 10))
            decay = float(getattr(self.config, "heatmap_decay", 0.0))
            gamma = float(getattr(self.config, "heatmap_gamma", 1.6))

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
            self.logger.info(f"[HEATMAP] enabled ({width}x{height}), interval={interval_sec}s, alpha={alpha}")
        else:
            self.heatmap_acc = None
        # ---- /HEATMAP INIT ----

        # Initialize video writer if needed
        if self.config.save_video:
            output_name = f"{video_path.stem}_output.mp4"
            self._initialize_video_writer(output_name)

        # Get total frame count for progress tracking
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Processing setup
        self.is_running = True
        self.stats = ProcessingStats()
        self.stats.start_time = time.time()
        self.current_segment = 0
        self._reset_segment()

        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Increment video frame counter and update video current time
                self.video_frame_number += 1
                self.video_current_time = self._get_current_timestamp()

                # Process frame
                processed_frame, events = self._process_frame_with_skip(frame)

                # Handle segment rollover
                if self._should_rollover_segment():
                    self._rollover_segment()

                # Update statistics
                self._update_stats(len(events) if events else 0)

                # Update progress window if available
                if progress is not None and self.stats.frames_processed % 10 == 0:  # Update every 10 frames
                    progress.update_frame_progress(self.stats.frames_processed, total_frames)
                    # Update stats display
                    stats_text = f"FPS: {self.stats.fps:.1f} | Events: {self.stats.total_events}"
                    progress.update_stats(stats_text)

                # Progress logging
                if self.stats.frames_processed % 300 == 0:  # Every 10 seconds at 30fps
                    self._log_progress()

        finally:
            self._finalize_current_segment()
            self.cap.release()
            if self.video_writer:
                self.video_writer.release()

        # Prepare results
        video_results = {
            "video_path": str(video_path),
            "stats": self.stats,
            "final_counts": self.counter.get_current_counts(),
            "events_summary": self.counter.get_events_summary()
        }

        # Export individual video results
        try:
            video_name = video_path.stem
            self.exporter.export_video_summary(video_results, video_name)
            self.logger.info(f"Exported results for {video_name}")
        except Exception as e:
            self.logger.error(f"Failed to export results for {video_name}: {e}")

        return video_results

    def _initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.config.input_source)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.config.input_source}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.display_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            # NEW: Set live source timing
            self.is_live_source = True
            self.video_start_time = datetime.now()
            self.video_fps = actual_fps if actual_fps > 0 else 30.0
            self.video_frame_number = 0
            self.video_current_time = self.video_start_time

            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

    def _get_current_timestamp(self) -> datetime:
        """Get the current timestamp based on video or live source"""
        if self.is_live_source:
            # For live sources, use system time
            return datetime.now()
        else:
            # For pre-recorded videos, calculate based on frame position
            if self.video_start_time and self.video_fps > 0:
                # Calculate time offset based on frames processed
                time_offset_seconds = self.video_frame_number / self.video_fps
                current_time = self.video_start_time + timedelta(seconds=time_offset_seconds)
                return current_time
            else:
                # Fallback to system time if we can't calculate
                return datetime.now()

    def _initialize_video_capture(self, video_path: str) -> bool:
        """Initialize video file capture"""
        try:
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return False

            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            duration = frame_count / fps if fps > 0 else 0

            # NEW: Get video file creation/modification time as baseline
            from pathlib import Path
            import os
            video_file = Path(video_path)

            # Try to get creation time, fall back to modification time
            if hasattr(os.stat(video_file), 'st_birthtime'):
                # macOS/Windows
                file_timestamp = os.stat(video_file).st_birthtime
            else:
                # Linux - use modification time as approximation
                file_timestamp = os.path.getmtime(video_file)

            # Set video timing properties
            self.video_start_time = datetime.fromtimestamp(file_timestamp)
            self.video_fps = fps if fps > 0 else 30.0
            self.video_frame_number = 0
            self.is_live_source = False  # This is a pre-recorded video
            self.video_current_time = self.video_start_time

            self.logger.info(f"Video loaded: {width}x{height}, {fps}fps, {duration:.1f}s, {frame_count} frames")
            self.logger.info(f"Video timestamp baseline: {self.video_start_time.isoformat()}")

            return True

        except Exception as e:
            self.logger.error(f"Video initialization failed: {e}")
            return False

    def _update_counter_frame_size(self):
        """Update counter with actual frame size"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Update counter with ACTUAL video frame size, not display size
            actual_frame_size = (width, height)

            # Recreate counters with correct frame size if needed
            if self.counter.frame_size != actual_frame_size:
                self.counter.set_frame_size(actual_frame_size)
                self.logger.info(f"Updated counter with actual frame size: {actual_frame_size}")

            # Check if we already have a counter with the right size
            if hasattr(self, 'counter') and self.counter:
                current_size = self.counter.frame_size
                if current_size == (width, height):
                    # Size matches, no need to recreate
                    return

                # Try to update existing counter's frame size if method exists
                if hasattr(self.counter, "set_frame_size"):
                    self.counter.set_frame_size((width, height))
                    self.logger.info(f"Updated counter frame size to {width}x{height}")
                    return

            # Only create new counter if we don't have one or can't update
            self.counter = self._create_configured_counter((width, height))

            # Reapply speed settings after creating new counter
            self.counter.configure_speed(
                enable=bool(getattr(self.config, "enable_speed", False)),
                units=str(getattr(self.config, "speed_units", "pxps")),
                meters_per_pixel=float(getattr(self.config, "meters_per_pixel", 0.0) or 0.0),
                smooth_window=int(getattr(self.config, "speed_smooth_window", 5) or 5),
                annotate=bool(getattr(self.config, "annotate_speed", True))
            )

            self.logger.info(f"Created new counter with frame size {width}x{height}")

    def _initialize_video_writer(self, filename: Optional[str] = None, frame: Optional[np.ndarray] = None) -> bool:
        """
        Initialize (or re-initialize) the video writer.

        It tries multiple combinations of:
          - Container: MP4 then fallback to AVI
          - Backend: FFMPEG, MSMF, DSHOW, and default
          - Codec/FourCC: mp4v/avc1/H264 for MP4 (may need OpenH264),
                          MJPG/XVID/DIVX/I420/IYUV/YUY2 for AVI
        Returns True on success, False otherwise.
        """
        if not self.cap:
            return False

        try:
            # Set output resolution based on config
            resolution_map = {
                "480p": (854, 480),
                "720p": (1280, 720),
                "1080p": (1920, 1080),
                "original": None
            }

            output_res = getattr(self.config, 'output_resolution', '720p')

            if output_res == "original" or output_res not in resolution_map:
                # Use original resolution
                if frame is not None:
                    h, w = frame.shape[:2]
                else:
                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            else:
                # Use specified resolution
                w, h = resolution_map[output_res]

            if w <= 0 or h <= 0:
                self.logger.warning("Video writer init: capture size invalid; waiting for a real frame.")
                return False

            # Some encoders reject odd sizes
            w -= (w % 2)
            h -= (h % 2)

            # FPS with sensible fallback
            fps_raw = self.cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_raw) if fps_raw and fps_raw > 0 else 30.0

            # ----- Resolve output path -----
            now = datetime.now()

            # Create nested directory structure: output/YYYY-MM/YYYY-MM-DD/

            out_dir = Path(self.config.output_folder) / "live_footage"
            out_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                if getattr(self.config, 'is_camera', False):
                    date_str = now.strftime("%Y-%m-%d")
                    hour_str = now.strftime("%H")
                    filename = f"live_{date_str}_{hour_str}.mp4"
                else:
                    filename = f"live_segment_{self.current_segment}.mp4"

            path = out_dir / filename

            # Helper: open with a specific backend and fourcc
            def open_with(api_pref, fourcc_str, out_path: Path) -> bool:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                vw = cv2.VideoWriter()
                try:
                    if api_pref is None:
                        ok = vw.open(str(out_path), fourcc, fps, (w, h), True)
                    else:
                        ok = vw.open(str(out_path), api_pref, fourcc, fps, (w, h), True)
                except Exception:
                    ok = False

                # Around line 788, inside the open_with function where it returns True:
                if ok and vw.isOpened():
                    self.video_writer = vw
                    self._video_writer_size = (w, h)
                    self.logger.info(
                        f"Video writer initialized: {out_path} ({fourcc_str}, {w}x{h}@{fps:.2f}, api={api_pref})"
                    )
                    try:
                        self.current_output_filename = out_path.name
                    except Exception:
                        pass
                    return True

                self.logger.warning(f"Video writer open failed with {fourcc_str} (api={api_pref}); trying next.")
                return False

            # Build attempts list
            attempts = []

            # If using .mp4, try MP4 first (may fail if OpenH264 mismatched)
            used_mp4 = path.suffix.lower() == ".mp4"
            if used_mp4:
                self.logger.warning("MP4 encoders may be unavailable (OpenH264). Will try MP4 then fall back to AVI.")
                attempts.extend([
                    (cv2.CAP_FFMPEG, "mp4v", path),
                    (cv2.CAP_FFMPEG, "avc1", path),
                    (cv2.CAP_FFMPEG, "H264", path),
                    (None, "mp4v", path),
                ])

            # AVI fallbacks (bypass H.264 entirely)
            avi_path = path if path.suffix.lower() == ".avi" else path.with_suffix(".avi")
            attempts.extend([
                (cv2.CAP_FFMPEG, "MJPG", avi_path),
                (cv2.CAP_MSMF, "MJPG", avi_path),
                (cv2.CAP_DSHOW, "MJPG", avi_path),
                (cv2.CAP_FFMPEG, "XVID", avi_path),
                (cv2.CAP_MSMF, "XVID", avi_path),
                (cv2.CAP_FFMPEG, "DIVX", avi_path),
                # Raw / YUV-ish options some backends accept:
                (cv2.CAP_MSMF, "I420", avi_path),
                (cv2.CAP_MSMF, "IYUV", avi_path),
                (cv2.CAP_MSMF, "YUY2", avi_path),
                (None, "MJPG", avi_path),
            ])

            for api, fourcc, p in attempts:
                if open_with(api, fourcc, p):
                    return True

            self.logger.error("Failed to initialize video writer after trying multiple backends/codecs/containers.")
            self.video_writer = None
            return False

        except Exception as e:
            self.logger.error(f"Video writer initialization error: {e}")
            self.video_writer = None
            return False

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[CountingEvent]]:
        """
        Process a single frame through the detection and counting pipeline

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, counting_events)
        """
        start_time = time.time()
        self.last_frame = frame

        # NEW: Update frame counter and current video time
        self.video_frame_number += 1
        self.video_current_time = self._get_current_timestamp()

        # Run detection and tracking (exclusion zones already filtered in detection engine)
        detection_start = time.time()
        detections = self.detection_engine.detect_and_track(frame)
        detection_time = time.time() - detection_start

        # Update counters - pass video timestamp (detections already filtered by exclusion mask)
        counting_start = time.time()
        events = self.counter.update_counts(detections, timestamp=self.video_current_time)
        counting_time = time.time() - counting_start

        # Training mode capture (use detections, not filtered_detections)
        if self.training_capture:
            captured_path = self.training_capture.capture_frame(frame, detections)
            if captured_path:
                self.logger.debug(f"Training frame saved: {captured_path}")

        # ---- HEATMAP UPDATE & EMIT ----
        if self.heatmap_acc is not None:
            # Build list of (x1,y1,x2,y2) from our Detection objects
            boxes_xyxy = []

            # Optional: respect allowed class filter from config if present
            allowed = set(self.config.allowed_classes) if getattr(self.config, "allowed_classes", None) else None

            for det in detections:
                if allowed is not None and det.class_id not in allowed:
                    continue
                x1, y1, x2, y2 = det.bbox  # Detection.bbox is already (x1,y1,x2,y2)
                boxes_xyxy.append((int(x1), int(y1), int(x2), int(y2)))

            if boxes_xyxy:
                self.heatmap_acc.update_from_boxes(boxes_xyxy, weight=1.2)

            # Note: Removed maybe_emit() - we only want ONE heatmap per video
            # The final heatmap is saved via _flush_heatmap() at video/segment end
        # ---- /HEATMAP UPDATE & EMIT ----

        # Draw visualizations with detections (already filtered by exclusion zones)
        visualization_start = time.time()
        processed_frame = self._draw_visualizations(frame.copy(), detections)
        visualization_time = time.time() - visualization_start

        # Update timing statistics
        total_time = time.time() - start_time
        self.stats.avg_detection_time = (
                                                self.stats.avg_detection_time * self.stats.frames_processed + detection_time
                                        ) / (self.stats.frames_processed + 1)
        self.stats.avg_processing_time = (
                                                 self.stats.avg_processing_time * self.stats.frames_processed + total_time
                                         ) / (self.stats.frames_processed + 1)

        return processed_frame, events

    def _draw_visualizations(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw all visualizations on the frame"""
        # Draw detection bounding boxes
        for detection in detections:
            self._draw_detection(frame, detection)

        # Draw counting overlays (lines, zones, exclusions)
        frame = self.counter.draw_overlays(frame, show_counts=True)

        # ===== SAVE TO VIDEO HERE (before stats) =====
        if self.config.save_video:
            self._save_video_frame(frame.copy())
        # ===== END VIDEO SAVE =====

        # Draw statistics overlay
        frame = self._draw_stats_overlay(frame)

        # Draw edit mode overlays
        if self.edit_mode and self.create_mode != "none":
            if self.create_mode == "line" and len(self.temp_points) == 1:  # Fixed: added ==
                cv2.line(frame, self.temp_points[0], tuple(self._last_mouse_pos), (0, 255, 255), 2)
                cv2.circle(frame, self.temp_points[0], 6, (0, 255, 255), -1)
                cv2.putText(frame, "Click second point",
                            (self.temp_points[0][0], self.temp_points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            elif self.create_mode == "zone":
                for p in self.temp_points:
                    cv2.circle(frame, p, 4, (255, 255, 0), -1)
                if len(self.temp_points) >= 2:
                    pts = np.array(self.temp_points, np.int32)
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                # helper line to cursor
                if self.temp_points:
                    cv2.line(frame, self.temp_points[-1], tuple(self._last_mouse_pos), (255, 255, 0), 1)

        # ===== ADD THIS NEW SECTION FOR NOTIFICATIONS =====
        # Draw notification banner if active
        if hasattr(self, 'notification') and self.notification:
            elapsed = time.time() - self.notification['start_time']
            if elapsed < self.notification['duration']:
                # Draw notification banner at top center
                text = self.notification['text']
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size for background
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # Position at top center of frame
                x = (frame.shape[1] - text_size[0]) // 2
                y = 50

                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (x - 10, y - text_size[1] - 10),
                              (x + text_size[0] + 10, y + 10),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                # Draw border
                cv2.rectangle(frame,
                              (x - 10, y - text_size[1] - 10),
                              (x + text_size[0] + 10, y + 10),
                              (0, 255, 255), 2)

                # Draw notification text
                cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)
            else:
                # Clear notification after duration expires
                self.notification = None

        # ===== OPTIONAL: Add frame skip indicator =====
        # Show frame skip status in top-left corner if not default
        if hasattr(self, 'frame_skip') and self.frame_skip > 1:
            skip_text = f"Skip: {self.frame_skip}x"
            if hasattr(self, 'interpolate_tracks'):
                skip_text += f" ({'Interp ON' if self.interpolate_tracks else 'Interp OFF'})"

            # Draw in top-left corner with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size = cv2.getTextSize(skip_text, font, font_scale, thickness)[0]

            # Background
            cv2.rectangle(frame, (10, 60), (15 + text_size[0], 80 + text_size[1]), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 60), (15 + text_size[0], 80 + text_size[1]), (255, 165, 0), 1)

            # Text
            cv2.putText(frame, skip_text, (12, 75), font, font_scale, (255, 165, 0), thickness)

        # Draw training mode indicator
        if self.training_capture and self.training_capture.is_active:
            # Draw training indicator in top-left
            cv2.putText(frame, "TRAINING", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status = self.training_capture.get_status()
            status_text = f"Captures: {status['captures']} | Interval: {status['interval_seconds']}s"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw minimize/maximize buttons
        self._draw_ui_toggle_buttons(frame)

        return frame

    def _save_video_frame(self, clean_frame: np.ndarray) -> None:
        """Save frame to video (called before stats are drawn)"""
        if self.video_writer is None or not self.video_writer.isOpened():
            self._initialize_video_writer(frame=clean_frame)
            if self.video_writer is None:
                return

        # Resize if needed
        if hasattr(self, '_video_writer_size'):
            h, w = clean_frame.shape[:2]
            if (w, h) != self._video_writer_size:
                clean_frame = cv2.resize(clean_frame, self._video_writer_size)

        self.video_writer.write(clean_frame)

    def _draw_ui_toggle_buttons(self, frame: np.ndarray):
        """Draw minimize/maximize buttons for UI panels"""
        h, w = frame.shape[:2]

        # Stats toggle button (top-left corner)
        stats_btn_x, stats_btn_y = 260, 20
        stats_btn_size = 20

        # Draw button background
        color = (0, 200, 0) if self.show_stats else (100, 100, 100)
        cv2.rectangle(frame,
                      (stats_btn_x, stats_btn_y),
                      (stats_btn_x + stats_btn_size, stats_btn_y + stats_btn_size),
                      color, -1)
        cv2.rectangle(frame,
                      (stats_btn_x, stats_btn_y),
                      (stats_btn_x + stats_btn_size, stats_btn_y + stats_btn_size),
                      (255, 255, 255), 1)

        # Draw minimize/maximize icon
        icon_text = "-" if self.show_stats else "+"
        cv2.putText(frame, icon_text,
                    (stats_btn_x + 6, stats_btn_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Controls toggle button (bottom-right corner)
        controls_btn_x = w - 400
        controls_btn_y = h - 220

        color = (0, 200, 0) if self.show_controls else (100, 100, 100)
        cv2.rectangle(frame,
                      (controls_btn_x, controls_btn_y),
                      (controls_btn_x + stats_btn_size, controls_btn_y + stats_btn_size),
                      color, -1)
        cv2.rectangle(frame,
                      (controls_btn_x, controls_btn_y),
                      (controls_btn_x + stats_btn_size, controls_btn_y + stats_btn_size),
                      (255, 255, 255), 1)

        icon_text = "-" if self.show_controls else "+"
        cv2.putText(frame, icon_text,
                    (controls_btn_x + 6, controls_btn_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_detection(self, frame: np.ndarray, detection: Detection):
        """Draw a single detection on the frame"""
        x1, y1, x2, y2 = detection.bbox

        # Choose color based on class
        colors = [(163, 207, 167), (247, 220, 236), (255, 225, 148), (255, 241, 222),
                  (146, 192, 212), (235, 185, 138), (187, 135, 170), (125, 122, 179)]
        color = colors[detection.class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{detection.class_name}"
        if detection.track_id is not None:
            label += f"#{detection.track_id}"
        label += f" {detection.confidence:.2f}"

        # NEW: append speed if enabled and available
        if getattr(self.config, "enable_speed", False) and getattr(self.counter, "annotate_speed", True):
            spd_map = getattr(self.counter, "_last_speeds", {})
            spd = spd_map.get(detection.track_id, None)
            if spd is not None:
                units = str(getattr(self.config, "speed_units", "pxps"))
                # e.g., "car#12 0.87 13.4 mph"
                label += f" {spd:.1f} {units}"

        # Label background (unchanged)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

        # Label text (unchanged)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Center point dot (unchanged)
        cv2.circle(frame, detection.center_point, 3, color, -1)

    def _draw_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance statistics and timestamp overlay"""

        # Get current timestamp - use video time if available
        if self.is_live_source:
            current_time = datetime.now()
            time_source = "LIVE"
        else:
            current_time = self.video_current_time if self.video_current_time else datetime.now()
            time_source = "VIDEO"

        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Only draw full stats if visible
        if self.show_stats:
            # Add frame skip info
            effective_fps = self.stats.fps * self.frame_skip if self.frame_skip > 1 else self.stats.fps

            # Prepare stats text - include time source indicator
            stats_text = [
                f"FPS: {self.stats.fps:.1f} (Effective: {effective_fps:.1f})",
                f"Frame Skip: {self.frame_skip}x",
                f"Frames: {self.stats.frames_processed}",
                f"Detections: {self.stats.total_detections}",
                f"Device: {self.detection_engine.device}",
                f"Events: {self.stats.total_events}",
                f"Segment: {self.current_segment}",
                f"Objects: {len(self.counter.object_states)}",
                f"Time Source: {time_source}"  # NEW: Show time source
            ]

            # Draw stats background (left side)
            text_height = 20
            bg_height = len(stats_text) * text_height + 20
            cv2.rectangle(frame, (10, 10), (250, bg_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, bg_height), (255, 255, 255), 1)

            # Draw stats text lines
            for i, text in enumerate(stats_text):
                y_pos = 30 + i * text_height
                cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Draw minimal stats indicator when minimized
            cv2.putText(frame, "STATS", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw timestamp overlay (top right corner)
        timestamp_font_scale = 0.7
        timestamp_thickness = 2
        timestamp_font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            timestamp_str, timestamp_font, timestamp_font_scale, timestamp_thickness
        )

        # Position for top-right corner
        margin = 10
        timestamp_x = frame.shape[1] - text_width - margin - 10
        timestamp_y = margin + text_height + 5

        # Draw semi-transparent background for timestamp
        bg_x1 = timestamp_x - 5
        bg_y1 = timestamp_y - text_height - 5
        bg_x2 = timestamp_x + text_width + 5
        bg_y2 = timestamp_y + 5

        # Create overlay for semi-transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw white border
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)

        # Draw timestamp text
        cv2.putText(frame, timestamp_str, (timestamp_x, timestamp_y),
                    timestamp_font, timestamp_font_scale, (0, 255, 255), timestamp_thickness)

        # Optional: Add recording indicator if saving video
        if self.config.save_video:
            # Draw red recording dot
            rec_center = (bg_x1 - 20, timestamp_y - text_height // 2)
            cv2.circle(frame, rec_center, 6, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(frame, rec_center, 6, (255, 255, 255), 1)  # White border

            # Add "REC" text
            cv2.putText(frame, "REC", (rec_center[0] - 40, rec_center[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Optional: Add segment time window if using clock-aligned segments
        if self.segment_split_minutes > 0 and self.align_segments_to_clock:
            window_text = f"Window: {self.current_segment_start_dt.strftime('%H:%M')}"
            if self.next_split_dt:
                window_text += f" - {self.next_split_dt.strftime('%H:%M')}"

            # Draw below timestamp
            window_y = timestamp_y + 25
            (window_width, _), _ = cv2.getTextSize(
                window_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            window_x = frame.shape[1] - window_width - margin - 10

            cv2.putText(frame, window_text, (window_x, window_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def _add_live_controls(self, frame: np.ndarray) -> np.ndarray:
        """Add live control instructions for camera mode"""

        if self.show_controls:
            controls = [
                "Controls:",
                "ESC - Exit   SPACE - Pause/Resume",
                "R - Reset counts   S - Save stats",
                f"E - Edit Mode - {'ON' if self.edit_mode else 'OFF'}",
                "M - Toggle stats   C - Toggle controls",
                # "T - Toggle training mode",
                # "+/- - Adjust training interval",
                # "1-5 - Set frame skip (process every Nth frame)",
                # "I - Toggle track interpolation",
                # f"Current: Skip={self.frame_skip}, Interp={'ON' if self.interpolate_tracks else 'OFF'}",
                # f"Training: {'ON' if self.training_capture and self.training_capture.is_active else 'OFF'}",
                # "EDIT: N=new line  Z=new zone  F=finish zone",
                # "EDIT: D/Del=delete item under cursor",
                # "EDIT: Drag endpoints/vertices to adjust"
            ]
            if self.training_capture:
                controls.append(
                    f"T - Training Mode - {'Active' if self.training_capture and self.training_capture.is_active else 'Inactive'}")
                controls.append("+/- - Adjust training interval")
                controls.append("1-5 - Set frame skip (process every Nth frame)")
                controls.append("I - Toggle track interpolation")
                controls.append(f"Current: Skip={self.frame_skip}, Interp={'ON' if self.interpolate_tracks else 'OFF'}")
            if self.edit_mode:
                controls.append("EDIT: N=new line  Z=new zone  F=finish zone")
                controls.append("EDIT: D/Del=delete item under cursor")
                controls.append("EDIT: Drag endpoints/vertices to adjust")

            # Position at bottom right
            start_y = frame.shape[0] - (len(controls) * 20 + 10)
            bg_width = 380
            bg_height = len(controls) * 20 + 10

            # Draw background
            cv2.rectangle(frame, (frame.shape[1] - bg_width - 10, start_y - 10),
                          (frame.shape[1] - 10, start_y + bg_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (frame.shape[1] - bg_width - 10, start_y - 10),
                          (frame.shape[1] - 10, start_y + bg_height), (255, 255, 255), 1)

            # Draw control text
            for i, control in enumerate(controls):
                y_pos = start_y + i * 20
                color = (0, 255, 255) if i == 0 else (255, 255, 255)
                cv2.putText(frame, control, (frame.shape[1] - bg_width, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Draw minimal controls indicator when minimized
            cv2.putText(frame, "CONTROLS",
                        (frame.shape[1] - 90, frame.shape[0] - 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def _rotate_video_writer(self, window_start: datetime, window_end: datetime):
        """Close current video writer and start a new one for the next segment"""
        try:
            # Close current writer if exists
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                self.logger.info(f"Closed video writer for segment {self.current_segment}")

            # Only start new writer if we're still saving video
            if self.config.save_video:
                # Format filename with time window
                time_str = f"{window_start.strftime('%H%M')}-{window_end.strftime('%H%M')}_{window_start.strftime('%Y%m%d')}"

                if self.config.is_camera:
                    filename = f"live_{time_str}.mp4"  # Ensure .mp4 extension
                else:
                    # For video files, include source name
                    source_name = Path(self.config.input_source).stem
                    filename = f"{source_name}_{time_str}.mp4"  # Ensure .mp4 extension

                # Re-initialize with new filename
                self._initialize_video_writer(filename)
                self.logger.info(f"Started new video writer: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to rotate video writer: {e}")

    def _show_notification(self, text: str, duration: float = 2.0):
        """Show on-screen notification (store for drawing)"""
        self.notification = {
            'text': text,
            'start_time': time.time(),
            'duration': duration
        }

    def _handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input during live processing

        Returns:
            True to continue processing, False to stop
        """
        # Ignore when no key was pressed
        if key == -1:
            return True

        if key == 27:  # ESC
            return False
        elif key == ord(' '):  # SPACE - pause/resume
            self.is_paused = not self.is_paused
            self.logger.info(f"Processing {'paused' if self.is_paused else 'resumed'}")
            while self.is_paused:
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    self.is_paused = False
                    break

        elif key == ord('r') or key == ord('R'):  # Reset counts
            self.counter.reset_all_counts()
            self.logger.info("Counts reset")

        elif key == ord('s') or key == ord('S'):  # Save stats
            self._save_current_stats()

        elif key == ord('m') or key == ord('M'):  # Toggle stats display
            self.show_stats = not self.show_stats
            self.logger.info(f"Stats display: {'ON' if self.show_stats else 'OFF'}")
            self._show_notification(f"Stats: {'ON' if self.show_stats else 'OFF'}", duration=1.0)
            return True

        elif key == ord('c') or key == ord('C'):  # Toggle controls display
            self.show_controls = not self.show_controls
            self.logger.info(f"Controls display: {'ON' if self.show_controls else 'OFF'}")
            self._show_notification(f"Controls: {'ON' if self.show_controls else 'OFF'}", duration=1.0)
            return True

        if key == ord('e') or key == ord('E'):
            self.edit_mode = not self.edit_mode
            self.create_mode = "none"
            self.temp_points.clear()
            self.logger.info(f"EDIT mode: {'ON' if self.edit_mode else 'OFF'}")
            return True

        # Add number keys 1-5 for frame skip adjustment
        if ord('1') <= key <= ord('5'):
            new_skip = key - ord('0')
            old_skip = self.frame_skip
            self.frame_skip = new_skip
            self.logger.info(f"Frame skip changed from {old_skip} to {new_skip}")

            # Show on-screen notification
            if hasattr(self, 'last_frame'):
                self._show_notification(f"Processing every {new_skip} frame(s)", duration=2.0)
            return True

        # Toggle interpolation with 'I' key
        if key == ord('i') or key == ord('I'):
            self.interpolate_tracks = not self.interpolate_tracks
            state = "ON" if self.interpolate_tracks else "OFF"
            self.logger.info(f"Track interpolation: {state}")
            self._show_notification(f"Track interpolation: {state}", duration=2.0)
            return True
        elif key == ord('t') or key == ord('T'):  # Toggle training mode
            if self.training_capture:
                is_active = self.training_capture.toggle()
                status = "ON" if is_active else "OFF"
                self.logger.info(f"Training mode: {status}")
                self._show_notification(f"Training mode: {status}", duration=2.0)
            return True

        elif key == ord('+') or key == ord('='):  # Increase capture interval
            if self.training_capture:
                current = self.training_capture.config.capture_interval_seconds
                new_interval = min(60, current + 1)
                self.training_capture.set_interval(new_interval)
                self._show_notification(f"Capture interval: {new_interval}s", duration=2.0)
            return True

        elif key == ord('-') or key == ord('_'):  # Decrease capture interval
            if self.training_capture:
                current = self.training_capture.config.capture_interval_seconds
                new_interval = max(0.5, current - 1)
                self.training_capture.set_interval(new_interval)
                self._show_notification(f"Capture interval: {new_interval}s", duration=2.0)
            return True

        # --- NEW editor keys, only when edit_mode is ON ---
        if self.edit_mode:
            if key == ord('n') or key == ord('N'):
                # start a new line
                self.create_mode = "line"
                self.temp_points.clear()
                self.logger.info("Create LINE: click 2 points")
                return True

            elif key == ord('z') or key == ord('Z'):
                # start a new zone
                self.create_mode = "zone"
                self.temp_points.clear()
                self.logger.info("Create ZONE: left-click to add vertices, right-click or 'F' to finish")
                return True

            elif key == ord('f') or key == ord('F'):
                # Finish zone via keyboard
                if self.create_mode == "zone" and len(self.temp_points) >= 3:
                    self._finalize_new_zone(self.temp_points)
                self.temp_points.clear()
                self.create_mode = "none"
                return True

            elif key in (127, ord('d'), ord('D')):  # 127 = ASCII DEL, 'd'/'D' = delete hotkey
                x, y = getattr(self, "_last_mouse_pos", (None, None))
                if x is not None:
                    self._delete_at_point((x, y))
                return True

            elif key == 27:  # ESC inside edit mode cancels drawing (but not exit app)
                if self.create_mode != "none":
                    self.temp_points.clear()
                    self.create_mode = "none"
                    self.logger.info("Create cancelled.")
                    return True
                # else: fall through to your global exit if you prefer
                return True

        if not self.edit_mode:
            return True

        return True

    def _mouse_callback(self, event, x, y, flags, param):
        self._last_mouse_pos = (x, y)

        if not self.edit_mode or self.counter is None:
            return

        # Check for toggle button clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # Stats button region (top-left)
            if 260 <= x <= 280 and 20 <= y <= 40:
                self.show_stats = not self.show_stats
                self.logger.info(f"Stats toggled: {self.show_stats}")
                return

            # Get frame dimensions from last frame
            if hasattr(self, 'last_frame') and self.last_frame is not None:
                h, w = self.last_frame.shape[:2]
                # Controls button region (bottom-right area)
                if (w - 400) <= x <= (w - 380) and (h - 220) <= y <= (h - 200):
                    self.show_controls = not self.show_controls
                    self.logger.info(f"Controls toggled: {self.show_controls}")
                    return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_target = self._find_nearest_handle((x, y))
            self.dragging = self.drag_target is not None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._apply_drag((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self._apply_drag((x, y))
            self.dragging = False
            self.drag_target = None

        if not self.edit_mode:
            return

        if self.create_mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_points.append((x, y))
                if len(self.temp_points) == 2:
                    # we have a segment; open properties, then add
                    self._finalize_new_line(self.temp_points[0], self.temp_points[1])
                    self.temp_points.clear()
                    self.create_mode = "none"

        elif self.create_mode == "zone":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # right click = finish zone quickly
                if len(self.temp_points) >= 3:
                    self._finalize_new_zone(self.temp_points)
                self.temp_points.clear()
                self.create_mode = "none"

    def _find_nearest_handle(self, pt):
        x, y = pt
        best = None
        best_dist2 = self.drag_threshold * self.drag_threshold

        # Check line endpoints
        for name, line in self.counter.line_counters.items():
            for which, p in (("start", line.start_px), ("end", line.end_px)):
                dx = p[0] - x
                dy = p[1] - y
                d2 = dx * dx + dy * dy
                if d2 <= best_dist2:
                    best_dist2 = d2
                    best = ('line', name, which)

        # Check zone vertices
        for name, zone in self.counter.zone_counters.items():
            for idx, p in enumerate(zone.points_px):
                dx = p[0] - x
                dy = p[1] - y
                d2 = dx * dx + dy * dy
                if d2 <= best_dist2:
                    best_dist2 = d2
                    best = ('zone', name, idx)

        return best

    def _apply_drag(self, pt):
        x, y = pt
        # clamp to frame
        w, h = self.counter.frame_size
        x = int(max(0, min(w - 1, x)))
        y = int(max(0, min(h - 1, y)))

        if not self.drag_target:
            return

        kind, name, extra = self.drag_target

        if kind == 'line':
            line = self.counter.line_counters.get(name)
            if line is not None:
                line.update_endpoint(extra, (x, y))  # uses the new helper

        elif kind == 'zone':
            zone = self.counter.zone_counters.get(name)
            if zone is not None and isinstance(extra, int):
                zone.update_point(extra, (x, y))  # uses the new helper

    def _should_rollover_segment(self) -> bool:
        """
        Rollover Segment Check
        - Camera: Rotate daily at midnight
        - Video file: Rotate based on original segment logic (e.g. hourly)
        """
        # Get current time (video time for recordings, real time for live)
        if self.is_live_source:
            now_dt = datetime.now()
        else:
            now_dt = self._get_current_timestamp()

        if self.current_segment_start_dt is None:
            self.current_segment_start_dt = now_dt
            return False

            # Use the standard clock boundary (60 minutes) for both camera and files
        if self.next_split_dt is None:
            self.next_split_dt = self._compute_next_clock_boundary(now_dt, 60)

        return now_dt >= self.next_split_dt

    def _rollover_segment(self) -> None:
        """Handle hourly rollover and export."""
        # Pass video time for accurate dwell calculation
        current_time = self.video_current_time if self.video_current_time else datetime.now()
        self.counter.update_events_with_final_stats(current_time)
        window_start = self.current_segment_start_dt
        window_end = self.next_split_dt if self.next_split_dt else datetime.now()

        self.logger.info(f"Hourly segment complete: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")

        # This will now save segments into the nested folders if you update exporter paths as well
        self._export_hourly_segment(window_start, window_end)

        self.current_segment += 1
        self.counter.reset_all_counts()
        self.current_segment_start_dt = window_end
        self.next_split_dt = self._compute_next_clock_boundary(window_end, 60)

        if self.config.save_video:
            self._rotate_hourly_video(window_start, window_end)

    def _export_hourly_segment(self, window_start: datetime, window_end: datetime):
        """Export data for the completed hour"""
        try:
            self._trigger_live_export(force=True)

            counts_dict = self.counter.get_current_counts()
            # Pass video time for accurate dwell calculation
            current_time = self.video_current_time if self.video_current_time else datetime.now()
            events_dict = self.counter.get_events_summary(current_time)

            # Add hourly metadata
            events_dict["hour_start"] = window_start.isoformat()
            events_dict["hour_end"] = window_end.isoformat()
            events_dict["hour_of_day"] = window_start.hour  # 0-23 for easy sorting
            events_dict["segment_number"] = self.current_segment + 1

            # Enhanced stats
            enhanced_stats = self.stats.__dict__.copy() if hasattr(self.stats, '__dict__') else {}
            enhanced_stats['video_time_based'] = not self.is_live_source

            # Export using existing exporter
            segment_id = f"hour_{window_start.hour:02d}"

            # Get current video name if available
            video_name = None
            if hasattr(self, 'current_video_path'):
                video_name = self.current_video_path.name

            original_master_setting = self.exporter.config.enable_master_log
            self.exporter.config.enable_master_log = False
            try:
                self.exporter.export_segment_results(
                    segment_id=segment_id,
                    counts=counts_dict,
                    events=events_dict,
                    stats=enhanced_stats,
                    video_source=video_name
                )
            finally:
                self.exporter.config.enable_master_log = original_master_setting

            self.logger.info(f"Exported hour {window_start.hour:02d}:00 data")

        except Exception as e:
            self.logger.error(f"Failed to export hourly data: {e}")

    def _rotate_hourly_video(self, window_start: datetime, window_end: datetime):
        """Rotate video file at hour boundary"""
        try:
            # Close current writer
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None

            if self.config.save_video:
                date_str = window_end.strftime("%Y%m%d")
                hour_str = window_end.strftime("%H")

                if getattr(self.config, 'is_camera', False):
                    filename = f"live_{date_str}_{hour_str}00.mp4"
                else:
                    source_name = Path(self.config.input_source).stem
                    filename = f"{source_name}_{hour_str}00.mp4"

                # _initialize_video_writer now handles the Month/Day folder creation
                self._initialize_video_writer(filename)

        except Exception as e:
            self.logger.error(f"Failed to rotate hourly video: {e}")

    def _reset_segment(self) -> None:
        """Reset segment tracking for new hour"""
        self.segment_start_time = time.time()

        # Reset the export tracker because self.counter events are about to be cleared
        self.last_exported_event_count = 0
        self.active_zone_events = {}  # Reset active zone tracking

        if hasattr(self, 'video_frame_number'):
            self.segment_start_frame = self.video_frame_number

        # Set segment start to current video/real time
        if self.is_live_source:
            self.current_segment_start_dt = datetime.now()
        else:
            self.current_segment_start_dt = self._get_current_timestamp()

    def _reset_segment(self) -> None:
        """Reset segment with proper timestamp"""
        self.segment_start_time = time.time()  # Keep for processing metrics

        # Track starting frame for duration-based rollover
        if hasattr(self, 'video_frame_number'):
            self.segment_start_frame = self.video_frame_number

        # Use video time for segment boundaries
        if self.is_live_source:
            self.current_segment_start_dt = datetime.now()
        else:
            self.current_segment_start_dt = self.video_current_time if self.video_current_time else datetime.now()

    def _finalize_current_segment(self) -> None:
        """Export the last open segment on shutdown."""
        try:
            # Pass video time for accurate dwell calculation
            current_time = self.video_current_time if self.video_current_time else datetime.now()
            
            # UPDATE EVENTS WITH FINAL STATS BEFORE FINAL EXPORT
            self.counter.update_events_with_final_stats(current_time)

            counts_dict = self.counter.get_current_counts()
            events_dict = self.counter.get_events_summary(current_time)

            # Determine final window
            window_start = self.current_segment_start_dt
            window_end = self.next_split_dt if (
                    self.segment_split_minutes > 0 and self.align_segments_to_clock
            ) else datetime.now()
            if window_end < window_start:
                window_end = datetime.now()

            # Attach meta
            events_dict["_window_start"] = window_start.isoformat()
            events_dict["_window_end"] = window_end.isoformat()

            # FIXED: Properly serialize object_states with all speed and dwell time data
            enhanced_stats = self.stats.__dict__.copy() if hasattr(self.stats, '__dict__') else {}

            # Serialize object states properly
            serialized_states = {}
            for track_id, obj_state in self.counter.object_states.items():
                if hasattr(obj_state, '__dict__'):
                    # Convert ObjectState to dict
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
                else:
                    serialized_states[track_id] = obj_state

            enhanced_stats['object_states'] = serialized_states
            enhanced_stats['speed_units'] = getattr(self.counter, 'speed_units', 'pxps')
            enhanced_stats['meters_per_pixel'] = getattr(self.counter, 'mpp', 0.0)

            # Force final live export to master_log before segment export
            self._trigger_live_export(force=True)
            
            # Disable master_log export here since _trigger_live_export already handled it
            # (prevents duplicate entries in master_log)
            original_master_setting = self.exporter.config.enable_master_log
            self.exporter.config.enable_master_log = False
            try:
                self.exporter.export_segment_results(
                    segment_id=self.current_segment,
                    counts=counts_dict,
                    events=events_dict,
                    stats=enhanced_stats
                )
            finally:
                self.exporter.config.enable_master_log = original_master_setting

            # Final heatmap & writer cleanup - use video time, not system time
            end_time = self.video_current_time if self.video_current_time else datetime.now()
            self._flush_heatmap(self.current_segment_start_dt, end_time, suffix="final")
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                self.logger.info("Closed final video segment")

        except Exception as e:
            self.logger.warning(f"Finalize current segment failed: {e}")

    def _update_stats(self, new_events_count: int):
        """Update processing statistics"""
        self.stats.frames_processed += 1
        self.stats.total_events += new_events_count
        self.stats.processing_time = time.time() - self.stats.start_time
        self.stats.update_fps()

    def _log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.stats.start_time
        self.logger.info(
            f"Progress: {self.stats.frames_processed} frames, "
            f"{self.stats.fps:.1f} FPS, "
            f"{elapsed / 60:.1f}m elapsed, "
            f"{self.stats.total_events} events"
        )

    def _save_current_stats(self):
        """Save current statistics to file"""
        try:
            stats_data = {
                "timestamp": time.time(),
                "processing_stats": self.stats.__dict__,
                "counts": self.counter.get_current_counts(),
                "events": self.counter.get_events_summary()
            }

            self.exporter.export_live_stats(stats_data)
            self.logger.info("Current stats saved")

        except Exception as e:
            self.logger.error(f"Failed to save current stats: {e}")

    def _combine_video_stats(self, video_results: List[Dict]) -> Dict:
        """Combine statistics from multiple videos"""
        combined = {
            "total_frames": sum(r["stats"].frames_processed for r in video_results),
            "total_events": sum(r["stats"].total_events for r in video_results),
            "avg_fps": np.mean([r["stats"].fps for r in video_results]),
            "total_processing_time": sum(r["stats"].processing_time for r in video_results)
        }
        return combined

    def _get_final_results(self) -> Dict:
        """Get final processing results"""
        return {
            "stats": self.stats,
            "final_counts": self.counter.get_current_counts(),
            "events_summary": self.counter.get_events_summary(),
            "segments_processed": self.current_segment + 1
        }

    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_requested = True
        self.is_running = False

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

            # Cleanup training mode
            if self.training_capture:
                self.training_capture.cleanup()
            
            # Flush any queued master log events before exit
            try:
                from results_export import get_master_log_writer
                writer = get_master_log_writer()
                self.logger.info("Flushing master log queue...")
                writer.flush_and_wait(timeout=15.0)
                self.logger.info("Master log queue flushed")
            except Exception as e:
                self.logger.warning(f"Could not flush master log queue: {e}")

            self.logger.info("Video processor cleanup completed")

        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

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

    def _ensure_tk_top(self):
        if self.tk_root is None or not self.tk_root.winfo_exists():
            self.tk_root = tk.Tk()
            self.tk_root.withdraw()
            self.tk_root.attributes('-topmost', True)
        return self.tk_root

    def _finalize_new_line(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        # Ask for name/direction/classes via existing dialog
        try:
            root = self._ensure_tk_top()
            # name
            name = simpledialog.askstring("Line Name", "Enter a name for this line:", parent=root)
            if not name:
                self.logger.info("Line creation cancelled (no name).")
                return

            # properties dialog
            dialog = LinePropertiesDialog(root, self.config, self.detection_engine.class_names)
            props = dialog.show()  # expects dict with keys: direction, classes (list[int])
            if not props:
                self.logger.info("Line creation cancelled in properties dialog.")
                return

            # Build dataclass with NORMALIZED coords
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            start_norm = (p1[0] / w, p1[1] / h)
            end_norm = (p2[0] / w, p2[1] / h)

            new_line = CountingLine(
                name=name,
                start_norm=start_norm,
                end_norm=end_norm,
                direction=props["direction"],
                classes=list(map(int, props["classes"])) if props.get("classes") else [],
                enabled=True,
                poi_mode=props.get("poi_mode", "center")
            )

            # Add to runtime counter + config
            self.counter.add_line(new_line)
            self.config.lines_config.append(new_line)
            self.logger.info(
                f"Added new line '{name}' ({props['direction']}) with {len(new_line.classes)} class filters.")

        except Exception as e:
            self.logger.error(f"Finalize new line failed: {e}")

    def _finalize_new_zone(self, pts: list[tuple[int, int]]) -> None:
        try:
            root = self._ensure_tk_top()
            name = simpledialog.askstring("Zone Name", "Enter a name for this zone:", parent=root)
            if not name:
                self.logger.info("Zone creation cancelled (no name).")
                return

            dialog = ZonePropertiesDialog(root, self.detection_engine.class_names)
            props = dialog.show()  # expects dict with key: classes (list[int])
            if not props:
                self.logger.info("Zone creation cancelled in properties dialog.")
                return

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            points_norm = [(x / w, y / h) for (x, y) in pts]

            new_zone = CountingZone(
                name=name,
                points_norm=points_norm,
                classes=list(map(int, props["classes"])) if props.get("classes") else [],
                enabled=True,
                track_max_concurrent=bool(props.get("track_max_concurrent", False)),
                show_peak_overlay=bool(props.get("show_peak_overlay", True)),
                poi_mode=props.get("poi_mode", "center")
            )

            self.counter.add_zone(new_zone)
            self.config.zones_config.append(new_zone)
            self.logger.info(
                f"Added new zone '{name}' with {len(new_zone.classes)} class filters and {len(points_norm)} vertices.")
        except Exception as e:
            self.logger.error(f"Finalize new zone failed: {e}")

    def _delete_at_point(self, pt: tuple[int, int]) -> None:
        # Prefer deleting zone (if inside) before nearest line
        zn = self.counter.zone_contains_point(pt)
        if zn:
            removed = self.counter.remove_zone(zn)
            if removed:
                # also remove from config
                self.config.zones_config = [z for z in self.config.zones_config if z.name != zn]
                self.logger.info(f"Deleted zone '{zn}'.")
                return

        ln = self.counter.find_nearest_line(pt, max_dist_px=self.drag_threshold * 1.5)
        if ln:
            removed = self.counter.remove_line(ln)
            if removed:
                self.config.lines_config = [l for l in self.config.lines_config if l.name != ln]
                self.logger.info(f"Deleted line '{ln}'.")
                return

        self.logger.info("Nothing to delete at cursor.")

    def _compute_next_clock_boundary(self, base_dt: datetime, interval_min: int) -> datetime:
        """
        Return the next hour boundary (e.g., 13:00, 14:00, 15:00)
        Simplified for fixed 60-minute intervals
        """
        # Round up to next hour
        next_hour = base_dt.replace(minute=0, second=0, microsecond=0)
        if base_dt.minute > 0 or base_dt.second > 0 or base_dt.microsecond > 0:
            next_hour += timedelta(hours=1)
        else:
            # If exactly on the hour, move to next hour
            next_hour += timedelta(hours=1)
        return next_hour

    def _flush_heatmap(self, window_start_dt, window_end_dt, suffix=""):
        if self.heatmap_acc is None:
            return
        try:
            frame = getattr(self, "last_frame", None)
            if frame is None:
                self.logger.warning("[HEATMAP] flush skipped: no last_frame available")
                return
            # Store start as float seconds
            self.heatmap_acc.last_emit_t = window_start_dt.timestamp()
            # Pass end as float seconds; suffix optional (e.g., "final")
            out_path = self.heatmap_acc.render_and_save(
                frame_bgr=frame,
                label=None if not suffix else suffix,  # or keep label separate if you prefer
                when=window_end_dt.timestamp(),
                suffix=None if not suffix else suffix
            )
            self.logger.info(f"[HEATMAP] snapshot saved: {out_path}")
        except Exception as e:
            self.logger.warning(f"[HEATMAP] flush failed: {e}")

    def fit_center(self, name, w, h, frac=0.9):
        r = tk.Tk();
        r.withdraw()
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        s = min((sw * frac) / w, (sh * frac) / h, 1.0)
        nw, nh = int(w * s), int(h * s)
        x, y = (sw - nw) // 2, (sh - nh) // 2
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, nw, nh)
        cv2.moveWindow(name, x, y)

    def _background_append_task(self, new_events: List, segment_id: str, video_name: str):
        """ Background thread to append events to master log without blocking."""
        if not new_events:
            return

        if self.export_lock.acquire(False):
            try:
                self.exporter._append_to_master_log(
                    new_events,
                    video_source=video_name,
                    segment_id=segment_id
                )
            except Exception as e:
                self.logger.error(f"Live master log update failed: {e}")
            finally:
                self.export_lock.release()

    def _trigger_live_export(self, force: bool = False):
        """
        Check if we should export new events to master_log.
        
        For zone events:
        - New zone events are appended immediately
        - Dwell times are updated periodically while object is in zone
        - Final dwell time is set when object exits or at segment end
        """
        now = time.time()

        # Only run if interval passed OR forced (e.g. at end of segment)
        if force or (now - self.last_live_export_time >= self.live_export_interval):
            # Pass video time for accurate dwell calculation
            current_time = self.video_current_time if self.video_current_time else datetime.now()
            
            # Update events with current video time before getting summary
            self.counter.update_events_with_final_stats(current_time)

            # Get all events currently in memory
            summary = self.counter.get_events_summary(current_time)
            all_events = summary['events']
            current_total = len(all_events)

            segment_id = f"hour_{self.current_segment_start_dt.hour:02d}"
            video_name = "LIVE CAMERA" if self.is_live_source else self.current_video_path.name

            # STEP 1: Append NEW events (both line and zone)
            if current_total > self.last_exported_event_count:
                new_events_slice = all_events[self.last_exported_event_count:]
                
                # Track new zone events for future dwell updates
                for event in new_events_slice:
                    if event.get('zone_name'):
                        key = (event.get('track_id'), event.get('zone_name'))
                        self.active_zone_events[key] = True
                
                # Append all new events to master log via background thread
                import copy
                events_to_export = copy.deepcopy(new_events_slice)
                if events_to_export:
                    threading.Thread(
                        target=self._background_append_task,
                        args=(events_to_export, segment_id, video_name),
                        daemon=True
                    ).start()
                
                self.last_exported_event_count = current_total

            # STEP 2: Update dwell times for active zone events
            if self.active_zone_events:
                zone_updates = []
                keys_to_remove = []
                
                for (track_id, zone_name), is_active in list(self.active_zone_events.items()):
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
                
                # Update dwell times in master log via background thread
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
                
                # Remove finalized zone events from tracking
                for key in keys_to_remove:
                    self.active_zone_events.pop(key, None)

            self.last_live_export_time = now
