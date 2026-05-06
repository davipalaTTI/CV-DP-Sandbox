import cv2
import time
import numpy as np
from datetime import datetime
from typing import List

from config_manager import AppConfig
from core.detection_engine import Detection


class Visualizer:
    """Handles all OpenCV drawing, overlays, and UI rendering."""

    def __init__(self, config: AppConfig):
        self.config = config

    def draw_all(self, processor_context, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Master drawing function.
        Takes the processor context to access current states (stats, UI toggles, etc.)
        """
        # 1. Draw detection bounding boxes
        for detection in detections:
            self._draw_detection(frame, detection, processor_context)

        # 2. Draw counting overlays (lines, zones, exclusions)
        frame = processor_context.counter.draw_overlays(frame, show_counts=True)

        # 3. Draw statistics overlay
        frame = self._draw_stats_overlay(frame, processor_context)

        # 4. Draw edit mode overlays
        self._draw_edit_mode(frame, processor_context)

        # 5. Draw notifications
        self._draw_notification(frame, processor_context)

        # 6. Draw minimize/maximize buttons
        self._draw_ui_toggle_buttons(frame, processor_context)

        # 7. Add live controls panel
        frame = self._add_live_controls(frame, processor_context)

        # 8. Add frame skip indicator
        if hasattr(processor_context, 'frame_skip') and processor_context.frame_skip > 1:
            skip_text = f"Skip: {processor_context.frame_skip}x"
            if hasattr(processor_context, 'interpolate_tracks'):
                skip_text += f" ({'Interp ON' if processor_context.interpolate_tracks else 'Interp OFF'})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size = cv2.getTextSize(skip_text, font, font_scale, thickness)[0]

            cv2.rectangle(frame, (10, 60), (15 + text_size[0], 80 + text_size[1]), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 60), (15 + text_size[0], 80 + text_size[1]), (255, 165, 0), 1)
            cv2.putText(frame, skip_text, (12, 75), font, font_scale, (255, 165, 0), thickness)

        # 9. Draw training mode indicator
        if processor_context.training_capture and processor_context.training_capture.is_active:
            cv2.putText(frame, "TRAINING", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status = processor_context.training_capture.get_status()
            status_text = f"Captures: {status['captures']} | Interval: {status['interval_seconds']}s"
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def _draw_detection(self, frame: np.ndarray, detection: Detection, context):
        """Draw a single detection on the frame"""
        x1, y1, x2, y2 = detection.bbox

        # Choose color based on class
        colors = [(163, 207, 167), (247, 220, 236), (255, 225, 148), (255, 241, 222),
                  (146, 192, 212), (235, 185, 138), (187, 135, 170), (125, 122, 179)]
        color = colors[detection.class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label
        label = f"{detection.class_name}"
        if detection.track_id is not None:
            label += f"#{detection.track_id}"
        label += f" {detection.confidence:.2f}"

        # append speed if enabled and available
        if getattr(self.config, "enable_speed", False) and getattr(context.counter, "annotate_speed", True):
            spd_map = getattr(context.counter, "_last_speeds", {})
            spd = spd_map.get(detection.track_id, None)
            if spd is not None:
                units = str(getattr(self.config, "speed_units", "pxps"))
                label += f" {spd:.1f} {units}"

        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), (int(x1) + label_size[0], int(y1)), color, -1)

        # Label text
        cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Center point dot
        cv2.circle(frame, detection.center_point, 3, color, -1)

    def _draw_stats_overlay(self, frame: np.ndarray, context) -> np.ndarray:
        """Draw performance statistics and timestamp overlay"""

        # Get current timestamp - use video time if available
        if context.is_live_source:
            current_time = datetime.now()
            time_source = "LIVE"
        else:
            current_time = context.video_current_time if context.video_current_time else datetime.now()
            time_source = "VIDEO"

        timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Only draw full stats if visible
        if context.show_stats:
            # Add frame skip info
            effective_fps = context.stats.fps * context.frame_skip if context.frame_skip > 1 else context.stats.fps

            # Prepare stats text - include time source indicator
            stats_text = [
                f"FPS: {context.stats.fps:.1f} (Effective: {effective_fps:.1f})",
                f"Frame Skip: {context.frame_skip}x",
                f"Frames: {context.stats.frames_processed}",
                f"Detections: {context.stats.total_detections}",
                f"Device: {context.detection_engine.device}",
                f"Events: {context.stats.total_events}",
                f"Segment: {context.current_segment}",
                f"Objects: {len(context.counter.object_states)}",
                f"Time Source: {time_source}"
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
        if context.segment_split_minutes > 0 and context.align_segments_to_clock:
            if context.current_segment_start_dt:
                window_text = f"Window: {context.current_segment_start_dt.strftime('%H:%M')}"
                if context.next_split_dt:
                    window_text += f" - {context.next_split_dt.strftime('%H:%M')}"

                # Draw below timestamp
                window_y = timestamp_y + 25
                (window_width, _), _ = cv2.getTextSize(
                    window_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                window_x = frame.shape[1] - window_width - margin - 10

                cv2.putText(frame, window_text, (window_x, window_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def _draw_edit_mode(self, frame: np.ndarray, context):
        if context.edit_mode and context.create_mode != "none":
            if context.create_mode == "line" and len(context.temp_points) == 1:
                cv2.line(frame, context.temp_points[0], tuple(context._last_mouse_pos), (0, 255, 255), 2)
                cv2.circle(frame, context.temp_points[0], 6, (0, 255, 255), -1)
                cv2.putText(frame, "Click second point",
                            (context.temp_points[0][0], context.temp_points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            elif context.create_mode == "zone":
                for p in context.temp_points:
                    cv2.circle(frame, p, 4, (255, 255, 0), -1)
                if len(context.temp_points) >= 2:
                    pts = np.array(context.temp_points, np.int32)
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                # helper line to cursor
                if context.temp_points:
                    cv2.line(frame, context.temp_points[-1], tuple(context._last_mouse_pos), (255, 255, 0), 1)

    def _draw_notification(self, frame: np.ndarray, context):
        if hasattr(context, 'notification') and context.notification:
            elapsed = time.time() - context.notification['start_time']
            if elapsed < context.notification['duration']:
                # Draw notification banner at top center
                text = context.notification['text']
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
                context.notification = None

    def _draw_ui_toggle_buttons(self, frame: np.ndarray, context):
        """Draw minimize/maximize buttons for UI panels"""
        h, w = frame.shape[:2]

        # Stats toggle button (top-left corner)
        stats_btn_x, stats_btn_y = 260, 20
        stats_btn_size = 20

        # Draw button background
        color = (0, 200, 0) if context.show_stats else (100, 100, 100)
        cv2.rectangle(frame,
                      (stats_btn_x, stats_btn_y),
                      (stats_btn_x + stats_btn_size, stats_btn_y + stats_btn_size),
                      color, -1)
        cv2.rectangle(frame,
                      (stats_btn_x, stats_btn_y),
                      (stats_btn_x + stats_btn_size, stats_btn_y + stats_btn_size),
                      (255, 255, 255), 1)

        # Draw minimize/maximize icon
        icon_text = "-" if context.show_stats else "+"
        cv2.putText(frame, icon_text,
                    (stats_btn_x + 6, stats_btn_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Controls toggle button (bottom-right corner)
        controls_btn_x = w - 400
        controls_btn_y = h - 220

        color = (0, 200, 0) if context.show_controls else (100, 100, 100)
        cv2.rectangle(frame,
                      (controls_btn_x, controls_btn_y),
                      (controls_btn_x + stats_btn_size, controls_btn_y + stats_btn_size),
                      color, -1)
        cv2.rectangle(frame,
                      (controls_btn_x, controls_btn_y),
                      (controls_btn_x + stats_btn_size, controls_btn_y + stats_btn_size),
                      (255, 255, 255), 1)

        icon_text = "-" if context.show_controls else "+"
        cv2.putText(frame, icon_text,
                    (controls_btn_x + 6, controls_btn_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _add_live_controls(self, frame: np.ndarray, context) -> np.ndarray:
        """Add live control instructions for camera mode"""

        if context.show_controls:
            controls = [
                "Controls:",
                "ESC - Exit   SPACE - Pause/Resume",
                "R - Reset counts   S - Save stats",
                f"E - Edit Mode - {'ON' if context.edit_mode else 'OFF'}",
                "M - Toggle stats   C - Toggle controls",
            ]
            if context.training_capture:
                controls.append(
                    f"T - Training Mode - {'Active' if context.training_capture and context.training_capture.is_active else 'Inactive'}")
                controls.append("+/- - Adjust training interval")
                controls.append("1-5 - Set frame skip (process every Nth frame)")
                controls.append("I - Toggle track interpolation")
                controls.append(
                    f"Current: Skip={context.frame_skip}, Interp={'ON' if context.interpolate_tracks else 'OFF'}")

            if context.edit_mode:
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