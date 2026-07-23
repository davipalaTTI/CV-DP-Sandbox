import cv2
import tkinter as tk
from tkinter import simpledialog

from config_manager import CountingLine, CountingZone

# If your GUI dialogs are in a different path, make sure this matches your structure
from gui.gui_setup import LinePropertiesDialog, ZonePropertiesDialog


class InteractionHandler:
    """Handles all user inputs: keyboard, mouse, and UI dialogs."""

    def __init__(self, processor_context):
        # We store a reference to the main VideoProcessor so we can read/change its state
        self.p = processor_context

    def handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input during live processing
        Returns:
            True to continue processing, False to stop
        """
        if key == -1:
            return True

        if key == 27:  # ESC
            return False

        elif key == ord(' '):  # SPACE - pause/resume
            self.p.is_paused = not self.p.is_paused
            self.p.logger.info(f"Processing {'paused' if self.p.is_paused else 'resumed'}")
            while self.p.is_paused:
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    self.p.is_paused = False
                    break

        elif key == ord('r') or key == ord('R'):  # Reset counts
            if self.p.counter:
                self.p.counter.reset_all_counts()
                self.p.logger.info("Counts reset")

        elif key == ord('s') or key == ord('S'):  # Save stats
            self.p._save_current_stats()

        elif key == ord('m') or key == ord('M'):  # Toggle stats display
            self.p.show_stats = not self.p.show_stats
            self.p.logger.info(f"Stats display: {'ON' if self.p.show_stats else 'OFF'}")
            self.p._show_notification(f"Stats: {'ON' if self.p.show_stats else 'OFF'}", duration=1.0)
            return True

        elif key == ord('c') or key == ord('C'):  # Toggle controls display
            self.p.show_controls = not self.p.show_controls
            self.p.logger.info(f"Controls display: {'ON' if self.p.show_controls else 'OFF'}")
            self.p._show_notification(f"Controls: {'ON' if self.p.show_controls else 'OFF'}", duration=1.0)
            return True

        elif key == ord('v') or key == ord('V'):  # Toggle full live video view
            if hasattr(self.p, 'toggle_live_view'):
                self.p.toggle_live_view()
            return True

        if key == ord('e') or key == ord('E'):
            if hasattr(self.p, 'live_view_enabled') and not self.p.live_view_enabled:
                self.p.toggle_live_view()
            self.p.edit_mode = not self.p.edit_mode
            self.p.create_mode = "none"
            self.p.temp_points.clear()
            self.p.logger.info(f"EDIT mode: {'ON' if self.p.edit_mode else 'OFF'}")
            return True

        # Frame skip adjustment
        if ord('1') <= key <= ord('5'):
            new_skip = key - ord('0')
            old_skip = self.p.frame_skip
            self.p.frame_skip = new_skip
            self.p.logger.info(f"Frame skip changed from {old_skip} to {new_skip}")
            if hasattr(self.p, 'last_frame'):
                self.p._show_notification(f"Processing every {new_skip} frame(s)", duration=2.0)
            return True

        # Toggle interpolation
        if key == ord('i') or key == ord('I'):
            self.p.interpolate_tracks = not self.p.interpolate_tracks
            state = "ON" if self.p.interpolate_tracks else "OFF"
            self.p.logger.info(f"Track interpolation: {state}")
            self.p._show_notification(f"Track interpolation: {state}", duration=2.0)
            return True

        elif key == ord('t') or key == ord('T'):  # Toggle training mode
            if self.p.training_capture:
                is_active = self.p.training_capture.toggle()
                status = "ON" if is_active else "OFF"
                self.p.logger.info(f"Training mode: {status}")
                self.p._show_notification(f"Training mode: {status}", duration=2.0)
            return True

        elif key == ord('+') or key == ord('='):  # Increase capture interval
            if self.p.training_capture:
                current = self.p.training_capture.config.capture_interval_seconds
                new_interval = min(60, current + 1)
                self.p.training_capture.set_interval(new_interval)
                self.p._show_notification(f"Capture interval: {new_interval}s", duration=2.0)
            return True

        elif key == ord('-') or key == ord('_'):  # Decrease capture interval
            if self.p.training_capture:
                current = self.p.training_capture.config.capture_interval_seconds
                new_interval = max(0.5, current - 1)
                self.p.training_capture.set_interval(new_interval)
                self.p._show_notification(f"Capture interval: {new_interval}s", duration=2.0)
            return True

        # --- Editor keys (Only when edit_mode is ON) ---
        if self.p.edit_mode:
            if key == ord('n') or key == ord('N'):
                self.p.create_mode = "line"
                self.p.temp_points.clear()
                self.p.logger.info("Create LINE: click 2 points")
                return True

            elif key == ord('z') or key == ord('Z'):
                self.p.create_mode = "zone"
                self.p.temp_points.clear()
                self.p.logger.info("Create ZONE: left-click to add vertices, right-click or 'F' to finish")
                return True

            elif key == ord('f') or key == ord('F'):
                if self.p.create_mode == "zone" and len(self.p.temp_points) >= 3:
                    self._finalize_new_zone(self.p.temp_points)
                self.p.temp_points.clear()
                self.p.create_mode = "none"
                return True

            elif key in (127, ord('d'), ord('D')):  # 127 = ASCII DEL
                x, y = getattr(self.p, "_last_mouse_pos", (None, None))
                if x is not None:
                    self._delete_at_point((x, y))
                return True

            elif key == 27:  # ESC inside edit mode cancels drawing
                if self.p.create_mode != "none":
                    self.p.temp_points.clear()
                    self.p.create_mode = "none"
                    self.p.logger.info("Create cancelled.")
                    return True
                return True

        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Bound to OpenCV window to handle mouse clicks and drags"""
        self.p._last_mouse_pos = (x, y)

        # 1. GLOBAL UI CLICKS
        if event == cv2.EVENT_LBUTTONDOWN:
            if 260 <= x <= 280 and 20 <= y <= 40:
                self.p.show_stats = not self.p.show_stats
                self.p.logger.info(f"Stats toggled: {self.p.show_stats}")
                return

            if hasattr(self.p, 'last_frame') and self.p.last_frame is not None:
                h, w = self.p.last_frame.shape[:2]
                if (w - 400) <= x <= (w - 380) and (h - 220) <= y <= (h - 200):
                    self.p.show_controls = not self.p.show_controls
                    self.p.logger.info(f"Controls toggled: {self.p.show_controls}")
                    return

                    # 2. EDIT MODE CLICKS
        if not self.p.edit_mode or self.p.counter is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.p.drag_target = self._find_nearest_handle((x, y))
            self.p.dragging = self.p.drag_target is not None

        elif event == cv2.EVENT_MOUSEMOVE and self.p.dragging:
            self._apply_drag((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.p.dragging:
            self._apply_drag((x, y))
            self.p.dragging = False
            self.p.drag_target = None

        # Line/Zone Creation Logic
        if self.p.create_mode == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.p.temp_points.append((x, y))
                if len(self.p.temp_points) == 2:
                    self._finalize_new_line(self.p.temp_points[0], self.p.temp_points[1])
                    self.p.temp_points.clear()
                    self.p.create_mode = "none"

        elif self.p.create_mode == "zone":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.p.temp_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.p.temp_points) >= 3:
                    self._finalize_new_zone(self.p.temp_points)
                self.p.temp_points.clear()
                self.p.create_mode = "none"

    def _find_nearest_handle(self, pt):
        x, y = pt
        best = None
        best_dist2 = self.p.drag_threshold * self.p.drag_threshold

        for name, line in self.p.counter.line_counters.items():
            for which, p in (("start", line.start_px), ("end", line.end_px)):
                dx, dy = p[0] - x, p[1] - y
                d2 = dx * dx + dy * dy
                if d2 <= best_dist2:
                    best_dist2 = d2
                    best = ('line', name, which)

        for name, zone in self.p.counter.zone_counters.items():
            for idx, p in enumerate(zone.points_px):
                dx, dy = p[0] - x, p[1] - y
                d2 = dx * dx + dy * dy
                if d2 <= best_dist2:
                    best_dist2 = d2
                    best = ('zone', name, idx)
        return best

    def _apply_drag(self, pt):
        x, y = pt
        w, h = self.p.counter.frame_size
        x = int(max(0, min(w - 1, x)))
        y = int(max(0, min(h - 1, y)))

        if not self.p.drag_target:
            return

        kind, name, extra = self.p.drag_target

        if kind == 'line':
            line = self.p.counter.line_counters.get(name)
            if line is not None:
                line.update_endpoint(extra, (x, y))

        elif kind == 'zone':
            zone = self.p.counter.zone_counters.get(name)
            if zone is not None and isinstance(extra, int):
                zone.update_point(extra, (x, y))

    def _ensure_tk_top(self):
        if self.p.tk_root is None or not self.p.tk_root.winfo_exists():
            self.p.tk_root = tk.Tk()
            self.p.tk_root.withdraw()
            self.p.tk_root.attributes('-topmost', True)
        return self.p.tk_root

    def _finalize_new_line(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        try:
            root = self._ensure_tk_top()
            name = simpledialog.askstring("Line Name", "Enter a name for this line:", parent=root)
            if not name:
                self.p.logger.info("Line creation cancelled (no name).")
                return

            dialog = LinePropertiesDialog(root, self.p.config, self.p.detection_engine.class_names)
            props = dialog.show()
            if not props:
                self.p.logger.info("Line creation cancelled in properties dialog.")
                return

            w = int(self.p.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.p.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

            self.p.counter.add_line(new_line)
            self.p.config.lines_config.append(new_line)
            self.p.logger.info(
                f"Added new line '{name}' ({props['direction']}) with {len(new_line.classes)} class filters.")

        except Exception as e:
            self.p.logger.error(f"Finalize new line failed: {e}")

    def _finalize_new_zone(self, pts: list[tuple[int, int]]) -> None:
        try:
            root = self._ensure_tk_top()
            name = simpledialog.askstring("Zone Name", "Enter a name for this zone:", parent=root)
            if not name:
                self.p.logger.info("Zone creation cancelled (no name).")
                return

            dialog = ZonePropertiesDialog(root, self.p.detection_engine.class_names)
            props = dialog.show()
            if not props:
                self.p.logger.info("Zone creation cancelled in properties dialog.")
                return

            w = int(self.p.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.p.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

            self.p.counter.add_zone(new_zone)
            self.p.config.zones_config.append(new_zone)
            self.p.logger.info(
                f"Added new zone '{name}' with {len(new_zone.classes)} class filters and {len(points_norm)} vertices.")
        except Exception as e:
            self.p.logger.error(f"Finalize new zone failed: {e}")

    def _delete_at_point(self, pt: tuple[int, int]) -> None:
        zn = self.p.counter.zone_contains_point(pt)
        if zn:
            removed = self.p.counter.remove_zone(zn)
            if removed:
                self.p.config.zones_config = [z for z in self.p.config.zones_config if z.name != zn]
                self.p.logger.info(f"Deleted zone '{zn}'.")
                return

        ln = self.p.counter.find_nearest_line(pt, max_dist_px=self.p.drag_threshold * 1.5)
        if ln:
            removed = self.p.counter.remove_line(ln)
            if removed:
                self.p.config.lines_config = [l for l in self.p.config.lines_config if l.name != ln]
                self.p.logger.info(f"Deleted line '{ln}'.")
                return

        self.p.logger.info("Nothing to delete at cursor.")