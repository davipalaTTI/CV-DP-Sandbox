"""
Interactive GUI Setup Module

Handles the interactive setup phase where users:
- Draw counting lines with direction and class filtering
- Draw exclusion zones and counting zones
- Configure properties for each element
- Preview their setup before processing
"""

import cv2
import numpy as np
from tkinter import messagebox, simpledialog
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import tkinter as tk

from config_manager import AppConfig, CountingLine, CountingZone, ExclusionZone
from utils.video_utils import load_source_preview


@dataclass
class GUIState:
    """Current state of the GUI interaction"""
    selection_mode: str = "line"  # "line", "zone", "exclusion"
    current_mouse_pos: Tuple[int, int] = (0, 0)
    current_line: List[Tuple[int, int]] = None
    current_zone: List[Tuple[int, int]] = None
    current_ref: list = None
    confirmed: bool = False

    def __post_init__(self):
        if self.current_line is None:
            self.current_line = []
        if self.current_zone is None:
            self.current_zone = []
        if self.current_ref is None:
            self.current_ref = []


class InteractiveGUI:
    """Interactive GUI for setting up counting lines and zones"""

    def __init__(
        self,
        config: AppConfig,
        class_names: Dict[int, str],
        preview_frame: Optional[np.ndarray] = None,
        dialog_parent: Optional[tk.Misc] = None,
    ):
        self.config = config
        self.class_names = class_names
        self.logger = logging.getLogger(__name__)

        # Display settings
        self.display_width = config.display_width
        self.display_height = config.display_height

        # GUI state
        self.state = GUIState()
        self.scale_factor = 1.0
        self._pending_actions: List[str] = []

        # Configuration storage
        self.lines_config = []
        self.zones_config = []
        self.exclusion_zones = []

        # UI elements
        self.buttons = self._define_buttons()
        self.preview_frame = None
        self._initial_preview_frame = (
            preview_frame.copy() if preview_frame is not None else None
        )
        self.original_frame_size = None

        self.dialog_root = dialog_parent
        self._owns_dialog_root = False
        self._dialog_parent_state = None
        self._dialog_parent_topmost = None

        self.logger.info("InteractiveGUI initialized")

    def _ensure_dialog_root(self) -> tk.Misc:
        try:
            if self.dialog_root is not None and self.dialog_root.winfo_exists():
                return self.dialog_root
        except tk.TclError:
            self.dialog_root = None

        if self.dialog_root is None:
            self.dialog_root = tk.Tk()
            self.dialog_root.withdraw()
            self.dialog_root.attributes('-topmost', True)
            self._owns_dialog_root = True
        return self.dialog_root

    def _define_buttons(self) -> Dict[str, Dict]:
        """Define UI buttons and their properties"""
        buttons = {
            "line": {
                "rect": (10, 10, 100, 50),
                "text": "Line",
                "color": (0, 255, 0),
                "text_color": (255, 255, 255)
            },
            "exclusion": {
                "rect": (110, 10, 200, 50),
                "text": "Exclusion",
                "color": (0, 0, 255),
                "text_color": (255, 255, 255)
            },
            "confirm": {
                "rect": (210, 10, 300, 50),
                "text": "Confirm",
                "color": (255, 0, 0),
                "text_color": (255, 255, 255)
            },
            "ref": {
                "rect": (510, 10, 600, 50),
                "text": "Ref",
                "color": (128, 0, 255),
                "text_color": (255, 255, 255)
            }
        }

        if self.config.enable_zones:
            buttons.update({
                "zone": {
                    "rect": (310, 10, 400, 50),
                    "text": "Zone",
                    "color": (255, 255, 0),
                    "text_color": (0, 0, 0)
                },
                "finish_zone": {
                    "rect": (410, 10, 500, 50),
                    "text": "Finish Zone",
                    "color": (255, 165, 0),
                    "text_color": (255, 255, 255)
                }
            })

        return buttons

    def run_setup(self) -> Tuple[List[CountingLine], List[CountingZone], List[ExclusionZone]]:
        """
        Run the interactive setup process

        Returns:
            Tuple of (lines_config, zones_config, exclusion_zones)
        """
        try:
            # Load preview frame
            if not self._load_preview_frame():
                self.logger.error("Failed to load preview frame")
                return [], [], []  # lines, zones, exclusion_zones

            # Run interactive loop
            self._run_interactive_loop()

            # Convert to normalized coordinates
            self._normalize_configurations()

            # Convert to dataclass objects
            lines_config = self._create_line_configs()
            zones_config = self._create_zone_configs()
            exclusion_zones = self._create_exclusion_configs()  # Add this

            return lines_config, zones_config, exclusion_zones  # Return all three

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return [], [], []

        finally:
            self._cleanup()

    def _load_preview_frame(self) -> bool:
        """Load the first frame for preview"""
        try:
            if self._initial_preview_frame is not None:
                frame = self._initial_preview_frame
                self._initial_preview_frame = None
            else:
                preview = load_source_preview(self.config)
                if preview.frame is None:
                    self.logger.error("Failed to load preview frame: %s", preview.error)
                    return False
                frame = preview.frame

            # Store original frame size
            self.original_frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

            # Resize for display if needed
            h, w = frame.shape[:2]
            if w > self.display_width:
                self.scale_factor = w / self.display_width
                new_w = self.display_width
                new_h = int(h / self.scale_factor)
                self.preview_frame = cv2.resize(frame, (new_w, new_h))
            else:
                self.scale_factor = 1.0
                self.preview_frame = frame.copy()

            self.logger.info(
                f"Preview frame loaded: {w}x{h} -> {self.preview_frame.shape[1]}x{self.preview_frame.shape[0]}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load preview frame: {e}")
            return False

    def _run_interactive_loop(self):
        """Run the main interactive loop"""
        window_name = "Setup - Draw Lines and Zones"
        self._hide_external_dialog_parent()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Changed from AUTOSIZE to NORMAL

        # Force it to match the preview frame size before centering
        h, w = self.preview_frame.shape[:2]
        cv2.resizeWindow(window_name, w, h)

        self._center_cv2_window(window_name)
        cv2.setMouseCallback("Setup - Draw Lines and Zones", self._mouse_callback)
        while not self.state.confirmed:
            # Create display canvas
            canvas = self.preview_frame.copy()

            # Draw interface elements
            self._draw_buttons(canvas)
            self._draw_existing_elements(canvas)
            self._draw_current_elements(canvas)
            self._draw_instructions(canvas)

            # Show canvas
            cv2.imshow("Setup - Draw Lines and Zones", canvas)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._process_pending_actions()
            if key == 27:  # ESC key
                if messagebox.askyesno("Exit", "Are you sure you want to exit setup?", parent=self._ensure_dialog_root()):
                    break
            elif key == ord('c'):  # Clear current drawing
                if self.state.selection_mode == "line":
                    self.state.current_line.clear()
                elif self.state.selection_mode in ["zone", "exclusion"]:
                    self.state.current_zone.clear()
            elif key == ord('u'):  # Undo last element
                self._undo_last_element()

        cv2.destroyAllWindows()

    def _hide_external_dialog_parent(self) -> None:
        """Keep the deployment editor hidden while OpenCV setup is active."""
        if self.dialog_root is None or self._owns_dialog_root:
            return
        try:
            self._dialog_parent_state = self.dialog_root.state()
            self._dialog_parent_topmost = self.dialog_root.attributes("-topmost")
            self.dialog_root.attributes("-topmost", True)
            self.dialog_root.withdraw()
            self.dialog_root.update_idletasks()
        except tk.TclError:
            self._dialog_parent_state = None

    def _restore_external_dialog_parent(self) -> None:
        if self.dialog_root is None or self._dialog_parent_state is None:
            return
        try:
            original_state = self._dialog_parent_state
            if original_state != "withdrawn":
                self.dialog_root.deiconify()
                if original_state == "zoomed":
                    self.dialog_root.state("zoomed")
                elif original_state == "iconic":
                    self.dialog_root.iconify()
                else:
                    self.dialog_root.lift()
            if self._dialog_parent_topmost is not None:
                self.dialog_root.attributes(
                    "-topmost", self._dialog_parent_topmost
                )
            self.dialog_root.update_idletasks()
        except tk.TclError:
            pass
        finally:
            self._dialog_parent_state = None
            self._dialog_parent_topmost = None

    def _queue_action(self, action: str) -> None:
        """Queue UI work so it never runs inside an OpenCV mouse callback."""
        if action not in self._pending_actions:
            self._pending_actions.append(action)

    def _process_pending_actions(self) -> None:
        """Run one queued modal action after cv2.waitKey has returned."""
        if not self._pending_actions:
            return

        action = self._pending_actions.pop(0)
        handlers = {
            "save_line": self._save_current_line,
            "finish_zone": self._finish_current_zone,
            "confirm": self._confirm_setup,
            "reference_length": self._finalize_reference_length,
        }
        handler = handlers.get(action)
        if handler is not None:
            handler()

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.state.current_mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check button clicks first
            clicked_button = self._check_button_click(x, y)
            if clicked_button:
                self._handle_button_click(clicked_button)
                return

            # Handle drawing based on current mode
            if self.state.selection_mode == "line":
                self._handle_line_click(x, y)
            elif self.state.selection_mode in ["zone", "exclusion"]:
                self._handle_zone_click(x, y)
            elif self.state.selection_mode == "ref":
                self._handle_ref_click(x, y)

    def _check_button_click(self, x: int, y: int) -> Optional[str]:
        """Check if click is on any button"""
        for button_name, button_info in self.buttons.items():
            x1, y1, x2, y2 = button_info["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return button_name
        return None

    def _handle_button_click(self, button_name: str):
        """Handle button clicks"""
        if button_name in ["line", "zone", "exclusion"]:
            self.state.selection_mode = button_name
            self.state.current_line.clear()
            self.state.current_zone.clear()

        elif button_name == "finish_zone" and self.config.enable_zones:
            self._queue_action("finish_zone")

        elif button_name == "confirm":
            self._queue_action("confirm")

        elif button_name == "ref":
            self.state.selection_mode = "ref"
            self.state.current_line.clear()
            self.state.current_zone.clear()
            self.state.current_ref.clear()

    def _handle_line_click(self, x: int, y: int):
        """Handle clicks in line drawing mode"""
        self.logger.debug(f"Line click at ({x}, {y}), current line has {len(self.state.current_line)} points")

        if len(self.state.current_line) == 0:
            # First point
            self.state.current_line.append((x, y))
            self.logger.info(f"Added first point: ({x}, {y})")

        elif len(self.state.current_line) == 1:
            # Second point - complete the line
            self.state.current_line.append((x, y))
            self.logger.info(f"Added second point: ({x}, {y}), completing line")

            self._queue_action("save_line")

        else:
            # This shouldn't happen, but let's handle it
            self.logger.warning(f"Unexpected line state: {len(self.state.current_line)} points")
            self.state.current_line.clear()
            self.state.current_line.append((x, y))

    def _handle_zone_click(self, x: int, y: int):
        """Handle clicks in zone/exclusion drawing mode"""
        self.state.current_zone.append((x, y))

    def _save_current_line(self):
        """Save the current line and ask for properties"""
        if len(self.state.current_line) != 2:
            self.logger.warning(f"Invalid line length: {len(self.state.current_line)}")
            return

        self.logger.info(f"Saving line with points: {self.state.current_line}")

        try:
            # Show the properties dialog
            properties = self._ask_line_properties()

            if properties:
                # Create line configuration
                line_config = {
                    "name": properties["name"],
                    "start": self.state.current_line[0],
                    "end": self.state.current_line[1],
                    "direction": properties["direction"],
                    "classes": properties["classes"],
                    "enabled": True,
                    "poi_mode": properties.get("poi_mode", "center")  # ADD THIS LINE
                }

                # Add to lines configuration (this saves it permanently)
                self.lines_config.append(line_config)
                self.logger.info(f"Line '{properties['name']}' saved successfully!")

                # Show success message
                messagebox.showinfo("Line Saved",
                                    f"Line '{properties['name']}' saved successfully!\n"
                                    f"Direction: {properties['direction']}\n"
                                    f"Classes: {len(properties['classes'])} selected",
                                    parent=self._ensure_dialog_root())

                # Clear current line ONLY after successful save
                self.state.current_line.clear()
                self.logger.info("Line saved and cleared from current drawing")

            else:
                # User cancelled - clear the current line
                self.state.current_line.clear()
                self.logger.info("Line creation cancelled, cleared current line")

        except Exception as e:
            self.logger.error(f"Error saving line: {e}")
            messagebox.showerror("Error", f"Failed to save line: {e}", parent=self._ensure_dialog_root())
            # Clear current line even on error
            self.state.current_line.clear()

    def _finish_current_zone(self):
        """Finish the current zone and ask for properties"""
        if len(self.state.current_zone) < 3:
            messagebox.showwarning("Incomplete Zone", "A zone needs at least 3 points.",
                                   parent=self._ensure_dialog_root())
            return

        if self.state.selection_mode == "zone":
            properties = self._ask_zone_properties()
            if properties:
                zone_config = {
                    "name": properties["name"],
                    "points": self.state.current_zone.copy(),
                    "classes": properties["classes"],
                    "enabled": True,
                    "track_max_concurrent": properties.get("track_max_concurrent", False),
                    "show_peak_overlay": properties.get("show_peak_overlay", True),
                    "poi_mode": properties.get("poi_mode", "center")  # ADD THIS LINE
                }
                self.zones_config.append(zone_config)
                messagebox.showinfo("Zone Saved", f"Zone '{properties['name']}' saved successfully!",
                                    parent=self._ensure_dialog_root())
        elif self.state.selection_mode == "exclusion":
            # Support multiple exclusion zones with names
            name = simpledialog.askstring(
                "Exclusion Zone Name",
                f"Enter a name for this exclusion zone (e.g., 'Exclusion {len(self.exclusion_zones) + 1}'):",
                parent=self._ensure_dialog_root()
            )
            if name:
                exclusion_config = {
                    "name": name,
                    "points": self.state.current_zone.copy(),
                    "enabled": True
                }
                self.exclusion_zones.append(exclusion_config)
                messagebox.showinfo("Exclusion Set", f"Exclusion zone '{name}' saved successfully!",
                                    parent=self._ensure_dialog_root())

        self.state.current_zone.clear()

    def _ask_line_properties(self) -> Optional[Dict]:
        """Ask user for line properties"""
        try:
            self.logger.info("Asking for line properties...")
            dialog_parent = self._ensure_dialog_root()

            # Get line name using simple dialog
            name = simpledialog.askstring(
                "Line Name",
                "Enter a name for this line:",
                parent=dialog_parent,
            )

            if not name:
                self.logger.info("No name provided, cancelling line creation")
                return None

            self.logger.info(f"Got line name: {name}")

            # Create dialog for class selection and direction
            dialog = LinePropertiesDialog(dialog_parent, self.config, self.class_names)
            result = dialog.show()

            if result:
                result["name"] = name
                self.logger.info(f"Line properties configured: {result}")
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error getting line properties: {e}")
            return None

    def _ask_zone_properties(self) -> Optional[Dict]:
        """Ask user for zone properties"""
        try:
            self.logger.info("Asking for zone properties...")
            dialog_parent = self._ensure_dialog_root()

            # Get zone name
            name = simpledialog.askstring(
                "Zone Name",
                "Enter a name for this zone:",
                parent=dialog_parent,
            )

            if not name:
                self.logger.info("No name provided, cancelling zone creation")
                return None

            self.logger.info(f"Got zone name: {name}")

            # Create dialog for class selection
            dialog = ZonePropertiesDialog(dialog_parent, self.class_names)
            result = dialog.show()

            if result:
                result["name"] = name
                self.logger.info(f"Zone properties configured: {result}")
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error getting zone properties: {e}")
            return None

    def _confirm_setup(self):
        """Confirm the current setup"""
        if not self.lines_config and not self.zones_config:
            messagebox.showwarning(
                "No Counting Geometry",
                "Please add at least one counting line or zone before confirming.",
                parent=self._ensure_dialog_root(),
            )
            return

        # Show summary
        summary = f"Setup Summary:\n"
        summary += f"- Counting Lines: {len(self.lines_config)}\n"
        summary += f"- Counting Zones: {len(self.zones_config)}\n"
        summary += f"- Exclusion Zones: {len(self.exclusion_zones)}\n"

        if messagebox.askyesno("Confirm Setup", summary + "\nProceed with this configuration?",
                               parent=self._ensure_dialog_root()):
            self.state.confirmed = True

    def _undo_last_element(self):
        """Undo the last added element"""
        if self.state.selection_mode == "line" and self.lines_config:
            removed = self.lines_config.pop()
            messagebox.showinfo("Undone", f"Removed line '{removed['name']}'", parent=self._ensure_dialog_root())
        elif self.state.selection_mode == "zone" and self.zones_config:
            removed = self.zones_config.pop()
            messagebox.showinfo("Undone", f"Removed zone '{removed['name']}'", parent=self._ensure_dialog_root())
        elif self.state.selection_mode == "exclusion" and self.exclusion_zones:
            removed = self.exclusion_zones.pop()
            messagebox.showinfo("Undone", f"Removed exclusion zone '{removed['name']}'",
                                parent=self._ensure_dialog_root())

    def _draw_buttons(self, canvas: np.ndarray):
        """Draw UI buttons on canvas"""
        for button_name, button_info in self.buttons.items():
            x1, y1, x2, y2 = button_info["rect"]
            color = button_info["color"]
            text_color = button_info["text_color"]
            text = button_info["text"]

            # Highlight current mode
            if button_name == self.state.selection_mode:
                cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

            # Draw button
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Draw text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # In gui_setup.py - replace the duplicate/incomplete _draw_existing_elements method with:
    def _draw_existing_elements(self, canvas: np.ndarray):
        """Draw existing lines and zones"""
        # Draw existing SAVED lines
        for i, line in enumerate(self.lines_config):
            start, end = line["start"], line["end"]
            cv2.line(canvas, start, end, (0, 255, 0), 3)
            cv2.circle(canvas, start, 8, (0, 255, 0), -1)
            cv2.circle(canvas, end, 8, (0, 255, 0), -1)
            self._draw_direction_arrow(canvas, start, end, line["direction"])

            # Draw label
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            text = line["name"]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(canvas,
                          (mid_point[0] - text_size[0] // 2 - 5, mid_point[1] - text_size[1] - 5),
                          (mid_point[0] + text_size[0] // 2 + 5, mid_point[1] + 5),
                          (0, 0, 0), -1)
            cv2.putText(canvas, text,
                        (mid_point[0] - text_size[0] // 2, mid_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw existing zones
        for zone in self.zones_config:
            points = np.array(zone["points"], np.int32)
            cv2.polylines(canvas, [points], True, (255, 255, 0), 2)
            if len(points) > 0:
                centroid = np.mean(points, axis=0).astype(int)
                cv2.putText(canvas, zone["name"], tuple(centroid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw exclusion zones (plural)
        for exclusion in self.exclusion_zones:
            points = np.array(exclusion["points"], np.int32)
            cv2.polylines(canvas, [points], True, (0, 0, 255), 2)

            # Draw name for each exclusion zone
            if len(points) > 0:
                centroid = np.mean(points, axis=0).astype(int)
                cv2.putText(canvas, exclusion["name"], tuple(centroid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _draw_current_elements(self, canvas: np.ndarray):
        """Draw elements currently being drawn (not saved yet)"""
        if self.state.selection_mode == "line":
            if len(self.state.current_line) == 1:
                # Draw line from first point to cursor (WHITE/YELLOW for "in progress")
                start = self.state.current_line[0]
                end = self.state.current_mouse_pos
                cv2.line(canvas, start, end, (0, 255, 255), 2)  # Yellow line
                cv2.circle(canvas, start, 6, (0, 255, 255), -1)  # Yellow dot

                # Add instruction text
                cv2.putText(canvas, "Click second point to complete line",
                            (start[0], start[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            elif len(self.state.current_line) == 2:
                # Show completed line in different color (waiting for properties)
                start, end = self.state.current_line
                cv2.line(canvas, start, end, (255, 255, 255), 2)  # White line
                cv2.circle(canvas, start, 6, (255, 255, 255), -1)
                cv2.circle(canvas, end, 6, (255, 255, 255), -1)

                cv2.putText(canvas, "Enter line properties in dialog",
                            (start[0], start[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.state.selection_mode in ["zone", "exclusion"]:
            color = (255, 255, 0) if self.state.selection_mode == "zone" else (0, 0, 255)

            # Draw existing points
            for point in self.state.current_zone:
                cv2.circle(canvas, point, 5, color, -1)

            # Draw lines between points
            if len(self.state.current_zone) > 1:
                points = np.array(self.state.current_zone, np.int32)
                cv2.polylines(canvas, [points], False, color, 2)

            # Draw line to cursor
            if self.state.current_zone:
                cv2.line(canvas, self.state.current_zone[-1], self.state.current_mouse_pos, color, 1)

        elif self.state.selection_mode == "ref":
            # Drawing for reference-length segment (meters-per-pixel helper)
            color = (0, 215, 255)  # gold-ish to distinguish from line/zones

            # Draw already-placed points
            if hasattr(self.state, "current_ref"):
                for p in self.state.current_ref:
                    cv2.circle(canvas, p, 6, color, -1)

                # If the first point exists, draw a rubber-band line to cursor
                if len(self.state.current_ref) == 1:
                    cv2.line(canvas, self.state.current_ref[0],
                             self.state.current_mouse_pos, color, 2)
                    cv2.putText(canvas, "Click second point to finish reference",
                                (self.state.current_ref[0][0], self.state.current_ref[0][1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # If two points are set, show the completed segment
                elif len(self.state.current_ref) == 2:
                    a, b = self.state.current_ref
                    cv2.line(canvas, a, b, color, 2)
                    cv2.circle(canvas, a, 6, color, -1)
                    cv2.circle(canvas, b, 6, color, -1)

                    # Optional live pixel-length label
                    px_len = int(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)
                    mid = ((a[0]+b[0])//2, (a[1]+b[1])//2)
                    cv2.putText(canvas, f"{px_len} px",
                                (mid[0]+8, mid[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    def _draw_direction_arrow(self, canvas: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], direction: str):
        """Draw direction arrow on line"""
        # Calculate arrow position and direction
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2

        # Arrow parameters
        arrow_length = 15
        arrow_angle = 0.5

        # Calculate line angle
        line_angle = np.arctan2(end[1] - start[1], end[0] - start[0])

        # Adjust based on direction
        if direction == "up":
            arrow_angle_offset = -np.pi / 2
        elif direction == "down":
            arrow_angle_offset = np.pi / 2
        elif direction == "left":
            arrow_angle_offset = np.pi
        else:  # right
            arrow_angle_offset = 0

        arrow_angle = line_angle + arrow_angle_offset

        # Calculate arrow points
        arrow_end_x = int(mid_x + arrow_length * np.cos(arrow_angle))
        arrow_end_y = int(mid_y + arrow_length * np.sin(arrow_angle))

        arrow_left_x = int(arrow_end_x - 8 * np.cos(arrow_angle - 0.3))
        arrow_left_y = int(arrow_end_y - 8 * np.sin(arrow_angle - 0.3))

        arrow_right_x = int(arrow_end_x - 8 * np.cos(arrow_angle + 0.3))
        arrow_right_y = int(arrow_end_y - 8 * np.sin(arrow_angle + 0.3))

        # Draw arrow
        cv2.line(canvas, (mid_x, mid_y), (arrow_end_x, arrow_end_y), (0, 255, 0), 2)
        cv2.line(canvas, (arrow_end_x, arrow_end_y), (arrow_left_x, arrow_left_y), (0, 255, 0), 2)
        cv2.line(canvas, (arrow_end_x, arrow_end_y), (arrow_right_x, arrow_right_y), (0, 255, 0), 2)

    def _draw_instructions(self, canvas: np.ndarray):
        """Draw instruction text"""
        instructions = []

        if self.state.selection_mode == "line":
            if not self.state.current_line:
                instructions.append("Click to start drawing a line")
            else:
                instructions.append("Click to finish the line")
        elif self.state.selection_mode in ["zone", "exclusion"]:
            instructions.append("Click to add points, then click 'Finish Zone'")
            instructions.append("Press 'C' to clear current drawing")

        instructions.extend([
            "Press 'U' to undo last element",
            "Press ESC to exit setup"
        ])

        # --- NEW: calculate widest line for background box ---
        text_sizes = [cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0] for instr in instructions]
        max_width = max(size[0] for size in text_sizes)
        box_width = max_width + 40  # extra padding to make it wider
        box_height = 20 * len(instructions) + 20
        x0, y0 = 5, canvas.shape[0] - box_height - 10
        x1, y1 = x0 + box_width, y0 + box_height

        # Draw semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        # Draw each instruction line
        for i, instruction in enumerate(instructions):
            cv2.putText(canvas, instruction,
                        (x0 + 10, y0 + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _normalize_configurations(self):
        """Convert pixel coordinates to normalized coordinates"""
        gui_w, gui_h = self.preview_frame.shape[1], self.preview_frame.shape[0]

        # Normalize lines
        for line in self.lines_config:
            start_x, start_y = line["start"]
            end_x, end_y = line["end"]

            # Convert to original frame coordinates
            orig_start_x = start_x * self.scale_factor
            orig_start_y = start_y * self.scale_factor
            orig_end_x = end_x * self.scale_factor
            orig_end_y = end_y * self.scale_factor

            # Normalize to 0-1 range
            line["start_norm"] = (orig_start_x / self.original_frame_size[0],
                                  orig_start_y / self.original_frame_size[1])
            line["end_norm"] = (orig_end_x / self.original_frame_size[0],
                                orig_end_y / self.original_frame_size[1])

        # Normalize zones
        for zone in self.zones_config:
            points_norm = []
            for x, y in zone["points"]:
                orig_x = x * self.scale_factor
                orig_y = y * self.scale_factor
                norm_x = orig_x / self.original_frame_size[0]
                norm_y = orig_y / self.original_frame_size[1]
                points_norm.append((norm_x, norm_y))
            zone["points_norm"] = points_norm

        # Normalize exclusion zones
        for exclusion in self.exclusion_zones:
            points_norm = []
            for x, y in exclusion["points"]:
                orig_x = x * self.scale_factor
                orig_y = y * self.scale_factor
                norm_x = orig_x / self.original_frame_size[0]
                norm_y = orig_y / self.original_frame_size[1]
                points_norm.append((norm_x, norm_y))
            exclusion["points_norm"] = points_norm

    def _create_exclusion_configs(self) -> List[ExclusionZone]:
        """Convert exclusion configurations to ExclusionZone objects"""
        return [
            ExclusionZone(
                name=exc["name"],
                points_norm=exc["points_norm"],
                enabled=exc.get("enabled", True)
            )
            for exc in self.exclusion_zones
        ]

    def _create_line_configs(self) -> List[CountingLine]:
        """Convert line configurations to CountingLine objects"""
        return [
            CountingLine(
                name=line["name"],
                start_norm=line["start_norm"],
                end_norm=line["end_norm"],
                direction=line["direction"],
                classes=line["classes"],
                enabled=line["enabled"],
                poi_mode=line.get("poi_mode", "center")
            )
            for line in self.lines_config
        ]

    def _create_zone_configs(self) -> List[CountingZone]:
        """Convert zone configurations to CountingZone objects"""
        return [
            CountingZone(
                name=zone["name"],
                points_norm=zone["points_norm"],
                classes=zone["classes"],
                enabled=zone["enabled"],
                track_max_concurrent=zone.get("track_max_concurrent", False),
                show_peak_overlay=zone.get("show_peak_overlay", True),
                poi_mode=zone.get("poi_mode", "center")
            )
            for zone in self.zones_config
        ]

    def _cleanup(self):
        """Clean up resources"""
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        self._restore_external_dialog_parent()
        try:
            if self.dialog_root and self._owns_dialog_root:
                self.dialog_root.destroy()
                self.dialog_root = None
        except tk.TclError:
            pass

    def _handle_ref_click(self, x: int, y: int):
        # two points define the pixel length
        self.state.current_ref.append((x, y))
        if len(self.state.current_ref) == 2:
            self._queue_action("reference_length")

    def _finalize_reference_length(self):
        try:
            (x1, y1), (x2, y2) = self.state.current_ref
            px_len = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if px_len <= 0:
                self.logger.warning("Reference length is zero; ignoring.")
                self.state.current_ref.clear()
                return

            # Ask user for real-world length and units
            root = self._ensure_dialog_root()
            length_str = simpledialog.askstring(
                "Reference length",
                "Enter real-world length (e.g., 1, 0.9144, 3):",
                parent=root
            )
            if not length_str:
                self.state.current_ref.clear()
                return
            try:
                real_len = float(length_str)
            except Exception:
                messagebox.showerror("Error", "Please enter a numeric length.", parent=root)
                self.state.current_ref.clear()
                return

            unit = simpledialog.askstring(
                "Units",
                "Units? (m, cm, mm, ft, in). Default = m",
                parent=root
            ) or "m"
            unit = unit.strip().lower()

            # Convert to meters
            factor = {
                "m": 1.0, "meter": 1.0, "meters": 1.0,
                "cm": 0.01, "mm": 0.001,
                "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
                "in": 0.0254, "inch": 0.0254, "inches": 0.0254
            }.get(unit, 1.0)

            real_m = real_len * factor
            mpp = real_m / px_len

            # Save to config so speed uses it
            self.config.meters_per_pixel = float(mpp)

            # Optional: if user picked speed units in the first dialog, speed overlay works out-of-the-box
            messagebox.showinfo(
                "Scale set",
                f"Computed scale: {mpp:.6f} meters/pixel\n"
                f"(Real {real_m:.4f} m over {px_len:.1f} px)",
                parent=root
            )

            # Clear ref points from the temp state
            self.state.current_ref.clear()

        except Exception as e:
            self.logger.error(f"Failed to set reference length: {e}")
            self.state.current_ref.clear()

    def _center_cv2_window(self, window_name: str):
        """Center an OpenCV window on the primary screen."""
        try:
            h, w = self.preview_frame.shape[:2]
            root = self._ensure_dialog_root()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()

            # --- HEADLESS SAFETY ---
            if screen_w > 3000 or screen_h > 2000:
                screen_w = 1920
                screen_h = 1080

            x = max(0, (screen_w - w) // 2)
            y = max(0, (screen_h - h) // 2)

            cv2.moveWindow(window_name, x, y)
        except Exception as e:
            self.logger.warning(f"Failed to center OpenCV window: {e}")

# in gui_setup.py
class LinePropertiesDialog:
    """Dialog for setting line properties"""

    def __init__(self, parent: tk.Misc, config: AppConfig, class_names: Dict[int, str]):
        self.parent = parent
        self.config = config
        self.class_names = class_names
        self.result = None

    def show(self) -> Optional[Dict]:
        """Show the dialog and return result"""
        # Create the Toplevel
        self.dialog = tk.Toplevel(self.parent)

        # Don’t make it transient to a withdrawn parent; just force it on top
        # (transient to a withdrawn parent can make it ‘invisible’ on some setups)
        # self.dialog.transient(self.parent)  # <- intentionally NOT used

        # Force it to the front and focus
        self.dialog.attributes('-topmost', True)
        self.dialog.lift()
        try:
            self.dialog.focus_force()
        except Exception:
            pass

        # Basic window setup
        self.dialog.title("Line Properties")
        self.dialog.geometry("540x640")

        # Ensure the window is actually mapped and visible before centering
        self.dialog.update_idletasks()
        try:
            self.dialog.deiconify()
            self.dialog.wait_visibility()
        except Exception:
            pass

            # Center on screen (parent is withdrawn)
            sw = self.dialog.winfo_screenwidth()
            sh = self.dialog.winfo_screenheight()

            if sw > 3000 or sh > 2000:
                sw = 1920
                sh = 1080

            x = max(0, (sw // 2) - 200)
            y = max(0, (sh // 2) - 250)
            self.dialog.geometry(f"540x640+{x}+{y}")

        # --- Direction selection ---
        direction_var = tk.StringVar(master=self.dialog, value="up")
        direction_frame = tk.LabelFrame(self.dialog, text="Count Direction")
        direction_frame.pack(fill="x", padx=10, pady=5)

        for text, value in [("Up ↑", "up"), ("Down ↓", "down"),
                            ("Left ←", "left"), ("Right →", "right")]:
            tk.Radiobutton(direction_frame, text=text,
                           variable=direction_var, value=value).pack(anchor="w")

        # --- Point of Interest (POI) ---
        poi_var = tk.StringVar(master=self.dialog, value="center")
        poi_frame = tk.LabelFrame(self.dialog, text="Point of Interest (POI)")
        poi_frame.pack(fill="x", padx=10, pady=5)
        tk.Radiobutton(poi_frame, text="Center", value="center", variable=poi_var).pack(anchor="w")
        tk.Radiobutton(poi_frame, text="Bottom middle", value="bottom", variable=poi_var).pack(anchor="w")

        # --- Class selection (scrollable) ---
        class_frame = tk.LabelFrame(self.dialog, text="Classes to Count")
        class_frame.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(class_frame)
        scrollbar = tk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        class_vars: Dict[int, tk.BooleanVar] = {}
        for cid, cname in self.class_names.items():
            var = tk.BooleanVar(master=self.dialog, value=False)
            tk.Checkbutton(scrollable, text=f"{cid}: {cname}", variable=var).pack(anchor="w")
            class_vars[cid] = var

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Buttons ---
        btns = tk.Frame(self.dialog)
        btns.pack(fill="x", padx=10, pady=5)

        def on_ok():
            selected = [cid for cid, v in class_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("No Classes",
                                       "Please select at least one class to count.",
                                       parent=self.dialog)
                return
            self.result = {
                "direction": direction_var.get(),
                "classes": selected,
                "poi_mode": poi_var.get()
            }

            try:
                self.dialog.grab_release()
            except Exception:
                pass
            self.dialog.destroy()

        def on_cancel():
            self.result = None
            try:
                self.dialog.grab_release()
            except Exception:
                pass
            self.dialog.destroy()

        tk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=5)
        tk.Button(btns, text="OK", command=on_ok).pack(side="right")

        # Finalize modality *after* it’s visible
        try:
            self.dialog.grab_set()
        except Exception:
            pass

        # Block until closed and return
        self.dialog.wait_window()
        return self.result


class ZonePropertiesDialog:
    """Dialog for setting zone properties"""

    def __init__(self, parent, class_names: Dict[int, str]):
        self.parent = parent
        self.class_names = class_names
        self.result = None
        self.dialog = None

    def show(self) -> Optional[Dict]:
        self.dialog = tk.Toplevel(self.parent)

        # force visible + on top (like your line dialog)
        self.dialog.attributes('-topmost', True)
        self.dialog.lift()
        try:
            self.dialog.focus_force()
        except Exception:
            pass
        self.dialog.title("Zone Properties")
        self.dialog.geometry("580x580")
        self.dialog.update_idletasks()
        try:
            self.dialog.deiconify()
            self.dialog.wait_visibility()
        except Exception:
            pass
        sw, sh = self.dialog.winfo_screenwidth(), self.dialog.winfo_screenheight()

        if sw > 3000 or sh > 2000:
            sw = 1920
            sh = 1080

        x = max(0, (sw // 2) - 290)
        y = max(0, (sh // 2) - 290)
        self.dialog.geometry(f"580x580+{x}+{y}")
        class_frame = tk.LabelFrame(self.dialog, text="Classes to Count in Zone")
        class_frame.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(class_frame)
        scrollbar = tk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # IMPORTANT: bind vars to this dialog
        class_vars: Dict[int, tk.BooleanVar] = {}
        for class_id, class_name in self.class_names.items():
            var = tk.BooleanVar(master=self.dialog, value=False)  # <-- master set
            tk.Checkbutton(scrollable_frame, text=f"{class_id}: {class_name}",
                           variable=var).pack(anchor="w")
            class_vars[class_id] = var

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        select_frame = tk.Frame(self.dialog)
        select_frame.pack(fill="x", padx=10, pady=5)

        def select_all():
            for v in class_vars.values():
                v.set(True)

        def deselect_all():
            for v in class_vars.values():
                v.set(False)

        tk.Button(select_frame, text="Select All", command=select_all).pack(side="left", padx=5)
        tk.Button(select_frame, text="Deselect All", command=deselect_all).pack(side="left", padx=5)

        # --- Point of Interest (POI) ---
        poi_var = tk.StringVar(master=self.dialog, value="center")
        poi_frame = tk.LabelFrame(self.dialog, text="Point of Interest (POI)")
        poi_frame.pack(fill="x", padx=10, pady=5)
        tk.Radiobutton(poi_frame, text="Center", value="center", variable=poi_var).pack(anchor="w")
        tk.Radiobutton(poi_frame, text="Bottom middle", value="bottom", variable=poi_var).pack(anchor="w")

        options_frame = tk.LabelFrame(self.dialog, text="Occupancy Options")
        options_frame.pack(fill="x", padx=10, pady=5)

        track_var = tk.BooleanVar(master=self.dialog, value=True)  # default ON is handy
        show_var = tk.BooleanVar(master=self.dialog, value=True)

        tk.Checkbutton(options_frame, text="Track max concurrent occupancy",
                       variable=track_var).pack(anchor="w")
        tk.Checkbutton(options_frame, text="Show peak on overlay",
                       variable=show_var).pack(anchor="w")

        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill="x", padx=10, pady=5)

        def on_ok():
            selected = [cid for cid, v in class_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("No Classes",
                                       "Please select at least one class to count in the zone.",
                                       parent=self.dialog)
                return
            self.result = {
                "classes": selected,
                "track_max_concurrent": track_var.get(),
                "show_peak_overlay": show_var.get(),
                "poi_mode": poi_var.get()
            }
            try:
                self.dialog.grab_release()
            except Exception:
                pass
            self.dialog.destroy()

        def on_cancel():
            self.result = None
            try:
                self.dialog.grab_release()
            except Exception:
                pass
            self.dialog.destroy()

        tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)
        tk.Button(button_frame, text="OK", command=on_ok).pack(side="right")

        try:
            self.dialog.grab_set()
        except Exception:
            pass
        self.dialog.wait_window()
        return self.result

    # Utility functions for coordinate transformations
    def normalize_point(point: Tuple[int, int], frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """Normalize pixel coordinates to 0-1 range"""
        x, y = point
        w, h = frame_size
        return (x / w, y / h)

    def denormalize_point(norm_point: Tuple[float, float], frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """Convert normalized coordinates back to pixels"""
        norm_x, norm_y = norm_point
        w, h = frame_size
        return (int(norm_x * w), int(norm_y * h))

    def scale_point(point: Tuple[int, int], scale_factor: float) -> Tuple[int, int]:
        """Scale a point by a factor"""
        x, y = point
        return (int(x * scale_factor), int(y * scale_factor))

    def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
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

    def calculate_line_angle(start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Calculate angle of line in radians"""
        return np.arctan2(end[1] - start[1], end[0] - start[0])

    def distance_point_to_line(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
        """Calculate perpendicular distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate the distance
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            # Line segment is actually a point
            return np.sqrt(A * A + B * B)

        param = dot / len_sq

        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = x0 - xx
        dy = y0 - yy
        return np.sqrt(dx * dx + dy * dy)


