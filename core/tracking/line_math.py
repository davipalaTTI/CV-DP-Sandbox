import numpy as np
import cv2
import math
import itertools
from typing import Tuple, Dict, Optional
from config_manager import CountingLine
from collections import defaultdict
import logging
import datetime
import time

from core.tracking.schemas import ObjectState, CountingEvent

# Process-wide monotonic counter for event_id uniqueness. time.time() alone
# collides at ms resolution on Windows when several crossings fire in the
# same frame; combining it with this counter guarantees uniqueness within
# the process lifetime (and is human-readable next to the timestamp prefix).
_event_seq = itertools.count()


class LineCounter:
    """Handles line crossing detection and counting"""

    def __init__(self, line_config: CountingLine, frame_size: Tuple[int, int]):
        self.config = line_config
        self.frame_size = frame_size

        # Convert normalized coordinates to pixel coordinates
        self.start_px = self._denormalize_point(line_config.start_norm)
        self.end_px = self._denormalize_point(line_config.end_norm)

        # Counting data
        self.counts = defaultdict(int)
        self.counted_objects = set()
        self.total_count = 0

        # Line geometry
        self._calculate_line_properties()

        # Adaptive proximity threshold
        self.base_proximity_threshold = 20  # pixels
        self.max_proximity_threshold = 50  # maximum for very fast objects

        self.logger = logging.getLogger(f"{__name__}.{line_config.name}")

    def _denormalize_point(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates to pixels"""
        norm_x, norm_y = norm_point
        w, h = self.frame_size
        return (int(norm_x * w), int(norm_y * h))

    def _calculate_line_properties(self):
        """Calculate line properties for crossing detection.

        We keep scalar copies of the line vector and normal alongside the
        numpy versions. The hot path (get_side, get_distance_to_point) reads
        the scalars to avoid building numpy arrays per detection per line
        per frame; the numpy versions are reserved for draw-time math."""
        x1, y1 = self.start_px
        x2, y2 = self.end_px

        # Scalar line vector for hot-path arithmetic
        self.line_vec_x = x2 - x1
        self.line_vec_y = y2 - y1
        self.line_length = math.hypot(self.line_vec_x, self.line_vec_y)

        # Scalar unit-normal (perpendicular to line, points to "left" side)
        if self.line_length > 0:
            inv_len = 1.0 / self.line_length
            self.normal_vec_x = -self.line_vec_y * inv_len
            self.normal_vec_y = self.line_vec_x * inv_len
        else:
            self.normal_vec_x = 0.0
            self.normal_vec_y = 1.0

        # Numpy aliases for code paths that still expect numpy (draw routines)
        self.line_vector = np.array([self.line_vec_x, self.line_vec_y])
        self.normal_vector = np.array([self.normal_vec_x, self.normal_vec_y])

    def _normalize_point(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        x, y = pt
        w, h = self.frame_size
        return (max(0, min(1, x / float(w))), max(0, min(1, y / float(h))))

    def update_endpoint(self, which: str, pt_px: Tuple[int, int]) -> None:
        """Move one endpoint in-place and refresh geometry; keeps counts intact."""
        if which == "start":
            self.start_px = pt_px
            self.config.start_norm = self._normalize_point(pt_px)
        else:
            self.end_px = pt_px
            self.config.end_norm = self._normalize_point(pt_px)
        self._calculate_line_properties()


    def get_side(self, point: Tuple[int, int]) -> str:
        """Determine which side of the line a point is on.

        Inlined 2-D cross product on scalars. np.cross on 2-vectors was the
        per-detection hot-path bottleneck and is deprecated in NumPy 2.x.
        """
        x, y = point
        x1, y1 = self.start_px
        # Cross product of (line_vec) x (point - start)
        cross = self.line_vec_x * (y - y1) - self.line_vec_y * (x - x1)
        if cross > 0:
            return "left"
        if cross < 0:
            return "right"
        return "on_line"

    def get_vertical_side(self, point: Tuple[int, int]) -> str:
        """Determine if point is above or below the line"""
        x, y = point
        x1, y1 = self.start_px
        x2, y2 = self.end_px

        # If the segment is vertical, compare to its midpoint Y
        if x1 == x2:
            y_mid = 0.5 * (y1 + y2)
            return "above" if y < y_mid else "below"

        # Otherwise compare y to the line's y at x
        line_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return "above" if y < line_y else "below"

    def check_crossing(self, object_state: ObjectState, current_position: Tuple[int, int], timestamp: Optional[datetime.datetime] = None) -> Optional[CountingEvent]:
        """
        Check if object crossed the line, using trajectory interpolation for fast objects
        """
        # Skip if object class not in line's class filter
        if object_state.class_id not in self.config.classes:
            return None

        # Skip if already counted
        if object_state.track_id in self.counted_objects:
            return None

        # Get previous position if available
        if len(object_state.positions) < 2:
            # Not enough history, use basic proximity check
            return self._check_basic_crossing(object_state, current_position, timestamp)

        # Get previous position
        prev_pos_data = object_state.positions[-2]
        prev_position = prev_pos_data.get('center') if self.config.poi_mode == "center" else prev_pos_data.get('bottom')

        # Calculate speed-based proximity threshold
        speed = object_state.current_speed_pxps if hasattr(object_state, 'current_speed_pxps') else 0
        proximity_threshold = min(
            self.base_proximity_threshold + speed * 0.5,  # Increase threshold with speed
            self.max_proximity_threshold
        )

        # Check if the trajectory from prev to current crosses the line
        crossing_point = self._check_trajectory_crossing(prev_position, current_position)

        if crossing_point:
            # Trajectory crosses the line - check direction
            direction_valid = self._validate_crossing_direction(prev_position, current_position)

            if direction_valid:
                # Mark as counted
                self.counted_objects.add(object_state.track_id)
                self.counts[object_state.class_id] += 1
                self.total_count += 1

                event = CountingEvent(
                    event_id=f"line_{self.config.name}_{object_state.track_id}_{int(time.time()*1000)}_{next(_event_seq)}",
                    track_id=object_state.track_id,
                    class_id=object_state.class_id,
                    class_name=object_state.class_name,
                    line_name=self.config.name,
                    direction=self.config.direction,
                    position=crossing_point,  # Use actual crossing point
                    actual_datetime=timestamp  # Add timestamp parameter

                )

                self.logger.info(
                    f"{object_state.class_name} (ID:{object_state.track_id}) crossed {self.config.name} "
                    f"going {self.config.direction} (interpolated)"
                )
                return event

        # If no trajectory crossing, check proximity-based crossing
        distance_to_line = self.get_distance_to_point(current_position)

        if distance_to_line <= proximity_threshold:
            return self._check_basic_crossing(object_state, current_position, timestamp)
        else:
            # Clear side tracking if too far
            if self.config.name in object_state.line_sides:
                del object_state.line_sides[self.config.name]

        return None

    def _check_trajectory_crossing(self, start_point: Tuple[int, int], end_point: Tuple[int, int]) -> Optional[
        Tuple[int, int]]:
        """
        Check if a trajectory segment crosses the counting line

        Returns the intersection point if crossing occurs, None otherwise
        """
        # Use line-line intersection algorithm
        x1, y1 = self.start_px
        x2, y2 = self.end_px
        x3, y3 = start_point
        x4, y4 = end_point

        # Calculate determinants
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection occurs within both segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            ix = int(x1 + t * (x2 - x1))
            iy = int(y1 + t * (y2 - y1))
            return (ix, iy)

        return None

    def _validate_crossing_direction(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> bool:
        """
        Validate that the crossing is in the configured direction
        """
        if self.config.direction in ["up", "down"]:
            prev_side = self.get_vertical_side(prev_pos)
            curr_side = self.get_vertical_side(curr_pos)

            if self.config.direction == "up":
                return prev_side == "below" and curr_side == "above"
            else:  # down
                return prev_side == "above" and curr_side == "below"
        else:  # left, right
            prev_side = self.get_side(prev_pos)
            curr_side = self.get_side(curr_pos)

            if self.config.direction == "left":
                return prev_side == "right" and curr_side == "left"
            else:  # right
                return prev_side == "left" and curr_side == "right"

    def _check_basic_crossing(self, object_state: ObjectState, current_position: Tuple[int, int], timestamp: Optional[datetime.datetime] = None) -> Optional[
        CountingEvent]:
        """
        Original proximity-based crossing check (fallback for when trajectory check doesn't apply)
        """
        # Determine current side
        if self.config.direction in ["up", "down"]:
            current_side = self.get_vertical_side(current_position)
        else:  # left, right
            current_side = self.get_side(current_position)

        # Get previous side
        previous_side = object_state.line_sides.get(self.config.name)

        # Update object's line side history
        object_state.line_sides[self.config.name] = current_side

        # Check for crossing
        if previous_side and previous_side != current_side and current_side != "on_line":
            # Validate direction
            transition_ok = False
            if self.config.direction == "up":
                transition_ok = (previous_side == "below" and current_side == "above")
            elif self.config.direction == "down":
                transition_ok = (previous_side == "above" and current_side == "below")
            elif self.config.direction == "left":
                transition_ok = (previous_side == "right" and current_side == "left")
            elif self.config.direction == "right":
                transition_ok = (previous_side == "left" and current_side == "right")

            if transition_ok:
                # Mark as counted
                self.counted_objects.add(object_state.track_id)
                self.counts[object_state.class_id] += 1
                self.total_count += 1

                event = CountingEvent(
                    event_id=f"line_{self.config.name}_{object_state.track_id}_{int(time.time()*1000)}_{next(_event_seq)}",
                    track_id=object_state.track_id,
                    class_id=object_state.class_id,
                    class_name=object_state.class_name,
                    line_name=self.config.name,
                    direction=self.config.direction,
                    position=current_position,
                    actual_datetime=timestamp  # NEW: Pass the datetime object
                )

                self.logger.info(
                    f"{object_state.class_name} (ID:{object_state.track_id}) crossed {self.config.name} "
                    f"going {self.config.direction}"
                )
                return event

        return None

    def get_distance_to_point(self, point: Tuple[int, int]) -> float:
        """Calculate perpendicular distance from point to the line segment.
        Pure Python arithmetic; runs per-detection per-line per-frame."""
        x0, y0 = point
        x1, y1 = self.start_px
        x2, y2 = self.end_px

        A = x2 - x1
        B = y2 - y1
        C = x0 - x1
        D = y0 - y1

        len_sq = A * A + B * B
        if len_sq == 0:
            return math.hypot(C, D)

        param = (C * A + D * B) / len_sq
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * A
            yy = y1 + param * B

        return math.hypot(x0 - xx, y0 - yy)

    def draw_line(self, frame: np.ndarray, show_count: bool = True) -> np.ndarray:
        """Draw the counting line on frame with optional count display"""
        if not self.config.enabled:
            return frame

        # Draw main line
        cv2.line(frame, self.start_px, self.end_px, (0, 255, 0), 2)

        # Draw endpoints
        cv2.circle(frame, self.start_px, 6, (0, 180, 0), -1)
        cv2.circle(frame, self.end_px, 6, (0, 180, 0), -1)

        # Draw direction arrow
        self._draw_direction_arrow(frame)

        # Draw count
        if show_count:
            count_text = f"{self.config.name}: {self.total_count}"
            text_pos = (self.start_px[0], max(20, self.start_px[1] - 10))
            cv2.putText(frame, count_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def _draw_direction_arrow(self, frame: np.ndarray):
        """Draw direction indicator arrow, aligned with the actual crossing direction."""
        # Midpoint of the line
        mid_x = (self.start_px[0] + self.end_px[0]) // 2
        mid_y = (self.start_px[1] + self.end_px[1]) // 2
        mid = np.array([mid_x, mid_y], dtype=float)

        # Base arrow settings
        arrow_len = 24
        head_len = 10
        arrow_color = (255, 0, 255)  # magenta (distinct from green line)
        thickness = 2

        # Unit normal vector (already computed in _calculate_line_properties)
        n = self.normal_vector  # points toward the line's "left" side (cross>0)

        # Decide which way along the normal corresponds to the configured 'direction'
        # We'll probe a tiny step along +n to see which "side" (+n) is.
        probe = (mid + 5.0 * n).astype(int).tolist()

        if self.config.direction in ["left", "right"]:
            # get_side(): "left"/"right" relative to line orientation
            probe_side = self.get_side(tuple(probe))
            # +n points to the "left" side by construction
            if self.config.direction == "left":
                dir_vec = n  # left means toward +n
            else:  # "right"
                dir_vec = -n  # right means toward -n
        else:
            # "up"/"down" are defined via get_vertical_side(): "above"/"below"
            probe_side = self.get_vertical_side(tuple(probe))
            # If +n points to "below", then:
            #  - 'up' (below->above) should point opposite n
            #  - 'down' should point along n
            # If +n points to "above", invert that logic.
            # Determine whether +n is "above" or "below"
            n_is_below = (probe_side == "below")
            if self.config.direction == "up":
                dir_vec = -n if n_is_below else n
            else:  # "down"
                dir_vec = n if n_is_below else -n

        # Arrow endpoints
        end = (mid + arrow_len * dir_vec).astype(int)
        end_pt = (int(end[0]), int(end[1]))
        mid_pt = (int(mid[0]), int(mid[1]))

        # Draw shaft
        cv2.line(frame, mid_pt, end_pt, arrow_color, thickness)

        # Draw head (two short legs)
        angle = np.arctan2(dir_vec[1], dir_vec[0])
        head_angle = 0.45
        h1 = (
            int(end[0] - head_len * np.cos(angle - head_angle)),
            int(end[1] - head_len * np.sin(angle - head_angle)),
        )
        h2 = (
            int(end[0] - head_len * np.cos(angle + head_angle)),
            int(end[1] - head_len * np.sin(angle + head_angle)),
        )
        cv2.line(frame, end_pt, h1, arrow_color, thickness)
        cv2.line(frame, end_pt, h2, arrow_color, thickness)

    def reset_counts(self):
        """Reset all counts"""
        self.counts.clear()
        self.counted_objects.clear()
        self.total_count = 0
        self.logger.info(f"Reset counts for line {self.config.name}")

    def get_stats(self) -> Dict:
        """Get counting statistics"""
        return {
            "name": self.config.name,
            "total_count": self.total_count,
            "class_counts": dict(self.counts),
            "direction": self.config.direction,
            "enabled": self.config.enabled,
            "counted_objects": list(self.counted_objects)
        }