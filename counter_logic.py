"""
Counter Logic Module

Handles all counting logic including:
- Line crossing detection with direction filtering
- Zone presence counting
- Object state tracking
- Counter management and statistics
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import time
from detection_engine import Detection
from config_manager import CountingLine, CountingZone
import datetime


def sanitize_dwell_time(dwell_seconds: float) -> float:
    """
    Validate and sanitize dwell time values.
    Returns 0.0 for invalid values (negative or unreasonably large).
    
    Args:
        dwell_seconds: Raw dwell time value
        
    Returns:
        Sanitized dwell time (0.0 if invalid, otherwise rounded to 2 decimals)
    """
    try:
        dwell = float(dwell_seconds) if dwell_seconds is not None else 0.0
        # Invalid if negative or greater than 24 hours (86400 seconds)
        if dwell < 0 or dwell > 86400:
            return 0.0
        return round(dwell, 2)
    except (ValueError, TypeError):
        return 0.0


class CrossingDirection(Enum):
    """Direction of line crossing"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class ObjectState:
    """Tracks the state of a detected object"""
    track_id: int
    class_id: int
    class_name: str
    positions: deque = field(default_factory=lambda: deque(maxlen=10))
    last_seen: float = field(default_factory=time.time)
    line_sides: Dict[str, str] = field(default_factory=dict)  # line_name -> side
    zone_presence: Dict[str, bool] = field(default_factory=dict)  # zone_name -> present
    current_speed_pxps: float = 0.0
    avg_speed_pxps: float = 0.0
    zone_entry_times: Dict[str, float] = field(default_factory=dict)  # zone_name -> entry_time
    zone_dwell_times: Dict[str, float] = field(default_factory=dict)  # zone_name -> total_dwell_seconds

    def update_position(self, center_point: Tuple[int, int], bottom_point: Tuple[int, int]):
        """Update object position history"""
        self.positions.append({
            'center': center_point,
            'bottom': bottom_point,
            'timestamp': time.time()
        })
        self.last_seen = time.time()

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if object tracking is stale"""
        return time.time() - self.last_seen > max_age_seconds

    def compute_speed_pxps(self, smooth_window: int = 5) -> float:
        """Compute smoothed speed in px/s from recent center points."""
        if len(self.positions) < 2:
            return 0.0

        # Take last N positions for smoothing
        hops = list(self.positions)[-min(smooth_window, len(self.positions)):]

        if len(hops) < 2:
            return 0.0

        total_distance = 0.0
        total_time = 0.0

        for i in range(len(hops) - 1):
            a, b = hops[i], hops[i + 1]
            (x1, y1), t1 = a['center'], a['timestamp']
            (x2, y2), t2 = b['center'], b['timestamp']

            dt = t2 - t1

            # Skip if time delta is too small or invalid
            if dt < 1e-6:
                continue

            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            total_distance += distance
            total_time += dt

        if total_time <= 0:
            return 0.0

        instant_speed = total_distance / total_time
        self.current_speed_pxps = instant_speed

        # Update running average with MORE smoothing for stability
        if self.avg_speed_pxps <= 0:
            self.avg_speed_pxps = instant_speed
        else:
            # Use exponential moving average with MUCH more weight on history
            # Lower alpha = smoother, less jittery speeds
            alpha = 0.15  # Reduced from 0.3 for better smoothing with frame skipping
            self.avg_speed_pxps = (1 - alpha) * self.avg_speed_pxps + alpha * instant_speed

        return self.current_speed_pxps


@dataclass
class CountingEvent:
    """Represents a counting event"""
    event_id: str
    track_id: int
    class_id: int
    class_name: str
    line_name: Optional[str] = None
    zone_name: Optional[str] = None
    direction: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    position: Optional[Tuple[int, int]] = None
    actual_datetime: Optional[datetime.datetime] = None  # Store actual video datetime
    # NEW: Add speed and dwell time fields
    avg_speed: float = 0.0  # Average speed at time of event
    speed_units: str = 'pxps'  # Units for the speed value
    dwell_seconds: float = 0.0  # Dwell time for zone events
    confidence: float = 0.0  # Detection confidence

    def to_dict(self) -> Dict:
        """Convert event to dictionary for export"""
        return {
            'event_id': self.event_id,
            'track_id': self.track_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'line_name': self.line_name or '',
            'zone_name': self.zone_name or '',
            'direction': self.direction or '',
            'timestamp': self.timestamp,
            'position': self.position,
            'actual_datetime': self.actual_datetime.isoformat() if self.actual_datetime else '',
            'speed': self.avg_speed,
            'speed_units': self.speed_units,
            'dwell_seconds': sanitize_dwell_time(self.dwell_seconds),
            'confidence': self.confidence
        }

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
        """Calculate line properties for crossing detection"""
        x1, y1 = self.start_px
        x2, y2 = self.end_px

        # Line vector
        self.line_vector = np.array([x2 - x1, y2 - y1])
        self.line_length = np.linalg.norm(self.line_vector)

        # Normal vector (perpendicular to line)
        if self.line_length > 0:
            self.normal_vector = np.array([-self.line_vector[1], self.line_vector[0]]) / self.line_length
        else:
            self.normal_vector = np.array([0, 1])

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
        """Determine which side of the line a point is on"""
        x, y = point
        x1, y1 = self.start_px

        # Vector from line start to point
        point_vector = np.array([x - x1, y - y1])

        # Calculate cross product to determine side
        cross_product = np.cross(self.line_vector, point_vector)

        if cross_product > 0:
            return "left"
        elif cross_product < 0:
            return "right"
        else:
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
                    event_id=f"line_{self.config.name}_{object_state.track_id}_{time.time()}",
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
                    event_id=f"line_{self.config.name}_{object_state.track_id}_{time.time()}",
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
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1 = self.start_px
        x2, y2 = self.end_px

        # Line segment vector
        A = x2 - x1
        B = y2 - y1

        # Vector from line start to point
        C = x0 - x1
        D = y0 - y1

        # Calculate distance
        dot = C * A + D * B
        len_sq = A * A + B * B

        if len_sq == 0:
            # Line is actually a point
            return np.sqrt(C * C + D * D)

        param = dot / len_sq

        # Find closest point on line segment
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * A
            yy = y1 + param * B

        # Calculate distance
        dx = x0 - xx
        dy = y0 - yy
        return np.sqrt(dx * dx + dy * dy)

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


class ZoneCounter:
    """Handles zone-based counting"""

    def __init__(self, zone_config: CountingZone, frame_size: Tuple[int, int]):
        self.config = zone_config
        self.frame_size = frame_size

        # Convert normalized coordinates to pixels
        self.points_px = [self._denormalize_point(p) for p in zone_config.points_norm]

        # Create mask for the zone
        self.mask = self._create_zone_mask()

        # Counting data
        self.objects_in_zone = set()  # track_ids currently in zone
        self.total_objects_seen = set()  # all track_ids that have been in zone
        self.class_counts = defaultdict(int)  # class_id -> count of unique objects
        self.max_concurrent = 0  # NEW: peak occupancy within the current segment

        self.logger = logging.getLogger(f"{__name__}.{zone_config.name}")

    def _denormalize_point(self, norm_point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized coordinates to pixels"""
        norm_x, norm_y = norm_point
        w, h = self.frame_size
        return (int(norm_x * w), int(norm_y * h))

    def _normalize_point(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        x, y = pt
        w, h = self.frame_size
        return (max(0, min(1, x / float(w))), max(0, min(1, y / float(h))))

    def update_point(self, idx: int, pt_px: Tuple[int, int]) -> None:
        """Move a vertex in-place and rebuild mask; keeps counts intact."""
        if 0 <= idx < len(self.points_px):
            self.points_px[idx] = pt_px
            if 0 <= idx < len(self.config.points_norm):
                self.config.points_norm[idx] = self._normalize_point(pt_px)
            # rebuild mask for hit-testing
            self.mask = self._create_zone_mask()


    def _create_zone_mask(self) -> np.ndarray:
        """Create binary mask for the zone"""
        mask = np.zeros(self.frame_size[::-1], dtype=np.uint8)  # (height, width)
        if len(self.points_px) >= 3:
            points = np.array(self.points_px, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

    def check_presence(self, object_state: ObjectState, current_position: Tuple[int, int], timestamp: Optional[datetime.datetime] = None) -> Optional[CountingEvent]:
        """
        Check if object is in the zone

        Args:
            object_state: Current object state
            current_position: Current position (center point)
            timestamp: Optional timestamp for the detection

        Returns:
            CountingEvent only on initial entry, None for updates
        """
        # Skip if object class not in zone's class filter
        if object_state.class_id not in self.config.classes:
            return None

        # Check if point is in zone
        x, y = current_position
        if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
            is_in_zone = self.mask[y, x] == 255
        else:
            is_in_zone = False

        # Get previous presence state
        was_in_zone = object_state.zone_presence.get(self.config.name, False)

        # Update object's zone presence
        object_state.zone_presence[self.config.name] = is_in_zone

        # Use timestamp for entry time tracking (convert to float if datetime)
        if timestamp is not None:
            if hasattr(timestamp, 'timestamp'):
                entry_timestamp = timestamp.timestamp()
            else:
                entry_timestamp = float(timestamp)
        else:
            entry_timestamp = time.time()

        # Handle zone entry
        if is_in_zone and not was_in_zone:
            # Object entered zone - record entry time using video timestamp
            object_state.zone_entry_times[self.config.name] = entry_timestamp

            self.objects_in_zone.add(object_state.track_id)

            # Count if it's a new object (first time in this zone)
            if object_state.track_id not in self.total_objects_seen:
                self.total_objects_seen.add(object_state.track_id)
                self.class_counts[object_state.class_id] += 1

                # Create entry event (will be updated with dwell time later)
                event = CountingEvent(
                    event_id=f"zone_{self.config.name}_{object_state.track_id}_{entry_timestamp}",
                    track_id=object_state.track_id,
                    class_id=object_state.class_id,
                    class_name=object_state.class_name,
                    zone_name=self.config.name,
                    position=current_position,
                    actual_datetime=timestamp,
                    dwell_seconds=0.0  # Will be updated continuously
                )

                self.logger.info(
                    f"{object_state.class_name} (ID:{object_state.track_id}) entered zone {self.config.name}"
                )

                # Update peak AFTER adding, using actual count
                if self.config.track_max_concurrent:
                    current_count = len(self.objects_in_zone)
                    if current_count > self.max_concurrent:
                        self.max_concurrent = current_count
                        self.logger.debug(f"Zone {self.config.name} new peak: {self.max_concurrent}")

                return event

        # Handle zone exit - calculate and store final dwell time
        elif not is_in_zone and was_in_zone:
            # Object exited zone - calculate dwell time
            if self.config.name in object_state.zone_entry_times:
                zone_entry_time = object_state.zone_entry_times[self.config.name]
                dwell_time = sanitize_dwell_time(entry_timestamp - zone_entry_time)  # FIXED: Sanitize calculation

                # Only record dwell time if >= 0.5 seconds (filter border cases)
                if dwell_time >= 0.5:
                    # Store accumulated dwell time
                    if self.config.name not in object_state.zone_dwell_times:
                        object_state.zone_dwell_times[self.config.name] = 0.0
                    # Add sanitized dwell time (already sanitized above, but double-check)
                    object_state.zone_dwell_times[self.config.name] += sanitize_dwell_time(dwell_time)

                    self.logger.info(
                        f"{object_state.class_name} (ID:{object_state.track_id}) exited zone {self.config.name}, "
                        f"dwell time: {dwell_time:.2f}s"
                    )
                else:
                    self.logger.debug(
                        f"{object_state.class_name} (ID:{object_state.track_id}) exited zone {self.config.name} "
                        f"too quickly (dwell time: {dwell_time:.2f}s) - likely border case, ignoring"
                    )

                # Clean up entry time regardless
                del object_state.zone_entry_times[self.config.name]

            self.objects_in_zone.discard(object_state.track_id)

        return None

    def cleanup_stale_objects(self, active_track_ids: Set[int]):
        """Remove objects that are no longer being tracked"""
        stale_ids = self.objects_in_zone - active_track_ids
        if stale_ids:
            self.objects_in_zone -= stale_ids
            self.logger.debug(f"Removed {len(stale_ids)} stale objects from zone {self.config.name}")

    def draw_zone(self, frame: np.ndarray, show_count: bool = True) -> np.ndarray:
        """Draw the counting zone on frame"""
        if not self.config.enabled or len(self.points_px) < 3:
            return frame

        # Draw zone boundary
        points = np.array(self.points_px, dtype=np.int32)
        cv2.polylines(frame, [points], True, (255, 255, 0), 2)

        # Fill with transparent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], (255, 255, 0))
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Draw zone name and count
        if show_count:
            # Calculate centroid for text position
            centroid = np.mean(points, axis=0).astype(int)
            count_text = f"{self.config.name}: {len(self.total_objects_seen)}"

            # Draw text background
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame,
                          (centroid[0] - text_size[0] // 2 - 5, centroid[1] - text_size[1] - 5),
                          (centroid[0] + text_size[0] // 2 + 5, centroid[1] + 5),
                          (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, count_text,
                        (centroid[0] - text_size[0] // 2, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if self.config.track_max_concurrent and self.config.show_peak_overlay:
                peak_text = f"peak {self.max_concurrent}"
                peak_size = cv2.getTextSize(peak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(
                    frame,
                    peak_text,
                    (centroid[0] - peak_size[0] // 2, centroid[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

        return frame

    def reset_counts(self):
        """Reset all counts"""
        self.objects_in_zone.clear()
        self.total_objects_seen.clear()
        self.class_counts.clear()
        self.max_concurrent = 0
        self.logger.info(f"Reset counts for zone {self.config.name}")

    def get_stats(self) -> Dict:
        """Get counting statistics"""
        return {
            "name": self.config.name,
            "total_unique_objects": len(self.total_objects_seen),
            "current_objects_in_zone": len(self.objects_in_zone),
            "class_counts": dict(self.class_counts),
            "enabled": self.config.enabled,
            "objects_seen": list(self.total_objects_seen),
            "objects_in_zone": list(self.objects_in_zone),
            "max_concurrent": self.max_concurrent
        }


class ObjectCounter:
    """Main counter class that manages all counting operations"""

    def __init__(self, lines_config: List[CountingLine], zones_config: List[CountingZone],
                 frame_size: Tuple[int, int], exclusion_zones: List = None, max_track_age: float = 60.0):
        self.frame_size = frame_size
        self.max_track_age = max_track_age

        # Initialize counters
        self.line_counters = {line.name: LineCounter(line, frame_size) for line in lines_config}
        self.zone_counters = {zone.name: ZoneCounter(zone, frame_size) for zone in zones_config}

        # Object tracking
        self.object_states = {}
        self.counting_events = []

        # Speed settings
        self.enable_speed = True
        self.speed_units = "pxps"
        self.mpp = 0.0
        self.speed_window = 5
        self.annotate_speed = True
        self._last_speeds = {}

        self.zone_entry_events = {}

        # Multiple exclusion zones
        self.exclusion_zones = exclusion_zones or []
        self.exclusion_masks = self._create_exclusion_masks()

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized counter with {len(self.line_counters)} lines, "
            f"{len(self.zone_counters)} zones, and {len(self.exclusion_zones)} exclusion zones")

    def _create_exclusion_masks(self) -> List[Dict]:
        """Create masks for all exclusion zones"""
        masks = []
        for exclusion in self.exclusion_zones:
            if hasattr(exclusion, 'points_norm'):
                # Convert normalized points to pixels
                points_px = []
                for norm_x, norm_y in exclusion.points_norm:
                    px_x = int(norm_x * self.frame_size[0])
                    px_y = int(norm_y * self.frame_size[1])
                    points_px.append((px_x, px_y))

                # Create mask
                mask = np.zeros(self.frame_size[::-1], dtype=np.uint8)
                if len(points_px) >= 3:
                    points_array = np.array(points_px, dtype=np.int32)
                    cv2.fillPoly(mask, [points_array], 255)

                masks.append({
                    'name': exclusion.name,
                    'mask': mask,
                    'points_px': points_px,
                    'enabled': exclusion.enabled
                })
        return masks

    def configure_speed(
            self,
            enable: bool = False,
            units: str = "pxps",
            meters_per_pixel: float = 0.0,
            smooth_window: int = 5,
            annotate: bool = True,
    ) -> None:
        """
        Configure per-object speed estimation.

        Args:
            enable: Turn speed estimation on/off.
            units: "pxps", "mps", "kmh", or "mph".
            meters_per_pixel: Real-world scale; required for non-"pxps" units.
            smooth_window: Number of recent hops used to smooth speed.
            annotate: Whether to draw speed text on labels.
        """
        self.enable_speed = bool(enable)
        self.speed_units = str(units).lower()
        self.mpp = float(meters_per_pixel or 0.0)
        self.speed_window = int(smooth_window or 5)
        self.annotate_speed = bool(annotate)

        # Clear old speeds if disabled so UI doesn't show stale values
        if not self.enable_speed:
            self._last_speeds.clear()

    def _convert_speed(self, v_pxps: float) -> float:
        """
        Convert pixels-per-second to the configured unit.
        If scale is missing for real units, returns 0.0 to avoid misleading values.
        """
        u = self.speed_units
        if u == "pxps":
            return float(v_pxps)

        # need a valid meters-per-pixel for real-world units
        if self.mpp <= 0.0:
            return 0.0

        mps = float(v_pxps) * self.mpp
        if u == "mps":
            return mps
        if u == "kmh":
            return mps * 3.6
        if u == "mph":
            return mps * 2.23693629
        # Fallback: unknown unit -> raw
        return float(v_pxps)

    def set_exclusion_zone(self, points: List[Tuple[int, int]]):
        """Set exclusion zone from pixel coordinates"""
        if len(points) >= 3:
            self.exclusion_mask = np.zeros(self.frame_size[::-1], dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(self.exclusion_mask, [points_array], 255)
            self.logger.info("Exclusion zone set")

    def update_counts(self, detections: List[Detection], timestamp: Optional[datetime.datetime] = None,
                      skip_speed_update: bool = False) -> List[CountingEvent]:
        """Update all counters based on new detections

        Args:
            detections: List of detections to process
            timestamp: Optional timestamp for the detections
            skip_speed_update: If True, skip updating speeds (for interpolated frames)
        """
        new_events = []

        # Use provided timestamp or current time
        current_timestamp = timestamp if timestamp else datetime.datetime.now()
        current_time = current_timestamp.timestamp()

        # Track active IDs for zone cleanup
        active_track_ids = set()

        # Update object states
        for detection in detections:
            # Skip objects in exclusion zone
            if self._is_in_exclusion(detection.center_point):
                continue

            # Track active IDs
            if detection.track_id is not None:
                active_track_ids.add(detection.track_id)

            # Get or create object state
            if detection.track_id not in self.object_states:
                self.object_states[detection.track_id] = ObjectState(
                    track_id=detection.track_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name
                )

            obj_state = self.object_states[detection.track_id]

            # ONLY update position history if NOT skipping speed update
            # This prevents interpolated positions from corrupting speed calculations
            if not skip_speed_update:
                # Update position with video timestamp
                obj_state.positions.append({
                    'center': detection.center_point,
                    'bottom': detection.bottom_point,
                    'timestamp': current_time  # Use video time
                })
                obj_state.last_seen = current_time

                # --- Speed estimation (instantaneous, smoothed) ---
                if self.enable_speed:
                    v_pxps = obj_state.compute_speed_pxps(self.speed_window)
                    v_unit = self._convert_speed(v_pxps)
                    self._last_speeds[detection.track_id] = v_unit

                    # ALWAYS persist converted speeds on the object
                    obj_state.avg_speed_converted = v_unit
                    obj_state.speed_units = self.speed_units

                    # Also store the raw px/s for reference
                    obj_state.avg_speed_pxps = obj_state.avg_speed_pxps
            else:
                # Still update last_seen time even when skipping speed updates
                obj_state.last_seen = current_time

            # Check line crossings (this happens regardless of skip_speed_update)
            for line_counter in self.line_counters.values():
                if not line_counter.config.enabled:
                    continue

                # Select the appropriate point based on the line's poi_mode
                current_pt = (
                    detection.bottom_point
                    if getattr(line_counter.config, "poi_mode", "center") == "bottom"
                    else detection.center_point
                )

                event = line_counter.check_crossing(obj_state, current_pt, current_timestamp)
                if event:
                    # Attach speed & units so exporter can pick them up
                    event.avg_speed = round(float(getattr(obj_state, "avg_speed_converted", 0.0)), 2)
                    event.speed_units = getattr(obj_state, "speed_units", "pxps")

                    new_events.append(event)
                    self.counting_events.append(event)

            #Clean up stale objects from zones BEFORE checking presence
            for zone_counter in self.zone_counters.values():
                zone_counter.cleanup_stale_objects(active_track_ids)

            # Check zone presence
            for zone_counter in self.zone_counters.values():
                if not zone_counter.config.enabled:
                    continue

                # Select the appropriate point based on the zone's poi_mode
                current_pt = (
                    detection.bottom_point
                    if getattr(zone_counter.config, "poi_mode", "center") == "bottom"
                    else detection.center_point
                )

                event = zone_counter.check_presence(obj_state, current_pt, current_timestamp)
                if event:
                    # NEW entry event - track it so we can update dwell time
                    event_key = (event.track_id, event.zone_name)
                    self.zone_entry_events[event_key] = event

                    # Attach speed & initial dwell time (0)
                    event.avg_speed = round(float(getattr(obj_state, "avg_speed_converted", 0.0)), 2)
                    event.speed_units = getattr(obj_state, "speed_units", "pxps")
                    event.dwell_seconds = 0.0  # ENSURE THIS IS 0.0, not some other value

                    new_events.append(event)
                    self.counting_events.append(event)
                else:
                    # Not a new event, but update existing event if object is still in zone
                    event_key = (detection.track_id, zone_counter.config.name)
                    if event_key in self.zone_entry_events:
                        # Object is still in zone - update dwell time
                        if zone_counter.config.name in obj_state.zone_entry_times:
                            entry_time = obj_state.zone_entry_times[zone_counter.config.name]
                            current_dwell = current_time - entry_time

                            # FIXED: Use sanitize function instead of manual validation
                            self.zone_entry_events[event_key].dwell_seconds = sanitize_dwell_time(current_dwell)

                        # Also check if object has exited (not in zone anymore)
                        if not obj_state.zone_presence.get(zone_counter.config.name, False):
                            # Object exited - finalize with total accumulated dwell time
                            if zone_counter.config.name in obj_state.zone_dwell_times:
                                final_dwell = obj_state.zone_dwell_times[zone_counter.config.name]
                                # FIXED: Sanitize final dwell time
                                self.zone_entry_events[event_key].dwell_seconds = sanitize_dwell_time(final_dwell)

                            # Remove from tracking since it's finalized
                            del self.zone_entry_events[event_key]

        # Clean up stale object states and finalize zone dwells
        stale_ids = [
            track_id for track_id, obj_state in self.object_states.items()
            if obj_state.is_stale(self.max_track_age) and track_id not in active_track_ids
        ]

        for track_id in stale_ids:
            obj_state = self.object_states[track_id]

            # Finalize any active zone dwells before removing
            for zone_name in list(obj_state.zone_presence.keys()):
                if obj_state.zone_presence.get(zone_name, False):
                    # Object is still marked as in zone but is now stale
                    if zone_name in obj_state.zone_entry_times:
                        entry_time = obj_state.zone_entry_times[zone_name]
                        dwell = current_time - entry_time

                        # Add to accumulated dwell time
                        if zone_name not in obj_state.zone_dwell_times:
                            obj_state.zone_dwell_times[zone_name] = 0.0
                        obj_state.zone_dwell_times[zone_name] += dwell

                        # Update the tracked event with final dwell time
                        event_key = (track_id, zone_name)
                        if event_key in self.zone_entry_events:
                            self.zone_entry_events[event_key].dwell_seconds = round(
                                obj_state.zone_dwell_times[zone_name], 2
                            )
                            del self.zone_entry_events[event_key]

                        # Mark as no longer present
                        obj_state.zone_presence[zone_name] = False
                        del obj_state.zone_entry_times[zone_name]

            del self.object_states[track_id]

        return new_events

    def _is_in_exclusion(self, point: Tuple[int, int]) -> bool:
        """Check if point is in any exclusion zone"""
        x, y = point

        for exclusion in self.exclusion_masks:
            if not exclusion['enabled']:
                continue

            mask = exclusion['mask']
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] == 255:
                    self.logger.debug(f"Point {point} is in exclusion zone '{exclusion['name']}'")
                    return True
        return False

    def _cleanup_stale_objects(self):
        """Remove stale object states"""
        stale_ids = [track_id for track_id, obj_state in self.object_states.items()
                     if obj_state.is_stale(self.max_track_age)]

        for track_id in stale_ids:
            del self.object_states[track_id]

        if stale_ids:
            self.logger.debug(f"Cleaned up {len(stale_ids)} stale object states")

    def draw_overlays(self, frame: np.ndarray, show_counts: bool = True) -> np.ndarray:
        """Draw all counting overlays on frame"""
        # Draw lines
        for line_counter in self.line_counters.values():
            frame = line_counter.draw_line(frame, show_counts)

        # Draw zones
        for zone_counter in self.zone_counters.values():
            frame = zone_counter.draw_zone(frame, show_counts)

        # Draw exclusion zones
        for exclusion in self.exclusion_masks:
            if exclusion['enabled']:
                # Create red overlay for exclusion
                points = np.array(exclusion['points_px'], dtype=np.int32)
                if len(points) >= 3:
                    # Draw boundary
                    cv2.polylines(frame, [points], True, (0, 0, 255), 2)

                    # Fill with transparent overlay
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points], (0, 0, 255))
                    frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)

                    # Draw name
                    centroid = np.mean(points, axis=0).astype(int)
                    cv2.putText(frame, f"EXCL: {exclusion['name']}", tuple(centroid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def get_current_counts(self) -> Dict:
        """Get current counting statistics"""
        stats = {
            "lines": {name: counter.get_stats() for name, counter in self.line_counters.items()},
            "zones": {name: counter.get_stats() for name, counter in self.zone_counters.items()},
            "total_events": len(self.counting_events),
            "active_objects": len(self.object_states)
        }
        return stats

    def reset_all_counts(self):
        """Reset all counters"""
        for line_counter in self.line_counters.values():
            line_counter.reset_counts()

        for zone_counter in self.zone_counters.values():
            zone_counter.reset_counts()

        self.counting_events.clear()
        self.object_states.clear()


        self.logger.info("All counters reset")

    def get_events_summary(self) -> Dict:
        """Get summary of all counting events with enriched data"""
        events_list = []

        for event in self.counting_events:
            event_dict = {
                'event_id': event.event_id,
                'track_id': event.track_id,
                'class_id': event.class_id,
                'class_name': event.class_name,
                'timestamp': event.timestamp,
                'position': event.position,
                'actual_datetime': str(event.actual_datetime) if event.actual_datetime else None
            }

            # Add line-specific data
            if event.line_name:
                event_dict['line_name'] = event.line_name
                event_dict['direction'] = event.direction

                # FIXED: Use speed from event if available, otherwise fallback
                speed_val = getattr(event, 'avg_speed', 0.0)
                if speed_val == 0.0 and event.track_id in self._last_speeds:
                    speed_val = self._last_speeds[event.track_id]

                event_dict['speed'] = float(speed_val or 0.0)
                event_dict['speed_units'] = getattr(event, 'speed_units', self.speed_units)

            # Add zone-specific data
            elif event.zone_name:
                event_dict['zone_name'] = event.zone_name

                # FIXED: Use speed from event if available, otherwise fallback
                speed_val = getattr(event, 'avg_speed', 0.0)
                if speed_val == 0.0 and event.track_id in self._last_speeds:
                    speed_val = self._last_speeds[event.track_id]

                event_dict['speed'] = float(speed_val or 0.0)
                event_dict['speed_units'] = getattr(event, 'speed_units', self.speed_units)

                # FIXED: Get dwell time from event or calculate from object state
                dwell_sec = getattr(event, 'dwell_seconds', 0.0)

                # Validate dwell time - if negative or unreasonably large, set to 0
                if dwell_sec < 0 or dwell_sec > 86400:  # More than 24 hours is probably an error
                    dwell_sec = 0.0

                if dwell_sec == 0.0 and event.track_id in self.object_states:
                    obj_state = self.object_states[event.track_id]

                    # Check if object is still in zone
                    if event.zone_name in obj_state.zone_presence and obj_state.zone_presence[event.zone_name]:
                        # Object still in zone - calculate current dwell time
                        if event.zone_name in obj_state.zone_entry_times:
                            entry_time = obj_state.zone_entry_times[event.zone_name]
                            dwell_sec = time.time() - entry_time

                            # Validate calculated dwell time
                            if dwell_sec < 0 or dwell_sec > 86400:
                                dwell_sec = 0.0
                    else:
                        # Object has left zone - use stored dwell time
                        if event.zone_name in obj_state.zone_dwell_times:
                            dwell_sec = obj_state.zone_dwell_times[event.zone_name]

                            # Validate stored dwell time
                            if dwell_sec < 0 or dwell_sec > 86400:
                                dwell_sec = 0.0

                event_dict['dwell_time'] = f"{dwell_sec:.2f}s" if dwell_sec > 0 else ""
                event_dict['dwell_seconds'] = sanitize_dwell_time(dwell_sec)

            events_list.append(event_dict)

        # Count events by type
        line_crossings = sum(1 for e in events_list if 'line_name' in e)
        zone_entries = sum(1 for e in events_list if 'zone_name' in e)

        return {
            'line_crossings': line_crossings,
            'zone_entries': zone_entries,
            'total_events': len(self.counting_events),
            'events': events_list
        }


    def add_line(self, line_cfg: CountingLine) -> None:
        """Add a new counting line at runtime."""
        if line_cfg.name in self.line_counters:
            self.logger.warning(f"Line '{line_cfg.name}' already exists; replacing.")
        self.line_counters[line_cfg.name] = LineCounter(line_cfg, self.frame_size)
        self.logger.info(f"Added line '{line_cfg.name}'")

    def remove_line(self, name: str) -> bool:
        """Remove a counting line by name. Returns True if removed."""
        removed = self.line_counters.pop(name, None) is not None
        if removed:
            self.logger.info(f"Removed line '{name}'")
        else:
            self.logger.warning(f"Line '{name}' not found")
        return removed

    def add_zone(self, zone_cfg: CountingZone) -> None:
        """Add a new counting zone at runtime."""
        if zone_cfg.name in self.zone_counters:
            self.logger.warning(f"Zone '{zone_cfg.name}' already exists; replacing.")
        self.zone_counters[zone_cfg.name] = ZoneCounter(zone_cfg, self.frame_size)
        self.logger.info(f"Added zone '{zone_cfg.name}'")

    def remove_zone(self, name: str) -> bool:
        """Remove a counting zone by name. Returns True if removed."""
        removed = self.zone_counters.pop(name, None) is not None
        if removed:
            self.logger.info(f"Removed zone '{name}'")
        else:
            self.logger.warning(f"Zone '{name}' not found")
        return removed

    def find_nearest_line(self, pt: tuple, max_dist_px: float = 20.0) -> Optional[str]:
        """
        Return the name of the nearest line within max_dist_px of point pt (x,y),
        else None.
        """
        best_name = None
        best_d = max_dist_px
        for name, lc in self.line_counters.items():
            d = lc.get_distance_to_point(pt)
            if d <= best_d:
                best_d = d
                best_name = name
        return best_name

    def zone_contains_point(self, pt: tuple) -> Optional[str]:
        """
        If the point falls inside any zone's mask, return that zone name; else None.
        """
        x, y = pt
        for name, zc in self.zone_counters.items():
            # ZoneCounter builds a binary mask (255=inside) for hit testing
            mask = getattr(zc, "mask", None)
            if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] == 255:
                    return name
        return None

    def set_frame_size(self, frame_size: tuple) -> None:
        """
        If you ever change the working frame size, call this to rebuild counters
        without losing their configs.
        """
        self.frame_size = frame_size
        # Keep configs and re-instantiate counters with new geometry
        line_cfgs = [lc.config for lc in self.line_counters.values()]
        zone_cfgs = [zc.config for zc in self.zone_counters.values()]
        self.line_counters = {cfg.name: LineCounter(cfg, frame_size) for cfg in line_cfgs}
        self.zone_counters = {cfg.name: ZoneCounter(cfg, frame_size) for cfg in zone_cfgs}
        self.exclusion_masks = self._create_exclusion_masks()
        self.logger.info(f"Rebuilt counters for frame size: {frame_size}")


    def update_events_with_final_stats(self) -> None:
        """Update all events with final/current statistics before export"""
        current_time = time.time()

        for event in self.counting_events:
            if event.track_id not in self.object_states:
                continue

            obj_state = self.object_states[event.track_id]

            # Update to FINAL average speed (across entire tracking lifetime)
            if hasattr(obj_state, 'avg_speed_converted'):
                event.avg_speed = round(float(obj_state.avg_speed_converted), 2)
            else:
                # Fallback: calculate from raw speed
                event.avg_speed = round(float(obj_state.avg_speed_pxps * self.mpp),
                                        2) if self.mpp > 0 else obj_state.avg_speed_pxps

            event.speed_units = self.speed_units

            # Update zone dwell time to TOTAL time spent in zone
            if event.zone_name:
                zone_name = event.zone_name

                # Calculate total dwell time
                total_dwell = obj_state.zone_dwell_times.get(zone_name, 0.0)

                # If still in zone, add current session time
                if obj_state.zone_presence.get(zone_name, False):
                    if zone_name in obj_state.zone_entry_times:
                        current_session = current_time - obj_state.zone_entry_times[zone_name]
                        total_dwell += current_session

                # FIXED: Sanitize final dwell time
                event.dwell_seconds = sanitize_dwell_time(total_dwell)