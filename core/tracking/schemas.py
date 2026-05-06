from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import time
import datetime

def sanitize_dwell_time(dwell_seconds: float, min_dwell: float = 0.0) -> float:
    """
    Validate and sanitize dwell time values.
    Returns 0.0 for invalid values (negative, below minimum, or unreasonably large).

    Args:
        dwell_seconds: Raw dwell time value
        min_dwell: Minimum dwell time threshold (default 0.0, use 0.3 for export filtering)

    Returns:
        Sanitized dwell time (0.0 if invalid, otherwise rounded to 2 decimals)
    """
    try:
        dwell = float(dwell_seconds) if dwell_seconds is not None else 0.0
        # Invalid if negative or greater than 24 hours (86400 seconds)
        if dwell < 0 or dwell > 86400:
            return 0.0
        # Filter below minimum threshold
        if dwell < min_dwell:
            return 0.0
        return round(dwell, 2)
    except (ValueError, TypeError):
        return 0.0

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