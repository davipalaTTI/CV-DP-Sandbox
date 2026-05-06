import numpy as np
import cv2
from typing import Tuple, Dict, Set, Optional
from config_manager import CountingZone
from collections import defaultdict
import logging
import datetime
import time

from core.tracking.schemas import ObjectState, CountingEvent, sanitize_dwell_time


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
        # Skip if object class not in zone's class filter
        if object_state.class_id not in self.config.classes:
            return None

        # Check if point is in zone
        x, y = current_position
        if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
            is_in_zone = self.mask[y, x] == 255
        else:
            is_in_zone = False

        was_in_zone = object_state.zone_presence.get(self.config.name, False)
        object_state.zone_presence[self.config.name] = is_in_zone

        # Use timestamp for entry time tracking
        if timestamp is not None:
            if hasattr(timestamp, 'timestamp'):
                entry_timestamp = timestamp.timestamp()
            else:
                entry_timestamp = float(timestamp)
        else:
            entry_timestamp = time.time()

        # Handle zone entry
        if is_in_zone and not was_in_zone:
            object_state.zone_entry_times[self.config.name] = entry_timestamp

            # Count if it's a new object (first time in this zone)
            if object_state.track_id not in self.total_objects_seen:
                self.total_objects_seen.add(object_state.track_id)
                self.class_counts[object_state.class_id] += 1

                event = CountingEvent(
                    event_id=f"zone_{self.config.name}_{object_state.track_id}_{entry_timestamp}",
                    track_id=object_state.track_id,
                    class_id=object_state.class_id,
                    class_name=object_state.class_name,
                    zone_name=self.config.name,
                    position=current_position,
                    actual_datetime=timestamp,
                    dwell_seconds=0.0
                )
                self.logger.info(f"{object_state.class_name} (ID:{object_state.track_id}) entered zone {self.config.name}")
                return event

        # Handle zone exit - calculate and store final dwell time
        elif not is_in_zone and was_in_zone:
            if self.config.name in object_state.zone_entry_times:
                zone_entry_time = object_state.zone_entry_times[self.config.name]
                dwell_time = sanitize_dwell_time(entry_timestamp - zone_entry_time)

                if dwell_time >= 0.5:
                    if self.config.name not in object_state.zone_dwell_times:
                        object_state.zone_dwell_times[self.config.name] = 0.0
                    object_state.zone_dwell_times[self.config.name] += sanitize_dwell_time(dwell_time)
                del object_state.zone_entry_times[self.config.name]

        return None

    def sync_occupancy(self, object_states: Dict, current_time: float):
        """Sync the zone's occupancy with the tracker's ground truth, filtering out ghosts."""
        self.objects_in_zone.clear()

        for track_id, state in object_states.items():
            # Check if the object is currently marked as inside the zone
            if state.zone_presence.get(self.config.name, False):

                # Filter out ghost tracks (ID switches) by ensuring the object
                # was explicitly detected within the last 0.5 seconds
                time_since_last_seen = current_time - state.last_seen

                if time_since_last_seen < 0.5:
                    self.objects_in_zone.add(track_id)
                else:
                    # It's a ghost track. Forcefully evict it so it doesn't inflate counts!
                    state.zone_presence[self.config.name] = False
                    if self.config.name in state.zone_entry_times:
                        del state.zone_entry_times[self.config.name]

        # Update peak tracking based on reality
        if self.config.track_max_concurrent:
            current_count = len(self.objects_in_zone)
            if current_count > self.max_concurrent:
                self.max_concurrent = current_count
                self.logger.debug(f"Zone {self.config.name} new peak: {self.max_concurrent}")

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

            # --- FIX: Change the text based on the UI mode! ---
            if self.config.track_max_concurrent:
                # Occupancy Mode: Show currently inside
                count_text = f"{self.config.name}: {len(self.objects_in_zone)} inside"
            else:
                # Standard Mode: Show all-time historical count
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