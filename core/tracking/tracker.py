import numpy as np
import cv2
import time
import datetime
from typing import Dict, List, Tuple, Optional
import logging

from core.detection_engine import Detection
from config_manager import CountingLine, CountingZone

# --- IMPORT FROM OUR NEW LOCAL FILES ---
from .schemas import ObjectState, CountingEvent, sanitize_dwell_time
from .line_math import LineCounter
from .zone_math import ZoneCounter


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

        # --- FIX: Synchronize Zone Reality ---
        # Evaluate all zones and forcefully evict any ghosts or stale IDs
        # so the UI and max concurrent tracking stay 100% accurate!
        for zone_counter in self.zone_counters.values():
            if zone_counter.config.enabled:
                zone_counter.sync_occupancy(self.object_states, current_time)

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

    def get_events_summary(self, current_video_time: datetime.datetime = None) -> Dict:
        """Get summary of all counting events with enriched data

        Args:
            current_video_time: The current video timestamp (datetime). If None, uses time.time()
                               which may cause issues with pre-recorded videos.
        """
        # Use video timestamp if provided, otherwise fall back to wall clock
        if current_video_time is not None:
            if hasattr(current_video_time, 'timestamp'):
                current_time = current_video_time.timestamp()
            else:
                current_time = float(current_video_time)
        else:
            current_time = time.time()

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
                        # Object still in zone - calculate current dwell time using VIDEO time
                        if event.zone_name in obj_state.zone_entry_times:
                            entry_time = obj_state.zone_entry_times[event.zone_name]
                            dwell_sec = current_time - entry_time

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

                # Apply minimum dwell threshold of 0.3 seconds for export
                event_dict['dwell_time'] = f"{dwell_sec:.2f}s" if dwell_sec >= 0.3 else ""
                event_dict['dwell_seconds'] = sanitize_dwell_time(dwell_sec, min_dwell=0.3)

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

    def update_events_with_final_stats(self, current_video_time: datetime.datetime = None) -> None:
        """Update all events with final/current statistics before export

        Args:
            current_video_time: The current video timestamp (datetime). If None, uses time.time()
                               which may cause issues with pre-recorded videos.
        """
        # Use video timestamp if provided, otherwise fall back to wall clock
        if current_video_time is not None:
            if hasattr(current_video_time, 'timestamp'):
                current_time = current_video_time.timestamp()
            else:
                current_time = float(current_video_time)
        else:
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
                        entry_time = obj_state.zone_entry_times[zone_name]
                        current_session = current_time - entry_time
                        # Only add if reasonable (positive and less than 24 hours)
                        if 0 <= current_session <= 86400:
                            total_dwell += current_session

                # FIXED: Sanitize final dwell time with minimum threshold of 0.3 seconds
                event.dwell_seconds = sanitize_dwell_time(total_dwell, min_dwell=0.3)