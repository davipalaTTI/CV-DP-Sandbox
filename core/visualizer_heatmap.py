
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, Iterable
import time

import cv2
import numpy as np

from utils.file_io import sanitize_filename, generate_unique_filename


class HeatmapAccumulator:
    """
    Accumulates object presence over a time window using detection boxes/points.
    Produces a single overlay snapshot per interval, then resets.

    Optimized with NumPy Vectorization for high-performance Jetson deployment.
    """

    def __init__(
            self,
            frame_size: Tuple[int, int],  # (h, w)
            alpha: float = 0.35,
            colormap: int = cv2.COLORMAP_HOT,
            out_dir: str = "outputs/heatmaps",
            interval_sec: float = 10.0,
            radius_px: int = 10,
            decay: float = 0.0,
            gamma: float = 1.6,
            saturation_boost: float = 1.0
    ) -> None:
        self.h, self.w = frame_size
        self.alpha = float(alpha)
        self.colormap = int(colormap)
        self.interval_sec = float(interval_sec)
        self.radius_px = int(max(0, radius_px))
        self.decay = float(max(0.0, min(1.0, decay)))
        self.gamma = float(max(0.1, gamma))
        self.saturation_boost = float(max(1.0, saturation_boost))
        self.accum = np.zeros((self.h, self.w), dtype=np.float32)
        self.last_emit_t = time.time()

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # --- PRECOMPUTE THE GAUSSIAN KERNEL ---
        # Instead of calculating pixel distance thousands of times per frame,
        # we calculate the perfect glowing dot once and stamp it later.
        if self.radius_px > 0:
            r = self.radius_px
            size = 2 * r + 1
            y, x = np.ogrid[-r:r + 1, -r:r + 1]
            dist_sq = x * x + y * y
            mask = dist_sq <= r * r
            sigma = r / 3.0
            kernel = np.exp(-dist_sq / (2 * sigma * sigma))
            self._kernel = (kernel * mask).astype(np.float32)
            self._kernel_size = size

    # -------------------- core state --------------------

    def reset(self) -> None:
        """Clear the accumulation buffer."""
        self.accum.fill(0.0)

    def _apply_decay(self) -> None:
        """Apply optional exponential-ish decay to gradually fade old heat."""
        if self.decay > 0.0:
            self.accum *= (1.0 - self.decay)

    # -------------------- updates --------------------

    def update_from_boxes(self, boxes_xyxy: Iterable[Tuple[int, int, int, int]], weight: float = 1.5) -> None:
        """
        Add heat from bounding boxes using ultra-fast NumPy matrix slicing
        """
        self._apply_decay()
        if not boxes_xyxy:
            return

        w, h = self.w, self.h
        wgt = float(max(0.0, weight))

        if self.radius_px > 0:
            # Pre-scale the master kernel by the weight
            scaled_kernel = self._kernel * wgt
            r = self.radius_px
            k_size = self._kernel_size

            for (x1, y1, x2, y2) in boxes_xyxy:
                cx = int(0.5 * (x1 + x2))
                cy = int(0.5 * (y1 + y2))

                # Target bounds on the master image
                y_min, y_max = cy - r, cy + r + 1
                x_min, x_max = cx - r, cx + r + 1

                # Clip bounds to ensure we don't stamp outside the screen
                valid_y_min, valid_y_max = max(0, y_min), min(h, y_max)
                valid_x_min, valid_x_max = max(0, x_min), min(w, x_max)

                # Calculate the exact slice of the stamp we need
                k_y_min = valid_y_min - y_min
                k_y_max = k_size - (y_max - valid_y_max)
                k_x_min = valid_x_min - x_min
                k_x_max = k_size - (x_max - valid_x_max)

                # Instantly stamp the matrix at C-level speed
                if valid_y_max > valid_y_min and valid_x_max > valid_x_min:
                    self.accum[valid_y_min:valid_y_max, valid_x_min:valid_x_max] += \
                        scaled_kernel[k_y_min:k_y_max, k_x_min:k_x_max]
        else:
            # Vectorized gradient filling for bounding boxes
            for (x1, y1, x2, y2) in boxes_xyxy:
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))

                box_w, box_h = x2 - x1, y2 - y1

                if box_w > 0 and box_h > 0:
                    cx_rel, cy_rel = box_w / 2.0, box_h / 2.0

                    # Create 1D distance arrays
                    x_idx = np.arange(box_w)
                    y_idx = np.arange(box_h)

                    # Normalize distances
                    dx = np.abs(x_idx - cx_rel) / max(cx_rel, 1.0)
                    dy = np.abs(y_idx - cy_rel) / max(cy_rel, 1.0)

                    # Create 2D weight matrix via NumPy broadcasting (dy as column, dx as row)
                    center_weight = 1.0 - 0.3 * np.maximum(dy[:, None], dx[None, :])

                    self.accum[y1:y2, x1:x2] += wgt * center_weight

    def update_from_points(self, points_xy: Iterable[Tuple[int, int]], weight: float = 1.0) -> None:
        """
        Add heat from a list of points using OpenCV's optimized C++ drawing
        """
        self._apply_decay()
        if not points_xy:
            return

        wgt = float(max(0.0, weight))
        r = max(1, self.radius_px)

        for (x, y) in points_xy:
            if 0 <= x < self.w and 0 <= y < self.h:
                cv2.circle(self.accum, (int(x), int(y)), r, wgt, thickness=-1)

    # -------------------- rendering/saving --------------------

    def _make_colormap(self) -> np.ndarray:
        """Create vibrant heatmap with enhanced contrast"""
        heat = self.accum
        if heat.size == 0:
            return np.zeros((self.h, self.w), dtype=np.uint8)

        vmax = float(heat.max())
        if vmax <= 0:
            return np.zeros_like(heat, dtype=np.uint8)

        norm = heat / vmax
        norm = np.power(norm, 1.0 / self.gamma)
        norm = 1.0 / (1.0 + np.exp(-10 * (norm - 0.3)))
        heat_u8 = (255.0 * norm).astype(np.uint8)

        return heat_u8

    def make_overlay(self, frame_bgr: Optional[np.ndarray]) -> np.ndarray:
        """Blend vibrant heatmap over frame with color enhancement"""
        heat_u8 = self._make_colormap()
        color_map = cv2.applyColorMap(heat_u8, self.colormap)

        if self.saturation_boost > 1.0:
            hsv = cv2.cvtColor(color_map, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation_boost, 0, 255)
            color_map = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if frame_bgr is None:
            base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w:
                color_map = cv2.resize(color_map, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            base = frame_bgr

        base_norm = base.astype(np.float32) / 255.0
        color_norm = color_map.astype(np.float32) / 255.0
        screen_blend = 1.0 - (1.0 - base_norm) * (1.0 - color_norm * self.alpha)

        return (screen_blend * 255).astype(np.uint8)

    def render_and_save(
            self,
            frame_bgr: Optional[np.ndarray],
            label: Optional[str] = None,
            when: Optional[float] = None,
            suffix: Optional[str] = None,
    ) -> str:
        """Render heatmap and write to PNG"""
        t_now = self._as_timestamp(when)
        start_t = getattr(self, "last_emit_t", None)
        if start_t is None:
            start_t = t_now - float(getattr(self, "interval_sec", 0) or 0)

        start_t = self._as_timestamp(start_t)
        start_dt = datetime.fromtimestamp(start_t)
        end_dt = datetime.fromtimestamp(t_now)

        overlay_bgr = self.make_overlay(frame_bgr)

        date_part = start_dt.strftime("%Y_%d_%m")
        stem = f"heatmap_{date_part}_{start_dt.strftime('%H%M')}_to_{end_dt.strftime('%H%M')}"

        if label:
            stem += f"_{sanitize_filename(str(label))[:32]}"
        if suffix and str(suffix) != str(label):
            stem += f"_{sanitize_filename(str(suffix))[:16]}"

        out_path = generate_unique_filename(Path(self.out_dir) / stem, extension=".png")

        ok = cv2.imwrite(str(out_path), overlay_bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for '{out_path}'")

        self.last_emit_t = float(t_now)
        self.reset()
        return str(out_path)

    def maybe_emit(
            self,
            frame_bgr: Optional[np.ndarray],
            t_now: Optional[float] = None,
            label: Optional[str] = None
    ) -> Optional[str]:
        t_now = self._as_timestamp(t_now)
        last = getattr(self, "last_emit_t", None)

        if last is None:
            self.last_emit_t = float(t_now)
            return None

        last_ts = self._as_timestamp(last)
        if (t_now - last_ts) >= float(getattr(self, "interval_sec", 0) or 0):
            return self.render_and_save(frame_bgr=frame_bgr, label=label, when=t_now)

        return None

    def _as_timestamp(self, t) -> float:
        if t is None:
            return time.time()
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, datetime):
            return t.timestamp()
        if isinstance(t, date):
            return datetime(t.year, t.month, t.day).timestamp()
        raise TypeError(f"Expected timestamp float/int or datetime/date, got {type(t)}")

    def set_interval_minutes(self, minutes: int) -> None:
        self.interval_sec = max(1.0, float(minutes) * 60.0)

    def set_colormap(self, colormap: int) -> None:
        self.colormap = int(colormap)

    def _build_filename(self, start_dt, end_dt, suffix: str = "") -> str:
        date_part = start_dt.strftime("%Y_%d_%m")
        t_start = start_dt.strftime("%H%M")
        t_end = end_dt.strftime("%H%M")
        stem = f"heatmap_{date_part}_{t_start}_to_{t_end}"
        if suffix:
            stem += f"_{suffix}"
        return stem