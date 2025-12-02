import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import logging
import torch


@dataclass
class Detection:
    """Standardized detection result"""
    track_id: Optional[int]
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center_point: Tuple[int, int]  # cx, cy
    bottom_point: Tuple[int, int]  # cx, bottom_y (for line crossing)


class DetectionEngine:
    """
    YOLO-based detection and tracking engine with configurable filtering
    """

    def __init__(self,
                 model_path: str,
                 confidence_threshold: float = 0.45,
                 tracker_config: Optional[str] = None,
                 allowed_classes: Optional[Set[int]] = None,
                 device: str = 'auto'):
        """
        Initialize detection engine
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.allowed_classes = allowed_classes
        self.device = device

        # Add exclusion mask
        self.exclusion_mask = None
        self.frame_shape = None

        # Initialize last_detections for frame skipping/interpolation
        self.last_detections = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load YOLO model with explicit device selection."""
        try:
            import torch

            # Resolve "auto" into a real target device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_str = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_str = "mps"
                else:
                    device_str = "cpu"
            else:
                device_str = self.device  # "cuda", "cpu", "mps", etc.

            # Load YOLO model (ONNX or PT)
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names

            suffix = Path(self.model_path).suffix.lower()

            #  Decide how we'll pass device into inference calls
            if device_str.startswith("cuda"):
                # use GPU 0 by default
                self.inference_device = 0
                self.device = "cuda:0"
            else:
                self.inference_device = device_str
                self.device = device_str

            # If it's a PyTorch model (.pt), try to physically move it to device.
            #    For ONNX, we SKIP .to() and let 'device' in track() control backend.
            if suffix == ".pt":
                try:
                    if hasattr(self.model, "to"):
                        self.model.to(device_str)
                    elif hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                        self.model.model.to(device_str)
                except Exception as move_err:
                    self.logger.warning(f"Could not move model to {device_str}: {move_err}")

            # Setup allowed classes if not specified
            if self.allowed_classes is None:
                if isinstance(self.class_names, dict):
                    self.allowed_classes = set(int(k) for k in self.class_names.keys())
                else:
                    self.allowed_classes = set(range(len(self.class_names)))

            self.logger.info(f"Model loaded: {self.model_path}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Available classes: {list(self.class_names.values())}")
            self.logger.info(
                f"Allowed classes: {[self.class_names[i] for i in self.allowed_classes]}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_path}: {e}")
            raise

    def detect_and_track(self,
                         frame: np.ndarray,
                         persist_tracks: bool = True) -> List[Detection]:
        """
        Run detection and tracking on a frame
        """
        try:
            # Apply exclusion mask to a COPY for detection only
            detection_frame = frame  # Use original by default

            if self.exclusion_mask is not None:
                # Process a masked copy, but don't modify the original
                detection_frame = self._apply_exclusion_mask(frame)

            # Run YOLO inference on the masked frame
            kwargs = {
                'persist': persist_tracks,
                'conf': self.confidence_threshold,
                'verbose': False,
                'tracker': "C:/Users/D-Palacios/PycharmProjects/CV-DP-Sandbox/.venv/Lib/site-packages/ultralytics/cfg/trackers/botsort.yaml"
            }

            # Add class filtering if specified
            if self.allowed_classes:
                kwargs['classes'] = sorted(list(self.allowed_classes))

            # Add tracker config if specified
            if self.tracker_config:
                kwargs['tracker'] = self.tracker_config

            # Run on detection_frame (masked), not original frame
            results = self.model.track(detection_frame, **kwargs)

            # Parse results - these coordinates are still valid for the original frame
            detections = self._parse_results(results, frame.shape)

            # Store for interpolation
            self.last_detections = detections

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def set_exclusion_zones(self, exclusion_zones: List, frame_shape: Tuple[int, int, int]):
        """
        Create a mask from exclusion zones to black out those regions

        Args:
            exclusion_zones: List of ExclusionZone objects with points_norm
            frame_shape: (height, width, channels) of the video frames
        """
        if not exclusion_zones:
            self.exclusion_mask = None
            self.frame_shape = None
            return

        h, w = frame_shape[:2]
        self.frame_shape = (h, w)

        # Create white mask (255 = process, 0 = exclude)
        self.exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Draw exclusion zones as black (0)
        for zone in exclusion_zones:
            # Denormalize points from normalized coordinates to pixel coordinates
            pts_pixel = [(int(x * w), int(y * h)) for x, y in zone.points_norm]
            pts = np.array(pts_pixel, dtype=np.int32)
            cv2.fillPoly(self.exclusion_mask, [pts], 0)

        self.logger.info(f"Exclusion mask created: {len(exclusion_zones)} zones")

    def _apply_exclusion_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply exclusion mask to frame before detection

        Args:
            frame: Input frame

        Returns:
            Masked frame (excluded regions are black)
        """
        if self.exclusion_mask is None:
            return frame

        # Verify frame shape matches mask
        if frame.shape[:2] != self.frame_shape:
            self.logger.warning("Frame shape doesn't match mask, regenerating mask")
            return frame

        # Apply mask - black out excluded regions
        masked_frame = frame.copy()
        masked_frame[self.exclusion_mask == 0] = 0

        return masked_frame

    def _parse_results(self, results, frame_shape: Tuple[int, int, int]) -> List[Detection]:
        """Optimized parsing with single GPU->CPU transfer"""
        detections = []

        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes

        # Get device used if set to auto
        try:
            # Only override if it's still "auto"
            if str(self.device).lower() in ("auto", "", "none", "unknown"):
                tensor = boxes.xyxy
                if tensor is not None and hasattr(tensor, "device"):
                    self.device = str(tensor.device)
        except Exception:
            pass

        if boxes.xyxy is None:
            return detections

        # Single batch transfer to CPU
        with torch.no_grad():
            # Stack all tensors and transfer once
            bbox_data = boxes.xyxy.cpu()
            conf_data = boxes.conf.cpu() if boxes.conf is not None else torch.ones(len(bbox_data))
            class_data = boxes.cls.cpu() if boxes.cls is not None else torch.zeros(len(bbox_data))
            track_data = boxes.id.cpu() if boxes.id is not None else None

            # Convert to numpy in one go
            bbox_np = bbox_data.numpy()
            conf_np = conf_data.numpy()
            class_np = class_data.numpy().astype(int)
            track_np = track_data.numpy().astype(int) if track_data is not None else [None] * len(bbox_np)

        # Vectorized filtering instead of loop
        valid_mask = conf_np >= self.confidence_threshold
        if self.allowed_classes:
            class_mask = np.isin(class_np, list(self.allowed_classes))
            valid_mask &= class_mask

        # Process only valid detections
        valid_indices = np.where(valid_mask)[0]

        h, w = frame_shape[:2]

        for i in valid_indices:
            # Use numpy operations
            x1, y1, x2, y2 = bbox_np[i].astype(int)

            # Vectorized clamping
            x1, x2 = np.clip([x1, x2], 0, w - 1)
            y1, y2 = np.clip([y1, y2], 0, h - 1)

            detection = Detection(
                track_id=int(track_np[i]) if track_np[i] is not None else None,
                class_id=int(class_np[i]),
                class_name=self.class_names.get(int(class_np[i]), f"class_{class_np[i]}"),
                bbox=(x1, y1, x2, y2),
                confidence=float(conf_np[i]),
                center_point=((x1 + x2) // 2, (y1 + y2) // 2),
                bottom_point=((x1 + x2) // 2, y2)
            )
            detections.append(detection)

        return detections

    def update_allowed_classes(self, new_classes: Set[int]):
        """Update the set of allowed classes"""
        self.allowed_classes = new_classes
        self.logger.info(f"Updated allowed classes: {[self.class_names[i] for i in new_classes]}")

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Updated confidence threshold: {self.confidence_threshold}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'class_names': self.class_names,
            'allowed_classes': list(self.allowed_classes) if self.allowed_classes else None,
            'confidence_threshold': self.confidence_threshold,
            'device': self.device
        }

    def warmup(self, frame_size: Tuple[int, int] = (640, 640)):
        """
        Warm up the model with a dummy inference

        Args:
            frame_size: (width, height) for dummy frame
        """
        try:
            dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            _ = self.detect_and_track(dummy_frame)
            self.logger.info("Model warmup completed")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")


class MultiModelEngine:
    """
    Engine that can handle multiple models for different scenarios
    """

    def __init__(self):
        self.engines = {}
        self.current_engine = None
        self.logger = logging.getLogger(__name__)

    def add_engine(self, name: str, engine: DetectionEngine):
        """Add a detection engine with a given name"""
        self.engines[name] = engine
        if self.current_engine is None:
            self.current_engine = name
        self.logger.info(f"Added engine '{name}'")

    def switch_engine(self, name: str):
        """Switch to a different engine"""
        if name not in self.engines:
            raise ValueError(f"Engine '{name}' not found")
        self.current_engine = name
        self.logger.info(f"Switched to engine '{name}'")

    def detect_and_track(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """Run detection using current engine"""
        if self.current_engine is None:
            raise RuntimeError("No engine selected")
        return self.engines[self.current_engine].detect_and_track(frame, **kwargs)

    def get_available_engines(self) -> List[str]:
        """Get list of available engine names"""
        return list(self.engines.keys())


# Utility functions for common detection tasks
def filter_detections_by_area(detections: List[Detection],
                              min_area: int = 0,
                              max_area: Optional[int] = None) -> List[Detection]:
    """Filter detections by bounding box area"""
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area and (max_area is None or area <= max_area):
            filtered.append(det)
    return filtered


def filter_detections_by_region(detections: List[Detection],
                                region_polygon: List[Tuple[int, int]]) -> List[Detection]:
    """Filter detections that have their center point inside a polygon region"""
    if len(region_polygon) < 3:
        return detections

    filtered = []
    for det in detections:
        if cv2.pointPolygonTest(np.array(region_polygon), det.center_point, False) >= 0:
            filtered.append(det)
    return filtered
