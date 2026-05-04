"""
Training Mode Module

Captures frames and annotations for creating training datasets.
Saves frames as images with corresponding YOLO format annotation files.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import logging
import json
from dataclasses import dataclass, asdict
import threading
import time


@dataclass
class TrainingConfig:
    """Configuration for training mode"""
    enabled: bool = False
    capture_interval_seconds: float = 5.0  # Capture every N seconds
    output_folder: str = "training_data"
    image_format: str = "jpg"  # jpg or png
    include_empty_frames: bool = False  # Save frames with no detections
    max_captures_per_session: int = 0  # 0 = unlimited
    auto_stop_after_hours: float = 0  # 0 = no auto-stop
    min_confidence: float = 0.5  # Minimum confidence for training annotations
    classes_to_capture: Optional[List[int]] = None  # None = all classes
    augment_captures: bool = False  # Apply augmentation variations
    save_metadata: bool = True  # Save additional metadata JSON


class TrainingModeCapture:
    """Handles training data capture during live processing"""

    def __init__(self, config: TrainingConfig, class_names: Dict[int, str]):
        self.config = config
        self.class_names = class_names
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.last_capture_time = 0
        self.capture_count = 0
        self.session_start_time = time.time()
        self.is_active = config.enabled

        # Setup output directories
        self._setup_directories()

        # Thread safety
        self.lock = threading.Lock()

        # Session metadata
        self.session_metadata = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "config": asdict(config),
            "class_names": class_names,
            "captures": []
        }

        self.logger.info(f"Training mode initialized: {'ENABLED' if self.is_active else 'DISABLED'}")

    def _setup_directories(self):
        """Create output directory structure"""
        self.base_path = Path(self.config.output_folder)
        self.session_path = self.base_path / datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.enabled:
            self.images_path = self.session_path / "images"
            self.labels_path = self.session_path / "labels"

            self.images_path.mkdir(parents=True, exist_ok=True)
            self.labels_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Training data folders created at: {self.session_path}")

    def should_capture(self) -> bool:
        """Check if it's time to capture a training frame"""
        if not self.is_active:
            return False

        # Check auto-stop
        if self.config.auto_stop_after_hours > 0:
            elapsed_hours = (time.time() - self.session_start_time) / 3600
            if elapsed_hours >= self.config.auto_stop_after_hours:
                self.stop()
                self.logger.info("Training mode auto-stopped after time limit")
                return False

        # Check max captures
        if self.config.max_captures_per_session > 0:
            if self.capture_count >= self.config.max_captures_per_session:
                self.stop()
                self.logger.info("Training mode stopped - max captures reached")
                return False

        # Check interval
        current_time = time.time()
        if current_time - self.last_capture_time >= self.config.capture_interval_seconds:
            return True

        return False

    def capture_frame(self, frame: np.ndarray, detections: List) -> Optional[str]:
        """
        Capture a training frame with annotations

        Args:
            frame: Current video frame
            detections: List of Detection objects from detection_engine

        Returns:
            Path to saved image file or None if not captured
        """
        if not self.should_capture():
            return None

        with self.lock:
            # Filter detections
            filtered_detections = self._filter_detections(detections)

            # Skip if no detections and we're not saving empty frames
            if not filtered_detections and not self.config.include_empty_frames:
                return None

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_filename = f"frame_{timestamp}.{self.config.image_format}"
            label_filename = f"frame_{timestamp}.txt"

            image_path = self.images_path / image_filename
            label_path = self.labels_path / label_filename

            try:
                # Save image
                if self.config.image_format == "jpg":
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(str(image_path), frame)

                # Save YOLO format annotations
                self._save_yolo_annotations(label_path, filtered_detections, frame.shape)

                # Apply augmentation if enabled
                if self.config.augment_captures and filtered_detections:
                    self._save_augmented_versions(frame, filtered_detections, timestamp)

                # Update metadata
                capture_info = {
                    "timestamp": datetime.now().isoformat(),
                    "image": str(image_path.name),
                    "label": str(label_path.name),
                    "detections": len(filtered_detections),
                    "frame_shape": frame.shape
                }
                self.session_metadata["captures"].append(capture_info)

                # Update state
                self.last_capture_time = time.time()
                self.capture_count += 1

                self.logger.debug(f"Training frame captured: {image_filename} ({len(filtered_detections)} detections)")

                return str(image_path)

            except Exception as e:
                self.logger.error(f"Failed to capture training frame: {e}")
                return None

    def _filter_detections(self, detections: List) -> List:
        """Filter detections based on training configuration"""
        filtered = []

        for det in detections:
            # Check confidence
            if det.confidence < self.config.min_confidence:
                continue

            # Check class filter
            if self.config.classes_to_capture is not None:
                if det.class_id not in self.config.classes_to_capture:
                    continue

            filtered.append(det)

        return filtered

    def _save_yolo_annotations(self, label_path: Path, detections: List, frame_shape: tuple):
        """
        Save detections in YOLO format

        YOLO format: class_id center_x center_y width height
        (all coordinates normalized to 0-1)
        """
        height, width = frame_shape[:2]

        with open(label_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det.bbox

                # Convert to YOLO format (normalized center coordinates)
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                # Ensure values are in valid range
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                box_width = max(0, min(1, box_width))
                box_height = max(0, min(1, box_height))

                # Write YOLO format line
                f.write(f"{det.class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

    def _save_augmented_versions(self, frame: np.ndarray, detections: List, base_timestamp: str):
        """Save augmented versions of the frame for training diversity"""
        augmentations = []

        # Horizontal flip
        flipped_frame = cv2.flip(frame, 1)
        flipped_detections = self._flip_detections_horizontal(detections, frame.shape[1])
        augmentations.append(("flip", flipped_frame, flipped_detections))

        # Brightness variations
        bright_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        dark_frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-30)
        augmentations.append(("bright", bright_frame, detections))
        augmentations.append(("dark", dark_frame, detections))

        # Save augmented versions
        for aug_type, aug_frame, aug_dets in augmentations:
            aug_image_filename = f"frame_{base_timestamp}_{aug_type}.{self.config.image_format}"
            aug_label_filename = f"frame_{base_timestamp}_{aug_type}.txt"

            aug_image_path = self.images_path / aug_image_filename
            aug_label_path = self.labels_path / aug_label_filename

            cv2.imwrite(str(aug_image_path), aug_frame)
            self._save_yolo_annotations(aug_label_path, aug_dets, aug_frame.shape)

    def _flip_detections_horizontal(self, detections: List, width: int) -> List:
        """Flip detection coordinates horizontally"""
        flipped = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Create a copy with flipped coordinates
            flipped_det = type(det)(
                track_id=det.track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=(width - x2, y1, width - x1, y2),
                confidence=det.confidence,
                center_point=(width - det.center_point[0], det.center_point[1]),
                bottom_point=(width - det.bottom_point[0], det.bottom_point[1])
            )
            flipped.append(flipped_det)
        return flipped

    def start(self):
        """Start or resume training mode"""
        with self.lock:
            self.is_active = True
            self.logger.info("Training mode STARTED")

    def stop(self):
        """Stop training mode"""
        with self.lock:
            self.is_active = False
            self.logger.info("Training mode STOPPED")
            self._save_session_metadata()

    def toggle(self) -> bool:
        """Toggle training mode on/off"""
        if self.is_active:
            self.stop()
        else:
            self.start()
        return self.is_active

    def set_interval(self, seconds: float):
        """Update capture interval"""
        self.config.capture_interval_seconds = max(0.1, seconds)
        self.logger.info(f"Capture interval set to {seconds} seconds")

    def _save_session_metadata(self):
        """Save session metadata to JSON file"""
        if not self.config.save_metadata:
            return

        try:
            self.session_metadata["end_time"] = datetime.now().isoformat()
            self.session_metadata["total_captures"] = self.capture_count
            self.session_metadata["duration_hours"] = (time.time() - self.session_start_time) / 3600

            metadata_path = self.session_path / "session_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.session_metadata, f, indent=2, default=str)

            # Create dataset.yaml for YOLO training
            self._create_yolo_dataset_yaml()

            self.logger.info(f"Session metadata saved to {metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to save session metadata: {e}")

    def _create_yolo_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        yaml_content = {
            "path": str(self.session_path.absolute()),
            "train": "images",
            "val": "images",  # You should split this properly for real training
            "names": self.class_names
        }

        yaml_path = self.session_path / "dataset.yaml"

        # Simple YAML writer (avoid yaml dependency)
        with open(yaml_path, 'w') as f:
            f.write(f"# Training dataset generated on {datetime.now().isoformat()}\n")
            f.write(f"path: {yaml_content['path']}\n")
            f.write(f"train: {yaml_content['train']}\n")
            f.write(f"val: {yaml_content['val']}\n")
            f.write("\nnames:\n")
            for class_id, class_name in yaml_content['names'].items():
                f.write(f"  {class_id}: {class_name}\n")

    def get_status(self) -> Dict:
        """Get current training mode status"""
        return {
            "active": self.is_active,
            "captures": self.capture_count,
            "interval_seconds": self.config.capture_interval_seconds,
            "session_path": str(self.session_path) if hasattr(self, 'session_path') else None,
            "elapsed_hours": (time.time() - self.session_start_time) / 3600,
            "last_capture": datetime.fromtimestamp(
                self.last_capture_time).isoformat() if self.last_capture_time else None
        }

    def cleanup(self):
        """Cleanup and save final metadata"""
        if self.is_active:
            self.stop()