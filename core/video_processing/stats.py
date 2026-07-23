from dataclasses import dataclass
from collections import deque
import time

@dataclass
class ProcessingStats:
    """Processing performance statistics"""
    frames_processed: int = 0
    total_detections: int = 0
    total_events: int = 0
    start_time: float = 0
    processing_time: float = 0
    fps: float = 0
    avg_fps: float = 0
    avg_detection_time: float = 0
    avg_processing_time: float = 0

    def __post_init__(self):
        self._fps_samples = deque(maxlen=60)

    def update_fps(self):
        """Update rolling FPS and lifetime average FPS."""
        now = time.perf_counter()

        if not hasattr(self, '_fps_samples'):
            self._fps_samples = deque(maxlen=60)

        self._fps_samples.append(now)

        if self.processing_time > 0:
            self.avg_fps = self.frames_processed / self.processing_time

        if len(self._fps_samples) >= 2:
            elapsed = self._fps_samples[-1] - self._fps_samples[0]
            if elapsed > 0:
                self.fps = (len(self._fps_samples) - 1) / elapsed
        elif self.processing_time > 0:
            self.fps = self.avg_fps
