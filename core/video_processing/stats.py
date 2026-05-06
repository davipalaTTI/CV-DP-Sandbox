from dataclasses import dataclass

@dataclass
class ProcessingStats:
    """Processing performance statistics"""
    frames_processed: int = 0
    total_detections: int = 0
    total_events: int = 0
    start_time: float = 0
    processing_time: float = 0
    fps: float = 0
    avg_detection_time: float = 0
    avg_processing_time: float = 0

    def update_fps(self):
        """Update FPS calculation"""
        if self.processing_time > 0:
            self.fps = self.frames_processed / self.processing_time