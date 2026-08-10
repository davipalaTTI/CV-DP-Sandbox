import logging
import threading
import time
from queue import Empty, Full, Queue


def put_latest(target_queue: Queue, item) -> bool:
    """Put without blocking, replacing one stale queued item when necessary.

    Returns True when an older item had to be dropped.
    """
    try:
        target_queue.put_nowait(item)
        return False
    except Full:
        try:
            target_queue.get_nowait()
            try:
                target_queue.task_done()
            except ValueError:
                pass
        except Empty:
            pass
        try:
            target_queue.put_nowait(item)
        except Full:
            return True
        return True


class AsyncVideoWriter:
    """
    Background thread video writer.
    Prevents the main AI loop from waiting on slow hard drive I/O operations.
    """

    def __init__(self, writer, queue_size=8):
        self.logger = logging.getLogger(__name__)
        # Wrap the successfully opened OpenCV writer
        self.writer = writer

        # Create a thread-safe holding box (queue) for frames
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = True
        self.dropped_frames = 0
        self.frames_written = 0
        self.last_error = None
        self.write_started_at = None
        self.last_write_completed_at = time.monotonic()

        # Start the background worker thread
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()
        self.logger.info("Async VideoWriter thread started")

    def isOpened(self):
        """Pass-through so the main loop knows it is ready"""
        return self.writer.isOpened()

    def _write_loop(self):
        """Background thread that endlessly pulls from the queue and writes to disk"""
        while self.running or not self.frame_queue.empty():
            try:
                # Grab the oldest frame from the box (times out so we can check if we are still running)
                frame = self.frame_queue.get(timeout=0.1)
                self.write_started_at = time.monotonic()
                self.writer.write(frame)
                self.frames_written += 1
                self.last_write_completed_at = time.monotonic()
                self.write_started_at = None
                self.frame_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.last_error = e
                self.write_started_at = None
                self.logger.error(f"Async writer error: {e}")

    def write(self, frame):
        """Called by the main loop to drop a frame into the holding box instantly"""
        if self.running:
            if put_latest(self.frame_queue, frame.copy()):
                self.dropped_frames += 1

    def release(self):
        """Safely flushes the queue and closes the file"""
        self.running = False
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            self.logger.error("Video writer did not flush within 10 seconds")
            return False
        self.writer.release()
        return True

class AsyncCameraReader:
    """
    Background thread camera reader.
    Continuously grabs the newest frame from the camera buffer.
    Prevents the main TensorRT loop from waiting on camera sensor I/O.
    """

    def __init__(self, cap):
        self.logger = logging.getLogger(__name__)
        self.cap = cap
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.sequence = 1 if self.ret and self.frame is not None else 0
        self.frames_read = self.sequence
        self.started_at = time.monotonic()
        self.last_frame_at = self.started_at if self.sequence else None
        self.last_error = None

        # Start the background reader thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.logger.info("Async CameraReader thread started")

    def _update(self):
        """Endlessly pull frames as fast as the camera can provide them"""
        while self.running:
            ret, frame = self.cap.read()
            now = time.monotonic()
            # Use a lock to safely overwrite the old frame with the absolute newest one
            with self.lock:
                self.ret = ret
                if ret and frame is not None:
                    self.frame = frame
                    self.sequence += 1
                    self.frames_read += 1
                    self.last_frame_at = now
                    self.last_error = None
                else:
                    self.last_error = "Camera read returned no frame"

            # If camera disconnects, cleanly shut down the thread
            if not ret:
                self.running = False
                break

    def read(self):
        """Called by the main AI loop to instantly grab the freshest frame in memory"""
        with self.lock:
            if self.frame is not None:
                # Return a copy so the AI can draw boxes on it without corrupting the background thread
                return self.ret, self.frame.copy()
            return self.ret, None

    def read_latest(self, last_sequence: int = -1, copy_frame: bool = False):
        """Return a frame only when it is newer than the caller's sequence."""
        with self.lock:
            sequence = self.sequence
            if not self.ret or self.frame is None or sequence == last_sequence:
                return self.ret, None, sequence
            frame = self.frame.copy() if copy_frame else self.frame
            return True, frame, sequence

    def seconds_since_last_frame(self) -> float:
        with self.lock:
            last_frame_at = self.last_frame_at
        reference = last_frame_at if last_frame_at is not None else self.started_at
        return max(0.0, time.monotonic() - reference)

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def set(self, prop_id, value):
        return self.cap.set(prop_id, value)

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()


class ThreadedDetectionEngine:
    """Background YOLO inference worker."""

    def __init__(
        self,
        engine,
        input_queue: Queue,
        output_queue: Queue,
        max_consecutive_errors: int = 30,
    ):
        self.engine = engine
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logging.getLogger("ThreadedYOLO")
        self.running = True
        self.max_consecutive_errors = max(1, int(max_consecutive_errors))
        self.fatal_error = None
        self.frames_processed = 0
        self.inference_started_at = None
        self.last_completed_at = time.monotonic()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        last_log_time = time.time()
        frames_processed = 0
        while self.running:
            try:
                # Wait for a frame from the camera
                frame = self.input_queue.get(timeout=0.5)

                # Run the heavy AI math (This no longer blocks the main thread!)
                self.inference_started_at = time.monotonic()
                detections = self.engine.detect_and_track(frame)
                if (
                    getattr(self.engine, "consecutive_detection_errors", 0)
                    >= self.max_consecutive_errors
                ):
                    raise RuntimeError(
                        f"Detection failed {self.max_consecutive_errors} consecutive times"
                    )

                # Push the results to the UI thread
                put_latest(self.output_queue, (frame, detections))

                frames_processed += 1
                self.frames_processed += 1
                self.last_completed_at = time.monotonic()
                self.inference_started_at = None
                try:
                    self.input_queue.task_done()
                except ValueError:
                    pass
                elapsed = time.time() - last_log_time
                if elapsed >= 10.0:
                    fps = frames_processed / elapsed if elapsed > 0 else 0.0
                    self.logger.debug(f"[THREADED YOLO] Alive. Processed ~{fps:.1f} FPS")
                    frames_processed = 0
                    last_log_time = time.time()
            except Empty:
                continue
            except Exception as e:
                self.fatal_error = e
                self.running = False
                self.logger.exception("Detection thread stopped: %s", e)

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
