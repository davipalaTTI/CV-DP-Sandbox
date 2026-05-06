import logging
import threading
import time
from queue import Queue, Empty
class AsyncVideoWriter:
    """
    Background thread video writer.
    Prevents the main AI loop from waiting on slow hard drive I/O operations.
    """

    def __init__(self, writer, queue_size=120):
        self.logger = logging.getLogger(__name__)
        # Wrap the successfully opened OpenCV writer
        self.writer = writer

        # Create a thread-safe holding box (queue) for frames
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = True

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
                self.writer.write(frame)
                self.frame_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Async writer error: {e}")

    def write(self, frame):
        """Called by the main loop to drop a frame into the holding box instantly"""
        if self.running:
            if self.frame_queue.full():
                # If disk is too slow and box fills up, drop the oldest frame to prevent memory crashes
                # This guarantees the AI NEVER slows down, even if the video drops a frame.
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass

            # Drop a COPY of the frame into the box so the main thread can keep modifying the original
            self.frame_queue.put(frame.copy())

    def release(self):
        """Safely flushes the queue and closes the file"""
        self.running = False
        self.thread.join()  # Wait for the background thread to finish writing the last frames
        self.writer.release()

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

        # Start the background reader thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        self.logger.info("Async CameraReader thread started")

    def _update(self):
        """Endlessly pull frames as fast as the camera can provide them"""
        while self.running:
            ret, frame = self.cap.read()
            # Use a lock to safely overwrite the old frame with the absolute newest one
            with self.lock:
                self.ret = ret
                self.frame = frame

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

    def __init__(self, engine, input_queue: Queue, output_queue: Queue):
        self.engine = engine
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logging.getLogger("ThreadedYOLO")
        self.running = True
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
                detections = self.engine.detect_and_track(frame)

                # Push the results to the UI thread
                if not self.output_queue.full():
                    self.output_queue.put((frame, detections))
                else:
                    try:
                        self.output_queue.get_nowait()  # Drop old frames if UI is lagging
                        self.output_queue.put((frame, detections))
                    except Empty:
                        pass

                frames_processed += 1
                if time.time() - last_log_time >= 10.0:
                    fps = frames_processed / 10.0
                    self.logger.debug(f"[THREADED YOLO] Alive. Processed ~{fps:.1f} FPS")
                    frames_processed = 0
                    last_log_time = time.time()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Detection Thread Error: {e}")

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
