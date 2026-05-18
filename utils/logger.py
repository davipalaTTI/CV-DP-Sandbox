import logging
import logging.handlers
import sys
import os
import queue
from pathlib import Path
from typing import Optional

_log_queue_listener = None


class _DropOldestQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that drops the OLDEST record when the bounded queue is full.

    Logging's default QueueHandler raises queue.Full once the queue is saturated,
    which silently drops the new record. Recent records carry the most diagnostic
    value, so we drop the oldest instead and always accept the newest.
    """

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                pass  # give up; another producer beat us to the slot


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    A RotatingFileHandler that gracefully handles file locking issues on Windows.

    When rotation fails (e.g., due to OneDrive sync or another process locking
    the file), it simply continues logging to the current file instead of raising
    an error. Rotation will be attempted again on the next write that exceeds maxBytes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotation_failed = False

    def doRollover(self):
        """Perform rotation, but handle errors gracefully"""
        try:
            # Close the stream before rotation
            if self.stream:
                self.stream.close()
                self.stream = None

            # Attempt rotation
            if self.backupCount > 0:
                # Delete oldest backup if it exists
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                    dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                    if os.path.exists(sfn):
                        try:
                            if os.path.exists(dfn):
                                os.remove(dfn)
                            os.rename(sfn, dfn)
                        except (OSError, PermissionError):
                            pass  # Ignore rotation errors for backups

                # Rotate current to .1
                dfn = self.rotation_filename(f"{self.baseFilename}.1")
                try:
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    if os.path.exists(self.baseFilename):
                        os.rename(self.baseFilename, dfn)
                    self._rotation_failed = False
                except (OSError, PermissionError):
                    # Rotation failed - file is locked (probably OneDrive)
                    # Just continue with the same file
                    self._rotation_failed = True

            # Reopen stream
            if not self.delay:
                self.stream = self._open()

        except Exception:
            # Any other error - just try to keep logging
            self._rotation_failed = True
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    pass

    def shouldRollover(self, record):
        """Check if rollover should occur, but be more lenient after failed rotation"""
        if self._rotation_failed:
            # After a failed rotation, only try again after file grows significantly more
            # This prevents constant rotation attempts
            if self.stream is None:
                try:
                    self.stream = self._open()
                except Exception:
                    return False

            try:
                self.stream.seek(0, 2)  # Seek to end
                # Only retry rotation after file grows another 50% beyond maxBytes
                if self.stream.tell() >= self.maxBytes * 1.5:
                    self._rotation_failed = False  # Reset and try again
                    return True
            except Exception:
                pass
            return False

        return super().shouldRollover(record)


def setup_logging(level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  console_output: bool = True,
                  format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging configuration with thread-safe file handling.

    Uses QueueHandler + QueueListener pattern with a SafeRotatingFileHandler
    to prevent file locking issues on Windows/OneDrive.

    Args:
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        format_string: Custom format string

    Returns:
        Configured logger
    """
    global _log_queue_listener

    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers and stop old listener
    root_logger.handlers.clear()
    if _log_queue_listener:
        _log_queue_listener.stop()
        _log_queue_listener = None

    # Create handlers list for QueueListener
    handlers = []

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # File handler with safe rotation (handles OneDrive/Windows locking)
    if log_file:
        file_handler = SafeRotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            delay=True  # Delay file opening until first write
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Use queue-based logging for thread safety
    if handlers:
        # Bounded queue: if a downstream handler stalls (e.g. log file locked
        # on OneDrive), memory growth is capped. _DropOldestQueueHandler drops
        # the oldest record on overflow so recent diagnostics survive.
        log_queue = queue.Queue(maxsize=10000)

        # QueueHandler for the root logger (all threads put logs here)
        queue_handler = _DropOldestQueueHandler(log_queue)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)

        # QueueListener processes the queue in a single thread (thread-safe file writes)
        _log_queue_listener = logging.handlers.QueueListener(
            log_queue,
            *handlers,
            respect_handler_level=True
        )
        _log_queue_listener.start()

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return root_logger


def stop_logging():
    """Stop the logging queue listener (call on application exit)"""
    global _log_queue_listener
    if _log_queue_listener:
        _log_queue_listener.stop()
        _log_queue_listener = None


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)