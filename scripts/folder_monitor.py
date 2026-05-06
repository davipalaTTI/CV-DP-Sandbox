"""
Folder Monitor Module

Monitors a folder for new video files and adds them to a processing queue in real-time.
"""

import time
import logging
from pathlib import Path
from typing import Set, Callable, List
from threading import Thread, Event


class FolderMonitor:
    """
    Monitors a folder for new video files and notifies when they appear.
    Uses polling instead of watchdog for simplicity and fewer dependencies.
    """

    # Supported video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}

    def __init__(self,
                 folder_path: str,
                 callback: Callable[[Path], None],
                 poll_interval: float = 2.0):
        """
        Initialize folder monitor

        Args:
            folder_path: Path to folder to monitor
            callback: Function to call when new video is detected (receives Path object)
            poll_interval: How often to check for new files (seconds)
        """
        self.folder_path = Path(folder_path)
        self.callback = callback
        self.poll_interval = poll_interval

        self.logger = logging.getLogger(__name__)

        # Tracking state
        self.known_files: Set[Path] = set()
        self.is_running = False
        self.stop_event = Event()
        self.monitor_thread: Thread = None

        # Initialize with existing files
        self._scan_existing_files()

        self.logger.info(f"Folder monitor initialized for: {self.folder_path}")
        self.logger.info(f"Found {len(self.known_files)} existing video files")

    def _scan_existing_files(self):
        """Scan folder and record existing video files"""
        if not self.folder_path.exists():
            self.logger.warning(f"Folder does not exist: {self.folder_path}")
            return

        for ext in self.VIDEO_EXTENSIONS:
            for video_file in self.folder_path.glob(f"*{ext}"):
                if video_file.is_file():
                    self.known_files.add(video_file)

        self.logger.debug(f"Scanned existing files: {len(self.known_files)} videos found")

    def _check_for_new_files(self):
        """Check folder for new video files"""
        try:
            current_files = set()

            # Scan all video extensions
            for ext in self.VIDEO_EXTENSIONS:
                for video_file in self.folder_path.glob(f"*{ext}"):
                    if video_file.is_file():
                        current_files.add(video_file)

            # Find new files
            new_files = current_files - self.known_files

            if new_files:
                # Sort by modification time (oldest first)
                new_files_sorted = sorted(new_files, key=lambda p: p.stat().st_mtime)

                for new_file in new_files_sorted:
                    self.logger.info(f"New video detected: {new_file.name}")

                    # Add to known files
                    self.known_files.add(new_file)

                    # Notify callback
                    try:
                        self.callback(new_file)
                    except Exception as e:
                        self.logger.error(f"Callback error for {new_file.name}: {e}")

        except Exception as e:
            self.logger.error(f"Error checking for new files: {e}")

    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        self.logger.info("Folder monitoring started")

        while not self.stop_event.is_set():
            self._check_for_new_files()

            # Wait for next poll interval (or until stop signal)
            self.stop_event.wait(self.poll_interval)

        self.logger.info("Folder monitoring stopped")

    def start(self):
        """Start monitoring the folder in a background thread"""
        if self.is_running:
            self.logger.warning("Monitor already running")
            return

        self.is_running = True
        self.stop_event.clear()

        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Folder monitor started")

    def stop(self):
        """Stop monitoring the folder"""
        if not self.is_running:
            return

        self.logger.info("Stopping folder monitor...")
        self.is_running = False
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        self.logger.info("Folder monitor stopped")

    def get_known_files(self) -> List[Path]:
        """Get list of all known video files"""
        return sorted(list(self.known_files), key=lambda p: p.stat().st_mtime)

    @staticmethod
    def is_file_growing(file_path: Path, check_interval: float = 2.0) -> bool:
        """
        Check if a file is still being written to by comparing sizes.

        Args:
            file_path: Path to the file to check
            check_interval: Seconds to wait between size checks

        Returns:
            True if file size changed (still growing), False if stable
        """
        try:
            if not file_path.exists():
                return False

            initial_size = file_path.stat().st_size
            time.sleep(check_interval)

            if not file_path.exists():
                return False

            final_size = file_path.stat().st_size
            return final_size > initial_size

        except Exception:
            return False

    @staticmethod
    def wait_for_file_stable(file_path: Path,
                             check_interval: float = 2.0,
                             timeout: float = 30.0,
                             logger: logging.Logger = None) -> bool:
        """
        Wait for a file to stop growing (recording complete).

        Args:
            file_path: Path to the file to monitor
            check_interval: Seconds between size checks
            timeout: Maximum seconds to wait for file to stabilize after last growth
            logger: Optional logger for status messages

        Returns:
            True if file is stable (no longer growing), False if timed out or error
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            if not file_path.exists():
                return False

            last_size = file_path.stat().st_size
            stable_start_time = time.time()

            while True:
                time.sleep(check_interval)

                if not file_path.exists():
                    logger.warning(f"File disappeared while waiting: {file_path.name}")
                    return False

                current_size = file_path.stat().st_size

                if current_size > last_size:
                    # File is still growing, reset the stable timer
                    logger.debug(f"File still growing: {file_path.name} ({last_size} -> {current_size})")
                    last_size = current_size
                    stable_start_time = time.time()
                else:
                    # File size unchanged
                    stable_duration = time.time() - stable_start_time

                    if stable_duration >= timeout:
                        logger.info(f"File stable for {timeout}s: {file_path.name}")
                        return True
                    else:
                        logger.debug(f"File stable for {stable_duration:.1f}s/{timeout}s: {file_path.name}")

        except Exception as e:
            logger.error(f"Error waiting for file stable: {e}")
            return False