import time
import signal
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from queue import Queue as ThreadQueue
from concurrent.futures import ThreadPoolExecutor

from config_manager import AppConfig
from core.detection_engine import DetectionEngine
from gui.progress_window import ProgressWindow
from scripts.folder_monitor import FolderMonitor
from utils.results_export import ResultsExporter, ExportConfig

from .worker import VideoWorker


class BatchRunner:
    """High-Throughput orchestrator for Video Files and Folder Monitoring"""

    def __init__(self, config: AppConfig, detection_engine: DetectionEngine):
        self.config = config
        self.detection_engine = detection_engine
        self.logger = logging.getLogger(__name__)

        # Results exporter
        export_config = ExportConfig(enable_api_upload=self.config.enable_api_upload)
        self.exporter = ResultsExporter(config.output_folder, export_config)

        # Thread-safe progress tracking
        self.progress_lock = threading.Lock()
        self.worker_progress = {}

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _combine_video_stats(self, video_results: List[Dict]) -> Dict:
        """Combine statistics from multiple videos"""
        combined = {
            "total_frames": sum(r["stats"].frames_processed for r in video_results if "stats" in r),
            "total_events": sum(r["stats"].total_events for r in video_results if "stats" in r),
            "avg_fps": np.mean([r["stats"].fps for r in video_results if "stats" in r]) if video_results else 0.0,
            "total_processing_time": sum(r["stats"].processing_time for r in video_results if "stats" in r)
        }
        return combined

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        # (ThreadPool will catch the shutdown event during execution)

    def run(self, max_workers: int = None) -> Dict:
        """Process video files from folder with concurrent workers."""
        if max_workers is None:
            max_workers = getattr(self.config, 'max_parallel_videos', 1)

        video_queue = ThreadQueue()
        folder_monitor = None
        pending_futures = {}
        total_results = []
        video_count = 0
        worker_id_counter = 0

        # Determine initial video files
        if self.config.input_type.value == "folder":
            folder_path = Path(self.config.input_source)

            def on_new_video(video_path: Path):
                self.logger.info(f"Adding new video to queue: {video_path.name}")
                video_queue.put(video_path)

            folder_monitor = FolderMonitor(
                folder_path=str(folder_path),
                callback=on_new_video,
                poll_interval=2.0
            )

            initial_files = folder_monitor.get_known_files()
            for video_file in initial_files:
                video_queue.put(video_file)

            folder_monitor.start()
            self.logger.info(f"Folder monitoring enabled for: {folder_path}")
            self.logger.info(f"Concurrent workers: {max_workers}")
        else:
            # Single file mode
            video_queue.put(Path(self.config.input_source))
            max_workers = 1

        if video_queue.empty():
            raise RuntimeError("No video files found")

        progress = ProgressWindow("Processing Videos")
        progress.set_total_queued(video_queue.qsize())
        overall_start_time = time.time()
        executor = ThreadPoolExecutor(max_workers=max_workers)

        def get_queue_size():
            return video_queue.qsize()

        def on_worker_progress(worker_id, frames_done, total_frames, fps, events,
                               waiting_until_recheck=0.0, remaining_stability_wait=0.0):
            with self.progress_lock:
                self.worker_progress[worker_id] = {
                    'frames': frames_done,
                    'total': total_frames,
                    'fps': fps,
                    'events': events,
                    'waiting_recheck': waiting_until_recheck,
                    'waiting_stability': remaining_stability_wait
                }

        try:
            active_workers = 0
            while True:
                # Submit new jobs
                while active_workers < max_workers and not video_queue.empty():
                    try:
                        video_path = video_queue.get_nowait()
                    except:
                        break

                    if not video_path.exists():
                        self.logger.warning(f"Video not found, skipping: {video_path}")
                        continue

                    video_count += 1
                    worker_id_counter += 1

                    worker = VideoWorker(self.config, video_path, worker_id_counter, queue_size_callback=get_queue_size)

                    # Quick check to register UI correctly
                    import cv2
                    temp_cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_cap.release()

                    progress.register_video(worker_id_counter, video_path.name, total_frames)
                    progress.set_total_queued(video_queue.qsize())

                    future = executor.submit(worker.process, on_worker_progress)
                    pending_futures[future] = {
                        'video_path': video_path,
                        'worker_id': worker_id_counter,
                        'start_time': time.time()
                    }
                    active_workers += 1

                # Update UI
                with self.progress_lock:
                    for wid, data in self.worker_progress.items():
                        progress.update_video_progress(
                            wid, data['frames'], data['fps'], data['events'],
                            data.get('waiting_recheck', 0.0), data.get('waiting_stability', 0.0)
                        )

                # Check completed futures
                done_futures = [f for f in pending_futures if f.done()]
                for future in done_futures:
                    info = pending_futures.pop(future)
                    active_workers -= 1
                    worker_id = info['worker_id']

                    try:
                        result = future.result(timeout=1.0)
                        total_results.append(result)
                        progress.complete_video(worker_id, success=True)
                    except Exception as e:
                        self.logger.error(f"Worker {worker_id} failed: {e}")
                        total_results.append({"error": str(e), "video_path": str(info['video_path'])})
                        progress.complete_video(worker_id, success=False)

                    with self.progress_lock:
                        self.worker_progress.pop(worker_id, None)

                progress.set_total_queued(video_queue.qsize())
                progress.update_elapsed_time(time.time() - overall_start_time)

                # Check exit condition
                if active_workers == 0 and video_queue.empty():
                    if self.config.input_type.value == "folder":
                        progress.set_status("Waiting for new files... (Ctrl+C to stop)")
                        while video_queue.empty():
                            time.sleep(0.1)
                            progress.keep_responsive()
                            if progress.is_closed:
                                return {
                                    "total_videos": len(total_results),
                                    "video_results": total_results,
                                    "combined_stats": self._combine_video_stats(total_results)
                                }
                    else:
                        break

                time.sleep(0.1)
                progress.keep_responsive()

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            executor.shutdown(wait=False, cancel_futures=True)

        finally:
            if folder_monitor:
                folder_monitor.stop()
            executor.shutdown(wait=True)
            progress.close()

            # Shut down the background threads cleanly!
            if hasattr(self, 'exporter') and self.exporter is not None:
                self.exporter.shutdown()

        return {
            "total_videos": len(total_results),
            "video_results": total_results,
            "combined_stats": self._combine_video_stats(total_results)
        }