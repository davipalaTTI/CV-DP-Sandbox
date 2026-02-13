"""
Progress Window Module

Displays processing progress for batch video processing with:
- Overall progress across all videos
- Per-video frame progress with percentage
- Real-time FPS display
- Processing statistics
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
import logging
import os


@dataclass
class VideoProgress:
    """Track progress for a single video"""
    video_name: str
    worker_id: int
    total_frames: int = 0
    frames_processed: int = 0
    fps: float = 0.0
    start_time: float = field(default_factory=time.time)
    events_count: int = 0
    status: str = "processing"  # processing, completed, error, waiting
    waiting_recheck: float = 0.0  # Seconds until next file recheck
    waiting_stability: float = 0.0  # Seconds until stability timeout

    @property
    def percentage(self) -> float:
        if self.total_frames > 0:
            return (self.frames_processed / self.total_frames) * 100
        return 0.0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float:
        if self.fps > 0 and self.total_frames > self.frames_processed:
            remaining_frames = self.total_frames - self.frames_processed
            return remaining_frames / self.fps
        return 0.0

    @property
    def is_waiting(self) -> bool:
        return self.waiting_recheck > 0 or self.waiting_stability > 0


class ProgressWindow:
    """
    Enhanced progress window for batch video processing.
    Shows overall progress, per-video details, and real-time FPS.
    """

    def __init__(self, title="Processing Videos"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("750x450")
        self.root.minsize(650, 350)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 375
        y = (self.root.winfo_screenheight() // 2) - 225
        self.root.geometry(f"750x450+{x}+{y}")

        # Track active videos
        self.active_videos: Dict[int, VideoProgress] = {}
        self.completed_count = 0
        self.total_queued = 0
        self.lock = threading.Lock()

        # FPS tracking - keep last 60 seconds of samples with timestamps
        self._fps_history = deque()  # (timestamp, fps) tuples
        self._fps_window_seconds = 60.0  # Average over last minute

        # Build UI
        self._build_ui()

        self.root.bind("<Escape>", self.on_close)
        self.root.bind("<Control-c>", self.on_close)
        self.root.bind("<Control-C>", self.on_close)

        self.logger = logging.getLogger(__name__)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.is_closed = False

    def _build_ui(self):
        """Build the UI components"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Overall Progress Section ===
        overall_frame = ttk.LabelFrame(main_frame, text="Overall Progress", padding="5")
        overall_frame.pack(fill=tk.X, pady=(0, 10))

        self.overall_progress = ttk.Progressbar(
            overall_frame, length=700, mode='determinate'
        )
        self.overall_progress.pack(fill=tk.X, pady=5)

        self.overall_label = ttk.Label(overall_frame, text="Initializing...")
        self.overall_label.pack()

        # === Current Videos Section ===
        videos_frame = ttk.LabelFrame(main_frame, text="Active Videos", padding="5")
        videos_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create a canvas with scrollbar for multiple videos
        canvas = tk.Canvas(videos_frame, height=180)
        scrollbar = ttk.Scrollbar(videos_frame, orient="vertical", command=canvas.yview)
        self.videos_container = ttk.Frame(canvas)

        self.videos_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.videos_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Dictionary to hold video progress widgets
        self.video_widgets: Dict[int, Dict] = {}

        # === Statistics Section ===
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X)

        # Stats in a grid
        self.stats_labels = {}
        stats_items = [
            ("completed", "Completed:"),
            ("active", "Active:"),
            ("queued", "Queued:"),
            ("avg_fps", "Avg FPS (1m):"),
            ("total_events", "Total Events:"),
            ("elapsed", "Elapsed:"),
        ]

        for i, (key, label_text) in enumerate(stats_items):
            row = i // 3
            col = (i % 3) * 2

            ttk.Label(stats_frame, text=label_text).grid(row=row, column=col, sticky="e", padx=(10, 2))
            self.stats_labels[key] = ttk.Label(stats_frame, text="0", width=12)
            self.stats_labels[key].grid(row=row, column=col+1, sticky="w", padx=(0, 15))

        # Configure grid weights
        for i in range(6):
            stats_frame.columnconfigure(i, weight=1)

    def register_video(self, worker_id: int, video_name: str, total_frames: int):
        """Register a new video being processed"""
        with self.lock:
            self.active_videos[worker_id] = VideoProgress(
                video_name=video_name,
                worker_id=worker_id,
                total_frames=total_frames,
                start_time=time.time()
            )
            self._create_video_widget(worker_id, video_name)
        self._update_display()

    def _create_video_widget(self, worker_id: int, video_name: str):
        """Create UI widgets for a video"""
        if self.is_closed:
            return

        frame = ttk.Frame(self.videos_container)
        frame.pack(fill=tk.X, pady=2)

        # Video name (truncated if too long)
        display_name = video_name if len(video_name) <= 35 else f"...{video_name[-32:]}"
        name_label = ttk.Label(frame, text=display_name, width=38, anchor="w")
        name_label.pack(side=tk.LEFT, padx=(0, 5))

        # Progress bar
        progress = ttk.Progressbar(frame, length=280, mode='determinate')
        progress.pack(side=tk.LEFT, padx=(0, 5))

        # Percentage, FPS and ETA - wider label
        stats_label = ttk.Label(frame, text="0% | 0.0 FPS | ETA: --", width=28, anchor="w")
        stats_label.pack(side=tk.LEFT)

        self.video_widgets[worker_id] = {
            'frame': frame,
            'name_label': name_label,
            'progress': progress,
            'stats_label': stats_label
        }

    def update_video_progress(self, worker_id: int, frames_processed: int,
                              fps: float = 0.0, events_count: int = 0,
                              waiting_recheck: float = 0.0, waiting_stability: float = 0.0):
        """Update progress for a specific video"""
        with self.lock:
            if worker_id in self.active_videos:
                video = self.active_videos[worker_id]
                video.frames_processed = frames_processed
                video.fps = fps
                video.events_count = events_count
                video.waiting_recheck = waiting_recheck
                video.waiting_stability = waiting_stability

                # Update status based on waiting state
                if waiting_recheck > 0 or waiting_stability > 0:
                    video.status = "waiting"
                else:
                    video.status = "processing"

                # Track FPS for averaging (with timestamp)
                if fps > 0:
                    self._fps_history.append((time.time(), fps))

        self._update_display()

    def complete_video(self, worker_id: int, success: bool = True):
        """Mark a video as completed"""
        with self.lock:
            if worker_id in self.active_videos:
                self.active_videos[worker_id].status = "completed" if success else "error"
                self.completed_count += 1

                # Remove widget after a short delay
                if worker_id in self.video_widgets:
                    widgets = self.video_widgets.pop(worker_id)
                    try:
                        widgets['frame'].destroy()
                    except:
                        pass

                # Remove from active tracking
                del self.active_videos[worker_id]

        self._update_display()

    def set_total_queued(self, count: int):
        """Set the total number of queued videos"""
        self.total_queued = count
        self._update_display()

    def _update_display(self):
        """Update all display elements"""
        if self.is_closed:
            return

        try:
            with self.lock:
                active_count = len(self.active_videos)
                total = self.completed_count + active_count + self.total_queued

                # Update overall progress
                if total > 0:
                    self.overall_progress['maximum'] = total
                    self.overall_progress['value'] = self.completed_count
                    pct = (self.completed_count / total) * 100
                    self.overall_label.config(
                        text=f"Completed {self.completed_count} of {total} videos ({pct:.1f}%)"
                    )

                # Update per-video progress bars
                for worker_id, video in self.active_videos.items():
                    if worker_id in self.video_widgets:
                        widgets = self.video_widgets[worker_id]

                        # Update progress bar
                        widgets['progress']['maximum'] = max(1, video.total_frames)
                        widgets['progress']['value'] = video.frames_processed

                        # Update stats label
                        pct = video.percentage
                        fps = video.fps
                        eta = video.eta_seconds

                        # Check if waiting for new content
                        if video.is_waiting:
                            # Show waiting status with countdown
                            wait_str = f"Waiting {video.waiting_stability:.0f}s"
                            widgets['stats_label'].config(
                                text=f"{pct:.1f}% | {wait_str} | Recheck: {video.waiting_recheck:.0f}s"
                            )
                        else:
                            if eta > 0:
                                eta_str = f" | ETA: {self._format_time(eta)}"
                            else:
                                eta_str = ""

                            widgets['stats_label'].config(
                                text=f"{pct:.1f}% | {fps:.1f} FPS{eta_str}"
                            )

                # Update statistics
                self.stats_labels['completed'].config(text=str(self.completed_count))
                self.stats_labels['active'].config(text=str(active_count))
                self.stats_labels['queued'].config(text=str(self.total_queued))

                # Average FPS (over last minute)
                if self._fps_history:
                    now = time.time()
                    cutoff = now - self._fps_window_seconds

                    # Remove old entries
                    while self._fps_history and self._fps_history[0][0] < cutoff:
                        self._fps_history.popleft()

                    # Calculate average from remaining entries
                    if self._fps_history:
                        avg_fps = sum(fps for _, fps in self._fps_history) / len(self._fps_history)
                        self.stats_labels['avg_fps'].config(text=f"{avg_fps:.1f}")
                    else:
                        self.stats_labels['avg_fps'].config(text="--")
                
                # Total events
                total_events = sum(v.events_count for v in self.active_videos.values())
                self.stats_labels['total_events'].config(text=str(total_events))
            
            self.root.update()
            
        except tk.TclError:
            # Window was closed
            self.is_closed = True
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def update_elapsed_time(self, elapsed_seconds: float):
        """Update the elapsed time display"""
        if not self.is_closed:
            try:
                self.stats_labels['elapsed'].config(text=self._format_time(elapsed_seconds))
            except:
                pass

    def set_status(self, status_text: str):
        """Set a custom status message in the overall label"""
        if not self.is_closed:
            try:
                self.overall_label.config(text=status_text)
                self.root.update()
            except tk.TclError:
                self.is_closed = True

    # === Legacy compatibility methods ===

    def update_frame_progress(self, current: int, total: int):
        """Legacy method for backward compatibility"""
        # Find first active video and update it
        with self.lock:
            if self.active_videos:
                worker_id = next(iter(self.active_videos))
                self.active_videos[worker_id].frames_processed = current
                self.active_videos[worker_id].total_frames = total
        self._update_display()

    def update_stats(self, text: str):
        """Legacy method - now shows in overall label"""
        if not self.is_closed:
            try:
                # Parse FPS from legacy format if present
                if "FPS:" in text:
                    parts = text.split("|")
                    for part in parts:
                        if "FPS:" in part:
                            try:
                                fps = float(part.split(":")[1].strip())
                                self._fps_history.append(fps)
                            except:
                                pass
            except:
                pass

    def on_close(self, event=None):
        """Handle window close"""
        self.is_closed = True
        print("\n[EXIT] Closing down... please wait a moment.")

        time.sleep(0.5)

        os._exit(0)
        # self.root.destroy()


    def close(self):
        """Close the progress window"""
        if not self.is_closed:
            self.on_close()


class SimpleProgressWindow:
    """
    Simplified progress window for single video or basic progress tracking.
    Use this when you don't need per-video tracking.
    """

    def __init__(self, title="Processing"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("500x180")

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 250
        y = (self.root.winfo_screenheight() // 2) - 90
        self.root.geometry(f"500x180+{x}+{y}")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video progress
        ttk.Label(main_frame, text="Overall Progress:").pack(pady=5)
        self.video_progress = ttk.Progressbar(main_frame, length=450, mode='determinate')
        self.video_progress.pack(pady=5)
        self.video_label = ttk.Label(main_frame, text="")
        self.video_label.pack()

        # Frame progress
        ttk.Label(main_frame, text="Current Video:").pack(pady=5)
        self.frame_progress = ttk.Progressbar(main_frame, length=450, mode='determinate')
        self.frame_progress.pack(pady=5)
        self.frame_label = ttk.Label(main_frame, text="")
        self.frame_label.pack()

        # Stats
        self.stats_label = ttk.Label(main_frame, text="", font=("Arial", 9))
        self.stats_label.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.is_closed = False

    def update_video_progress(self, current: int, total: int, name: str = ""):
        if not self.is_closed:
            self.video_progress['maximum'] = total
            self.video_progress['value'] = current
            percentage = (current / total * 100) if total > 0 else 0
            self.video_label.config(text=f"Video {current}/{total} ({percentage:.1f}%): {name}")
            self.root.update()

    def update_frame_progress(self, current: int, total: int, fps: float = 0.0):
        if not self.is_closed:
            self.frame_progress['maximum'] = total
            self.frame_progress['value'] = current
            percentage = (current / total * 100) if total > 0 else 0
            fps_str = f" | {fps:.1f} FPS" if fps > 0 else ""
            self.frame_label.config(text=f"Frame {current}/{total} ({percentage:.1f}%){fps_str}")
            self.root.update()

    def update_stats(self, text: str):
        if not self.is_closed:
            self.stats_label.config(text=text)
            self.root.update()

    def on_close(self):
        self.is_closed = True
        self.root.destroy()

    def close(self):
        if not self.is_closed:
            self.on_close()