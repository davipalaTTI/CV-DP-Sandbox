import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional


class ProgressWindow:
    def __init__(self, title="Processing Videos"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("500x200")

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 250
        y = (self.root.winfo_screenheight() // 2) - 100
        self.root.geometry(f"500x200+{x}+{y}")

        # Video progress
        tk.Label(self.root, text="Overall Progress:").pack(pady=5)
        self.video_progress = ttk.Progressbar(
            self.root, length=450, mode='determinate'
        )
        self.video_progress.pack(pady=5)
        self.video_label = tk.Label(self.root, text="")
        self.video_label.pack()

        # Frame progress
        tk.Label(self.root, text="Current Video:").pack(pady=5)
        self.frame_progress = ttk.Progressbar(
            self.root, length=450, mode='determinate'
        )
        self.frame_progress.pack(pady=5)
        self.frame_label = tk.Label(self.root, text="")
        self.frame_label.pack()

        # Stats label
        self.stats_label = tk.Label(self.root, text="", font=("Arial", 9))
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

    def update_frame_progress(self, current: int, total: int):
        if not self.is_closed:
            self.frame_progress['maximum'] = total
            self.frame_progress['value'] = current
            percentage = (current / total * 100) if total > 0 else 0
            self.frame_label.config(text=f"Frame {current}/{total} ({percentage:.1f}%)")
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