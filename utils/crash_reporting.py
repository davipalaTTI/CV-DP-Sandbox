"""Low-overhead crash diagnostics for long-running camera processes."""

from __future__ import annotations

import faulthandler
import os
import platform
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import psutil


class CrashReporter:
    def __init__(
        self,
        report_dir: Path,
        source_name: str = "application",
    ) -> None:
        safe_source = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", source_name.strip()
        ).strip("._") or "application"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.report_dir = Path(report_dir)
        self.report_path = self.report_dir / (
            f"crash_{safe_source}_{timestamp}_pid{os.getpid()}.log"
        )
        self.source_name = source_name or "application"
        self._stream = None
        self._lock = threading.RLock()
        self._reported_exception_ids = set()
        self._previous_sys_hook = None
        self._previous_thread_hook = None
        self._previous_unraisable_hook = None
        self._faulthandler_was_enabled = False

    def install(self) -> "CrashReporter":
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._stream = self.report_path.open("w", encoding="utf-8", buffering=1)
        self._faulthandler_was_enabled = faulthandler.is_enabled()
        faulthandler.enable(file=self._stream, all_threads=True)

        self._previous_sys_hook = sys.excepthook
        sys.excepthook = self._sys_exception_hook
        if hasattr(threading, "excepthook"):
            self._previous_thread_hook = threading.excepthook
            threading.excepthook = self._thread_exception_hook
        if hasattr(sys, "unraisablehook"):
            self._previous_unraisable_hook = sys.unraisablehook
            sys.unraisablehook = self._unraisable_hook
        return self

    def _write_environment(self, context: str) -> None:
        process = psutil.Process()
        memory = process.memory_info()
        self._stream.write("=" * 78 + "\n")
        self._stream.write(f"Crash context: {context}\n")
        self._stream.write(f"Time: {datetime.now().astimezone().isoformat()}\n")
        self._stream.write(f"Source: {self.source_name}\n")
        self._stream.write(f"PID: {os.getpid()}\n")
        self._stream.write(f"Python: {sys.version.replace(os.linesep, ' ')}\n")
        self._stream.write(f"Platform: {platform.platform()}\n")
        self._stream.write(f"Working directory: {Path.cwd()}\n")
        self._stream.write(f"RSS memory: {memory.rss / (1024 * 1024):.1f} MB\n")
        self._stream.write(f"Threads: {process.num_threads()}\n")

    def report_exception(
        self,
        context: str,
        exc: BaseException,
        tb: Optional[TracebackType] = None,
    ) -> None:
        with self._lock:
            if self._stream is None:
                return
            exception_id = id(exc)
            if exception_id in self._reported_exception_ids:
                return
            self._reported_exception_ids.add(exception_id)
            self._write_environment(context)
            traceback.print_exception(
                type(exc), exc, tb if tb is not None else exc.__traceback__, file=self._stream
            )
            self._write_all_thread_stacks()
            self._flush()

    def report_message(self, context: str, message: str) -> None:
        with self._lock:
            if self._stream is None:
                return
            self._write_environment(context)
            self._stream.write(f"Message: {message}\n")
            self._write_all_thread_stacks()
            self._flush()

    def _write_all_thread_stacks(self) -> None:
        self._stream.write("\nPython thread stacks:\n")
        frames = sys._current_frames()
        for thread in threading.enumerate():
            self._stream.write(
                f"\n--- thread name={thread.name!r} ident={thread.ident} "
                f"daemon={thread.daemon} ---\n"
            )
            frame = frames.get(thread.ident)
            if frame is not None:
                traceback.print_stack(frame, file=self._stream)

    def _flush(self) -> None:
        self._stream.flush()
        try:
            os.fsync(self._stream.fileno())
        except OSError:
            pass

    def _sys_exception_hook(
        self,
        exc_type: Type[BaseException],
        exc: BaseException,
        tb: Optional[TracebackType],
    ) -> None:
        self.report_exception("Unhandled main-thread exception", exc, tb)
        if self._previous_sys_hook is not None:
            self._previous_sys_hook(exc_type, exc, tb)

    def _thread_exception_hook(self, args) -> None:
        self.report_exception(
            f"Unhandled thread exception ({args.thread.name})",
            args.exc_value,
            args.exc_traceback,
        )
        if self._previous_thread_hook is not None:
            self._previous_thread_hook(args)

    def _unraisable_hook(self, args) -> None:
        exc = args.exc_value or RuntimeError(str(args.err_msg or "Unraisable error"))
        self.report_exception("Unraisable Python exception", exc, args.exc_traceback)
        if self._previous_unraisable_hook is not None:
            self._previous_unraisable_hook(args)

    def close(self) -> Optional[Path]:
        with self._lock:
            if self._stream is None:
                return None
            sys.excepthook = self._previous_sys_hook
            if self._previous_thread_hook is not None:
                threading.excepthook = self._previous_thread_hook
            if self._previous_unraisable_hook is not None:
                sys.unraisablehook = self._previous_unraisable_hook

            faulthandler.disable()
            if self._faulthandler_was_enabled:
                faulthandler.enable(all_threads=True)
            self._stream.close()
            self._stream = None

            try:
                if self.report_path.stat().st_size == 0:
                    self.report_path.unlink()
                    return None
            except OSError:
                pass
            return self.report_path


_active_reporter: Optional[CrashReporter] = None


def install_crash_reporting(
    log_file: Optional[str] = None,
    source_name: Optional[str] = None,
    report_dir: Optional[str] = None,
) -> CrashReporter:
    global _active_reporter
    if _active_reporter is not None:
        _active_reporter.close()
    if report_dir:
        directory = Path(report_dir)
    elif log_file:
        directory = Path(log_file).expanduser().resolve().parent / "crash_reports"
    else:
        directory = Path("logs") / "crash_reports"
    _active_reporter = CrashReporter(
        directory,
        source_name or "application",
    ).install()
    return _active_reporter


def report_exception(context: str, exc: BaseException) -> None:
    if _active_reporter is not None:
        _active_reporter.report_exception(context, exc)


def close_crash_reporting() -> Optional[Path]:
    global _active_reporter
    if _active_reporter is None:
        return None
    path = _active_reporter.close()
    _active_reporter = None
    return path
