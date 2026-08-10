#!/usr/bin/env python3
"""Run one counter process per source according to a local-time schedule."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_manager import (
    CameraJob,
    DailySchedule,
    Deployment,
    ManifestError,
    load_deployment,
)

LOGGER = logging.getLogger("camera_scheduler")


class ManagedCamera:
    def __init__(
        self,
        job: CameraJob,
        deployment: Deployment,
        headless: bool = False,
        window_index: int = 0,
        window_count: int = 1,
    ):
        self.job = job
        self.deployment = deployment
        self.process: Optional[subprocess.Popen] = None
        self.restart_not_before = datetime.min
        self.stop_requested_at: Optional[float] = None
        self.stop_signal_sent = False
        self.completed_window_start: Optional[datetime] = None
        self.headless = headless
        self.window_index = max(0, int(window_index))
        self.window_count = max(1, int(window_count))

    def _command(self, window_end: Optional[datetime]) -> List[str]:
        command = [
            str(self.deployment.python_executable),
            str(self.deployment.project_root / "main.py"),
            "--config",
            str(self.job.config_path),
            "--no-gui",
            "--source-name",
            self.job.source_name,
            "--log-file",
            str(self.job.log_path),
            "--window-index",
            str(self.window_index),
            "--window-count",
            str(self.window_count),
            "--crash-report-dir",
            str(self.job.log_path.parent / "crash_reports"),
        ]
        if self.headless:
            command.append("--headless")
        if window_end is not None:
            command.extend(["--stop-at", window_end.isoformat(timespec="seconds")])
        if self.deployment.debug:
            command.append("--debug")
        return command

    def start(self, window_end: Optional[datetime]) -> None:
        self.job.log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: Dict[str, Any] = {"cwd": str(self.deployment.project_root)}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        self.process = subprocess.Popen(self._command(window_end), **kwargs)
        self.stop_requested_at = None
        self.stop_signal_sent = False
        stop_label = window_end.isoformat(timespec="seconds") if window_end else "continuous"
        LOGGER.info(
            "Started camera %s (pid=%s, scheduled stop=%s)",
            self.job.name,
            self.process.pid,
            stop_label,
        )

    def observe_exit(
        self, now: datetime, active_window_start: Optional[datetime]
    ) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        log_method = LOGGER.info if exit_code == 0 else LOGGER.warning
        log_method("Source %s exited with code %s", self.job.name, exit_code)
        process_id = self.process.pid
        self.process = None
        self.stop_requested_at = None
        self.stop_signal_sent = False
        self.restart_not_before = now + timedelta(
            seconds=self.deployment.restart_delay_seconds
        )
        if exit_code != 0:
            self._record_crash_exit(now, process_id, exit_code)
        if exit_code == 0 and not self.job.restart_on_success:
            self.completed_window_start = active_window_start

    def _record_crash_exit(
        self,
        occurred_at: datetime,
        process_id: int,
        exit_code: int,
    ) -> None:
        """Append bounded supervisor metadata for a failed child process."""
        report_dir = self.job.log_path.parent / "crash_reports"
        history_path = report_dir / "supervisor_exit_history.jsonl"
        backup_path = report_dir / "supervisor_exit_history.jsonl.1"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            if history_path.exists() and history_path.stat().st_size >= 5 * 1024 * 1024:
                backup_path.unlink(missing_ok=True)
                history_path.replace(backup_path)
            record = {
                "time": occurred_at.astimezone().isoformat(),
                "source": self.job.source_name,
                "camera": self.job.name,
                "pid": process_id,
                "exit_code": exit_code,
                "config": str(self.job.config_path),
                "process_log": str(self.job.log_path),
                "restart_not_before": self.restart_not_before.astimezone().isoformat(),
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            LOGGER.warning(
                "Could not write crash history for %s: %s", self.job.name, exc
            )

    def request_stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            LOGGER.info("Requested graceful stop for camera %s", self.job.name)
        except OSError as exc:
            LOGGER.warning("Could not stop camera %s: %s", self.job.name, exc)

    def force_stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            LOGGER.error("Force-stopping camera %s after grace period", self.job.name)
            self.process.kill()


class ScheduleSupervisor:
    def __init__(
        self,
        deployment: Deployment,
        headless: bool = False,
        manifest_path: Optional[Path] = None,
    ):
        self.deployment = deployment
        self.headless = headless
        self.manifest_path = manifest_path.expanduser().resolve() if manifest_path else None
        self.stop_file_path = (
            Path(f"{self.manifest_path}.scheduler-stop")
            if self.manifest_path is not None
            else None
        )
        self._manifest_file_signature = self._read_manifest_signature()
        self.cameras = self._build_cameras(deployment)
        self.stop_requested = False

    def _build_cameras(self, deployment: Deployment) -> List[ManagedCamera]:
        window_count = max(1, len(deployment.jobs))
        return [
            ManagedCamera(
                job,
                deployment,
                headless=self.headless,
                window_index=index,
                window_count=window_count,
            )
            for index, job in enumerate(deployment.jobs)
        ]

    def _read_manifest_signature(self) -> Optional[tuple]:
        if self.manifest_path is None:
            return None
        try:
            stat = self.manifest_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _reload_if_changed(self) -> bool:
        if self.manifest_path is None:
            return False
        signature = self._read_manifest_signature()
        if signature is None or signature == self._manifest_file_signature:
            return False
        try:
            deployment = load_deployment(self.manifest_path)
        except ManifestError as exc:
            self._manifest_file_signature = signature
            LOGGER.error("Could not reload changed deployment manifest: %s", exc)
            return False

        LOGGER.info("Deployment manifest changed; applying the saved schedule")
        self.shutdown()
        self.deployment = deployment
        self.cameras = self._build_cameras(deployment)
        self._manifest_file_signature = self._read_manifest_signature()
        LOGGER.info("Reloaded deployment with %d camera(s)", len(self.cameras))
        return True

    def _consume_external_stop_request(self) -> bool:
        if self.stop_file_path is None or not self.stop_file_path.exists():
            return False
        LOGGER.info("External startup-task removal requested")
        try:
            self.stop_file_path.unlink()
        except OSError as exc:
            LOGGER.warning("Could not remove scheduler stop marker: %s", exc)
        return True

    def _handle_signal(self, signum, _frame) -> None:
        LOGGER.info("Scheduler received signal %s", signum)
        self.stop_requested = True

    def _sync_camera(self, camera: ManagedCamera, now: datetime) -> None:
        active_window = camera.job.schedule.active_window(now) if camera.job.enabled else None
        active_window_start = active_window[0] if active_window is not None else None
        camera.observe_exit(now, active_window_start)

        if active_window is not None:
            if (
                camera.process is None
                and camera.completed_window_start != active_window_start
                and now >= camera.restart_not_before
            ):
                camera.start(active_window[1])
            return

        if camera.process is None:
            return

        if camera.stop_requested_at is None:
            # The child normally exits itself at --stop-at. Give it a short grace
            # period before sending a signal in case this loop reaches the boundary first.
            camera.stop_requested_at = time.monotonic()
            return

        elapsed = time.monotonic() - camera.stop_requested_at
        if (
            not camera.stop_signal_sent
            and elapsed >= min(2.0, self.deployment.shutdown_grace_seconds)
        ):
            camera.request_stop()
            camera.stop_signal_sent = True
        if elapsed >= self.deployment.shutdown_grace_seconds:
            camera.force_stop()

    def shutdown(self) -> None:
        running = [camera for camera in self.cameras if camera.process is not None]
        for camera in running:
            camera.request_stop()

        deadline = time.monotonic() + self.deployment.shutdown_grace_seconds
        while time.monotonic() < deadline:
            if all(
                camera.process is None or camera.process.poll() is not None
                for camera in running
            ):
                return
            time.sleep(0.2)

        for camera in running:
            camera.force_stop()

    def cleanup(self) -> None:
        self.stop_requested = True
        self.shutdown()

    def run(self) -> int:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._handle_signal)

        LOGGER.info("Scheduler started with %d camera(s)", len(self.cameras))
        try:
            while not self.stop_requested:
                if self._consume_external_stop_request():
                    self.stop_requested = True
                    break
                self._reload_if_changed()
                now = datetime.now()
                for camera in self.cameras:
                    self._sync_camera(camera, now)
                time.sleep(self.deployment.poll_seconds)
        finally:
            self.shutdown()
        return 0


def _configure_logging(log_path: Path, debug: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Deployment JSON/YAML path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the manifest and camera configs, then exit",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Disable runtime camera/status windows in child processes",
    )
    parser.add_argument(
        "--show-windows",
        dest="headless",
        action="store_false",
        help="Show source-named live/status windows for an interactive run",
    )
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)
    try:
        deployment = load_deployment(Path(args.manifest))
    except ManifestError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(deployment.project_root / "logs" / "scheduler.log", deployment.debug)
    LOGGER.info("Validated deployment manifest: %s", Path(args.manifest).resolve())
    if args.check:
        return 0
    return ScheduleSupervisor(
        deployment,
        headless=args.headless,
        manifest_path=Path(args.manifest),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
