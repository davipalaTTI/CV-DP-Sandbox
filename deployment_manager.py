"""Shared deployment manifest models, parsing, and validation."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


DAY_NAMES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


class ManifestError(ValueError):
    """Raised when a deployment manifest cannot be run safely."""


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def absolute_path_preserving_symlink(value: str, base_dir: Path) -> Path:
    """Build an absolute path without dereferencing a virtualenv executable symlink."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Path(os.path.abspath(str(path)))


def _parse_clock(value: Any, field_name: str) -> clock_time:
    if not isinstance(value, str):
        raise ManifestError(f"{field_name} must be a string in HH:MM format")
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ManifestError(f"{field_name} must use 24-hour HH:MM format") from exc


def _parse_days(value: Any) -> FrozenSet[int]:
    if value is None:
        return frozenset(range(7))
    if not isinstance(value, list) or not value:
        raise ManifestError("schedule.days must be a non-empty list")

    parsed = set()
    for raw_day in value:
        key = str(raw_day).strip().lower()
        if key not in DAY_NAMES:
            raise ManifestError(f"Unknown schedule day: {raw_day!r}")
        parsed.add(DAY_NAMES[key])
    return frozenset(parsed)


@dataclass(frozen=True)
class DailySchedule:
    days: FrozenSet[int] = frozenset(range(7))
    start: Optional[clock_time] = None
    end: Optional[clock_time] = None
    always: bool = False

    def __post_init__(self) -> None:
        if self.always:
            return
        if self.start is None or self.end is None:
            raise ManifestError("Scheduled cameras require start and end times")
        if self.start == self.end:
            raise ManifestError("schedule.start and schedule.end cannot be equal")

    def active_window(
        self, now: datetime
    ) -> Optional[Tuple[datetime, Optional[datetime]]]:
        """Return the active [start, end) window, including overnight windows."""
        if self.always:
            return datetime.min, None

        for start_date in (now.date(), now.date() - timedelta(days=1)):
            if start_date.weekday() not in self.days:
                continue
            start_dt = datetime.combine(start_date, self.start)
            end_date = start_date if self.end > self.start else start_date + timedelta(days=1)
            end_dt = datetime.combine(end_date, self.end)
            if start_dt <= now < end_dt:
                return start_dt, end_dt
        return None


@dataclass(frozen=True)
class CameraJob:
    name: str
    source_name: str
    config_path: Path
    log_path: Path
    schedule: DailySchedule
    enabled: bool = True
    restart_on_success: bool = True


@dataclass(frozen=True)
class Deployment:
    project_root: Path
    python_executable: Path
    poll_seconds: float
    restart_delay_seconds: float
    shutdown_grace_seconds: float
    debug: bool
    jobs: Tuple[CameraJob, ...]


def read_document(path: Path) -> Dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError(f"{path} must contain an object at its root")
    return data


def _positive_number(data: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{key} must be a number") from exc
    if value <= 0:
        raise ManifestError(f"{key} must be greater than zero")
    return value


def validate_source_config(config_path: Path, project_root: Path) -> Path:
    if not config_path.is_file():
        raise ManifestError(f"Source config does not exist: {config_path}")
    config = read_document(config_path)

    if not (config.get("lines_config") or config.get("zones_config")):
        raise ManifestError(
            f"Source config has no saved counting lines or zones: {config_path}"
        )
    output_folder = config.get("output_folder")
    if not isinstance(output_folder, str) or not output_folder.strip():
        raise ManifestError(f"Source config has no output_folder: {config_path}")
    return resolve_path(output_folder, project_root)


def load_deployment(manifest_path: Path) -> Deployment:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"Deployment manifest does not exist: {manifest_path}")

    data = read_document(manifest_path)
    manifest_dir = manifest_path.parent
    default_root = Path(__file__).resolve().parent
    project_root = resolve_path(str(data.get("project_root", default_root)), manifest_dir)
    python_value = str(data.get("python_executable", sys.executable))
    python_executable = absolute_path_preserving_symlink(python_value, manifest_dir)

    if not (project_root / "main.py").is_file():
        raise ManifestError(f"project_root does not contain main.py: {project_root}")
    if not python_executable.is_file():
        raise ManifestError(f"Python executable does not exist: {python_executable}")

    raw_jobs = data.get("cameras")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ManifestError("cameras must be a non-empty list")

    names = set()
    source_names = set()
    output_folders: Dict[Path, str] = {}
    log_paths: Dict[Path, str] = {}
    jobs: List[CameraJob] = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ManifestError(f"cameras[{index}] must be an object")

        name = str(raw_job.get("name", "")).strip()
        if not name:
            raise ManifestError(f"cameras[{index}].name is required")
        name_key = name.casefold()
        if name_key in names:
            raise ManifestError(f"Duplicate camera name: {name}")
        names.add(name_key)

        config_value = raw_job.get("config")
        if not isinstance(config_value, str) or not config_value.strip():
            raise ManifestError(f"Camera {name!r} requires a config path")
        config_path = resolve_path(config_value, manifest_dir)
        source_config = read_document(config_path)
        output_folder = validate_source_config(config_path, project_root)
        if output_folder in output_folders:
            other_name = output_folders[output_folder]
            raise ManifestError(
                f"Sources {other_name!r} and {name!r} share output_folder {output_folder}"
            )
        output_folders[output_folder] = name

        raw_schedule = raw_job.get("schedule")
        if not isinstance(raw_schedule, dict):
            raise ManifestError(f"Camera {name!r} requires a schedule object")
        always = raw_schedule.get("always", False)
        if not isinstance(always, bool):
            raise ManifestError(f"Camera {name!r} schedule.always must be true or false")
        schedule = DailySchedule(
            always=always,
            days=_parse_days(raw_schedule.get("days")) if not always else frozenset(range(7)),
            start=(
                _parse_clock(raw_schedule.get("start"), f"{name}.schedule.start")
                if not always
                else None
            ),
            end=(
                _parse_clock(raw_schedule.get("end"), f"{name}.schedule.end")
                if not always
                else None
            ),
        )

        source_name = str(raw_job.get("source_name", name)).strip() or name
        source_name_key = source_name.casefold()
        if source_name_key in source_names:
            raise ManifestError(f"Duplicate camera source_name: {source_name}")
        source_names.add(source_name_key)

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "camera"
        log_value = raw_job.get("log_file", f"logs/{safe_name}.log")
        if not isinstance(log_value, str) or not log_value.strip():
            raise ManifestError(f"Camera {name!r} has an invalid log_file")
        log_path = resolve_path(log_value, project_root)
        if log_path in log_paths:
            other_name = log_paths[log_path]
            raise ManifestError(
                f"Sources {other_name!r} and {name!r} share log_file {log_path}"
            )
        log_paths[log_path] = name

        enabled = raw_job.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestError(f"Camera {name!r} enabled must be true or false")

        jobs.append(
            CameraJob(
                name=name,
                source_name=source_name,
                config_path=config_path,
                log_path=log_path,
                schedule=schedule,
                enabled=enabled,
                restart_on_success=bool(
                    source_config.get("is_camera")
                    or source_config.get("input_type") in {"camera", "rtsp"}
                ),
            )
        )

    debug = data.get("debug", False)
    if not isinstance(debug, bool):
        raise ManifestError("debug must be true or false")

    return Deployment(
        project_root=project_root,
        python_executable=python_executable,
        poll_seconds=_positive_number(data, "poll_seconds", 5.0),
        restart_delay_seconds=_positive_number(data, "restart_delay_seconds", 15.0),
        shutdown_grace_seconds=_positive_number(data, "shutdown_grace_seconds", 30.0),
        debug=debug,
        jobs=tuple(jobs),
    )
