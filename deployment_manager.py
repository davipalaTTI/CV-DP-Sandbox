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
    output_folder: Path
    log_path: Path
    schedule: DailySchedule
    enabled: bool = True
    save_footage: bool = False
    footage_retention_days: int = 0
    restart_on_success: bool = True


@dataclass(frozen=True)
class StartupRegistration:
    configured: bool = False
    enabled: bool = False
    windows_task_name: str = "CV-DP Camera Scheduler"
    linux_service_name: str = "cv-dp-camera-scheduler.service"


@dataclass(frozen=True)
class Deployment:
    project_root: Path
    python_executable: Path
    poll_seconds: float
    restart_delay_seconds: float
    shutdown_grace_seconds: float
    debug: bool
    jobs: Tuple[CameraJob, ...]
    startup: StartupRegistration = StartupRegistration()


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


def set_startup_enabled(manifest_path: Path, enabled: bool) -> Path:
    """Atomically persist the desired boot-registration state in a deployment."""
    manifest_path = manifest_path.expanduser().resolve()
    data = read_document(manifest_path)
    startup = data.get("startup")
    if startup is None:
        startup = {}
    elif not isinstance(startup, dict):
        raise ManifestError("startup must be an object")
    startup.update(
        {
            "enabled": bool(enabled),
            "windows_task_name": str(
                startup.get("windows_task_name", "CV-DP Camera Scheduler")
            ),
            "linux_service_name": str(
                startup.get(
                    "linux_service_name",
                    "cv-dp-camera-scheduler.service",
                )
            ),
        }
    )
    data["startup"] = startup

    temp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        if manifest_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        else:
            content = json.dumps(data, indent=2) + "\n"
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(manifest_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return manifest_path


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

        save_footage = raw_job.get(
            "save_footage",
            source_config.get("save_video", True),
        )
        if not isinstance(save_footage, bool):
            raise ManifestError(f"Camera {name!r} save_footage must be true or false")

        retention_days = raw_job.get(
            "footage_retention_days",
            source_config.get("footage_retention_days", 0),
        )
        if isinstance(retention_days, bool) or not isinstance(retention_days, int):
            raise ManifestError(
                f"Camera {name!r} footage_retention_days must be a whole number"
            )
        if retention_days < 0:
            raise ManifestError(
                f"Camera {name!r} footage_retention_days cannot be negative"
            )

        jobs.append(
            CameraJob(
                name=name,
                source_name=source_name,
                config_path=config_path,
                output_folder=output_folder,
                log_path=log_path,
                schedule=schedule,
                enabled=enabled,
                save_footage=save_footage,
                footage_retention_days=retention_days,
                restart_on_success=bool(
                    source_config.get("is_camera")
                    or source_config.get("input_type") in {"camera", "rtsp"}
                ),
            )
        )

    debug = data.get("debug", False)
    if not isinstance(debug, bool):
        raise ManifestError("debug must be true or false")

    raw_startup = data.get("startup")
    if raw_startup is None:
        startup = StartupRegistration()
    elif not isinstance(raw_startup, dict):
        raise ManifestError("startup must be an object")
    else:
        startup_enabled = raw_startup.get("enabled", False)
        if not isinstance(startup_enabled, bool):
            raise ManifestError("startup.enabled must be true or false")
        windows_task_name = str(
            raw_startup.get("windows_task_name", "CV-DP Camera Scheduler")
        ).strip()
        linux_service_name = str(
            raw_startup.get(
                "linux_service_name",
                "cv-dp-camera-scheduler.service",
            )
        ).strip()
        if not windows_task_name:
            raise ManifestError("startup.windows_task_name cannot be empty")
        if not re.match(r"^CV[-_ ]?DP(?:[-_ ]|$)", windows_task_name, re.IGNORECASE):
            raise ManifestError("startup.windows_task_name must begin with CV-DP")
        if not re.fullmatch(
            r"cv-dp[A-Za-z0-9_.@-]*\.service",
            linux_service_name,
        ):
            raise ManifestError(
                "startup.linux_service_name must begin with cv-dp and end with .service"
            )
        startup = StartupRegistration(
            configured=True,
            enabled=startup_enabled,
            windows_task_name=windows_task_name,
            linux_service_name=linux_service_name,
        )

    return Deployment(
        project_root=project_root,
        python_executable=python_executable,
        poll_seconds=_positive_number(data, "poll_seconds", 5.0),
        restart_delay_seconds=_positive_number(data, "restart_delay_seconds", 15.0),
        shutdown_grace_seconds=_positive_number(data, "shutdown_grace_seconds", 30.0),
        debug=debug,
        jobs=tuple(jobs),
        startup=startup,
    )
