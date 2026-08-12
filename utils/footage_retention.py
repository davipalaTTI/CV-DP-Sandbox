"""Retention policy helpers for managed live-camera recordings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


SECONDS_PER_DAY = 24 * 60 * 60
POLICY_OFF = "Off"
POLICY_KEEP = "Keep indefinitely"
POLICY_DELETE = "Delete after"
POLICY_OPTIONS = (POLICY_OFF, POLICY_KEEP, POLICY_DELETE)


@dataclass(frozen=True)
class CleanupResult:
    deleted_files: int = 0
    freed_bytes: int = 0
    errors: int = 0


def validate_retention_days(value: object, field_name: str = "footage_retention_days") -> int:
    """Return a non-negative integer retention period or raise ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a whole number of days")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a whole number of days")
    try:
        days = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a whole number of days") from exc
    if days < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return days


def footage_policy_label(save_footage: bool, retention_days: int) -> str:
    if not save_footage:
        return "Off"
    days = max(0, int(retention_days))
    return f"{days} day{'s' if days != 1 else ''}" if days else "Keep"


def policy_to_settings(policy: str, retention_days: object) -> tuple[bool, int]:
    """Convert a user-facing policy choice to durable config values."""
    if policy == POLICY_OFF:
        return False, 0
    if policy == POLICY_KEEP:
        return True, 0
    if policy != POLICY_DELETE:
        raise ValueError("Select a valid footage retention policy")
    days = validate_retention_days(retention_days, "Footage retention")
    if days == 0:
        raise ValueError("Footage retention must be at least 1 day")
    return True, days


def settings_to_policy(save_footage: bool, retention_days: object) -> tuple[str, int]:
    days = validate_retention_days(retention_days)
    if not save_footage:
        return POLICY_OFF, max(1, days or 2)
    if days == 0:
        return POLICY_KEEP, 2
    return POLICY_DELETE, days


def cleanup_live_footage(
    output_folder: Union[str, Path],
    retention_days: int,
    *,
    now: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> CleanupResult:
    """Delete expired regular files only from ``output_folder/live_footage``."""
    return cleanup_footage_directory(
        Path(output_folder).expanduser() / "live_footage",
        retention_days,
        now=now,
        logger=logger,
    )


def cleanup_footage_directory(
    footage_folder: Union[str, Path],
    retention_days: int,
    *,
    now: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> CleanupResult:
    """Delete expired regular files beneath one explicitly managed folder."""
    days = validate_retention_days(retention_days)
    if days == 0:
        return CleanupResult()

    live_footage = Path(footage_folder).expanduser()
    if not live_footage.is_dir():
        return CleanupResult()

    cutoff = (time.time() if now is None else float(now)) - (days * SECONDS_PER_DAY)
    deleted_files = 0
    freed_bytes = 0
    errors = 0

    try:
        candidates = list(live_footage.rglob("*"))
    except OSError as exc:
        if logger:
            logger.warning("Could not scan live footage folder %s: %s", live_footage, exc)
        return CleanupResult(errors=1)

    directories = []
    for path in candidates:
        try:
            if path.is_symlink():
                continue
            if path.is_dir():
                directories.append(path)
                continue
            if not path.is_file():
                continue
            stat = path.stat()
            if stat.st_mtime >= cutoff:
                continue
            path.unlink()
            deleted_files += 1
            freed_bytes += stat.st_size
        except OSError as exc:
            errors += 1
            if logger:
                logger.warning("Could not delete expired footage %s: %s", path, exc)

    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    return CleanupResult(deleted_files, freed_bytes, errors)
