"""Platform-specific boot service installation and status helpers."""

from __future__ import annotations

import base64
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


WINDOWS_TASK_NAME = "CV-DP Camera Scheduler"
LINUX_SERVICE_NAME = "cv-dp-camera-scheduler.service"


def startup_service_supported(platform: Optional[str] = None) -> bool:
    platform = platform or sys.platform
    return platform == "win32" or platform.startswith("linux")


def startup_service_kind(platform: Optional[str] = None) -> str:
    return "Task" if (platform or sys.platform) == "win32" else "Service"


def startup_boot_label(platform: Optional[str] = None) -> str:
    return "Windows boot" if (platform or sys.platform) == "win32" else "device boot"


def query_startup_status(project_root: Path) -> Dict:
    """Return the primary startup entry for backward-compatible callers."""
    entries = query_startup_entries(project_root)
    if entries:
        return entries[0]
    return {
        "installed": False,
        "task_name": WINDOWS_TASK_NAME if sys.platform == "win32" else LINUX_SERVICE_NAME,
        "task_path": "\\" if sys.platform == "win32" else "",
        "state": "Not installed",
        "manifest": "",
        "last_run_time": "",
        "last_result": "",
    }


def query_startup_entries(project_root: Path) -> List[Dict]:
    if sys.platform == "win32":
        return _query_windows_entries(project_root)
    if sys.platform.startswith("linux"):
        status = _query_linux_status()
        return [status] if status.get("installed", False) else []
    raise OSError(f"Boot service management is not supported on {sys.platform}")


def launch_startup_install(
    project_root: Path,
    manifest_path: Path,
    python_executable: str,
) -> str:
    if sys.platform == "win32":
        installer = project_root / "scripts" / "install_windows_startup_task.ps1"
        parameters = (
            f'-NoProfile -ExecutionPolicy Bypass -File "{installer}" '
            f'-Manifest "{manifest_path}" -PythonExecutable "{python_executable}"'
        )
        _launch_windows_elevated(project_root, parameters)
        return (
            "Accept the administrator prompt to enable startup after every Windows boot. "
            "A boot during an active schedule starts the camera immediately."
        )

    if sys.platform.startswith("linux"):
        installer = project_root / "scripts" / "install_linux_startup_service.sh"
        _launch_linux_elevated(
            project_root,
            installer,
            [
                "--manifest",
                str(manifest_path),
                "--python-executable",
                python_executable,
            ],
        )
        return (
            "Complete the administrator authentication prompt to enable the systemd "
            "service at device boot. A boot during an active schedule starts the camera "
            "immediately."
        )

    raise OSError(f"Boot service installation is not supported on {sys.platform}")


def launch_startup_stop(
    project_root: Path,
    task_name: str = WINDOWS_TASK_NAME,
    task_path: str = "\\",
) -> str:
    if sys.platform == "win32":
        _launch_windows_task_operation(project_root, "Stop", task_name, task_path)
        return (
            "Accept the administrator prompt to stop the selected run. Its startup "
            "registration will remain installed."
        )

    if sys.platform.startswith("linux"):
        manager = project_root / "scripts" / "manage_linux_startup_service.sh"
        _launch_linux_elevated(project_root, manager, ["--operation", "stop"])
        return (
            "Complete the administrator authentication prompt to stop the service. "
            "It remains enabled for the next device boot."
        )

    raise OSError(f"Boot service stop is not supported on {sys.platform}")


def launch_startup_remove(
    project_root: Path,
    task_name: str = WINDOWS_TASK_NAME,
    task_path: str = "\\",
) -> str:
    if sys.platform == "win32":
        _launch_windows_task_operation(project_root, "Remove", task_name, task_path)
        return "Accept the administrator prompt to stop and remove the Windows startup task."

    if sys.platform.startswith("linux"):
        manager = project_root / "scripts" / "manage_linux_startup_service.sh"
        _launch_linux_elevated(project_root, manager, ["--operation", "remove"])
        return (
            "Complete the administrator authentication prompt to stop and remove the "
            "systemd startup service."
        )

    raise OSError(f"Boot service removal is not supported on {sys.platform}")


def _query_windows_entries(project_root: Path) -> List[Dict]:
    manager = project_root / "scripts" / "manage_windows_startup_task.ps1"
    if not manager.is_file():
        raise OSError(f"Startup task manager was not found: {manager}")
    kwargs = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(manager),
            "-Operation",
            "List",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        **kwargs,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OSError(detail or "Windows could not query the startup task")
    json_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.lstrip().startswith(("[", "{"))
    ]
    if not json_lines:
        raise OSError("Windows returned no startup task status")
    try:
        result = json.loads(json_lines[-1])
    except json.JSONDecodeError as exc:
        raise OSError("Windows returned invalid startup task status") from exc
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        raise OSError("Windows returned an unexpected startup task status")
    return result


def _launch_windows_task_operation(
    project_root: Path,
    operation: str,
    task_name: str,
    task_path: str,
) -> None:
    manager = project_root / "scripts" / "manage_windows_startup_task.ps1"
    if not manager.is_file():
        raise OSError(f"Startup task manager was not found: {manager}")

    def quote(value) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    command = (
        f"& {quote(manager)} -Operation {quote(operation)} "
        f"-TaskName {quote(task_name)} -TaskPath {quote(task_path)}"
    )
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    parameters = f"-NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"
    _launch_windows_elevated(project_root, parameters)


def _query_linux_status() -> Dict:
    if shutil.which("systemctl") is None:
        raise OSError("systemctl is not installed; this Jetson cannot manage a systemd service")
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "ExecMainStartTimestamp",
        "ExecMainStatus",
        "Result",
    )
    completed = subprocess.run(
        [
            "systemctl",
            "show",
            LINUX_SERVICE_NAME,
            "--no-pager",
            *(f"--property={name}" for name in properties),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    if completed.returncode != 0 and "LoadState" not in values:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OSError(detail or "systemd could not query the startup service")
    load_state = values.get("LoadState", "not-found")
    if load_state != "loaded":
        return {
            "installed": False,
            "task_name": LINUX_SERVICE_NAME,
            "state": "Not installed",
            "manifest": "",
            "last_run_time": "",
            "last_result": "",
        }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OSError(detail or "systemd could not query the startup service")

    manifest = ""
    unit = subprocess.run(
        ["systemctl", "cat", LINUX_SERVICE_NAME, "--no-pager"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if unit.returncode == 0:
        for line in unit.stdout.splitlines():
            if not line.startswith("ExecStart="):
                continue
            try:
                command = shlex.split(line.split("=", 1)[1])
                manifest_index = command.index("--manifest") + 1
                manifest = command[manifest_index]
            except (ValueError, IndexError):
                pass
            break

    active = values.get("ActiveState", "unknown")
    substate = values.get("SubState", "unknown")
    state = f"{active}/{substate}"
    unit_state = values.get("UnitFileState", "unknown")
    result = values.get("Result", "unknown")
    exit_status = values.get("ExecMainStatus", "")
    return {
        "installed": True,
        "task_name": LINUX_SERVICE_NAME,
        "state": f"{state} ({unit_state})",
        "manifest": manifest,
        "last_run_time": values.get("ExecMainStartTimestamp", ""),
        "last_result": f"{result} (exit {exit_status})",
    }


def _launch_windows_elevated(project_root: Path, parameters: str) -> None:
    import ctypes

    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        "powershell.exe",
        parameters,
        str(project_root),
        1,
    )
    if result <= 32:
        raise OSError(f"Windows ShellExecute error {result}")


def _launch_linux_elevated(
    project_root: Path,
    script_path: Path,
    arguments: list,
) -> None:
    if not script_path.is_file():
        raise OSError(f"Startup service script was not found: {script_path}")
    pkexec = shutil.which("pkexec")
    if pkexec is None:
        argument_text = " ".join(shlex.quote(value) for value in arguments)
        raise OSError(
            "pkexec is not installed. Run this from a Jetson terminal instead:\n"
            f"sudo /bin/bash {shlex.quote(str(script_path))} {argument_text}"
        )
    subprocess.Popen(
        [pkexec, "/bin/bash", str(script_path), *arguments],
        cwd=str(project_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
