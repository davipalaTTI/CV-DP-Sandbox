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
        "registered": False,
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
        return _query_linux_entries()
    raise OSError(f"Boot service management is not supported on {sys.platform}")


def launch_startup_install(
    project_root: Path,
    manifest_path: Path,
    python_executable: str,
    registration_name: Optional[str] = None,
    start_now: bool = False,
) -> str:
    if sys.platform == "win32":
        task_name = registration_name or WINDOWS_TASK_NAME
        installer = project_root / "scripts" / "install_windows_startup_task.ps1"
        command = (
            f"& {_powershell_quote(installer)} "
            f"-Manifest {_powershell_quote(manifest_path)} "
            f"-PythonExecutable {_powershell_quote(python_executable)} "
            f"-TaskName {_powershell_quote(task_name)}"
        )
        if start_now:
            command += " -StartNow"
        parameters = _powershell_encoded_parameters(command)
        _launch_windows_elevated(project_root, parameters)
        action = " and start it now" if start_now else ""
        return (
            "Accept the administrator prompt to enable startup after every Windows boot. "
            f"The task will be registered{action}. A boot during an active schedule "
            "starts the camera immediately."
        )

    if sys.platform.startswith("linux"):
        service_name = registration_name or LINUX_SERVICE_NAME
        installer = project_root / "scripts" / "install_linux_startup_service.sh"
        arguments = [
            "--manifest",
            str(manifest_path),
            "--python-executable",
            python_executable,
            "--service-name",
            service_name,
        ]
        if start_now:
            arguments.append("--start-now")
        log_path = _launch_linux_elevated(
            project_root,
            installer,
            arguments,
        )
        action = " and start it now" if start_now else ""
        return (
            "Complete the administrator authentication prompt to enable the systemd "
            f"service at device boot{action}. A boot during an active schedule starts "
            f"the camera immediately. Operation details: {log_path}"
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
        log_path = _launch_linux_elevated(
            project_root,
            manager,
            ["--operation", "stop", "--service-name", task_name],
        )
        return (
            "Complete the administrator authentication prompt to stop the service. "
            f"It remains enabled for the next device boot. Operation details: {log_path}"
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
        log_path = _launch_linux_elevated(
            project_root,
            manager,
            ["--operation", "remove", "--service-name", task_name],
        )
        return (
            "Complete the administrator authentication prompt to stop and remove the "
            f"systemd startup service. Operation details: {log_path}"
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

    command = (
        f"& {_powershell_quote(manager)} -Operation {_powershell_quote(operation)} "
        f"-TaskName {_powershell_quote(task_name)} "
        f"-TaskPath {_powershell_quote(task_path)}"
    )
    parameters = _powershell_encoded_parameters(command)
    _launch_windows_elevated(project_root, parameters)


def _powershell_quote(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _powershell_encoded_parameters(command: str) -> str:
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return f"-NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"


def _query_linux_entries() -> List[Dict]:
    if shutil.which("systemctl") is None:
        raise OSError("systemctl is not installed; this Jetson cannot manage a systemd service")
    names = {LINUX_SERVICE_NAME}
    completed = subprocess.run(
        [
            "systemctl",
            "list-unit-files",
            "--type=service",
            "--no-legend",
            "--no-pager",
            "cv-dp*.service",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = line.split()
            if fields and fields[0].startswith("cv-dp") and fields[0].endswith(".service"):
                names.add(fields[0])

    entries = []
    for service_name in sorted(names):
        status = _query_linux_status(service_name)
        if status.get("installed", False):
            entries.append(status)
    return entries


def _query_linux_status(service_name: str = LINUX_SERVICE_NAME) -> Dict:
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
        "FragmentPath",
        "LoadError",
    )
    completed = subprocess.run(
        [
            "systemctl",
            "show",
            service_name,
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
    if load_state == "not-found":
        return {
            "installed": False,
            "registered": False,
            "task_name": service_name,
            "task_path": "",
            "state": "Not installed",
            "manifest": "",
            "last_run_time": "",
            "last_result": "",
        }
    manifest = ""
    unit = subprocess.run(
        ["systemctl", "cat", service_name, "--no-pager"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    unit_text = unit.stdout if unit.returncode == 0 else ""
    fragment_path = values.get("FragmentPath", "")
    if not unit_text and fragment_path:
        try:
            unit_text = Path(fragment_path).read_text(encoding="utf-8")
        except OSError:
            pass
    for line in unit_text.splitlines():
        line = line.strip()
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
    load_error = values.get("LoadError", "").strip()
    if load_state != "loaded":
        state = f"{load_state}; {state}"
    return {
        "installed": True,
        "registered": True,
        "task_name": service_name,
        "task_path": "",
        "state": f"{state} ({unit_state})",
        "manifest": manifest,
        "last_run_time": values.get("ExecMainStartTimestamp", ""),
        "last_result": load_error or f"{result} (exit {exit_status})",
        "load_state": load_state,
        "fragment_path": fragment_path,
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
) -> Path:
    if not script_path.is_file():
        raise OSError(f"Startup service script was not found: {script_path}")
    pkexec = shutil.which("pkexec")
    if pkexec is None:
        argument_text = " ".join(shlex.quote(value) for value in arguments)
        raise OSError(
            "pkexec is not installed. Run this from a Jetson terminal instead:\n"
            f"sudo /bin/bash {shlex.quote(str(script_path))} {argument_text}"
        )
    log_path = project_root / "logs" / "startup_service_operation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as operation_log:
        subprocess.Popen(
            [pkexec, "/bin/bash", str(script_path), *arguments],
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=operation_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return log_path
