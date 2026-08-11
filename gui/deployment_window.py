"""Tk editor for scheduled and multi-camera deployment manifests."""

from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config_manager import DeploymentRequest
from deployment_manager import (
    ManifestError,
    load_deployment,
    read_document,
    validate_source_config,
)
from utils.startup_service import (
    launch_startup_install,
    launch_startup_remove,
    launch_startup_stop,
    query_startup_entries,
    startup_boot_label,
    startup_service_kind,
    startup_service_supported,
)


DAY_OPTIONS = (
    ("Mon", "mon"),
    ("Tue", "tue"),
    ("Wed", "wed"),
    ("Thu", "thu"),
    ("Fri", "fri"),
    ("Sat", "sat"),
    ("Sun", "sun"),
)

def configured_video_source(config_path: str) -> str:
    """Read the exported video source identifier from an existing source config."""
    try:
        value = read_document(Path(config_path).expanduser()).get("source_name", "")
    except ManifestError:
        return ""
    return str(value).strip()


class DeploymentWindow:
    """Create and edit a deployment manifest from saved camera configs."""

    def __init__(
        self,
        schedule_enabled: bool,
        multiple_cameras: bool,
        new_camera_callback: Optional[
            Callable[[Set[str], tk.Misc], Optional[str]]
        ] = None,
    ):
        self.project_root = Path(__file__).resolve().parents[1]
        self.initial_schedule_enabled = schedule_enabled
        self.initial_multiple_cameras = multiple_cameras
        self.new_camera_callback = new_camera_callback
        self.cameras: List[Dict] = []
        self.result: Optional[DeploymentRequest] = None
        self.root: Optional[tk.Tk] = None

    def show(self) -> Optional[DeploymentRequest]:
        self.root = tk.Tk()
        self.root.title("Source Deployment")
        self.root.geometry("1180x620")
        self.root.minsize(900, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

        self.manifest_var = tk.StringVar(value=str(self.project_root / "deployment.json"))
        self.schedule_var = tk.BooleanVar(value=self.initial_schedule_enabled)
        self.multiple_var = tk.BooleanVar(value=self.initial_multiple_cameras)
        self.auto_start_var = tk.BooleanVar(value=self.initial_schedule_enabled)
        self.status_var = tk.StringVar(value="No camera configurations added")

        self._build_layout()
        self._refresh_tree()
        self._center_window()
        self.root.mainloop()
        self.root.destroy()
        return self.result

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        manifest_frame = ttk.LabelFrame(outer, text="Deployment File", padding=10)
        manifest_frame.grid(row=0, column=0, sticky="ew")
        manifest_frame.columnconfigure(1, weight=1)
        ttk.Label(manifest_frame, text="Manifest:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(manifest_frame, textvariable=self.manifest_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(manifest_frame, text="Browse", command=self._browse_manifest).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Button(manifest_frame, text="Load", command=self._load_manifest).grid(
            row=0, column=3, padx=(8, 0)
        )

        mode_frame = ttk.Frame(outer, padding=(0, 12, 0, 8))
        mode_frame.grid(row=1, column=0, sticky="ew")
        ttk.Checkbutton(
            mode_frame,
            text="Use operating schedule",
            variable=self.schedule_var,
            command=self._on_schedule_changed,
        ).pack(side="left")
        ttk.Checkbutton(
            mode_frame,
            text="Process multiple cameras",
            variable=self.multiple_var,
            command=self._on_multiple_changed,
        ).pack(side="left", padx=(24, 0))
        if startup_service_supported():
            ttk.Checkbutton(
                mode_frame,
                text=f"Start at {startup_boot_label()}",
                variable=self.auto_start_var,
                command=self._on_auto_start_changed,
            ).pack(side="left", padx=(24, 0))

        table_frame = ttk.Frame(outer)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("name", "source_name", "config", "output", "schedule", "days", "enabled")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("name", text="Deployment Name")
        self.tree.heading("source_name", text="Video Source")
        self.tree.heading("config", text="Saved Config")
        self.tree.heading("output", text="Output Folder")
        self.tree.heading("schedule", text="Hours")
        self.tree.heading("days", text="Days")
        self.tree.heading("enabled", text="Enabled")
        self.tree.column("name", width=125, minwidth=100)
        self.tree.column("source_name", width=125, minwidth=100)
        self.tree.column("config", width=235, minwidth=170)
        self.tree.column("output", width=225, minwidth=170)
        self.tree.column("schedule", width=125, minwidth=100, anchor="center")
        self.tree.column("days", width=150, minwidth=100, anchor="center")
        self.tree.column("enabled", width=70, minwidth=60, anchor="center", stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _event: self._edit_selected())

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        tools = ttk.Frame(outer, padding=(0, 10, 0, 4))
        tools.grid(row=3, column=0, sticky="ew")
        self.create_button = ttk.Button(
            tools,
            text="Create Source",
            command=self._create_camera,
            state="normal" if self.new_camera_callback else "disabled",
        )
        self.create_button.pack(side="left")
        self.add_button = ttk.Button(tools, text="Add Saved Config", command=self._add_camera)
        self.add_button.pack(side="left", padx=(8, 0))
        ttk.Button(tools, text="Edit", command=self._edit_selected).pack(side="left", padx=(8, 0))
        ttk.Button(tools, text="Remove", command=self._remove_selected).pack(side="left", padx=(8, 0))
        ttk.Label(tools, textvariable=self.status_var).pack(side="right")

        actions = ttk.Frame(outer, padding=(0, 10, 0, 0))
        actions.grid(row=4, column=0, sticky="e")
        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="left")
        if startup_service_supported():
            service_kind = startup_service_kind()
            ttk.Button(
                actions,
                text=f"Manage Startup {service_kind}s",
                command=self._show_startup_status,
            ).pack(side="left", padx=(8, 0))
            ttk.Button(
                actions,
                text=f"Install Startup {service_kind}",
                command=self._install_startup_service,
            ).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Save", command=lambda: self._save_manifest(False)).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="Save and Run", command=lambda: self._save_manifest(True)).pack(
            side="left", padx=(8, 0)
        )

    def _browse_manifest(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Deployment manifest",
            initialdir=str(Path(self.manifest_var.get()).parent),
            initialfile=Path(self.manifest_var.get()).name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.manifest_var.set(filename)

    def _load_manifest(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Load deployment manifest",
            initialdir=str(Path(self.manifest_var.get()).parent),
            filetypes=[("Deployment files", "*.json *.yaml *.yml"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            deployment = load_deployment(Path(filename))
            self.project_root = deployment.project_root
            self.cameras = []
            for job in deployment.jobs:
                schedule = job.schedule
                self.cameras.append(
                    {
                        "name": job.name,
                        "source_name": job.source_name,
                        "config": str(job.config_path),
                        "output_folder": str(
                            validate_source_config(job.config_path, deployment.project_root)
                        ),
                        "log_file": str(job.log_path),
                        "enabled": job.enabled,
                        "start": schedule.start.strftime("%H:%M") if schedule.start else "07:00",
                        "end": schedule.end.strftime("%H:%M") if schedule.end else "18:30",
                        "days": [key for index, (_label, key) in enumerate(DAY_OPTIONS) if index in schedule.days],
                    }
                )
            self.manifest_var.set(filename)
            self.multiple_var.set(len(self.cameras) > 1)
            has_schedule = any(not job.schedule.always for job in deployment.jobs)
            self.schedule_var.set(has_schedule)
            if startup_service_supported() and has_schedule:
                self.auto_start_var.set(True)
            self._refresh_tree()
        except (ManifestError, OSError) as exc:
            messagebox.showerror("Deployment Error", str(exc), parent=self.root)

    def _add_camera(self) -> None:
        if not self.multiple_var.get() and self.cameras:
            messagebox.showinfo(
                "Single Source Mode",
                "Enable multiple cameras before adding another source.",
                parent=self.root,
            )
            return
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Select saved camera config",
            initialdir=str(self.project_root),
            filetypes=[("Config files", "*.json *.yaml *.yml"), ("All files", "*.*")],
        )
        if filename:
            self._open_camera_editor(None, filename)

    def _create_camera(self) -> None:
        if self.new_camera_callback is None:
            return
        if not self.multiple_var.get() and self.cameras:
            messagebox.showinfo(
                "Single Source Mode",
                "Enable multiple cameras before creating another source.",
                parent=self.root,
            )
            return

        reserved_outputs = {
            str(Path(camera["output_folder"]).resolve()).casefold()
            for camera in self.cameras
        }
        config_path = self.new_camera_callback(reserved_outputs, self.root)
        if config_path:
            self._open_camera_editor(None, config_path)

    def _selected_index(self) -> Optional[int]:
        selection = self.tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def _edit_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            self._open_camera_editor(index, self.cameras[index]["config"])

    def _remove_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            del self.cameras[index]
            self._refresh_tree()

    def _open_camera_editor(self, index: Optional[int], config_path: str) -> None:
        existing = self.cameras[index] if index is not None else None
        dialog = tk.Toplevel(self.root)
        dialog.title("Source Configuration")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        default_name = Path(config_path).parent.name or Path(config_path).stem
        name_var = tk.StringVar(value=existing["name"] if existing else default_name)
        default_source = configured_video_source(config_path) or default_name
        source_var = tk.StringVar(value=existing["source_name"] if existing else default_source)
        config_var = tk.StringVar(value=config_path)
        enabled_var = tk.BooleanVar(value=existing["enabled"] if existing else True)
        start_var = tk.StringVar(value=existing["start"] if existing else "07:00")
        end_var = tk.StringVar(value=existing["end"] if existing else "18:30")
        selected_days = set(existing["days"] if existing else [key for _label, key in DAY_OPTIONS])
        day_vars = {key: tk.BooleanVar(value=key in selected_days) for _label, key in DAY_OPTIONS}

        body = ttk.Frame(dialog, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Deployment name:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(body, textvariable=name_var, width=42).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(body, text="Video Source ID:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(body, textvariable=source_var, width=42).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(body, text="Saved config:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(body, textvariable=config_var, width=58).grid(row=2, column=1, sticky="ew", pady=4)

        def browse_config() -> None:
            value = filedialog.askopenfilename(
                parent=dialog,
                title="Select saved camera config",
                filetypes=[("Config files", "*.json *.yaml *.yml"), ("All files", "*.*")],
            )
            if value:
                config_var.set(value)
                source_from_config = configured_video_source(value)
                if source_from_config:
                    source_var.set(source_from_config)

        ttk.Button(body, text="Browse", command=browse_config).grid(row=2, column=2, padx=(8, 0))
        ttk.Checkbutton(body, text="Enabled", variable=enabled_var).grid(
            row=3, column=1, sticky="w", pady=(6, 4)
        )

        if self.schedule_var.get():
            schedule_frame = ttk.LabelFrame(body, text="Operating Schedule", padding=10)
            schedule_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 4))
            ttk.Label(schedule_frame, text="Start:").grid(row=0, column=0, sticky="w")
            ttk.Entry(schedule_frame, textvariable=start_var, width=8).grid(row=0, column=1, padx=(4, 16))
            ttk.Label(schedule_frame, text="End:").grid(row=0, column=2, sticky="w")
            ttk.Entry(schedule_frame, textvariable=end_var, width=8).grid(row=0, column=3, padx=(4, 0))
            days_frame = ttk.Frame(schedule_frame)
            days_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))
            for label, key in DAY_OPTIONS:
                ttk.Checkbutton(days_frame, text=label, variable=day_vars[key]).pack(side="left", padx=(0, 8))

        accepted = {"value": False}

        def accept() -> None:
            name = name_var.get().strip()
            source_name = source_var.get().strip()
            path = Path(config_var.get()).expanduser()
            days = [key for _label, key in DAY_OPTIONS if day_vars[key].get()]
            if not name or not source_name:
                messagebox.showerror(
                    "Source Error",
                    "Deployment name and Video Source ID are required.",
                    parent=dialog,
                )
                return
            if not path.is_file():
                messagebox.showerror("Source Error", f"Config file does not exist:\n{path}", parent=dialog)
                return
            try:
                output_folder = validate_source_config(path.resolve(), self.project_root)
            except ManifestError as exc:
                messagebox.showerror("Source Error", str(exc), parent=dialog)
                return
            if any(
                i != index and Path(item["output_folder"]) == output_folder
                for i, item in enumerate(self.cameras)
            ):
                messagebox.showerror(
                    "Output Folder",
                    f"Each source requires a separate output folder:\n{output_folder}",
                    parent=dialog,
                )
                return
            if any(i != index and item["name"].casefold() == name.casefold() for i, item in enumerate(self.cameras)):
                messagebox.showerror("Source Error", f"Source name already exists: {name}", parent=dialog)
                return
            if any(
                i != index and item["source_name"].casefold() == source_name.casefold()
                for i, item in enumerate(self.cameras)
            ):
                messagebox.showerror(
                    "Source Error",
                    f"Video Source ID already exists: {source_name}",
                    parent=dialog,
                )
                return
            if self.schedule_var.get():
                try:
                    start_time = datetime.strptime(start_var.get().strip(), "%H:%M").time()
                    end_time = datetime.strptime(end_var.get().strip(), "%H:%M").time()
                except ValueError:
                    messagebox.showerror("Schedule Error", "Start and end must use 24-hour HH:MM format.", parent=dialog)
                    return
                if start_time == end_time or not days:
                    messagebox.showerror("Schedule Error", "Select at least one day and use different start/end times.", parent=dialog)
                    return

            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "camera"
            log_file = (
                existing["log_file"]
                if existing
                else str(output_folder / f"{safe_name}.log")
            )
            camera = {
                "name": name,
                "source_name": source_name,
                "config": str(path.resolve()),
                "output_folder": str(output_folder),
                "log_file": log_file,
                "enabled": enabled_var.get(),
                "start": start_var.get().strip(),
                "end": end_var.get().strip(),
                "days": days,
            }
            if index is None:
                self.cameras.append(camera)
            else:
                self.cameras[index] = camera
            accepted["value"] = True
            dialog.destroy()

        buttons = ttk.Frame(body, padding=(0, 12, 0, 0))
        buttons.grid(row=5, column=0, columnspan=3, sticky="e")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left")
        ttk.Button(buttons, text="Apply", command=accept).pack(side="left", padx=(8, 0))
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()
        if accepted["value"]:
            self._refresh_tree()

    def _on_multiple_changed(self) -> None:
        if not self.multiple_var.get() and len(self.cameras) > 1:
            messagebox.showwarning(
                "Multiple Cameras",
                "Remove cameras until only one remains before disabling this option.",
                parent=self.root,
            )
            self.multiple_var.set(True)
        self._refresh_tree()

    def _on_schedule_changed(self) -> None:
        if startup_service_supported() and self.schedule_var.get():
            self.auto_start_var.set(True)
        self._refresh_tree()

    def _on_auto_start_changed(self) -> None:
        if self.auto_start_var.get():
            return

        self.status_var.set("Checking for registered startup schedules...")
        state = {"done": False, "entries": [], "error": None}

        def worker() -> None:
            try:
                state["entries"] = query_startup_entries(self.project_root)
            except Exception as exc:
                state["error"] = str(exc)
            finally:
                state["done"] = True

        def display() -> None:
            if state["error"] is not None:
                self.auto_start_var.set(True)
                self.status_var.set("Startup schedule status unavailable")
                messagebox.showerror(
                    "Startup Service",
                    state["error"],
                    parent=self.root,
                )
                return
            if not state["entries"]:
                self.status_var.set("Startup at boot disabled")
                return

            self.auto_start_var.set(True)
            count = len(state["entries"])
            noun = "schedule" if count == 1 else "schedules"
            if messagebox.askyesno(
                "Manage Startup Schedules",
                f"{count} registered CV-DP startup {noun} must be stopped or removed "
                "explicitly. Open the startup manager now?",
                parent=self.root,
            ):
                self._show_startup_status()

        def poll() -> None:
            if state["done"]:
                display()
            else:
                self.root.after(100, poll)

        threading.Thread(
            target=worker,
            name="startup-service-disable-check",
            daemon=True,
        ).start()
        self.root.after(100, poll)

    def _refresh_tree(self) -> None:
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        scheduled = self.schedule_var.get()
        for index, camera in enumerate(self.cameras):
            hours = f"{camera['start']} - {camera['end']}" if scheduled else "Continuous"
            days = " ".join(day.title() for day in camera["days"]) if scheduled else "Every day"
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    camera["name"],
                    camera["source_name"],
                    camera["config"],
                    camera["output_folder"],
                    hours,
                    days,
                    "Yes" if camera["enabled"] else "No",
                ),
            )
        self.add_button.configure(
            state="normal" if self.multiple_var.get() or not self.cameras else "disabled"
        )
        self.create_button.configure(
            state=(
                "normal"
                if self.new_camera_callback
                and (self.multiple_var.get() or not self.cameras)
                else "disabled"
            )
        )
        count = len(self.cameras)
        self.status_var.set(f"{count} source configuration{'s' if count != 1 else ''}")

    def _manifest_data(self) -> Dict:
        scheduled = self.schedule_var.get()
        camera_records = []
        for camera in self.cameras:
            schedule = (
                {"days": camera["days"], "start": camera["start"], "end": camera["end"]}
                if scheduled
                else {"always": True}
            )
            camera_records.append(
                {
                    "name": camera["name"],
                    "source_name": camera["source_name"],
                    "config": camera["config"],
                    "log_file": camera["log_file"],
                    "enabled": camera["enabled"],
                    "schedule": schedule,
                }
            )
        return {
            "project_root": str(self.project_root),
            "python_executable": sys.executable,
            "poll_seconds": 5,
            "restart_delay_seconds": 15,
            "shutdown_grace_seconds": 30,
            "debug": False,
            "cameras": camera_records,
        }

    def _write_manifest(self) -> Optional[Path]:
        if not self.cameras:
            messagebox.showerror("Deployment Error", "Add at least one camera config.", parent=self.root)
            return None
        if not self.multiple_var.get() and len(self.cameras) != 1:
            messagebox.showerror("Deployment Error", "Single-camera mode requires exactly one config.", parent=self.root)
            return None

        manifest_path = Path(self.manifest_var.get()).expanduser()
        if manifest_path.suffix.lower() != ".json":
            manifest_path = manifest_path.with_suffix(".json")
            self.manifest_var.set(str(manifest_path))
        temp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(self._manifest_data(), indent=2), encoding="utf-8")
            load_deployment(temp_path)
            temp_path.replace(manifest_path)
        except (ManifestError, OSError) as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            messagebox.showerror("Deployment Error", str(exc), parent=self.root)
            return None

        return manifest_path.resolve()

    def _save_manifest(self, run_after_save: bool) -> None:
        manifest_path = self._write_manifest()
        if manifest_path is None:
            return

        if (
            startup_service_supported()
            and getattr(self, "auto_start_var", None) is not None
            and self.auto_start_var.get()
            and not self._launch_startup_installer(manifest_path)
        ):
            return

        if run_after_save:
            self.result = DeploymentRequest(str(manifest_path))
            self.root.quit()
        else:
            messagebox.showinfo("Deployment Saved", f"Saved to:\n{manifest_path}", parent=self.root)

    def _install_startup_service(self) -> None:
        manifest_path = self._write_manifest()
        if manifest_path is None:
            return

        self._launch_startup_installer(manifest_path)

    def _launch_startup_installer(self, manifest_path: Path) -> bool:
        """Open the platform authentication prompt for persistent boot startup."""
        try:
            prompt = launch_startup_install(
                self.project_root,
                manifest_path,
                sys.executable,
            )
        except OSError as exc:
            messagebox.showerror("Startup Service", str(exc), parent=self.root)
            return False

        messagebox.showinfo(
            "Startup Service",
            prompt,
            parent=self.root,
        )
        return True

    def _show_startup_status(self) -> None:
        service_kind = startup_service_kind().lower()
        service_label = startup_service_kind()
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Startup {service_label} Manager")
        dialog.geometry("980x430")
        dialog.minsize(760, 340)
        dialog.transient(self.root)

        outer = ttk.Frame(dialog, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        columns = ("name", "state", "manifest", "last_run", "result")
        task_tree = ttk.Treeview(
            outer,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        task_tree.heading("name", text=service_label)
        task_tree.heading("state", text="State")
        task_tree.heading("manifest", text="Deployment Manifest")
        task_tree.heading("last_run", text="Last Run")
        task_tree.heading("result", text="Last Result")
        task_tree.column("name", width=210, minwidth=150)
        task_tree.column("state", width=110, minwidth=90, anchor="center")
        task_tree.column("manifest", width=310, minwidth=180)
        task_tree.column("last_run", width=150, minwidth=120)
        task_tree.column("result", width=100, minwidth=80, anchor="center")
        task_tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=task_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        task_tree.configure(yscrollcommand=scrollbar.set)

        detail_var = tk.StringVar(value=f"Checking registered startup {service_kind}s...")
        ttk.Label(outer, textvariable=detail_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        actions = ttk.Frame(outer, padding=(0, 10, 0, 0))
        actions.grid(row=2, column=0, columnspan=2, sticky="e")
        entries = {}
        refresh_generation = [0]

        def selected_entry() -> Optional[Dict]:
            selection = task_tree.selection()
            return entries.get(selection[0]) if selection else None

        stop_button = ttk.Button(actions, text="Stop Running", state="disabled")
        stop_button.pack(side="left")
        remove_button = ttk.Button(actions, text="Remove Schedule", state="disabled")
        remove_button.pack(side="left", padx=(8, 0))
        refresh_button = ttk.Button(actions, text="Refresh")
        refresh_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Close", command=dialog.destroy).pack(
            side="left", padx=(8, 0)
        )

        def update_selection(_event=None) -> None:
            entry = selected_entry()
            if entry is None:
                stop_button.configure(state="disabled")
                remove_button.configure(state="disabled")
                return
            state_text = str(entry.get("state", "Unknown"))
            running = "running" in state_text.lower()
            stop_button.configure(state="normal" if running else "disabled")
            remove_button.configure(state="normal")
            manifest = str(entry.get("manifest", "")) or "Unknown"
            detail_var.set(f"Manifest: {manifest}")

        def display_results(result_state: Dict, generation: int) -> None:
            if generation != refresh_generation[0] or not dialog.winfo_exists():
                return
            refresh_button.configure(state="normal")
            error = result_state["error"]
            if error is not None:
                detail_var.set(f"Startup {service_kind} status unavailable")
                messagebox.showerror("Startup Service", error, parent=dialog)
                return

            entries.clear()
            task_tree.delete(*task_tree.get_children())
            for index, entry in enumerate(result_state["entries"]):
                task_name = str(entry.get("task_name", "Unknown"))
                task_path = str(entry.get("task_path", ""))
                display_name = f"{task_path}{task_name}" if task_path else task_name
                item_id = f"startup-{index}"
                entries[item_id] = entry
                task_tree.insert(
                    "",
                    "end",
                    iid=item_id,
                    values=(
                        display_name,
                        entry.get("state", "Unknown"),
                        entry.get("manifest", "") or "Unknown",
                        entry.get("last_run_time", "") or "Never",
                        entry.get("last_result", "Unknown"),
                    ),
                )

            count = len(entries)
            self.status_var.set(f"Startup {service_kind}s found: {count}")
            if count:
                first = next(iter(entries))
                task_tree.selection_set(first)
                task_tree.focus(first)
                update_selection()
            else:
                detail_var.set(f"No CV-DP startup {service_kind}s are registered.")
                stop_button.configure(state="disabled")
                remove_button.configure(state="disabled")

        def refresh() -> None:
            refresh_generation[0] += 1
            generation = refresh_generation[0]
            refresh_button.configure(state="disabled")
            stop_button.configure(state="disabled")
            remove_button.configure(state="disabled")
            detail_var.set(f"Checking registered startup {service_kind}s...")
            result_state = {"done": False, "entries": [], "error": None}

            def worker() -> None:
                try:
                    result_state["entries"] = query_startup_entries(self.project_root)
                except Exception as exc:
                    result_state["error"] = str(exc)
                finally:
                    result_state["done"] = True

            def poll() -> None:
                if not dialog.winfo_exists() or generation != refresh_generation[0]:
                    return
                if result_state["done"]:
                    display_results(result_state, generation)
                else:
                    dialog.after(100, poll)

            threading.Thread(
                target=worker,
                name="startup-service-status",
                daemon=True,
            ).start()
            dialog.after(100, poll)

        def run_operation(operation: str) -> None:
            entry = selected_entry()
            if entry is None:
                messagebox.showwarning(
                    "Startup Service",
                    f"Select a startup {service_kind} first.",
                    parent=dialog,
                )
                return

            task_name = str(entry.get("task_name", ""))
            task_path = str(entry.get("task_path", "\\"))
            display_name = f"{task_path}{task_name}" if task_path else task_name
            if operation == "stop":
                question = (
                    f"Stop the currently running scheduler for {display_name}?\n\n"
                    "The schedule will remain registered for its next startup trigger."
                )
            else:
                question = (
                    f"Stop and remove the schedule {display_name}?\n\n"
                    "Saved deployment, camera configs, and output data will not be deleted."
                )
            if not messagebox.askyesno("Startup Service", question, parent=dialog):
                return

            try:
                if operation == "stop":
                    prompt = launch_startup_stop(
                        self.project_root,
                        task_name,
                        task_path,
                    )
                else:
                    prompt = launch_startup_remove(
                        self.project_root,
                        task_name,
                        task_path,
                    )
                    self.auto_start_var.set(False)
            except OSError as exc:
                messagebox.showerror("Startup Service", str(exc), parent=dialog)
                return

            messagebox.showinfo(
                "Startup Service",
                f"{prompt}\n\nUse Refresh after authentication completes.",
                parent=dialog,
            )

            def refresh_if_open() -> None:
                if dialog.winfo_exists():
                    refresh()

            dialog.after(2500, refresh_if_open)

        task_tree.bind("<<TreeviewSelect>>", update_selection)
        stop_button.configure(command=lambda: run_operation("stop"))
        remove_button.configure(command=lambda: run_operation("remove"))
        refresh_button.configure(command=refresh)
        refresh()

    def _cancel(self) -> None:
        self.result = None
        self.root.quit()

    def _center_window(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"+{x}+{y}")
