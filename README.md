# Multi-Line Object Counter

A comprehensive computer vision application for counting objects crossing predefined lines and zones using YOLO detection and tracking.

![Version](https://img.shields.io/badge/version-1.0.1-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🎯 Overview

This application provides real-time object counting across multiple counting lines and zones with direction filtering, speed estimation, heatmap visualization, and training data capture capabilities. It supports both live camera feeds and pre-recorded video files (including folders of videos), with hourly segmentation and comprehensive event logging.

**Key Highlights:**
- Process single videos, folders of videos, or live camera streams
- Draw counting lines and zones interactively or load from saved configurations
- Run fully headless for automated/scheduled processing
- Export events to JSON, CSV, Excel, and optionally to PostgreSQL cloud databases
- Parallel video processing for batch operations

---

## ✨ Features

### Core Features
- **Multi-line counting** with configurable direction filtering (up/down/left/right)
- **Zone-based counting** with occupancy tracking and dwell time measurement
- **Exclusion zones** to ignore specific areas (e.g., static objects, irrelevant regions)
- **Real-time object tracking** using YOLO v11 with BoTSORT tracker
- **Class-specific filtering** - count only specific object types per line/zone
- **Speed estimation** in multiple units (px/s, m/s, km/h, mph)
- **Automated hourly segmentation** with clock-aligned boundaries

### Advanced Features
- **Parallel video processing** - Process 1-4 videos simultaneously
- **Folder monitoring** - Automatically process new videos as they appear
- **Growing file support** - Handle files still being recorded/written
- **Training mode** - Capture annotated frames for dataset creation
- **Heatmap generation** - Visualize object presence over time
- **Frame skipping** with track interpolation for performance optimization
- **Live editing** - Add/modify/delete lines and zones during runtime
- **Headless operation** - Run without GUI using saved configuration files
- **Cloud database upload** - Optionally push events to PostgreSQL

### Export Formats
- **JSON** - Machine-readable event logs
- **CSV** - Spreadsheet-compatible data
- **Excel** - Formatted workbooks with auto-sized columns
- **Master Event Log** - Cumulative Excel file updated in real-time
- **PostgreSQL** - Optional cloud database upload

---

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows, Linux, or macOS
- GPU recommended (NVIDIA CUDA, Apple MPS, or CPU fallback)
- 8GB+ RAM recommended for parallel processing

### Python Dependencies

```
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.23.0,<2.0.0
torch>=2.0.0
pandas>=2.0.0
openpyxl>=3.1.0
psutil>=5.9.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.0.0
requests>=2.31.0          # cloud API upload (always imported)
python-dotenv>=1.0.0      # loads API_URL / API_KEY from .env
```

**Optional dependencies:**
```
onnxruntime>=1.16.0       # required to run .onnx models
onnx>=1.14.0              # fallback for ONNX input-shape probing
memory-profiler>=0.61.0   # enables the --memory-profile CLI flag
# TensorRT must be installed manually from NVIDIA (no pip package).
```

---

## 🚀 Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/yourusername/multi-line-counter.git
cd multi-line-counter
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: ONNX model support
pip install onnxruntime onnx

# Optional: --memory-profile CLI flag
pip install memory-profiler
```

### 4. Download YOLO Model

The application will auto-download `yolo11s.pt` on first run, or you can:

```bash
# Download manually from Ultralytics
# https://github.com/ultralytics/assets/releases/

# Or use a custom-trained model (.pt, .onnx, or .engine format)
```

---

## 🎮 Usage

### Interactive Mode (Recommended for First Run)

```bash
python main.py
```

This launches the GUI configuration wizard:
1. **Run Mode** - Choose scheduling and/or multiple cameras
2. **Configuration** - Configure one source, or select saved camera configs for a deployment
3. **Interactive Setup** - Draw counting lines, zones, and exclusion areas for a new camera config
4. **Processing** - Run the selected source or camera deployment

### Reusing Saved Setup

After your first run, a `config.json` file is saved in your output folder. Reuse it:

```bash
# Skip configuration/drawing and use saved lines/zones directly.
# The runtime camera view still follows show_live_video in the config.
python main.py --config path/to/config.json --no-gui

# Disable all runtime windows for unattended execution
python main.py --config path/to/config.json --headless

# Load config but allow GUI modifications
python main.py --config path/to/config.json
```

### Scheduled Multi-Camera Deployment

Lines, zones, exclusion areas, and processing settings are already stored in each
camera's `config.json`. Keep one config and one output folder per camera. The
deployment scheduler starts one isolated process for every camera whose local-time
schedule is active.

The normal user workflow is available from `python main.py`:

- Leave both run-mode options off for the existing single-source setup.
- Enable **Use an operating schedule** for one scheduled camera.
- Enable **Process multiple cameras** for a continuous multi-camera deployment.
- Enable both options for scheduled multi-camera processing.

The deployment editor supports both first-time and existing cameras:

- **Create Source** opens the original settings dialog with model selection and all
  four input types: Folder, Video, Camera, and RTSP Stream. It then runs the existing
  line and zone drawing workflow. When setup is complete, the config is saved and
  the deployment list returns so the next source can be created.
- **Add Saved Config** adds a camera whose setup was completed previously.
- **Edit Source Settings** reopens a selected camera's model, input, output, and
  processing settings without discarding its saved lines or zones.
- **Edit Deployment** changes the schedule, enabled state, source ID, and the
  per-camera live-footage policy.

Each camera has an independent live-footage policy: **Off** writes no recordings,
**Keep indefinitely** preserves all recordings, and **Delete after** removes files
older than the selected number of days. Automatic cleanup only operates inside
that camera output folder's `live_footage` directory. It runs when a direct camera
starts and at hourly rollover, and the deployment scheduler also checks every 15
minutes, including outside camera operating hours. The **Video Source ID** is
written to the `video_source` column for every record produced by that camera.

Footage recording defaults to **off** for new source configs and new example
deployments. When it is off, live cameras do not create or write MP4 files under
`live_footage`, and file/folder inputs do not create processed output videos.
Counts, event exports, logs, crash reports, heatmaps, and API uploads continue.
Selecting **Off** does not delete existing footage; select **Delete after** to
apply automatic retention. Older deployment manifests without `save_footage`
inherit their saved source config's `save_video` value. A missing
`footage_retention_days` value means keep indefinitely for backward compatibility.

Each camera must use a different output folder. The editor shows that folder in the
camera list and rejects duplicates before drawing or adding the camera. Segment
files, master logs, video, training data, and the camera process log therefore stay
separate. The generated manifest remains compatible with unattended startup and
can still be edited manually when needed. On Windows, leave **Start at Windows
boot** enabled. On Jetson/Linux, leave **Start at device boot** enabled. Saving
opens the platform's administrator authentication prompt and installs the boot
task or service.

To manage an existing schedule, load its deployment manifest and use **Edit
Deployment** to change days, hours, footage policy, or the source's **Enabled**
setting. Saving the same manifest causes a running supervisor to reload it within
`poll_seconds` and gracefully restart the managed camera processes with the new
setting. Use **Manage Startup Tasks** on Windows or **Manage Startup Services** on
Jetson to list registered CV-DP schedulers and the deployment currently selected
in the editor. Broken systemd units remain visible with their full load error.
**Edit Schedule** loads the selected manifest, **Install / Repair** rebuilds and
starts its boot registration, and **Stop Running** ends only the current run.
**Remove From Startup** stops and unregisters it while keeping the editable
deployment manifest, camera configs, and output data.

Deployment manifests persist the desired registration under `startup`, including
`enabled`, `windows_task_name`, and `linux_service_name`. This lets the manager
show a configured schedule even when Task Scheduler or systemd registration is
missing.

1. Run `python main.py` and select the deployment run-mode options.
2. Use **Create Source** for each new setup, choosing a unique output folder during
   its settings step. Use **Add Saved Config** only for existing setups.
3. Save or run the deployment. `deployment.example.json` remains available as a
   platform-neutral manual template; when `python_executable` is omitted, the
   scheduler uses the interpreter that launched it.
3. Validate without starting cameras:

```powershell
.venv\Scripts\python.exe scripts\scheduled_runner.py --manifest deployment.json --check
```

4. Run the scheduler manually for an end-to-end check:

```powershell
.venv\Scripts\python.exe scripts\scheduled_runner.py --manifest deployment.json --show-windows
```

Manual runs and **Save and Run** honor each camera config's **Show Live Video**
setting. Multi-camera windows include the source name and are arranged into screen
tiles. Press `V` in a camera's status window to open that camera's live view.

If the computer boots at any point inside an active window, that camera starts
immediately and receives the window end as a graceful stop time. A failed camera
process or disconnected stream is restarted while its window remains active.
Overnight windows such as `22:00` to `06:00` are supported. Schedule times use the
computer's local timezone, so automatic clock synchronization should be enabled.

On Windows, run PowerShell as Administrator to install the scheduler at system
startup under the `SYSTEM` account:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_startup_task.ps1 `
  -Manifest .\deployment.json
```

The GUI task manager is preferred when more than one CV-DP task exists. The
PowerShell equivalent for listing every detected CV-DP task is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\manage_windows_startup_task.ps1 `
  -Operation List
```

On NVIDIA Jetson/Ubuntu, the GUI uses the normal PolicyKit authentication prompt
to install `cv-dp-camera-scheduler.service`. It runs headlessly as the user who
installed it, waits for networking, and is enabled through `systemd`. The terminal
equivalent is:

```bash
sudo bash scripts/install_linux_startup_service.sh \
  --manifest "$(realpath deployment.json)" \
  --python-executable "$PWD/.venv/bin/python" \
  --start-now
```

The installer runs `systemd-analyze verify`, reloads systemd, confirms
`LoadState=loaded`, enables the service for future boots, and optionally starts it
immediately. To repair an existing `Loaded: bad-setting` service, use **Install /
Repair** or rerun the command above. GUI operation output is written to
`logs/startup_service_operation.log`.

Inspect or remove the Jetson service with:

```bash
systemctl status cv-dp-camera-scheduler.service --no-pager --full
systemctl is-enabled cv-dp-camera-scheduler.service
journalctl -u cv-dp-camera-scheduler.service -f
sudo bash scripts/manage_linux_startup_service.sh --operation remove
```

The boot task/service does not turn the device on. Configure the Jetson carrier
board or power controller to restore power after AC loss, or use the platform's
supported wake mechanism. Let the application reach its scheduled stop before a
smart plug removes power. Abrupt power removal can lose buffered events or corrupt
the open video segment. A practical external-power schedule should leave several
minutes between the application's end time and power removal.

Each camera process loads its own model. Confirm that the GPU has enough memory for
the intended camera count; CPU mode is available through each camera config when
process isolation is more important than throughput.

### Performance and Long-Running Diagnostics

Live cameras use a latest-frame pipeline. A captured frame is submitted once, and
if inference is slower than the camera, the one waiting frame is replaced with the
newest frame. This avoids processing stale or duplicate frames and keeps latency
low without imposing an artificial FPS limit. Recording also uses a small bounded
writer queue so slow storage cannot consume unbounded memory.

Every 30 seconds, each camera log reports capture FPS, inference FPS, analytics
FPS, stale input drops, recording queue depth/drops, and process RSS memory. Stale
input drops are expected whenever camera FPS is higher than model inference FPS;
they show that the process is staying current rather than building latency.

Advanced per-camera config values and defaults are:

```json
{
  "camera_stall_timeout_seconds": 20.0,
  "inference_stall_timeout_seconds": 120.0,
  "max_consecutive_detection_errors": 30,
  "performance_log_interval_seconds": 30.0,
  "video_writer_queue_size": 8,
  "video_writer_stall_timeout_seconds": 30.0
}
```

A stalled camera read, hung inference, or repeated detection failure exits the
camera child with a nonzero code so the deployment supervisor can restart it.
Python exceptions and native faults are written under the process log directory's
`crash_reports/` folder. `supervisor_exit_history.jsonl` records failed process
exit codes and restart timing. Clean exits remove their unused empty crash file.

### Command Line Options

```bash
python main.py [options]

Options:
  -c, --config FILE     Load configuration from JSON/YAML file
  --no-gui              Skip interactive setup (requires --config with saved lines)
  --headless            Disable runtime windows and skip interactive setup
  --crash-report-dir    Override the crash report directory
  -d, --debug           Enable debug logging (verbose output)
  -l, --log-file FILE   Specify log file path (default: logs/app.log)
  -v, --version         Show version information
  --profile             Run with performance profiling
  --memory-profile      Run with memory profiling
```

### Examples

```bash
# Interactive mode - full GUI
python main.py

# Load config, modify lines/zones in GUI, then process
python main.py --config outputs/config.json

# Reuse saved geometry and keep the configured live camera view
python main.py --config outputs/config.json --no-gui

# Headless with debug logging
python main.py --config outputs/config.json --headless --debug

# Custom log file location
python main.py --config outputs/config.json --headless --log-file /var/log/counter.log
```

---

## ⚙️ GUI Configuration Wizard

When you run `python main.py`, the configuration dialog appears:

### Basic Settings Section

| Setting | Description | Default |
|---------|-------------|---------|
| **YOLO Model File** | Path to model (.pt, .onnx, .engine) | `yolo11s.pt` |
| **Input Type** | Folder / Video File / Camera | Folder |
| **Input Source** | Path to folder/video or camera index | - |
| **Output Folder** | Where results and recordings are saved | - |

### Options Section

| Option | Description | Default |
|--------|-------------|---------|
| **Enable Zone Counters** | Activate zone-based occupancy counting | Off |
| **Save footage recordings** | Record processed MP4 footage with overlays | Off |
| **Enable Heatmap Overlay** | Generate occupancy heatmaps | Off |
| **Confidence** | Detection confidence threshold (0.1-0.9) | 0.45 |
| **Device** | Processing device: auto/cpu/cuda/mps | auto |

### Speed Estimation Section (Optional)

| Setting | Description | Default |
|---------|-------------|---------|
| **Enable speed estimation** | Calculate object speeds | Off |
| **Draw speed labels** | Show speed on bounding boxes | On |
| **Units** | pxps (pixels/sec), mps, kmh, mph | pxps |
| **Meters per pixel** | Real-world scale calibration | 0.0 |
| **Smooth window** | Frames to average (3, 5, 7, 9) | 5 |

### Performance Optimization Section

| Setting | Description | Default |
|---------|-------------|---------|
| **Process every N frame(s)** | Frame skip for speed (1-5) | 1 |
| **Interpolate tracks** | Smooth tracking between skips | On |
| **Parallel videos** | Simultaneous video processing (1-4) | 1 |
| **Playback speed multiplier** | Saved-video export FPS multiplier (config-only field `playback_speed_multiplier`). `1.0` = real-time; `>1.0` produces sped-up review video and desyncs the timestamp overlay | 1.0 |

### Training Mode Section (Optional)

| Setting | Description | Default |
|---------|-------------|---------|
| **Enable training capture** | Capture annotated frames | Off |
| **Capture interval** | Seconds between captures | 5.0 |
| **Auto-stop after (hours)** | Automatic shutdown | 2.0 |
| **Min confidence** | Minimum confidence for captures | 0.5 |
| **Include empty frames** | Save frames with no detections | Off |
| **Apply augmentation** | Flip/brightness variations | Off |

### Cloud API Upload Section (Optional)

| Setting | Description | Default |
|---------|-------------|---------|
| **Enable API Upload** | Toggle cloud upload via HTTPS POST to the configured FastAPI bridge (`enable_api_upload`) | Off |

The API URL and key are **not** entered in the GUI — they're read from a
`.env` file in the working directory at upload time:

```env
API_URL=https://your-bridge.example.com/events
API_KEY=your-api-key
```

> **Note:** Leave **Enable API Upload** unchecked to run local-only.

### Load Config Button

Click **"Load Config..."** to load a previously saved configuration file. This populates all fields and optionally lets you skip to processing if lines/zones are already configured.

---

## 🎨 Interactive Line/Zone Setup

After the configuration dialog, the interactive setup window appears:

### Drawing Modes

| Mode | How to Draw | Purpose |
|------|-------------|---------|
| **Line** | Click start point, click end point | Counting line with direction |
| **Zone** | Click vertices, right-click to finish | Occupancy/dwell zone |
| **Exclusion** | Click vertices, right-click to finish | Area to ignore detections |

### Keyboard Shortcuts (Setup Mode)

| Key | Action |
|-----|--------|
| `L` | Enter Line drawing mode |
| `Z` | Enter Zone drawing mode |
| `X` | Enter Exclusion zone mode |
| `ESC` | Cancel current drawing |
| `D` / `Delete` | Delete item under cursor |

Completed runtime edits are saved automatically. Moving an endpoint/vertex, adding
a line or zone, or deleting one atomically updates the exact loaded config file
(normally `<output_folder>/config.json`). A later restart therefore uses the most
recent geometry. The `S` key continues to save a statistics snapshot; no separate
config-save key is required in runtime edit mode.
| `Enter` | Finish setup and start processing |
| `S` | Save current configuration |
| `Q` | Quit setup |

### Mouse Controls

- **Left-click**: Place point or select item
- **Right-click**: Finish polygon (zone/exclusion)
- **Drag**: Move line endpoints or zone vertices
- **Double-click**: Edit properties of line/zone

### Line Properties Dialog

When creating or editing a line:
- **Name**: Display name for the line
- **Direction**: up, down, left, right (objects crossing in this direction are counted)
- **Classes**: Which object classes to count (checkboxes)
- **Point of Interest**: center or bottom (which part of bounding box triggers crossing)

### Zone Properties Dialog

When creating or editing a zone:
- **Name**: Display name for the zone
- **Classes**: Which object classes to count
- **Track max concurrent**: Record peak occupancy
- **Show peak overlay**: Display peak count on video
- **Point of Interest**: center or bottom

---

## 🖥️ Runtime Controls

During video processing, these controls are available:

### General Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `SPACE` | Pause/Resume processing |
| `R` | Reset all counts to zero |
| `S` | Save current statistics snapshot |
| `M` | Toggle stats panel visibility |
| `C` | Toggle controls panel visibility |
| `H` | Toggle UI minimize mode |

### Frame Skip Controls

| Key | Action |
|-----|--------|
| `1` | Process every frame (no skip) |
| `2` | Process every 2nd frame |
| `3` | Process every 3rd frame |
| `4` | Process every 4th frame |
| `5` | Process every 5th frame |
| `I` | Toggle track interpolation |

### Edit Mode (Camera/Live Only)

| Key | Action |
|-----|--------|
| `E` | Toggle edit mode |
| `N` | Create new line (in edit mode) |
| `Z` | Create new zone (in edit mode) |
| `F` | Finish zone drawing |
| `D` / `Delete` | Delete item under cursor |

### Training Mode

| Key | Action |
|-----|--------|
| `T` | Toggle training capture on/off |
| `+` / `=` | Increase capture interval |
| `-` | Decrease capture interval |

---

## 📁 Output Structure

```
output_folder/
├── config.json                    # Saved configuration (reusable)
├── master_event_log.xlsx          # Cumulative event log (all events)
│
├── segments/                      # Hourly segment exports
│   ├── events_0800-0900_20240119.json
│   ├── events_0800-0900_20240119.csv
│   ├── events_0800-0900_20240119.xlsx
│   ├── events_0900-1000_20240119.json
│   └── ...
│
├── heatmaps/                      # Heatmap snapshots (if enabled)
│   ├── heatmap_20240119_0800_to_0900.png
│   └── ...
│
├── summaries/                     # Per-video summaries
│   └── video_name_summary_20240119_120000.json
│
├── training_data/                 # Training captures (if enabled)
│   └── 20240119_120000/
│       ├── images/
│       │   ├── frame_001.jpg
│       │   └── ...
│       └── labels/
│           ├── frame_001.txt
│           └── ...
│
└── *.mp4                          # Recorded video segments (if enabled)
    ├── live_20240119_0800.mp4
    └── ...
```

---

## 📊 Event Log Format

### Line Crossing Event

```json
{
  "actual_datetime": "2024-01-19T12:34:56.789",
  "event_type": "line_crossing",
  "track_id": 42,
  "class_id": 2,
  "class_name": "car",
  "line_name": "Entry Gate",
  "zone_name": "",
  "direction": "up",
  "confidence": 0.87,
  "speed": 15.7,
  "speed_units": "kmh",
  "dwell_seconds": null,
  "video_source": "traffic_video.mp4",
  "segment_id": "0800-0900_20240119"
}
```

### Zone Entry Event

```json
{
  "actual_datetime": "2024-01-19T12:35:10.123",
  "event_type": "zone_entry",
  "track_id": 43,
  "class_id": 2,
  "class_name": "car",
  "line_name": "",
  "zone_name": "Parking Area",
  "direction": "",
  "confidence": 0.91,
  "speed": 5.2,
  "speed_units": "kmh",
  "dwell_seconds": 125.3,
  "video_source": "traffic_video.mp4",
  "segment_id": "0800-0900_20240119"
}
```

---

## ☁️ Cloud API Upload (Optional)

When **Enable API Upload** is checked, each hourly segment is also POSTed
to a remote FastAPI bridge over HTTPS as soon as the local segment
export finishes. The upload runs on the exporter's background thread
pool, so the camera/processing loop is never blocked waiting for the
network.

### Setup

1. **Dependencies** — `requests` and `python-dotenv` are listed in
   `requirements.txt` and installed automatically.

2. **Create a `.env` file** in the working directory:
   ```env
   API_URL=https://your-bridge.example.com/events
   API_KEY=your-api-key
   ```

3. **Enable in the GUI** — check the **Enable API Upload** option (or
   set `"enable_api_upload": true` in `config.json`).

### Cloud Upload Behavior

- Events are written to local files first, then uploaded — local export
  is preserved even if the upload fails.
- The POST sends JSON with an `X-API-Key` header and a 15 s timeout.
- If `API_URL` or `API_KEY` is missing from `.env`, the upload is
  skipped and the failure is logged.
- The upload is fire-and-forget: failures are logged but do not stop
  processing.

### Cloud Data Transformation

Before upload, events are normalized to the bridge's schema:
- Columns dropped: `event_id`, `class_id`, `speed_units`, `segment_id`,
  `confidence`, `timestamp`, `position`.
- Columns renamed: `actual_datetime` → `time`, `speed` → `speed_mph`.
- Values rounded: `speed_mph` and `track_id` (nearest), `dwell_seconds`
  (ceiling).
- `NaN` / `±inf` values are converted to `None`.

---

## 🔧 Advanced Configuration

### Configuration File Format (config.json)

The configuration file is automatically saved and contains all settings:

```json
{
  "model_path": "path/to/model.pt",
  "input_source": "path/to/videos",
  "output_folder": "path/to/output",
  "source_name": "north_gate",
  "confidence_threshold": 0.45,
  "device": "cuda",
  "input_type": "folder",
  "is_camera": false,
  "enable_zones": true,
  "save_video": false,
  "footage_retention_days": 0,
  "playback_speed_multiplier": 1.0,
  "frame_skip": 2,
  "interpolate_tracks": true,
  "max_parallel_videos": 2,

  "enable_speed": true,
  "speed_units": "mph",
  "meters_per_pixel": 0.064,

  "enable_heatmap": true,
  "heatmap_interval_sec": 600,

  "lines_config": [
    {
      "name": "Entry Line",
      "start_norm": [0.1, 0.5],
      "end_norm": [0.9, 0.5],
      "direction": "up",
      "classes": [2, 3, 5, 7],
      "enabled": true,
      "poi_mode": "bottom"
    }
  ],

  "zones_config": [...],
  "exclusion_zones": [...],

  "enable_api_upload": false
}
```

> Cloud upload credentials (`API_URL`, `API_KEY`) live in a `.env` file
> alongside `config.json`, not in `config.json` itself.

### Calibrating Real-World Speed

For accurate speed in real-world units (m/s, km/h, mph):

1. Measure a known distance in your camera view (e.g., lane width = 3.5 meters)
2. Count the pixels spanning that distance in the video
3. Calculate: `meters_per_pixel = real_distance / pixel_distance`
4. Enter this value in the "Meters per pixel" field

**Example:** If 3.5 meters spans 55 pixels: `3.5 / 55 = 0.0636 meters per pixel`

### Growing File Support (Live Recording)

When processing folders with files being actively recorded:

| Setting | Description | Default |
|---------|-------------|---------|
| `wait_for_growing_files` | Wait for files still being written | true |
| `growing_file_check_interval` | Seconds between size checks | 2.0 |
| `growing_file_timeout` | Seconds to wait before considering complete | 30.0 |
| `pre_process_stability_seconds` | File must be stable this long before processing | 10.0 |
| `folder_idle_timeout` | Exit after N seconds with no new files (0=forever) | 0.0 |

### YOLO Model Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PyTorch | `.pt` | Default, works everywhere |
| ONNX | `.onnx` | Cross-platform, good for CPU |
| TensorRT | `.engine` | Fastest on NVIDIA GPUs, GPU-specific |

---

## 🐛 Troubleshooting

### No Detections

- Lower confidence threshold (try 0.25-0.35)
- Check model compatibility with your objects
- Verify GPU/CUDA setup with `--debug` flag
- Ensure objects are within the non-excluded areas

### Poor Tracking / ID Switches

- Reduce frame skip value (use 1 or 2)
- Enable track interpolation
- Increase `max_track_age` in config
- Use a larger/more accurate model

### Video Not Saving

- Check available disk space
- Verify write permissions on output folder
- Try different output resolution setting
- Check logs for codec errors

### Memory Issues

- Increase frame skip value
- Reduce `max_parallel_videos` to 1
- Reduce output resolution
- Close other applications
- Process videos sequentially instead of in parallel

### Cloud Upload Failing

- Verify `.env` exists in the working directory and defines both
  `API_URL` and `API_KEY` (missing values are logged but do not raise).
- Confirm the bridge URL is reachable from this machine (the POST has a
  15 s timeout).
- Check the API key — a non-200 response is logged with the bridge's
  body for diagnosis.
- Verify the local segment files exist in `output_folder/segments/`;
  the local export always runs first, so its presence rules out a
  config problem upstream of the upload step.

### Headless Mode Not Working

- Ensure config file has saved `lines_config` (not empty)
- Use both `--config` and `--headless` flags together
- Check that input/output paths are still valid

---

## 🎯 Use Cases

- **Traffic Monitoring**: Count vehicles at intersections, measure queue lengths
- **Border/Checkpoint Analysis**: Track crossing times and throughput
- **Retail Analytics**: Customer flow, zone occupancy, dwell time analysis
- **Security**: Monitor restricted area access, count entries/exits
- **Wildlife Studies**: Count animal movements, track migration patterns
- **Industrial**: Production line counting, inventory monitoring
- **Sports Analytics**: Player movement analysis, zone coverage
- **Smart Cities**: Pedestrian counting, bicycle lane usage

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Include log files (`logs/app.log`) and configuration when reporting bugs

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for video processing
- [BoTSORT](https://github.com/NirAharon/BoT-SORT) for object tracking
- [pandas](https://pandas.pydata.org/) and [openpyxl](https://openpyxl.readthedocs.io/) for data export

---

**Version:** 1.0.1  
**Author:** TH  
**Last Updated:** May 2026

