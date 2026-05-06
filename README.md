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
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
psutil>=5.9.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.0.0
```

**Optional dependencies:**
```
psycopg2-binary>=2.9.0  # For PostgreSQL cloud upload
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

# Optional: For cloud database support
pip install psycopg2-binary
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
1. **Configuration Dialog** - Set model, input source, output folder, and options
2. **Interactive Setup** - Draw counting lines, zones, and exclusion areas
3. **Processing** - Real-time display with overlays and controls

### Headless Mode (Automated/Scheduled Processing)

After your first run, a `config.json` file is saved in your output folder. Reuse it:

```bash
# Skip all GUI - use saved lines/zones directly
python main.py --config path/to/config.json --no-gui

# Load config but allow GUI modifications
python main.py --config path/to/config.json
```

### Command Line Options

```bash
python main.py [options]

Options:
  -c, --config FILE     Load configuration from JSON/YAML file
  --no-gui              Skip interactive setup (requires --config with saved lines)
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

# Fully headless - no GUI at all (for automation)
python main.py --config outputs/config.json --no-gui

# Headless with debug logging
python main.py --config outputs/config.json --no-gui --debug

# Custom log file location
python main.py --config outputs/config.json --no-gui --log-file /var/log/counter.log
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
| **Save Video Output** | Record processed video with overlays | On |
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

### Training Mode Section (Optional)

| Setting | Description | Default |
|---------|-------------|---------|
| **Enable training capture** | Capture annotated frames | Off |
| **Capture interval** | Seconds between captures | 5.0 |
| **Auto-stop after (hours)** | Automatic shutdown | 2.0 |
| **Min confidence** | Minimum confidence for captures | 0.5 |
| **Include empty frames** | Save frames with no detections | Off |
| **Apply augmentation** | Flip/brightness variations | Off |

### Cloud Database Upload Section (Optional)

| Setting | Description | Default |
|---------|-------------|---------|
| **DB Config File** | Path to database config file (.conf) | (blank) |
| **DB Section Name** | Section name in config file | (blank) |
| **Table Name** | PostgreSQL table for uploads | (blank) |

> **Note:** Leave all cloud fields blank to skip cloud upload (local-only mode).

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

## ☁️ Cloud Database Upload (Optional)

The application can optionally upload events to a PostgreSQL database after each segment export.

### Setup

1. **Install psycopg2:**
   ```bash
   pip install psycopg2-binary
   ```

2. **Create a database config file** (e.g., `config/dbconfig.conf`):
   ```ini
   [cv-database]
   host=your-database-host.com
   port=5432
   database=your_database_name
   user=your_username
   password=your_password
   ```

3. **Configure in the GUI:**
   - **DB Config File**: Browse to your `dbconfig.conf`
   - **DB Section Name**: `cv-database` (or your section name)
   - **Table Name**: Your target table name

### Cloud Upload Behavior

- Events are exported locally first, then uploaded to the cloud
- If upload fails, local export is preserved (no data loss)
- If `psycopg2` is not installed, cloud upload is silently skipped
- Leave all cloud fields blank to use local-only mode

### Cloud Data Transformation

When uploading to the cloud, the data is transformed:
- Columns dropped: `class_id`, `speed_units`, `segment_id`, `confidence`
- Columns renamed: `actual_datetime` → `time`, `speed` → `speed_mph`
- Values rounded: `speed_mph`, `track_id`, `dwell_seconds` (ceiling)

---

## 🔧 Advanced Configuration

### Configuration File Format (config.json)

The configuration file is automatically saved and contains all settings:

```json
{
  "model_path": "path/to/model.pt",
  "input_source": "path/to/videos",
  "output_folder": "path/to/output",
  "confidence_threshold": 0.45,
  "device": "cuda",
  "input_type": "folder",
  "is_camera": false,
  "enable_zones": true,
  "save_video": true,
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
  
  "cloud_db_name": "cv-database",
  "cloud_table_name": "events",
  "cloud_db_config_path": "config/dbconfig.conf"
}
```

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

- Verify `psycopg2` is installed: `pip install psycopg2-binary`
- Check database config file path and format
- Verify database credentials and connectivity
- Check logs for specific error messages

### Headless Mode Not Working

- Ensure config file has saved `lines_config` (not empty)
- Use both `--config` and `--no-gui` flags together
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
**Last Updated:** February 2026

