# YOLO-RTSP Spatial Detector

> Coarse-grained spatial relationship detection on live RTSP camera feeds.  
> Built on **YOLO26 + OpenCV** — designed as an Agent Skill for AI assistants.

## What It Does

This project monitors RTSP camera streams and detects **spatial relationships**
between objects in real time. For example:

- **Pet on the dining table** → `dog/cat` + `dining table` + `on_or_over`
- **Cat on the sofa** → `cat` + `couch` + `on_top`
- **Person sitting in a chair** → `person` + `chair` + `within`

When a match is triggered, annotated screenshots are saved to disk for a
**downstream Agent** (e.g., a multimodal LLM) to perform fine-grained review
and decide whether to alert the user.

### Design Philosophy

```
User intent ("Watch the dog — don't let it on the table")
       ↓  Agent maps to
  Spatial relationship (dog + dining table + on_or_over)
       ↓
  This Skill: YOLO coarse detection → suspicious → screenshot saved
       ↓
  Agent reads screenshot → fine-grained judgment
       ↓
  Confirmed → notify user; False positive → no disturbance
```

**This skill = coarse detection + screenshot provider.** It does NOT perform
fine-grained behavior recognition and does NOT notify the user directly.

## Project Structure

```
yolo-rtsp-spatial-detector/
├── SKILL.md                        # Agent Skill definition (English)
├── references/
│   └── tuning.md                   # Threshold tuning & RTSP hardening guide
└── scripts/
    ├── main.py                     # Main pipeline: stream → inference → check → screenshot
    ├── spatial_detector.py         # YOLO inference wrapper
    ├── scenario.py                 # Scenario rule parsing & evaluation
    ├── geometry.py                 # Spatial geometry checks (pure math, no dependencies)
    ├── model_catalog.py            # Model capability catalog (COCO 80 classes)
    ├── downstream.py               # Event Sink abstraction (File / HTTP / Callable)
    ├── vision_review.py            # Screenshot queue (pending/ reviewed/ directories)
    ├── hud.py                      # HUD status panel overlay (Chinese text via PIL)
    ├── preview.py                  # MJPEG live streaming server
    ├── settings.json               # All configuration
    ├── requirements.txt            # Python dependencies
    └── yolov8n.pt                  # Default YOLOv8n model weights
tests/
├── test_geometry.py                # Spatial geometry unit tests
└── test_scenarios.py               # Capability mapping + scenario unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r yolo-rtsp-spatial-detector/scripts/requirements.txt
```

Requires **Python 3.10+**. Core dependencies:
- `ultralytics` (YOLO26)
- `opencv-python`
- `numpy`
- `requests` (optional, for HTTP downstream)

### 2. Configure Scenarios

Edit [`scripts/settings.json`](yolo-rtsp-spatial-detector/scripts/settings.json):

```json
{
  "scenarios": [
    {
      "name": "pet_on_table",
      "subject": ["cat", "dog"],
      "surface": "dining table",
      "relationship": "on_or_over",
      "surface_ratio": 0.6,
      "min_conf": 0.35,
      "enabled": true
    }
  ],
  "rtsp_url": "rtsp://your-camera-address:554/stream"
}
```

### 3. Run

```bash
# Using RTSP URL from settings.json
python scripts/main.py

# Override RTSP URL via CLI
python scripts/main.py --rtsp "rtsp://user:pass@192.168.1.64:554/stream"

# Use local webcam for testing
python scripts/main.py --source 0

# Use a video file for testing
python scripts/main.py --source demo.mp4

# Show preview window with HUD
python scripts/main.py --show

# Print available COCO classes
python scripts/main.py --list-classes
```

## CLI Arguments

| Argument | Description |
|---|---|
| `--rtsp "rtsp://..."` | Override RTSP stream URL |
| `--source 0` | Use local webcam (index) for debugging |
| `--source demo.mp4` | Use a video file for debugging |
| `--frame-skip N` | Process 1 of every N frames (improves real-time perf) |
| `--model path.pt` | Override model path (e.g., `yolov8m.pt` for accuracy) |
| `--conf 0.35` | Override global confidence threshold |
| `--cooldown 10` | Min interval between screenshots (seconds) |
| `--surface-ratio 0.6` | Override surface zone ratio for `on_top` scenarios |
| `--show` | Open OpenCV preview window with HUD |
| `--mjpeg-port 8090` | Enable MJPEG live stream on port |
| `--list-classes` | Print model capability table and exit |
| `--alert-hold 3.0` | HUD alert display duration (seconds) |

## Spatial Relationships

Four built-in relationship types:

| Relationship | Logic | Use Case |
|---|---|---|
| `on_top` | Subject's **foot point** falls within the surface's upper zone | Pet standing on table |
| `body_over` | Subject's **head/upper body** intrudes into the surface zone | Pet leaning over table edge |
| `on_or_over` | `on_top` OR `body_over` | **Recommended for pets** |
| `within` | Subject's **center point** falls inside the surface box | Person sitting in chair |

Custom relationships can be registered:

```python
from geometry import register_relationship, Box

def is_above(subject: Box, surface: Box, gap: float = 10.0, **_kw) -> bool:
    return subject.bottom <= surface.y1 - gap

register_relationship("above", is_above)
```

## Configuration Reference

### `settings.json`

| Field | Type | Default | Description |
|---|---|---|---|
| `rtsp_url` | string | — | RTSP stream URL |
| `model_path` | string | `"yolov8n.pt"` | Model weights path |
| `conf_threshold` | float | `0.20` | Global detection confidence |
| `iou_threshold` | float | `0.45` | NMS deduplication threshold |
| `frame_skip` | int | `0` | Skip N frames between processing |
| `save_cooldown` | float | `10.0` | Global screenshot cooldown (seconds) |
| `buffer_size` | int | `2` | RTSP frame buffer size |
| `device` | string | `""` | `"0"` for CUDA, `""` for auto/CPU |
| `scenarios` | array | `[]` | Detection scenario rules |

### Scenario Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Scenario identifier |
| `subject` | int/str/list | Yes | Subject class(es) — "who" |
| `surface` | int/str/list | Yes | Surface/zone class(es) — "where" |
| `relationship` | string | No | Spatial relationship (default: `on_top`) |
| `surface_ratio` | float | No | Surface zone height ratio (default: 0.6) |
| `margin_x_ratio` | float | No | Horizontal margin shrink (default: 0.0) |
| `margin_y_ratio` | float | No | Vertical margin shrink for `within` (default: 0.0) |
| `min_overlap_ratio` | float | No | Min body overlap for `body_over` (default: 0.25) |
| `min_conf` | float | No | Minimum subject confidence |
| `cooldown` | float | No | Per-scenario cooldown (0 = use global) |
| `enabled` | bool | No | Enable/disable (default: true) |

## Output

### Screenshots

Triggered screenshots are saved to `scripts/captures/`:

```
captures/
├── pending/
│   ├── 20260907_143022.jpg            # Annotated trigger frame
│   └── 20260907_143022.json           # Event metadata
├── reviewed/
│   └── 20260907_143022.json           # Agent verdict (written back)
├── capture_20260907_143022.jpg        # Raw frame (FileSink)
└── capture_20260907_143022_annotated.jpg  # Annotated (FileSink)
```

### Event Metadata (`pending/<id>.json`)

```json
{
  "event_id": "20260907_143022",
  "scenario": "pet_on_table",
  "relationship": "on_or_over",
  "subject": "dog",
  "subject_conf": 0.82,
  "surface": "dining table",
  "suspicious": true,
  "created_at": 1757234422.0
}
```

## Extensibility

- **New models**: Set `model.class_names` in `settings.json` or register a
  `ModelProfile` in `model_catalog.py`
- **New scenarios**: Add entries to the `scenarios` array — zero code changes
- **New relationships**: Use `geometry.register_relationship()` — pluggable evaluator system
- **Downstream hooks**: Implement a custom `Sink` or use `CallableSink` for
  integration with multimodal models, message queues, databases, etc.

## Testing

```bash
# Geometry unit tests (no torch required)
python -m pytest tests/test_geometry.py -v

# Scenario + capability mapping tests (no torch required)
python -m pytest tests/test_scenarios.py -v
```

## MJPEG Preview

Enable browser-based live preview:

```bash
python scripts/main.py --mjpeg-port 8090
```

Then open `http://127.0.0.1:8090` in any browser or VLC.

## Requirements

- **Python** 3.10+
- **ultralytics** >= 8.0.0 (YOLOv8)
- **opencv-python** >= 4.7.0
- **numpy** >= 1.23.0
- **requests** >= 2.28.0 (optional, for HTTP downstream)
- **Pillow** (optional, for Chinese HUD text rendering)

## License

This project is provided as an Agent Skill for AI assistant integration.
