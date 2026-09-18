---
name: rtsp-multi-tracker
description: |-
  Multi-camera person tracking across RTSP feeds.
  Built on Ultralytics track() + Torchreid OSNet + topology constraints.
  Tracks the same person across multiple cameras with persistent global IDs,
  handles simultaneous appearances and brief disappearances.
  Uses VLM for key-node confirmation when needed.
---

# Multi-Camera Person Tracking Skill

This skill provides **cross-camera person tracking**: given multiple RTSP camera
feeds and a topology configuration describing camera adjacency and transit times,
it tracks the same person across cameras with a persistent global ID.

Key capabilities:
- **Single-camera tracking**: Ultralytics track() with BoT-SORT for persistent per-camera IDs
- **Appearance features**: Torchreid OSNet extracts 512-dim ReID embeddings per person crop
- **Cross-camera association**: matches appearance + topology/time constraints to link identities
- **Overlap handling**: supports cameras with overlapping fields of view (same person in multiple views)
- **Brief disappearance recovery**: gallery buffer retains recent embeddings for re-identification

## Architecture

```
Camera 1 RTSP ──→ YOLO track() ──→ OSNet features ──┐
Camera 2 RTSP ──→ YOLO track() ──→ OSNet features ──┤──→ Cross-camera ──→ Trajectory
Camera 3 RTSP ──→ YOLO track() ──→ OSNet features ──┘    Association      Output
                                                          (topology +       (global IDs,
                                                           time window)      timeline)
```

## Role

```
User intent ("Watch the dog — don't let it steal food from the table")
       ↓  Agent maps to
  Spatial relationship (dog + dining table + on_or_over)
       ↓
  This Skill: YOLO coarse detection → suspicious match → screenshot saved
       ↓
  Agent reads screenshot → fine-grained judgment (stealing? sleeping? passing by?)
       ↓
  Confirmed violation → notify user; false positive → no disturbance
```

**This skill = coarse detection + screenshot provider.** It does not perform
fine-grained review and does not notify the user.

## Workflow

### 1. Obtain RTSP Stream URL

Use `@xpai-camera-control` to connect to the target camera and retrieve the live
stream URL (`rtsp://...`). This skill does not manage cameras — it only consumes
RTSP streams.

### 2. Configure Detection Scenarios

Edit `scripts/settings.json` and declare monitoring scenarios in the `scenarios`
array. The Agent must map user intent to COCO 80-class combinations (see "Semantic
Mapping" below).

Write the RTSP URL into the `rtsp_url` field, or pass it via `--rtsp` at startup.

### 3. Start Detection

```bash
python scripts/main.py --rtsp "rtsp://..."
```

The skill runs as a **persistent background process**: continuously reading the
stream → YOLO inference → spatial relationship check → trigger screenshots. After
startup, the Agent can handle other tasks and periodically check the screenshot
directory.

Common CLI arguments:

| Argument | Description |
|---|---|
| `--rtsp "rtsp://..."` | Override RTSP stream URL |
| `--source 0` | Use local webcam for debugging |
| `--source demo.mp4` | Use a video file for debugging |
| `--frame-skip N` | Process 1 out of every N frames for better real-time performance |
| `--conf 0.35` | Override global confidence threshold |
| `--cooldown 10` | Minimum interval between screenshots (seconds) |
| `--show` | Open OpenCV preview window (with HUD overlay) |
| `--mjpeg-port 8090` | MJPEG live stream port for browser viewing |
| `--list-classes` | Print model capability table (COCO 80 classes) |

### 4. Read Coarse Detection Screenshots

Triggered screenshots are stored in `scripts/captures/pending/`:

- `<id>.jpg` — raw trigger frame
- `<id>_annotated.jpg` — annotated frame (red box = subject / blue box = surface)
- `<id>.json` — event metadata:

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

### 5. Agent Secondary Review

The Agent acts as a secondary reviewer:

1. Read `captures/pending/<id>.jpg` (screenshot) and `<id>.json` (event description)
2. Combined with the user's monitoring intent, determine whether the coarse
   detection trigger **truly constitutes** an alertable behavior
3. Confirmed violation → notify user; false positive → no disturbance

The Agent can write the verdict back to `captures/reviewed/<id>.json`, and the HUD
will automatically display the review status.

## Semantic Mapping Guide

The model is YOLOv8n, recognizing **COCO 80 classes** (run
`python scripts/main.py --list-classes` to see the full list). Common classes:

| Class Name | Description | Class Name | Description |
|---|---|---|---|
| `person` | Person | `cat` | Cat |
| `dog` | Dog | `bird` | Bird |
| `dining table` | Dining table | `couch` | Sofa/Couch |
| `chair` | Chair | `bed` | Bed |
| `potted plant` | Potted plant | `tv` | TV |
| `laptop` | Laptop | `keyboard` | Keyboard |
| `cell phone` | Cell phone | `refrigerator` | Refrigerator |
| `backpack` | Backpack | `suitcase` | Suitcase |

### Mapping Principles

YOLO can only identify "what's in the frame and where" — it **cannot understand
behavioral semantics**. The Agent needs to decompose user intent into COCO class +
spatial relationship combinations:

- "Dog stealing food from table" → `subject: dog`, `surface: dining table`, `relationship: on_or_over`
- "Cat entered the bedroom" → `subject: cat`, `surface: bed`, `relationship: on_top` (bed as a zone proxy)
- "Pet chewing cables" → `subject: [cat, dog]`, requires custom surface class or ROI zone
- "Child approaching pool" → `subject: person`, `surface: swimming pool`, `relationship: within`
- "Package taken away" → beyond spatial relationship capability; suggest alternative solution

**This skill's coarse detection has blind spots**: a pet lying on the table
sleeping vs. standing on the table eating — the spatial relationship result is
identical. This is precisely why the Agent's secondary review is needed.

### Scenario Configuration Fields

Each entry in the `scenarios` array:

- `name` — scenario identifier
- `subject` — subject class (name, id, or list), i.e., "who"
- `surface` — surface/zone class, i.e., "where"
- `relationship` — spatial relationship:
  - `on_top`: contact on top surface (foot point falls within surface zone)
  - `within`: entered interior (center point falls within bounding box)
  - `body_over`: body intruding (head/upper body crosses over surface)
  - `on_or_over`: on_top or body_over (recommended for pet scenarios)
- `surface_ratio`: surface zone height ratio relative to full bounding box (0–1, default 0.6)
- `min_conf`: minimum subject confidence
- `cooldown`: per-scenario screenshot cooldown in seconds (overrides global `save_cooldown`)
- `enabled`: whether the scenario is active

## Configuration Reference

`scripts/settings.json` core fields:

- `rtsp_url` — RTSP stream URL (written by Agent at startup or passed via `--rtsp`)
- `model_path` — model path (`yolov8n.pt` for speed / `yolov8m.pt` for accuracy)
- `conf_threshold` — global confidence threshold (recommended 0.20–0.30; surfaces partially occluded yield lower confidence)
- `iou_threshold` — NMS deduplication threshold
- `frame_skip` — process 1 out of every N frames (0 = no skipping)
- `save_cooldown` — global minimum interval between screenshots in seconds (default 10)
- `scenarios` — scenario rules array (see field descriptions above)

## Extensibility

- **New models**: set `model.class_names` in settings (a list ordered by class id)
- **New scenarios**: add entries to the `scenarios` array — zero code changes
- **New spatial relationships**: `geometry.register_relationship("above", fn)` to register a custom evaluator

## Collaboration with @xpai-camera-control

This skill is used in conjunction with `@xpai-camera-control`:

1. Agent calls `@xpai-camera-control` to connect to a camera and obtain the RTSP URL
2. Pass the URL to this skill (write to `settings.json` `rtsp_url` or via `--rtsp`)
3. Start this skill for continuous monitoring
4. When screenshots are triggered, the Agent reads them for fine-grained review and decides whether to notify the user

## File Index

| File | Responsibility |
|---|---|
| `scripts/main.py` | Main pipeline: stream input → inference → spatial check → screenshot |
| `scripts/spatial_detector.py` | YOLO inference wrapper |
| `scripts/scenario.py` | Scenario rule parsing and evaluation |
| `scripts/geometry.py` | Spatial relationship geometry checks (pure math, unit-testable) |
| `scripts/model_catalog.py` | Model capability catalog (COCO 80 classes) |
| `scripts/downstream.py` | Event Sink abstraction (File / HTTP / Callable) |
| `scripts/vision_review.py` | Screenshot queue management (pending / reviewed directories) |
| `scripts/hud.py` | HUD status panel overlay |
| `scripts/preview.py` | MJPEG live streaming server |
| `scripts/settings.json` | All configuration |
| `references/tuning.md` | Threshold tuning & RTSP hardening guide |

Test files are located in the `tests/` directory (not part of skill runtime):

| File | Responsibility |
|---|---|
| `tests/test_geometry.py` | Spatial geometry unit tests |
| `tests/test_scenarios.py` | Capability mapping + scenario rule unit tests |
