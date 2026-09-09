# Tuning & Deployment Notes

> The skill is **config-driven**: detection targets and the spatial relationship
> are declared via the model capability map + `scenarios` in `settings.json`.
> Nothing about "animal" or "table" is hard-coded. This document covers how to
> tune thresholds, swap models, and extend relationships.

## 1. Model capability map (what the model can see)

Before writing a scenario, know the model's class vocabulary. COCO 80 classes are
built in; run `python scripts/main.py --list-classes` to print the full `id name`
list. Common entries you'll reuse:

| id | name        | id | name         | id | name        |
|----|-------------|----|--------------|----|-------------|
| 0  | person      | 56 | chair        | 60 | dining table|
| 15 | cat         | 57 | couch        | 61 | toilet      |
| 16 | dog         | 58 | potted plant | 62 | tv          |
| 59 | bed         | 57 | couch        |    |             |

A `subject`/`surface` reference accepts a **name** (`"cat"`), an **id** (`15`), a
**numeric string** (`"15"`), or a **list** (`["cat","dog"]`). Resolution is done
against the active capability map; unknown references are skipped with a warning
at startup — so a typo fails loud and early, not silently.

### Custom / non-COCO model

For a model trained on your own dataset, set `model.class_names` in `settings.json`
(a list where the **index is the class id**):

```json
"model": { "name": "my_model", "class_names": ["background", "table", "cat", "dog"] }
```

Or register a `ModelProfile` in `model_catalog.py`:

```python
from model_catalog import register_model, ModelProfile
register_model(ModelProfile("my_model", ["background", "table", "cat", "dog"]))
```

Then reference classes by name in `scenarios`. No detector code changes.

## 2. Spatial-relationship thresholds

### `on_top` (subject's feet on the surface)

Uses the subject's **bottom-center point** against the surface's *upper-surface
zone*:

```
zone.x1 = surface.x1 + margin_x_ratio * surface.width
zone.x2 = surface.x2 - margin_x_ratio * surface.width
zone.y1 = surface.y1
zone.y2 = surface.y1 + surface_ratio * surface.height
on_top = (zone.x1 <= foot_x <= zone.x2) and (zone.y1 <= foot_y <= zone.y2)
```

- **Top-down camera** (looking down): raise `surface_ratio` toward `0.85–0.95` —
  most of the box *is* the surface.
- **Eye-level / low-angle**: lower to `0.45–0.6` — legs and floor gap occupy the
  lower box.
- **Perspective skew / box over-estimation**: add `margin_x_ratio` (`0.05–0.15`)
  to shrink the horizontal zone so a subject slightly off-edge is not included.

Start at `surface_ratio=0.6`, `margin_x_ratio=0.0`; nudge using the annotated
captures in `captures/` (surface in blue, subject in red, label on top).

### `within` (subject center inside the surface box)

Use for "person sits in chair", "object placed in container". Thresholds:
`margin_x_ratio`, `margin_y_ratio` shrink the surface box; the subject's **center**
must fall inside.

### Adding a new relationship

Relationships are registrable, so the geometry is never a closed `if/else`:

```python
from geometry import register_relationship, Box

def is_above(subject: Box, surface: Box, gap: float = 10.0, **_kw) -> bool:
    # subject entirely above the surface with a small vertical gap
    return subject.bottom <= surface.y1 - gap

register_relationship("above", is_above)
```

Then `"relationship": "above"` in a scenario. Signature must be
`fn(subject, surface, **thresholds) -> bool`; extra thresholds are ignored via
`**_ignored`.

## 3. Detection thresholds

- `conf_threshold`: `0.30` default. Raise to `0.4–0.5` for spurious boxes; lower
  to `0.2` only in low-light streams. A per-scenario `min_conf` can also gate the
  subject.
- `iou_threshold`: keep `0.45` (YOLOv8 default) unless boxes overlap heavily.

## 4. Real-time performance

- **Model size:** `yolov8n.pt` (~6 MB, fastest) vs `yolov8m.pt` (~25 MB, better
  small-object accuracy). `n` for edge/CPU, `m` when the target is far/small.
- **`frame_skip`:** if inference lags the stream FPS, set `frame_skip=1` (every
  2nd frame) or `2` (every 3rd). The event is still caught within a fraction of a
  second.
- **`CAP_PROP_BUFFERSIZE = 2`** is set for RTSP to prevent stale-frame buildup.
- GPU: set `device: "0"` in `settings.json` to use CUDA; leave empty for auto/CPU.
- **Warm-up:** `main.py` runs one dummy inference before the loop.

## 5. RTSP hardening

- Auth in the URL: `rtsp://user:pass@host:port/path`.
- Auto-reconnect on read failure (re-opens after a short delay). For unstable
  networks, use TCP transport: append `?tcp` or set
  `cv2.CAP_PROP_RTSP_TRANSPORT = cv2.CAP_RTSP_TCP` before `read()`.
- Constant drops → verify the camera sub-stream URL and the FFmpeg decoder
  (`opencv-python` uses FFmpeg under the hood).

## 6. Downstream multimodal integration

`scripts/downstream.py` defines a `Sink` interface. `TriggerEvent` carries
`relation` (`SpatialRelation`: `scenario`, `relationship`, `subject`, `surface`,
`on`, `iou`) and the raw `frame` (BGR `numpy.ndarray`).

Minimal custom hook (add beside / instead of `HttpSink` in `main.py::run`):

```python
from downstream import CallableSink, TriggerEvent

def confirm_with_multimodal(event: TriggerEvent):
    r = event.relation
    ok = my_multimodal_api(event.frame, subject=r.subject.label,
                           surface=r.surface.label, scenario=r.scenario)
    if ok:
        send_alert(f"{r.subject.label} on {r.surface.label} ({r.scenario})")

composite.add(CallableSink(confirm_with_multimodal))
```

`HttpSink` POSTs the saved JPEG (multipart `image`) plus JSON metadata
(`scenario`, `relationship`, `subject`, `surface`, `matched`) to
`downstream_http_url`.

## 7. Multiple scenarios & cooldown

You can run several scenes at once (e.g. `pet_on_table` + `cat_on_sofa`). Each
scene is evaluated independently; the highest-confidence subject per scene is
emitted, debounced by that scene's `cooldown` (falls back to global
`save_cooldown` when `0`). Set per-scene `cooldown` to avoid cross-talk between
busy scenes.

## 8. Testing

- `python scripts/test_geometry.py` — pure-geometry unit tests (no torch).
- `python scripts/test_scenarios.py` — capability-map + scenario parsing + multi-
  scene evaluation (no torch).
- End-to-end dry run without a camera: `--source demo.mp4` or `--source 0`;
  confirm `captures/` fills with annotated frames on a positive match.
