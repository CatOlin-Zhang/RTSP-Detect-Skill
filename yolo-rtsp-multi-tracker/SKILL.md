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

This skill runs as a **persistent background service** that tracks persons across
multiple RTSP cameras. It exposes **6 tools via HTTP API** for the Agent to query
tracking data and trigger trajectory analysis on demand.

## Architecture

```
                        Agent (decision maker)
                              │
                     HTTP tool calls (6 tools)
                              │
┌─────────────────────────────▼─────────────────────────────┐
│              Tracking Service (background)                  │
│                                                            │
│  Camera 1 ──→ YOLO track() ──→ OSNet ReID ──┐             │
│  Camera 2 ──→ YOLO track() ──→ OSNet ReID ──┤─→ 跨摄关联   │
│  Camera 3 ──→ YOLO track() ──→ OSNet ReID ──┘   │          │
│                                                    ▼        │
│                                              EventStore     │
│                                              (append-only)  │
│                                                    │        │
│                                         API Server (:8765)  │
└────────────────────────────────────────────────────────────┘
```

**Key design decision**: The tracker runs passively and accumulates data.
The Agent decides WHEN and WHAT to analyze by calling tools.
No automatic VLM calls — all analysis is Agent-triggered.

## Agent Tools (HTTP API)

Base URL: `http://127.0.0.1:8765` (configurable in settings.json `api_server`)

All responses are JSON. Start the tracker with:
```bash
python scripts/main.py --mode tracking
```

---

### Tool 1: `tracker_status`

**When to use**: Check if the tracker is running, camera health, how many people tracked.

```
GET /api/status
```

**Response**:
```json
{
  "tool": "tracker_status",
  "status": "running",
  "uptime_sec": 3600.5,
  "uptime_human": "1h0min",
  "fps": 12.3,
  "total_persons": 3,
  "total_events": 847,
  "cameras": {
    "dining": {"connected": true, "frames_read": 12340},
    "hallway": {"connected": true, "frames_read": 12200},
    "living": {"connected": false, "last_error": "read failed"}
  },
  "topology": {"cameras": ["dining", "hallway", "living"], "links": 3}
}
```

---

### Tool 2: `list_persons`

**When to use**: Get an overview of all detected persons — who appeared, where they were last seen.

```
GET /api/persons
```

**Response**:
```json
{
  "tool": "list_persons",
  "total": 3,
  "persons": [
    {
      "global_id": 1,
      "first_seen": "2026-09-20 08:00:15",
      "last_seen": "2026-09-20 08:32:05",
      "duration_hours": 0.53,
      "total_events": 47,
      "appearances": 4,
      "cameras_seen": ["dining", "hallway", "living"],
      "last_camera": "living",
      "last_position": [0.52, 0.68],
      "is_active": false
    }
  ]
}
```

---

### Tool 3: `get_person_detail`

**When to use**: Get full timeline and analysis history for a specific person.

```
GET /api/person/<global_id>
```

**Response**:
```json
{
  "tool": "get_person_detail",
  "summary": {
    "global_id": 1,
    "first_seen": "2026-09-20 08:00:15",
    "last_seen": "2026-09-20 08:32:05",
    "appearances": 4,
    "cameras_seen": ["dining", "hallway", "living"],
    "is_active": false
  },
  "timeline": [
    {"event_type": "new_person", "camera": "dining", "timestamp": 1758340815.0, "track_id": 1},
    {"event_type": "lingered", "camera": "dining", "timestamp": 1758340830.0, "duration_sec": 15.2},
    {"event_type": "vanished", "camera": "dining", "timestamp": 1758340820.0, "duration_sec": 15.2},
    {"event_type": "cross_camera", "camera": "hallway", "timestamp": 1758340825.0, "transit_sec": 5.0},
    {"event_type": "vanished", "camera": "hallway", "timestamp": 1758340840.0},
    {"event_type": "appeared", "camera": "living", "timestamp": 1758340857.0}
  ],
  "analyses": [
    {"since": "08:00:15", "until": "08:32:05", "vlm_summary": {"1": {"summary": "从餐厅逗留后经走廊至客厅"}}}
  ]
}
```

---

### Tool 4: `analyze_trajectory` (核心分析工具)

**When to use**: Trigger a retrospective trajectory analysis for a person. Renders a
trajectory image on the floor plan and optionally calls VLM for semantic annotation.
This is the "回溯截断" tool — it analyzes everything up to the specified time.

```
POST /api/analyze
Content-Type: application/json

{
  "global_id": 1,
  "since": "08:00:00",
  "until": "09:30:00",
  "include_vlm": true
}
```

**Parameters**:
| Field | Type | Required | Description |
|---|---|---|---|
| `global_id` | int | Yes | Person to analyze |
| `since` | string | No | Start time (HH:MM:SS or ISO). Default = all history |
| `until` | string | No | End time. Default = now |
| `include_vlm` | bool | No | Whether to call VLM for semantic labels (default: true) |

**Response**:
```json
{
  "tool": "analyze_trajectory",
  "global_id": 1,
  "time_range": {"since": "08:00:00", "until": "09:30:00"},
  "event_count": 12,
  "key_events": [
    {"type": "new_person", "time": "08:00:15", "camera": "dining", "position": [0.24, 0.19]},
    {"type": "lingered", "time": "08:00:30", "camera": "dining", "duration_sec": 15.2},
    {"type": "vanished", "time": "08:00:20", "camera": "dining"},
    {"type": "cross_camera", "time": "08:00:25", "camera": "hallway", "details": "match_source=neighbor"},
    {"type": "appeared", "time": "08:00:57", "camera": "living", "position": [0.52, 0.68]}
  ],
  "trajectory_image": "./trajectories/analysis_1_20260920_093000.png",
  "vlm_result": {
    "labels": {"1": {"labels": {"①": "进门逗留", "②": "走廊转移"}, "summary": "从入口进餐厅逗留后经走廊到客厅"}}
  }
}
```

**Agent decision pattern**: Call this tool when:
- User asks "what has person X been doing?"
- A significant event occurs (long disappearance, unusual location, etc.)
- You need to generate a report for the user

---

### Tool 5: `get_event_log`

**When to use**: Query raw events in a time range. Useful for "what happened between 8am and 9am?"

```
GET /api/events?since=08:00:00&until=09:00:00&global_id=1&limit=50
```

**Parameters** (all optional query params):
| Param | Description |
|---|---|
| `since` | Start time (HH:MM:SS or ISO or unix timestamp) |
| `until` | End time |
| `global_id` | Filter by person |
| `event_type` | Filter by type: appeared/lingered/vanished/cross_camera/new_person/analysis |
| `limit` | Max results (default 100) |

---

### Tool 6: `render_snapshot`

**When to use**: Get the current real-time state — who is where RIGHT NOW.

```
GET /api/snapshot?format=json
GET /api/snapshot?format=png
```

**JSON Response**:
```json
{
  "tool": "render_snapshot",
  "timestamp": "2026-09-20 14:30:22",
  "active_persons": {
    "1": {
      "camera": "living",
      "position": [0.52, 0.68],
      "last_event": "track_update",
      "last_seen": "14:30:20"
    }
  },
  "total_tracked": 3,
  "currently_active": 1
}
```

**PNG Response**: Returns the current trajectory rendered on the floor plan image.

---

## Agent Usage Patterns

### Pattern 1: User asks "Where is everyone?"
```
Agent calls: GET /api/snapshot
→ Returns who is currently active and where
```

### Pattern 2: User asks "What did person 1 do today?"
```
Agent calls: GET /api/person/1
→ Gets full timeline
Agent decides: "There's enough data to analyze"
Agent calls: POST /api/analyze {"global_id": 1, "include_vlm": true}
→ Gets trajectory image + VLM summary
Agent presents: image + summary to user
```

### Pattern 3: Suspicious activity detection
```
Agent monitors: GET /api/status (periodic)
Agent notices: person appeared at unusual time (3am)
Agent calls: POST /api/analyze {"global_id": 2, "since": "03:00:00"}
→ Analyzes the suspicious segment
Agent alerts user with trajectory image
```

### Pattern 4: "What happened while I was away?"
```
Agent calls: GET /api/events?since=09:00:00&until=18:00:00&event_type=appeared
→ Lists all appearances during absence
Agent calls: POST /api/analyze for each person who appeared
→ Generates summary for each
```

## Event Types Reference

| Event | Trigger | Fields |
|---|---|---|
| `new_person` | First time seeing this person | global_id, camera, bbox, confidence |
| `appeared` | Person appears in a camera (re-enter) | global_id, camera, bbox, confidence |
| `lingered` | Person stayed > threshold (default 8s) | global_id, camera, duration_sec |
| `vanished` | Person disappeared from camera | global_id, camera, duration_sec |
| `cross_camera` | Same person matched across cameras | global_id, camera, transit_sec, confidence |
| `analysis` | Agent-triggered analysis result | global_id, analysis_result |

## Starting the Service

```bash
# Start tracking mode (background service)
python scripts/run.py --mode tracking

# With preview
python scripts/run.py --mode tracking --show

# With custom port
# Edit settings.json: api_server.port = 9000
python scripts/run.py --mode tracking
```

The service runs continuously. The Agent connects to the API to query and analyze.

## Configuration

See `scripts/settings.json` for all options. Key sections:

- `cameras[]` — camera IDs, names, RTSP sources
- `topology.links[]` — adjacency, transit times, overlap flags
- `tracking` — ReID threshold, gallery age, tracker type
- `trajectory` — floor plan image, camera positions on plan, linger threshold
- `api_server` — host/port for Agent tool API

## File Index

```
scripts/
├── run.py                          # Entry point (sys.path setup + delegate)
├── main.py                         # CLI 解析 + 两种模式主循环
├── settings.json                   # All configuration
├── requirements.txt                # Python dependencies
│
├── core/                           # 共享基础设施
│   ├── geometry.py                 # Box 定义 + 空间关系几何判定
│   ├── model_catalog.py            # 模型能力表 (COCO 80类)
│   └── event_store.py              # 事件日志 (append-only, 线程安全)
│
├── detection/                      # 检测层
│   ├── spatial_detector.py         # YOLO detect() + track() 封装
│   ├── scenario.py                 # 场景规则 (声明式配置 → 评估器)
│   ├── downstream.py               # Sink 链 (文件/HTTP/回调)
│   ├── hud.py                      # HUD 状态面板渲染
│   ├── vision_review.py            # 截图队列 (粗检 → Agent 精判)
│   └── preview.py                  # MJPEG 直播服务
│
├── tracking/                       # 跨摄追踪层
│   ├── tracking_models.py          # 数据模型 (CameraTrack, GlobalPerson, Topology)
│   ├── reid_extractor.py           # OSNet 512维特征提取
│   ├── cross_camera.py             # 跨摄关联引擎 (外观+时间+拓扑)
│   └── multi_camera.py             # 多路 RTSP 流并行管理
│
├── trajectory/                     # 轨迹标注层
│   ├── trajectory_annotator.py     # 节点/路径段构建 + 位置推算
│   ├── trajectory_renderer.py      # PIL 平面图精确渲染
│   └── vlm_annotate_prompt.py      # VLM prompt 构建 + 响应解析
│
└── agent/                          # Agent 工具层
    └── api_server.py               # HTTP API (6 tools for Agent)
```
