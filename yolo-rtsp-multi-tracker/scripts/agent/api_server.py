"""Agent 工具 API 服务器：HTTP 接口暴露追踪系统的查询和分析能力。

设计原则
--------
* 追踪引擎是被动的后台服务，Agent 通过 HTTP 主动调用工具。
* 6 个工具覆盖：状态查询 / 人员列表 / 详情 / 轨迹分析 / 事件日志 / 当前快照。
* 使用 Python 标准库 http.server，不引入额外依赖。
* API 线程 + 追踪线程通过 EventStore 解耦。

工具列表
--------
GET /api/status                  → 系统状态（摄像头/人数/FPS）
GET /api/persons                 → 所有已跟踪人员列表
GET /api/person/<id>             → 某人详情（历史 + 当前位置）
POST /api/analyze                → 触发轨迹分析（回溯截断）
GET /api/events                  → 事件日志查询
GET /api/snapshot                → 当前状态快照渲染

所有响应均为 JSON 格式。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("api_server")


# ---------------------------------------------------------------------------
# 共享上下文（追踪主循环注入）
# ---------------------------------------------------------------------------
class TrackerContext:
    """追踪系统共享上下文，API 服务器通过它访问追踪数据。

    由 main.py 在启动时创建并注入给 API 服务器。
    """

    def __init__(self):
        self.event_store = None          # EventStore 实例
        self.annotator = None            # TrajectoryAnnotator 实例
        self.renderer = None             # TrajectoryRenderer 实例
        self.vlm_annotator = None        # VLMAnnotator 实例
        self.topology = None             # Topology 实例
        self.camera_manager = None       # MultiCameraManager 实例
        self.output_dir = "./trajectories"
        self.fps = 0.0
        self.start_time = 0.0


# 全局上下文引用（线程安全由 EventStore 的锁保证）
_ctx: Optional[TrackerContext] = None


# ---------------------------------------------------------------------------
# 请求处理器
# ---------------------------------------------------------------------------
class ToolHandler(BaseHTTPRequestHandler):
    """处理 Agent 工具调用。"""

    def log_message(self, format, *args):
        """覆盖默认日志，使用 logger。"""
        logger.debug(format, *args)

    def _json_response(self, data: dict, status: int = 200) -> None:
        """返回 JSON 响应。"""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _image_response(self, image_bytes: bytes, content_type: str = "image/png") -> None:
        """返回图片响应。"""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(image_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(image_bytes)

    def _error(self, message: str, status: int = 400) -> None:
        self._json_response({"error": message}, status)

    # -------------------------------------------------------------------
    # GET 路由
    # -------------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/status":
            self._handle_status()
        elif path == "/api/persons":
            self._handle_persons()
        elif path.startswith("/api/person/"):
            gid_str = path.split("/")[-1]
            try:
                gid = int(gid_str)
            except ValueError:
                self._error(f"无效的 global_id: {gid_str}")
                return
            self._handle_person_detail(gid)
        elif path == "/api/events":
            self._handle_events(params)
        elif path == "/api/snapshot":
            self._handle_snapshot(params)
        else:
            self._error(f"未知路径: {path}", 404)

    # -------------------------------------------------------------------
    # POST 路由
    # -------------------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._error("无效的 JSON body")
                return
            self._handle_analyze(data)
        else:
            self._error(f"未知 POST 路径: {path}", 404)

    # -------------------------------------------------------------------
    # 工具实现
    # -------------------------------------------------------------------
    def _handle_status(self):
        """工具 1: tracker_status — 系统运行状态。"""
        if not _ctx:
            self._error("追踪系统未初始化", 503)
            return

        uptime = time.time() - _ctx.start_time if _ctx.start_time else 0
        cameras_status = {}
        if _ctx.camera_manager:
            cameras_status = _ctx.camera_manager.status()

        self._json_response({
            "tool": "tracker_status",
            "status": "running",
            "uptime_sec": round(uptime, 1),
            "uptime_human": _format_duration(uptime),
            "fps": round(_ctx.fps, 1),
            "total_persons": _ctx.event_store.total_persons if _ctx.event_store else 0,
            "total_events": _ctx.event_store.total_events if _ctx.event_store else 0,
            "cameras": cameras_status,
            "topology": {
                "cameras": list(_ctx.topology.cameras.keys()) if _ctx.topology else [],
                "links": len(_ctx.topology.links) if _ctx.topology else 0,
            },
        })

    def _handle_persons(self):
        """工具 2: list_persons — 所有已跟踪人员列表。"""
        if not _ctx or not _ctx.event_store:
            self._error("追踪系统未初始化", 503)
            return

        gids = _ctx.event_store.get_person_ids()
        persons = []
        for gid in gids:
            summary = _ctx.event_store.get_person_summary(gid)
            persons.append(summary)

        self._json_response({
            "tool": "list_persons",
            "total": len(persons),
            "persons": persons,
        })

    def _handle_person_detail(self, global_id: int):
        """工具 3: get_person_detail — 某人详情。"""
        if not _ctx or not _ctx.event_store:
            self._error("追踪系统未初始化", 503)
            return

        summary = _ctx.event_store.get_person_summary(global_id)
        if "error" in summary:
            self._error(f"未找到 global_id={global_id}", 404)
            return

        # 关键事件时间线（只返回有意义的节点）
        key_events = _ctx.event_store.get_person_events(
            global_id,
            event_types=["appeared", "lingered", "vanished", "reappeared", "cross_camera", "new_person"],
        )

        # 分析历史
        analyses = _ctx.event_store.get_person_events(
            global_id, event_types=["analysis"],
        )

        self._json_response({
            "tool": "get_person_detail",
            "summary": summary,
            "timeline": key_events,
            "analyses": [a.get("analysis_result") for a in analyses if a.get("analysis_result")],
        })

    def _handle_analyze(self, data: dict):
        """工具 4: analyze_trajectory — 触发轨迹分析（回溯截断）。

        POST body:
        {
            "global_id": 1,           // 必填：要分析的人员 ID
            "since": "2026-09-20T08:00:00",  // 可选：起始时间
            "until": "2026-09-20T09:30:00",  // 可选：截止时间（默认=现在）
            "include_vlm": true        // 可选：是否调用 VLM 语义标注
        }
        """
        if not _ctx or not _ctx.event_store:
            self._error("追踪系统未初始化", 503)
            return

        global_id = data.get("global_id")
        if global_id is None:
            self._error("缺少 global_id 参数")
            return

        # 解析时间范围
        since = _parse_time(data.get("since")) or 0.0
        until = _parse_time(data.get("until")) or time.time()
        include_vlm = data.get("include_vlm", True)

        # 获取该时间范围内的事件
        events = _ctx.event_store.get_person_events(global_id, since=since, until=until)
        if not events:
            self._error(f"global_id={global_id} 在该时间范围内没有事件")
            return

        # 构建 annotation（利用 annotator 的已有数据）
        if _ctx.annotator:
            annotation = _ctx.annotator.build_annotation(
                floor_plan_image=os.path.join(_ctx.output_dir, "floor_plan.jpg"),
            )
        else:
            annotation = {"persons": [], "camera_positions": {}}

        # 渲染轨迹图
        image_path = ""
        if _ctx.renderer:
            os.makedirs(_ctx.output_dir, exist_ok=True)
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(
                _ctx.output_dir, f"analysis_{global_id}_{ts_str}.png",
            )
            try:
                _ctx.renderer.render(annotation, image_path)
            except Exception as exc:
                logger.error("轨迹渲染失败: %s", exc)
                image_path = ""

        # VLM 语义标注（可选）
        vlm_result = None
        if include_vlm and _ctx.vlm_annotator and _ctx.vlm_annotator.call_fn and _ctx.renderer:
            try:
                image_bytes = _ctx.renderer.render_to_bytes(annotation)
                annotation = _ctx.vlm_annotator.annotate(annotation, image_bytes)
                vlm_result = {
                    "labels": {
                        p["global_id"]: {
                            "labels": {n["id"]: n.get("vlm_label", "") for n in p["nodes"] if n.get("vlm_label")},
                            "summary": p.get("vlm_summary", ""),
                        }
                        for p in annotation.get("persons", [])
                        if p.get("vlm_summary")
                    }
                }
            except Exception as exc:
                logger.error("VLM 标注失败: %s", exc)

        # 记录分析事件
        from event_store import TrackingEvent
        analysis_event = TrackingEvent(
            event_type="analysis",
            global_id=global_id,
            camera="",
            timestamp=time.time(),
            analysis_result={
                "since": time.strftime("%H:%M:%S", time.localtime(since)) if since > 0 else "start",
                "until": time.strftime("%H:%M:%S", time.localtime(until)),
                "event_count": len(events),
                "image_path": image_path,
                "vlm_summary": vlm_result,
            },
        )
        _ctx.event_store.append(analysis_event)

        self._json_response({
            "tool": "analyze_trajectory",
            "global_id": global_id,
            "time_range": {
                "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)) if since > 0 else "start",
                "until": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)),
            },
            "event_count": len(events),
            "key_events": [
                {
                    "type": e["event_type"],
                    "time": time.strftime("%H:%M:%S", time.localtime(e["timestamp"])),
                    "camera": e.get("camera", ""),
                    "position": e.get("position"),
                    "details": e.get("details", ""),
                    "duration_sec": e.get("duration_sec"),
                }
                for e in events
                if e["event_type"] != "track_update"  # 过滤掉高频更新事件
            ],
            "trajectory_image": image_path,
            "vlm_result": vlm_result,
        })

    def _handle_events(self, params: dict):
        """工具 5: get_event_log — 事件日志查询。

        Query params:
            since: 起始时间 (ISO 或 unix timestamp)
            until: 截止时间
            global_id: 按人过滤
            event_type: 按类型过滤
            limit: 最大返回条数
        """
        if not _ctx or not _ctx.event_store:
            self._error("追踪系统未初始化", 503)
            return

        since = _parse_time(params.get("since", [None])[0]) or 0.0
        until = _parse_time(params.get("until", [None])[0]) or time.time()
        gid = int(params["global_id"][0]) if "global_id" in params else None
        event_type = params.get("event_type", [None])[0]
        limit = int(params.get("limit", ["100"])[0])

        if gid is not None:
            types = [event_type] if event_type else None
            events = _ctx.event_store.get_person_events(gid, since=since, until=until, event_types=types)
        else:
            events = _ctx.event_store.get_all_events(since=since, until=until)
            if event_type:
                events = [e for e in events if e["event_type"] == event_type]

        # 限制数量
        events = events[-limit:]

        self._json_response({
            "tool": "get_event_log",
            "count": len(events),
            "time_range": {
                "since": time.strftime("%H:%M:%S", time.localtime(since)) if since > 0 else "all",
                "until": time.strftime("%H:%M:%S", time.localtime(until)),
            },
            "events": events,
        })

    def _handle_snapshot(self, params: dict):
        """工具 6: render_snapshot — 当前状态快照渲染。

        Query params:
            format: json(默认) | png
        """
        if not _ctx or not _ctx.event_store:
            self._error("追踪系统未初始化", 503)
            return

        fmt = params.get("format", ["json"])[0]

        # 获取当前所有活跃人员
        latest = _ctx.event_store.get_latest_event_per_person()
        active_persons = {}
        for gid, event in latest.items():
            if (time.time() - event["timestamp"]) < 60:  # 60s 内有事件算活跃
                active_persons[gid] = event

        if fmt == "png" and _ctx.annotator and _ctx.renderer:
            # 渲染当前状态的 PNG
            annotation = _ctx.annotator.build_annotation("")
            image_bytes = _ctx.renderer.render_to_bytes(annotation)
            self._image_response(image_bytes)
        else:
            # 返回 JSON 快照
            self._json_response({
                "tool": "render_snapshot",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "active_persons": {
                    gid: {
                        "camera": e.get("camera", ""),
                        "position": e.get("position"),
                        "last_event": e["event_type"],
                        "last_seen": time.strftime("%H:%M:%S", time.localtime(e["timestamp"])),
                    }
                    for gid, e in active_persons.items()
                },
                "total_tracked": _ctx.event_store.total_persons,
                "currently_active": len(active_persons),
            })


# ---------------------------------------------------------------------------
# 服务器启动
# ---------------------------------------------------------------------------
class APIServer:
    """后台 API 服务器（独立线程）。

    Parameters
    ----------
    host : str
        监听地址。
    port : int
        监听端口。
    context : TrackerContext
        追踪系统上下文。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, context: Optional[TrackerContext] = None):
        self.host = host
        self.port = port
        self._context = context
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> str:
        """启动 API 服务器（后台线程）。"""
        global _ctx
        _ctx = self._context

        self._server = HTTPServer((self.host, self.port), ToolHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="api-server",
            daemon=True,
        )
        self._thread.start()
        url = f"http://{self.host}:{self.port}"
        logger.info("Agent 工具 API 已启动: %s", url)
        return url

    def stop(self) -> None:
        """停止服务器。"""
        if self._server:
            self._server.shutdown()
            logger.info("Agent 工具 API 已停止")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _parse_time(value) -> Optional[float]:
    """解析时间参数（ISO 格式或 Unix 时间戳）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 尝试 Unix 时间戳
        try:
            return float(value)
        except ValueError:
            pass
        # 尝试 ISO 格式
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(value)
            return dt.timestamp()
        except (ValueError, TypeError):
            pass
        # 尝试 HH:MM:SS 格式（今天的）
        try:
            from datetime import datetime, date
            parts = value.split(":")
            if len(parts) == 3:
                today = date.today()
                dt = datetime(today.year, today.month, today.day,
                              int(parts[0]), int(parts[1]), int(parts[2]))
                return dt.timestamp()
        except (ValueError, TypeError):
            pass
    return None


def _format_duration(seconds: float) -> str:
    """格式化时长为人类可读字符串。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m}min"
