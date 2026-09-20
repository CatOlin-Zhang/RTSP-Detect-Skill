"""事件存储：追踪过程的持久化日志，供 Agent 工具查询。

设计原则
--------
* 追踪引擎只管往里写事件，不做任何分析决策。
* Agent 工具从这里读数据，自行决定何时分析。
* 内存 + JSONL 双写：内存供实时查询，JSONL 供回溯。
* 事件是不可变的 append-only 日志。

事件类型
--------
appeared     : 新轨迹出现（首次在某摄像头看到某人）
track_update : 轨迹持续更新（同一人在同一摄像头移动）
lingered     : 停留超过阈值
vanished     : 轨迹从摄像头消失
reappeared   : 消失后在同一摄像头重现
cross_camera : 跨摄像头匹配成功（人从 A 摄像头转移到 B）
new_person   : 分配了新的 global_id（从未见过的人）
analysis     : Agent 触发的分析结果（回填记录）
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("event_store")


# ---------------------------------------------------------------------------
# 事件定义
# ---------------------------------------------------------------------------
@dataclass
class TrackingEvent:
    """一条追踪事件。"""

    event_type: str          # appeared / track_update / lingered / vanished / ...
    global_id: int
    camera: str
    timestamp: float
    # 可选字段（不同事件类型使用不同子集）
    bbox: Optional[Tuple[float, float, float, float]] = None
    position: Optional[Tuple[float, float]] = None    # 平面图归一化坐标
    duration_sec: Optional[float] = None              # lingered 的停留时长
    from_camera: Optional[str] = None                 # cross_camera 的来源摄像头
    transit_sec: Optional[float] = None               # cross_camera 的转移时间
    track_id: Optional[int] = None                    # 局部 track_id
    confidence: float = 0.0                           # 检测/匹配置信度
    details: str = ""                                 # 额外说明
    # 分析结果（仅 event_type=analysis 时使用）
    analysis_result: Optional[dict] = None

    def to_dict(self) -> dict:
        """序列化为 dict（去掉 None 字段，保持 JSON 精简）。"""
        d = asdict(self)
        # 移除 None 值
        return {k: v for k, v in d.items() if v is not None}

    def to_json_line(self) -> str:
        """序列化为单行 JSON（用于 JSONL 文件）。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 事件存储
# ---------------------------------------------------------------------------
class EventStore:
    """追踪事件的 append-only 存储。

    线程安全（追踪线程写，API 线程读）。

    Parameters
    ----------
    output_dir : str
        JSONL 文件存放目录。
    max_memory_events : int
        内存中最多保留多少条事件（超出后只保留文件）。
    """

    def __init__(self, output_dir: str = "./events", max_memory_events: int = 10000):
        self._lock = threading.Lock()
        self._events: List[TrackingEvent] = []
        self._max_memory = max_memory_events

        # JSONL 文件
        os.makedirs(output_dir, exist_ok=True)
        date_str = time.strftime("%Y%m%d")
        self._jsonl_path = os.path.join(output_dir, f"events_{date_str}.jsonl")
        self._file = open(self._jsonl_path, "a", encoding="utf-8")

        # 索引：global_id -> [event_index]
        self._person_index: Dict[int, List[int]] = {}

        logger.info("EventStore 初始化: %s", self._jsonl_path)

    def append(self, event: TrackingEvent) -> None:
        """写入一条事件。"""
        with self._lock:
            idx = len(self._events)
            self._events.append(event)
            # 更新索引
            if event.global_id not in self._person_index:
                self._person_index[event.global_id] = []
            self._person_index[event.global_id].append(idx)
            # 写文件
            self._file.write(event.to_json_line() + "\n")
            self._file.flush()
            # 内存淘汰
            if len(self._events) > self._max_memory:
                self._events = self._events[-self._max_memory // 2:]

    # -------------------------------------------------------------------
    # 查询接口（供 Agent 工具使用）
    # -------------------------------------------------------------------
    def get_all_events(
        self,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> List[dict]:
        """获取所有事件（可选时间范围过滤）。"""
        with self._lock:
            result = []
            for e in self._events:
                if since and e.timestamp < since:
                    continue
                if until and e.timestamp > until:
                    continue
                result.append(e.to_dict())
            return result

    def get_person_events(
        self,
        global_id: int,
        since: Optional[float] = None,
        until: Optional[float] = None,
        event_types: Optional[List[str]] = None,
    ) -> List[dict]:
        """获取某人的事件列表。"""
        with self._lock:
            indices = self._person_index.get(global_id, [])
            result = []
            for idx in indices:
                if idx >= len(self._events):
                    continue
                e = self._events[idx]
                if since and e.timestamp < since:
                    continue
                if until and e.timestamp > until:
                    continue
                if event_types and e.event_type not in event_types:
                    continue
                result.append(e.to_dict())
            return result

    def get_person_ids(self) -> List[int]:
        """获取所有已知的 global_id 列表。"""
        with self._lock:
            return sorted(self._person_index.keys())

    def get_person_summary(self, global_id: int) -> dict:
        """获取某人的摘要信息（最新位置、总出现次数、时间跨度等）。"""
        events = self.get_person_events(global_id)
        if not events:
            return {"global_id": global_id, "error": "no events"}

        first_ts = events[0]["timestamp"]
        last_ts = events[-1]["timestamp"]
        cameras_seen = set()
        appearances = 0
        last_camera = ""
        last_position = None

        for e in events:
            if e["event_type"] in ("appeared", "reappeared", "cross_camera"):
                appearances += 1
            if e.get("camera"):
                cameras_seen.add(e["camera"])
                last_camera = e["camera"]
            if e.get("position"):
                last_position = e["position"]

        return {
            "global_id": global_id,
            "first_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first_ts)),
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)),
            "duration_hours": round((last_ts - first_ts) / 3600, 2),
            "total_events": len(events),
            "appearances": appearances,
            "cameras_seen": sorted(cameras_seen),
            "last_camera": last_camera,
            "last_position": last_position,
            "is_active": (time.time() - last_ts) < 30,  # 30s 内有事件算活跃
        }

    def get_latest_event_per_person(self) -> Dict[int, dict]:
        """获取每人最新一条事件。"""
        with self._lock:
            result = {}
            for gid, indices in self._person_index.items():
                if indices:
                    last_idx = indices[-1]
                    if last_idx < len(self._events):
                        result[gid] = self._events[last_idx].to_dict()
            return result

    def get_events_in_range(
        self,
        since: float,
        until: float,
        global_id: Optional[int] = None,
    ) -> List[dict]:
        """获取时间范围内的事件（可选按人过滤）。"""
        if global_id is not None:
            return self.get_person_events(global_id, since=since, until=until)
        return self.get_all_events(since=since, until=until)

    def get_analysis_history(self, global_id: Optional[int] = None) -> List[dict]:
        """获取历史分析结果。"""
        return self.get_person_events(
            global_id, event_types=["analysis"],
        ) if global_id else self.get_all_events()

    # -------------------------------------------------------------------
    # 生命周期
    # -------------------------------------------------------------------
    def close(self) -> None:
        """关闭文件句柄。"""
        if self._file:
            self._file.close()
            logger.info("EventStore 已关闭: %s", self._jsonl_path)

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def total_persons(self) -> int:
        with self._lock:
            return len(self._person_index)
