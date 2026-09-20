"""跨摄像头追踪的数据模型：单摄轨迹、全局人员、特征画廊。

设计原则
--------
* 纯 Python dataclass，不依赖 torch / cv2，可独立单元测试。
* 所有时间戳使用 time.time()（Unix epoch 秒），与主流程一致。
* 特征向量存为 numpy ndarray，但本文件不 import numpy——由调用方保证类型。

核心概念
--------
CameraTrack   : 单摄像头内的一条连续跟踪轨迹（来自 YOLO track 的 track_id）。
GlobalPerson  : 跨摄像头的全局人员身份，关联多条 CameraTrack。
FeatureGallery: 缓存最近出现的特征向量，用于短暂消失后的重识别。
TopologyLink  : 两个摄像头之间的邻接关系和转移时间约束。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 拓扑配置（从 settings.json 解析）
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    """单个摄像头的配置。"""

    id: str
    name: str
    source: str
    description: str = ""


@dataclass
class TopologyLink:
    """两个摄像头之间的邻接关系。

    transit_sec: (最短秒, 最长秒) — 从 from_cam 消失到在 to_cam 出现的合理时间窗口。
    overlap: 两摄像头视野有重叠，允许同一人同时出现在两个画面。
    bidirectional: 是否双向可达（默认 True，单向场景如单行道设为 False）。
    """

    from_cam: str
    to_cam: str
    transit_sec: Tuple[float, float] = (0.0, 30.0)
    overlap: bool = False
    bidirectional: bool = True


@dataclass
class Topology:
    """摄像头拓扑图，提供快速查询接口。"""

    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    links: List[TopologyLink] = field(default_factory=list)

    def neighbors(self, cam_id: str) -> List[str]:
        """返回与 cam_id 相邻的所有摄像头 id。"""
        result = []
        for link in self.links:
            if link.from_cam == cam_id:
                result.append(link.to_cam)
            if link.bidirectional and link.to_cam == cam_id:
                result.append(link.from_cam)
        return result

    def get_link(self, cam_a: str, cam_b: str) -> Optional[TopologyLink]:
        """查询两个摄像头之间的连接关系（双向查找）。"""
        for link in self.links:
            if link.from_cam == cam_a and link.to_cam == cam_b:
                return link
            if link.bidirectional and link.from_cam == cam_b and link.to_cam == cam_a:
                return link
        return None

    def is_adjacent(self, cam_a: str, cam_b: str) -> bool:
        """两个摄像头是否相邻。"""
        return self.get_link(cam_a, cam_b) is not None

    def is_overlapping(self, cam_a: str, cam_b: str) -> bool:
        """两个摄像头是否有视野重叠。"""
        link = self.get_link(cam_a, cam_b)
        return link.overlap if link else False


# ---------------------------------------------------------------------------
# 单摄像头轨迹
# ---------------------------------------------------------------------------
@dataclass
class CameraTrack:
    """单摄像头内的一条连续跟踪轨迹。

    track_id: 该摄像头内 YOLO track 分配的局部 ID。
    camera_id: 所属摄像头 id。
    global_id: 关联到的全局人员 ID（None 表示尚未关联）。
    feature: OSNet 提取的 512 维归一化特征向量（取轨迹期间多次特征的平均）。
    first_seen / last_seen: 该轨迹在本摄像头的首次/末次出现时间。
    bbox_history: 最近的边界框历史 [(timestamp, x1, y1, x2, y2), ...]。
    """

    track_id: int
    camera_id: str
    global_id: Optional[int] = None
    feature: Optional[np.ndarray] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    bbox_history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    frame_count: int = 0

    @property
    def duration(self) -> float:
        """轨迹持续时间（秒）。"""
        return self.last_seen - self.first_seen

    @property
    def is_active(self) -> bool:
        """轨迹是否仍在活跃（最近 2 秒内有更新）。"""
        return (time.time() - self.last_seen) < 2.0

    def update(self, bbox: Tuple[float, float, float, float], timestamp: float = None) -> None:
        """更新轨迹：记录新位置和刷新时间。"""
        ts = timestamp or time.time()
        self.last_seen = ts
        self.frame_count += 1
        self.bbox_history.append((ts, *bbox))
        # 只保留最近 30 个框，避免内存膨胀
        if len(self.bbox_history) > 30:
            self.bbox_history = self.bbox_history[-30:]


# ---------------------------------------------------------------------------
# 全局人员身份
# ---------------------------------------------------------------------------
@dataclass
class GlobalPerson:
    """跨摄像头的全局人员身份。

    global_id: 系统分配的全局唯一 ID（递增整数）。
    tracks: 关联到的所有 CameraTrack（可能跨多个摄像头）。
    feature: 累积的外观特征（多条轨迹特征的加权平均）。
    trajectory: 移动轨迹记录 [(timestamp, camera_id, description), ...]。
    """

    global_id: int
    tracks: List[CameraTrack] = field(default_factory=list)
    feature: Optional[np.ndarray] = None
    trajectory: List[Tuple[float, str, str]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def add_track(self, track: CameraTrack) -> None:
        """关联一条新的 CameraTrack。"""
        track.global_id = self.global_id
        self.tracks.append(track)
        self.last_seen = max(self.last_seen, track.last_seen)
        # 记录轨迹
        cam_name = track.camera_id
        self.trajectory.append((track.first_seen, cam_name, f"track_id={track.track_id}"))
        # 更新累积特征（简单平均，后续可改为加权）
        if track.feature is not None:
            if self.feature is None:
                self.feature = track.feature.copy()
            else:
                self.feature = (self.feature + track.feature) / 2.0
                # 重新归一化
                norm = np.linalg.norm(self.feature)
                if norm > 0:
                    self.feature /= norm

    @property
    def active_cameras(self) -> List[str]:
        """当前活跃出现的摄像头列表。"""
        return [t.camera_id for t in self.tracks if t.is_active]

    def timeline_summary(self) -> str:
        """返回可读的时间线摘要。"""
        lines = [f"GlobalPerson #{self.global_id}:"]
        for ts, cam, detail in sorted(self.trajectory):
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            lines.append(f"  {t_str} -> {cam} ({detail})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 特征画廊（用于消失后重识别）
# ---------------------------------------------------------------------------
@dataclass
class GalleryEntry:
    """画廊中的一条记录。"""

    global_id: int
    feature: np.ndarray
    camera_id: str
    timestamp: float = field(default_factory=time.time)


class FeatureGallery:
    """缓存最近出现的人员特征，用于短暂消失后的重识别。

    max_age_sec: 特征最大保留时间（秒），超过则淘汰。
    """

    def __init__(self, max_age_sec: float = 120.0):
        self.max_age_sec = max_age_sec
        self._entries: List[GalleryEntry] = []

    def add(self, global_id: int, feature: np.ndarray, camera_id: str) -> None:
        """添加一条特征记录。"""
        self._entries.append(GalleryEntry(global_id, feature, camera_id))
        self._prune()

    def search(
        self,
        query_feature: np.ndarray,
        threshold: float = 0.45,
        exclude_cameras: Optional[List[str]] = None,
    ) -> Optional[Tuple[int, float]]:
        """在画廊中搜索最匹配的全局 ID。

        返回 (global_id, similarity) 或 None（无匹配）。
        exclude_cameras: 排除来自这些摄像头的记录（避免同摄像头内重复匹配）。
        """
        self._prune()
        best_id = None
        best_sim = -1.0
        for entry in self._entries:
            if exclude_cameras and entry.camera_id in exclude_cameras:
                continue
            sim = float(np.dot(query_feature, entry.feature))
            if sim > best_sim:
                best_sim = sim
                best_id = entry.global_id
        if best_id is not None and best_sim >= threshold:
            return (best_id, best_sim)
        return None

    def remove(self, global_id: int) -> None:
        """移除某个全局 ID 的所有特征（已确认离开场景时调用）。"""
        self._entries = [e for e in self._entries if e.global_id != global_id]

    def _prune(self) -> None:
        """淘汰过期特征。"""
        cutoff = time.time() - self.max_age_sec
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]

    @property
    def size(self) -> int:
        return len(self._entries)
