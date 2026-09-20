"""轨迹标注器：从追踪事件流中提取关键节点，构建平面图上的精确标注。

设计原则
--------
* 代码完成 90% 的几何工作（路径、节点标记、编号、时间戳）。
* VLM 只负责最后的语义标注（~2000 token / 人）。
* 位置精度：基于 bbox 中心在摄像头 FoV 内的相对位置推算，
  不是笼统的"餐厅"而是 FoV 扇形内的具体点。
* 消失→重现使用虚线路径推测。

核心概念
--------
Node          : 关键节点（appear/linger/vanish/reappear/exit）。
PathSegment   : 节点之间的连接（solid/dashed/dotted）。
PositionEstimator : 根据摄像头位置 + bbox 位置推算平面图坐标。

事件过滤
--------
* VLM 可见节点: appear, linger, vanish, reappear, exit
* 静默节点(只画图不发给 VLM): pass, transit
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("trajectory_annotator")


# ---------------------------------------------------------------------------
# 节点类型
# ---------------------------------------------------------------------------
# 进入 VLM 摘要的节点类型
VLM_NODE_TYPES = {"appear", "linger", "vanish", "reappear", "exit"}
# 只画在图上，不占 VLM token
SILENT_NODE_TYPES = {"pass", "transit"}

# 节点视觉属性
NODE_STYLE = {
    "appear":   {"symbol": "circle_filled", "color": "#2ECC71", "label_zh": "出现"},
    "linger":   {"symbol": "circle_double", "color": "#F39C12", "label_zh": "停留"},
    "transit":  {"symbol": "diamond",       "color": "#3498DB", "label_zh": "转移"},
    "pass":     {"symbol": "dot",           "color": "#95A5A6", "label_zh": "经过"},
    "vanish":   {"symbol": "circle_hollow", "color": "#E74C3C", "label_zh": "消失"},
    "reappear": {"symbol": "circle_hollow", "color": "#27AE60", "label_zh": "重现"},
    "exit":     {"symbol": "square_filled", "color": "#C0392B", "label_zh": "离开"},
}


# ---------------------------------------------------------------------------
# 位置推算
# ---------------------------------------------------------------------------
@dataclass
class CameraPose:
    """摄像头在平面图上的位姿。"""
    x: float          # 归一化坐标 (0~1)
    y: float
    fov_direction: float = 0.0   # 朝向角度（度，0=右，90=下，180=左，270=上）
    fov_angle: float = 90.0      # 视场角（度）
    coverage_radius: float = 0.12  # 覆盖半径（归一化距离，表示 FoV 扇形大小）

    @classmethod
    def from_config(cls, cfg: dict) -> "CameraPose":
        """从 settings.json 的 camera_positions 配置构建。"""
        direction_map = {
            "east": 0, "southeast": 45, "south": 90, "southwest": 135,
            "west": 180, "northwest": 225, "north": 270, "northeast": 315,
        }
        fov_dir = cfg.get("fov_direction", "east")
        if isinstance(fov_dir, str):
            fov_dir = direction_map.get(fov_dir.lower(), 0)
        return cls(
            x=cfg["x"],
            y=cfg["y"],
            fov_direction=float(fov_dir),
            fov_angle=float(cfg.get("fov_angle", 90)),
            coverage_radius=float(cfg.get("coverage_radius", 0.12)),
        )


class PositionEstimator:
    """根据摄像头位姿 + bbox 在画面中的相对位置，推算平面图坐标。

    原理
    ----
    假设摄像头 FoV 是一个扇形:
    * bbox 水平位置(frame_x_ratio 0~1) → 扇形内的左右偏移
    * bbox 垂直位置(frame_y_ratio 0~1) → 扇形内的远近偏移

    frame_x=0.5, frame_y=0.5 → 正好在摄像头位置（FoV 中心）
    frame_x=0.0 → 扇形左边缘
    frame_x=1.0 → 扇形右边缘
    frame_y=0.0 → 扇形远端
    frame_y=1.0 → 扇形近端（摄像头附近）
    """

    def estimate(
        self,
        pose: CameraPose,
        frame_x_ratio: float,
        frame_y_ratio: float,
    ) -> Tuple[float, float]:
        """推算平面图上的归一化坐标。

        Parameters
        ----------
        pose : CameraPose
            摄像头位姿。
        frame_x_ratio : float
            bbox 中心在帧中的水平位置 (0=左边缘, 1=右边缘)。
        frame_y_ratio : float
            bbox 中心在帧中的垂直位置 (0=上边缘/远处, 1=下边缘/近处)。

        Returns
        -------
        (x, y) : 平面图上的归一化坐标。
        """
        # 距离: frame_y 越小 = 离摄像头越远
        distance = pose.coverage_radius * (1.0 - frame_y_ratio * 0.7)

        # 角度偏移: frame_x 映射到 FoV 角度范围
        half_fov = pose.fov_angle / 2.0
        angle_offset = (frame_x_ratio - 0.5) * pose.fov_angle  # -half_fov ~ +half_fov
        absolute_angle = math.radians(pose.fov_direction + angle_offset)

        # 极坐标 → 笛卡尔
        dx = distance * math.cos(absolute_angle)
        dy = distance * math.sin(absolute_angle)

        # 限制在 [0, 1] 范围
        x = max(0.02, min(0.98, pose.x + dx))
        y = max(0.02, min(0.98, pose.y + dy))
        return (x, y)


# ---------------------------------------------------------------------------
# 节点 & 路径段
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryNode:
    """关键节点。"""
    id: str                    # "①", "②", ...
    node_type: str             # appear, linger, transit, ...
    camera: str                # 摄像头 ID
    time_str: str              # "14:30:05"
    timestamp: float           # unix epoch
    position: Tuple[float, float]  # 平面图归一化坐标
    label: str = ""            # 简短标签（代码自动生成或 VLM 填充）
    duration_sec: float = 0.0  # 停留时长（仅 linger 类型）
    is_vlm_visible: bool = True  # 是否进入 VLM 摘要

    @property
    def style(self) -> dict:
        return NODE_STYLE.get(self.node_type, NODE_STYLE["pass"])


@dataclass
class PathSegment:
    """节点之间的路径段。"""
    from_id: str               # 起始节点 id
    to_id: str                 # 终点节点 id
    from_pos: Tuple[float, float]
    to_pos: Tuple[float, float]
    style: str = "solid"       # solid / dashed / dotted
    label: str = ""            # 路径标签（如 "4s"）
    is_inferred: bool = False  # 是否为推测路径（vanish→reappear）


# ---------------------------------------------------------------------------
# 主标注器
# ---------------------------------------------------------------------------
# 多人颜色
PERSON_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]


@dataclass
class PersonTrajectory:
    """单人的完整轨迹标注。"""
    global_id: int
    color: str
    nodes: List[TrajectoryNode] = field(default_factory=list)
    segments: List[PathSegment] = field(default_factory=list)

    @property
    def vlm_nodes(self) -> List[TrajectoryNode]:
        """只返回 VLM 可见的关键节点。"""
        return [n for n in self.nodes if n.is_vlm_visible]

    @property
    def summary_draft(self) -> str:
        """代码自动生成的轨迹摘要草稿（供 VLM 参考）。"""
        parts = []
        for node in self.vlm_nodes:
            if node.node_type == "appear":
                parts.append(f"{node.time_str}出现")
            elif node.node_type == "linger":
                parts.append(f"停留{int(node.duration_sec)}s")
            elif node.node_type == "vanish":
                parts.append("消失")
            elif node.node_type == "reappear":
                parts.append("重现")
            elif node.node_type == "exit":
                parts.append(f"{node.time_str}离开")
        return "→".join(parts) if parts else ""


# 编号符号
NODE_SYMBOLS = [
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
]


class TrajectoryAnnotator:
    """轨迹标注器：监听追踪事件，构建精确的平面图标注。

    Parameters
    ----------
    camera_poses : dict
        {camera_id: CameraPose}
    linger_threshold_sec : float
        停留超过此时间才算 linger（秒）。
    vanish_timeout_sec : float
        消失超过此时间后，如果再出现则标记为 reappear（而非正常 appear）。
    """

    def __init__(
        self,
        camera_poses: Dict[str, CameraPose],
        linger_threshold_sec: float = 8.0,
        vanish_timeout_sec: float = 30.0,
    ):
        self.poses = camera_poses
        self.linger_threshold = linger_threshold_sec
        self.vanish_timeout = vanish_timeout_sec
        self.estimator = PositionEstimator()

        # 轨迹状态
        self._trajectories: Dict[int, PersonTrajectory] = {}
        self._next_global_idx = 0  # 用于分配颜色
        self._node_counter = 0     # 节点编号计数器（每人独立）
        self._node_counters: Dict[int, int] = {}

        # 消失追踪: {global_id: (timestamp, camera, position)}
        self._vanished: Dict[int, Tuple[float, str, Tuple[float, float]]] = {}

    # -------------------------------------------------------------------
    # 事件接口
    # -------------------------------------------------------------------
    def on_new_person(self, global_id: int) -> None:
        """新人员首次出现。"""
        color = PERSON_COLORS[self._next_global_idx % len(PERSON_COLORS)]
        self._next_global_idx += 1
        self._trajectories[global_id] = PersonTrajectory(
            global_id=global_id, color=color,
        )
        self._node_counters[global_id] = 0

    def on_track_update(
        self,
        global_id: int,
        camera_id: str,
        bbox: Tuple[float, float, float, float],
        frame_size: Tuple[int, int],
        timestamp: float,
        track_is_new: bool = False,
    ) -> None:
        """单摄像头内轨迹更新（每帧调用）。

        内部自动判断：首次出现 / 停留 / 经过。
        """
        if global_id not in self._trajectories:
            self.on_new_person(global_id)

        traj = self._trajectories[global_id]
        pose = self.poses.get(camera_id)
        if pose is None:
            return

        # 推算精确位置
        fw, fh = frame_size
        bx1, by1, bx2, by2 = bbox
        cx_ratio = (bx1 + bx2) / 2.0 / fw
        cy_ratio = (by1 + by2) / 2.0 / fh
        position = self.estimator.estimate(pose, cx_ratio, cy_ratio)

        # 检查是否需要生成节点
        last_node = traj.nodes[-1] if traj.nodes else None
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        if track_is_new or last_node is None:
            # 首次出现 / 新轨迹段
            if global_id in self._vanished:
                # 消失后重现
                self._add_reappear(traj, global_id, camera_id, position, timestamp, time_str)
            else:
                # 全新出现
                self._add_node(traj, "appear", camera_id, position, timestamp, time_str)
        else:
            # 同一轨迹持续中：检查是否需要标记 linger
            if (last_node.node_type in ("appear", "pass")
                    and last_node.camera == camera_id):
                duration = timestamp - last_node.timestamp
                if duration >= self.linger_threshold:
                    # 升级为 linger
                    self._upgrade_to_linger(last_node, duration)

        # 清理消失标记（人已重现）
        if global_id in self._vanished and not track_is_new:
            del self._vanished[global_id]

    def on_cross_camera_match(
        self,
        global_id: int,
        from_camera: str,
        to_camera: str,
        bbox: Tuple[float, float, float, float],
        frame_size: Tuple[int, int],
        timestamp: float,
        transit_sec: float,
    ) -> None:
        """跨摄像头匹配成功（人从 from_camera 转移到 to_camera）。"""
        if global_id not in self._trajectories:
            self.on_new_person(global_id)

        traj = self._trajectories[global_id]
        pose = self.poses.get(to_camera)
        if pose is None:
            return

        fw, fh = frame_size
        bx1, by1, bx2, by2 = bbox
        cx_ratio = (bx1 + bx2) / 2.0 / fw
        cy_ratio = (by1 + by2) / 2.0 / fh
        position = self.estimator.estimate(pose, cx_ratio, cy_ratio)
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        # 添加 transit 节点（静默，只画图不进 VLM）
        transit_node = self._add_node(
            traj, "transit", to_camera, position, timestamp, time_str,
            is_vlm_visible=False,
        )
        transit_node.label = f"{transit_sec:.0f}s"

        # 添加从前一个节点到 transit 的路径段
        if len(traj.nodes) >= 2:
            prev_node = traj.nodes[-2]
            traj.segments.append(PathSegment(
                from_id=prev_node.id,
                to_id=transit_node.id,
                from_pos=prev_node.position,
                to_pos=position,
                style="solid",
                label=f"{transit_sec:.0f}s",
            ))

    def on_track_lost(
        self,
        global_id: int,
        camera_id: str,
        last_bbox: Optional[Tuple[float, float, float, float]],
        frame_size: Tuple[int, int],
        timestamp: float,
    ) -> None:
        """轨迹从摄像头中消失。"""
        if global_id not in self._trajectories:
            return

        traj = self._trajectories[global_id]
        pose = self.poses.get(camera_id)
        if pose is None:
            return

        # 推算消失位置
        if last_bbox and pose:
            fw, fh = frame_size
            bx1, by1, bx2, by2 = last_bbox
            cx_ratio = (bx1 + bx2) / 2.0 / fw
            cy_ratio = (by1 + by2) / 2.0 / fh
            position = self.estimator.estimate(pose, cx_ratio, cy_ratio)
        else:
            # 使用最后一个节点的位置
            position = traj.nodes[-1].position if traj.nodes else (pose.x, pose.y)

        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        # 记录消失状态
        self._vanished[global_id] = (timestamp, camera_id, position)

        # 标记 vanish 节点
        vanish_node = self._add_node(
            traj, "vanish", camera_id, position, timestamp, time_str,
        )

    def on_tracking_end(self, timestamp: float) -> None:
        """追踪结束：为所有仍活跃的人添加 exit 节点。"""
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        for gid, traj in self._trajectories.items():
            if gid in self._vanished:
                continue  # 已经 vanish 了
            last_node = traj.nodes[-1] if traj.nodes else None
            if last_node and last_node.node_type != "exit":
                self._add_node(
                    traj, "exit", last_node.camera,
                    last_node.position, timestamp, time_str,
                )

    # -------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------
    def _add_node(
        self,
        traj: PersonTrajectory,
        node_type: str,
        camera: str,
        position: Tuple[float, float],
        timestamp: float,
        time_str: str,
        is_vlm_visible: bool = None,
    ) -> TrajectoryNode:
        """添加一个节点。"""
        if is_vlm_visible is None:
            is_vlm_visible = node_type in VLM_NODE_TYPES

        self._node_counters[traj.global_id] = self._node_counters.get(traj.global_id, 0) + 1
        idx = self._node_counters[traj.global_id]
        node_id = NODE_SYMBOLS[idx - 1] if idx <= len(NODE_SYMBOLS) else f"[{idx}]"

        style = NODE_STYLE.get(node_type, {})
        node = TrajectoryNode(
            id=node_id,
            node_type=node_type,
            camera=camera,
            time_str=time_str,
            timestamp=timestamp,
            position=position,
            label=style.get("label_zh", ""),
            is_vlm_visible=is_vlm_visible,
        )
        traj.nodes.append(node)

        # 自动连接前一个节点的路径段（reappear 除外，由 _add_reappear 处理）
        if node_type != "reappear" and len(traj.nodes) >= 2:
            prev = traj.nodes[-2]
            # 判断路径类型
            if prev.camera != camera:
                seg_style = "solid"
                is_inferred = False
            else:
                seg_style = "solid"
                is_inferred = False

            traj.segments.append(PathSegment(
                from_id=prev.id,
                to_id=node.id,
                from_pos=prev.position,
                to_pos=position,
                style=seg_style,
                is_inferred=is_inferred,
            ))

        return node

    def _add_reappear(
        self,
        traj: PersonTrajectory,
        global_id: int,
        camera_id: str,
        position: Tuple[float, float],
        timestamp: float,
        time_str: str,
    ) -> None:
        """消失后重现。"""
        vanish_info = self._vanished.pop(global_id, None)
        node = self._add_node(traj, "reappear", camera_id, position, timestamp, time_str)

        # 添加推测路径（虚线）
        if vanish_info and len(traj.nodes) >= 2:
            vanish_ts, vanish_cam, vanish_pos = vanish_info
            # 找到 vanish 节点
            vanish_node = None
            for n in reversed(traj.nodes[:-1]):
                if n.node_type == "vanish":
                    vanish_node = n
                    break
            if vanish_node:
                elapsed = timestamp - vanish_ts
                traj.segments.append(PathSegment(
                    from_id=vanish_node.id,
                    to_id=node.id,
                    from_pos=vanish_pos,
                    to_pos=position,
                    style="dashed",
                    label=f"?{elapsed:.0f}s",
                    is_inferred=True,
                ))

    def _upgrade_to_linger(self, node: TrajectoryNode, duration: float) -> None:
        """将 appear/pass 节点升级为 linger。"""
        node.node_type = "linger"
        node.duration_sec = duration
        node.is_vlm_visible = True  # linger 始终可见
        node.label = f"停留{int(duration)}s"

    # -------------------------------------------------------------------
    # 输出接口
    # -------------------------------------------------------------------
    def get_trajectory(self, global_id: int) -> Optional[PersonTrajectory]:
        """获取单人的轨迹标注。"""
        return self._trajectories.get(global_id)

    def get_all_trajectories(self) -> Dict[int, PersonTrajectory]:
        """获取所有人的轨迹。"""
        return dict(self._trajectories)

    def build_annotation(self, floor_plan_image: str = "") -> dict:
        """构建完整的标注 JSON（供 renderer 和 VLM 使用）。"""
        persons = []
        for gid, traj in sorted(self._trajectories.items()):
            person_data = {
                "global_id": gid,
                "color": traj.color,
                "summary_draft": traj.summary_draft,
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.node_type,
                        "camera": n.camera,
                        "time": n.time_str,
                        "position": {"x": round(n.position[0], 4), "y": round(n.position[1], 4)},
                        "label": n.label,
                        "duration_sec": n.duration_sec if n.duration_sec > 0 else None,
                        "vlm_visible": n.is_vlm_visible,
                    }
                    for n in traj.nodes
                ],
                "segments": [
                    {
                        "from": s.from_id,
                        "to": s.to_id,
                        "style": s.style,
                        "label": s.label,
                        "inferred": s.is_inferred,
                    }
                    for s in traj.segments
                ],
                "vlm_nodes_summary": [
                    f"{n.id} {n.time_str} {n.camera} {n.label}"
                    for n in traj.vlm_nodes
                ],
            }
            persons.append(person_data)

        return {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "floor_plan_image": floor_plan_image,
            "camera_positions": {
                cam_id: {"x": p.x, "y": p.y, "fov_direction": p.fov_direction}
                for cam_id, p in self.poses.items()
            },
            "persons": persons,
            "total_persons": len(persons),
        }
