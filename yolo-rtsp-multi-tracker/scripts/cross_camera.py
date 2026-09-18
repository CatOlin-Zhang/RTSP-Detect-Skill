"""跨摄像头关联引擎：结合外观特征 + 拓扑约束 + 时间窗口，把不同摄像头里的
同一人关联到同一个 GlobalPerson。

设计要点
--------
* 核心算法：对每一条新的 CameraTrack，在已知全局人员和特征画廊中搜索匹配。
* 匹配得分 = α × 外观相似度 + β × 时间合理性 + γ × 空间合理性
  - 外观相似度：OSNet 特征的 cosine similarity
  - 时间合理性：消失到重现的时间差是否在拓扑定义的窗口内
  - 空间合理性：两个摄像头是否相邻（拓扑约束）
* 处理三种场景：
  1. 首次出现 → 分配新 global_id
  2. 相邻摄像头出现 → 拓扑+外观匹配
  3. 重叠视野同时出现 → 实时帧间特征匹配

依赖
----
* reid_extractor.ReIDExtractor  — 特征提取
* tracking_models              — 数据模型
* 不依赖 cv2 / torch（特征提取在调用方完成）
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking_models import (
    CameraTrack,
    FeatureGallery,
    GlobalPerson,
    Topology,
)

logger = logging.getLogger("cross_camera")


# ---------------------------------------------------------------------------
# 关联结果
# ---------------------------------------------------------------------------
@dataclass
class AssociationResult:
    """一次跨摄关联的结果。"""

    matched: bool
    global_id: Optional[int] = None
    similarity: float = 0.0
    match_source: str = ""  # 'gallery' | 'active_track' | 'new'
    confidence: float = 0.0
    details: str = ""


# ---------------------------------------------------------------------------
# 关联引擎
# ---------------------------------------------------------------------------
@dataclass
class AssociationConfig:
    """关联参数配置。"""

    reid_threshold: float = 0.45      # 外观相似度阈值
    gallery_max_age_sec: float = 120.0  # 特征画廊最大保留时间
    time_window_weight: float = 0.2   # 时间合理性权重
    topology_weight: float = 0.1      # 空间合理性权重
    appearance_weight: float = 0.7    # 外观相似度权重
    new_person_threshold: float = 0.35  # 低于此值才分配新 global_id（避免重复分配）
    overlap_threshold: float = 0.55   # 重叠视野匹配阈值（更严格）


class CrossCameraAssociator:
    """跨摄像头关联引擎。

    Parameters
    ----------
    topology : Topology
        摄像头拓扑图。
    config : AssociationConfig
        关联参数。
    """

    def __init__(self, topology: Topology, config: Optional[AssociationConfig] = None):
        self.topology = topology
        self.config = config or AssociationConfig()

        # 全局人员库
        self._persons: Dict[int, GlobalPerson] = {}
        self._next_global_id: int = 1

        # 特征画廊（消失后重识别用）
        self._gallery = FeatureGallery(max_age_sec=self.config.gallery_max_age_sec)

        # 每条活跃轨迹的最后特征（用于重叠视野实时匹配）
        self._active_features: Dict[str, Dict[int, np.ndarray]] = {}
        # camera_id -> {track_id: feature}

        # 统计
        self.stats = {
            "total_associations": 0,
            "new_persons": 0,
            "gallery_reids": 0,
            "overlap_matches": 0,
            "failed_matches": 0,
        }

    # -------------------------------------------------------------------
    # 核心关联流程
    # -------------------------------------------------------------------
    def associate(
        self,
        track: CameraTrack,
        feature: np.ndarray,
        active_tracks_in_other_cams: Optional[Dict[str, List[Tuple[int, np.ndarray]]]] = None,
    ) -> AssociationResult:
        """对一条新轨迹/更新轨迹进行跨摄关联。

        Parameters
        ----------
        track : CameraTrack
            当前摄像头的轨迹。
        feature : np.ndarray
            该轨迹对应的外观特征向量（512 维，已归一化）。
        active_tracks_in_other_cams : dict, optional
            其他摄像头中当前活跃的轨迹：{camera_id: [(track_id, feature), ...]}

        Returns
        -------
        AssociationResult
            关联结果。
        """
        cam_id = track.camera_id

        # Step 1: 如果该轨迹已有 global_id，只做特征更新
        if track.global_id is not None:
            self._update_feature(track, feature)
            return AssociationResult(
                matched=True,
                global_id=track.global_id,
                similarity=1.0,
                match_source="existing",
                confidence=1.0,
            )

        # Step 2: 重叠视野 — 检查其他摄像头是否有同时出现的匹配
        if active_tracks_in_other_cams:
            overlap_result = self._match_overlap(cam_id, feature, active_tracks_in_other_cams)
            if overlap_result.matched:
                self._link_to_person(track, feature, overlap_result.global_id)
                self.stats["overlap_matches"] += 1
                return overlap_result

        # Step 3: 拓扑约束匹配 — 在相邻摄像头的最近消失轨迹中搜索
        neighbor_result = self._match_neighbors(cam_id, feature, track)
        if neighbor_result.matched:
            self._link_to_person(track, feature, neighbor_result.global_id)
            self.stats["gallery_reids"] += 1
            return neighbor_result

        # Step 4: 全局画廊搜索（非相邻摄像头，但时间合理）
        gallery_result = self._gallery.search(
            feature,
            threshold=self.config.reid_threshold,
            exclude_cameras=[cam_id],
        )
        if gallery_result is not None:
            gid, sim = gallery_result
            result = AssociationResult(
                matched=True,
                global_id=gid,
                similarity=sim,
                match_source="gallery",
                confidence=sim,
                details=f"global gallery match (sim={sim:.3f})",
            )
            self._link_to_person(track, feature, gid)
            self.stats["gallery_reids"] += 1
            return result

        # Step 5: 无法匹配 → 分配新的 global_id
        new_gid = self._new_global_id()
        self._link_to_person(track, feature, new_gid)
        self.stats["new_persons"] += 1
        logger.info(
            "新人员 global_id=%d 首次出现在 %s (track_id=%d)",
            new_gid, cam_id, track.track_id,
        )
        return AssociationResult(
            matched=True,
            global_id=new_gid,
            similarity=0.0,
            match_source="new",
            confidence=1.0,
            details=f"new person in {cam_id}",
        )

    # -------------------------------------------------------------------
    # 轨迹消失处理
    # -------------------------------------------------------------------
    def on_track_lost(self, track: CameraTrack) -> None:
        """轨迹从摄像头中消失时调用：将特征存入画廊供后续重识别。"""
        if track.global_id is not None and track.feature is not None:
            self._gallery.add(track.global_id, track.feature, track.camera_id)
            logger.debug(
                "轨迹消失: cam=%s track_id=%d global_id=%d -> 存入画廊",
                track.camera_id, track.track_id, track.global_id,
            )

        # 清理活跃特征缓存
        if track.camera_id in self._active_features:
            self._active_features[track.camera_id].pop(track.track_id, None)

    # -------------------------------------------------------------------
    # 内部匹配逻辑
    # -------------------------------------------------------------------
    def _match_overlap(
        self,
        cam_id: str,
        feature: np.ndarray,
        active_tracks: Dict[str, List[Tuple[int, np.ndarray]]],
    ) -> AssociationResult:
        """在重叠视野的摄像头中搜索同时出现的同一人。"""
        for other_cam, tracks in active_tracks.items():
            if not self.topology.is_overlapping(cam_id, other_cam):
                continue
            for other_track_id, other_feat in tracks:
                sim = float(np.dot(feature, other_feat))
                if sim >= self.config.overlap_threshold:
                    # 找到重叠视野匹配
                    person = self._find_person_by_track(other_cam, other_track_id)
                    if person is not None:
                        return AssociationResult(
                            matched=True,
                            global_id=person.global_id,
                            similarity=sim,
                            match_source="overlap",
                            confidence=sim,
                            details=f"overlap match in {other_cam} track={other_track_id} (sim={sim:.3f})",
                        )
        return AssociationResult(matched=False)

    def _match_neighbors(
        self,
        cam_id: str,
        feature: np.ndarray,
        track: CameraTrack,
    ) -> AssociationResult:
        """在相邻摄像头的画廊中搜索（带时间窗口约束）。"""
        neighbors = self.topology.neighbors(cam_id)
        if not neighbors:
            return AssociationResult(matched=False)

        best_gid = None
        best_score = -1.0
        best_sim = 0.0
        best_source = ""

        for neighbor_cam in neighbors:
            link = self.topology.get_link(cam_id, neighbor_cam)
            if link is None:
                continue

            # 在画廊中搜索来自相邻摄像头的特征
            for entry in self._gallery._entries:
                if entry.camera_id != neighbor_cam:
                    continue

                # 时间窗口检查
                elapsed = time.time() - entry.timestamp
                min_t, max_t = link.transit_sec
                if elapsed < min_t or elapsed > max_t + 30:  # 额外 30s 宽容度
                    continue

                # 外观相似度
                sim = float(np.dot(feature, entry.feature))
                if sim < self.config.reid_threshold:
                    continue

                # 综合得分
                time_score = self._time_score(elapsed, link.transit_sec)
                score = (
                    self.config.appearance_weight * sim
                    + self.config.time_window_weight * time_score
                    + self.config.topology_weight * 1.0  # 相邻 = 满分
                )

                if score > best_score:
                    best_score = score
                    best_gid = entry.global_id
                    best_sim = sim
                    best_source = f"neighbor {neighbor_cam} (sim={sim:.3f}, time={elapsed:.1f}s)"

        if best_gid is not None:
            return AssociationResult(
                matched=True,
                global_id=best_gid,
                similarity=best_sim,
                match_source="neighbor",
                confidence=best_score,
                details=best_source,
            )
        return AssociationResult(matched=False)

    # -------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------
    def _link_to_person(self, track: CameraTrack, feature: np.ndarray, global_id: int) -> None:
        """将轨迹关联到全局人员。"""
        if global_id not in self._persons:
            self._persons[global_id] = GlobalPerson(global_id=global_id)
        person = self._persons[global_id]
        person.add_track(track)

        # 更新活跃特征缓存
        cam = track.camera_id
        if cam not in self._active_features:
            self._active_features[cam] = {}
        self._active_features[cam][track.track_id] = feature

        self.stats["total_associations"] += 1

    def _update_feature(self, track: CameraTrack, feature: np.ndarray) -> None:
        """更新已有轨迹的特征。"""
        cam = track.camera_id
        if cam not in self._active_features:
            self._active_features[cam] = {}
        # 指数移动平均
        old = self._active_features[cam].get(track.track_id)
        if old is not None:
            new_feat = 0.3 * old + 0.7 * feature
            norm = np.linalg.norm(new_feat)
            if norm > 0:
                new_feat /= norm
            self._active_features[cam][track.track_id] = new_feat
        else:
            self._active_features[cam][track.track_id] = feature

        # 更新 GlobalPerson 特征
        if track.global_id in self._persons:
            self._persons[track.global_id].feature = self._active_features[cam][track.track_id]

        track.feature = self._active_features[cam][track.track_id]

    def _find_person_by_track(self, camera_id: str, track_id: int) -> Optional[GlobalPerson]:
        """通过摄像头 ID 和 track_id 找到对应的 GlobalPerson。"""
        for person in self._persons.values():
            for t in person.tracks:
                if t.camera_id == camera_id and t.track_id == track_id:
                    return person
        return None

    def _new_global_id(self) -> int:
        """分配新的全局 ID。"""
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    @staticmethod
    def _time_score(elapsed: float, transit_sec: Tuple[float, float]) -> float:
        """计算时间合理性得分（0~1）。在窗口内为 1，越偏离越低。"""
        min_t, max_t = transit_sec
        if min_t <= elapsed <= max_t:
            return 1.0
        if elapsed < min_t:
            return max(0.0, 1.0 - (min_t - elapsed) / max(min_t, 1.0))
        # elapsed > max_t
        overshoot = elapsed - max_t
        return max(0.0, 1.0 - overshoot / 30.0)  # 30s 内衰减到 0

    # -------------------------------------------------------------------
    # 查询接口
    # -------------------------------------------------------------------
    def get_person(self, global_id: int) -> Optional[GlobalPerson]:
        """获取全局人员信息。"""
        return self._persons.get(global_id)

    def get_all_persons(self) -> Dict[int, GlobalPerson]:
        """获取所有全局人员。"""
        return dict(self._persons)

    def get_active_persons(self) -> List[GlobalPerson]:
        """获取当前活跃的全局人员（至少在 1 个摄像头中出现）。"""
        return [p for p in self._persons.values() if p.active_cameras]

    def trajectory_report(self) -> str:
        """生成所有人员的轨迹摘要报告。"""
        lines = [f"=== 跨摄追踪报告 ({len(self._persons)} 人) ===\n"]
        for gid, person in sorted(self._persons.items()):
            lines.append(person.timeline_summary())
            lines.append("")
        lines.append(f"统计: {self.stats}")
        return "\n".join(lines)
