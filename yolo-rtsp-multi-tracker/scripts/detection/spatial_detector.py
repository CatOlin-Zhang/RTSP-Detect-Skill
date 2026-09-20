"""Ultralytics YOLO + OpenCV 检测/跟踪器：RTSP 逐帧推理并解析为 Box 对象。

设计要点
--------
* ultralytics 的 import 延迟到 YoloDetector.__init__ 内部，使 geometry / scenario /
  downstream 等模块在无 torch 环境也能导入，方便单独做单元测试。
* 类别标签统一来自 ModelProfile 能力表（单一事实来源），不再在代码里硬编码
  任何 COCO 类 id。
* 推理一次返回一帧（RTSP 实时流标准做法：「读一帧 -> 推一帧」）。
* 两种推理模式：
  - detect(): model.predict() — 独立帧检测，无时序关联
  - track():  model.track()  — 时序跟踪，分配持久 track_id
* 空间关系判定完全委托给 scenario.evaluate_scenarios，本文件不持有任何业务语义。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from core.geometry import Box, SpatialRelation
from core.model_catalog import COCO_PROFILE, ModelProfile
from detection.scenario import ResolvedScenario, evaluate_scenarios

logger = logging.getLogger("spatial_detector")


@dataclass
class DetectConfig:
    """检测相关配置（纯模型/推理参数，不含任何场景语义）。"""

    model_path: str = "yolo26n.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    device: str = ""   # '' => 自动选择 (cpu / cuda)
    tracker: str = "botsort.yaml"  # 跟踪器配置：botsort.yaml | bytetrack.yaml


class YoloDetector:
    """封装 ultralytics YOLO，把推理结果转换成 geometry.Box 列表。"""

    def __init__(
        self,
        cfg: Optional[DetectConfig] = None,
        profile: Optional[ModelProfile] = None,
    ):
        self.cfg = cfg or DetectConfig()
        self.profile = profile or COCO_PROFILE
        from ultralytics import YOLO  # 延迟导入，避免无 torch 环境无法 import

        logger.info("加载 YOLO 模型: %s (能力表: %s)", self.cfg.model_path, self.profile.name)
        self.model = YOLO(self.cfg.model_path)
        if self.cfg.device:
            self.model.to(self.cfg.device)

    def infer(self, frame: np.ndarray) -> List[Box]:
        """对单帧做推理，返回 Box 列表（label 由 ModelProfile 解析）。"""
        results = self.model.predict(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            verbose=False,
            device=self.cfg.device or None,
        )
        return self._parse(results[0])

    @staticmethod
    def _parse(result) -> List[Box]:
        """把 ultralytics 的 Results 解析为 Box 列表。"""
        boxes: List[Box] = []
        if result.boxes is None:
            return boxes
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            boxes.append(Box(x1, y1, x2, y2, conf, cls, label=""))  # label 在 detect 时补
        return boxes

    def detect(self, frame: np.ndarray) -> List[Box]:
        """推理并回填类别标签（来自 ModelProfile 能力表，单一事实来源）。"""
        boxes = self.infer(frame)
        for b in boxes:
            b.label = self.profile.id_to_name(b.cls)
        return boxes

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Box]:
        """跟踪模式：使用 model.track() 代替 predict()，分配持久 track_id。

        persist=True 保持帧间跟踪状态（必须设为 True 才能跨帧关联 ID）。
        返回的 Box 包含有效的 track_id（>= 0）。
        """
        results = self.model.track(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            persist=persist,
            tracker=self.cfg.tracker,
            verbose=False,
            device=self.cfg.device or None,
        )
        return self._parse_tracked(results[0])

    def _parse_tracked(self, result) -> List[Box]:
        """解析 track() 结果，包含 track_id。"""
        boxes: List[Box] = []
        if result.boxes is None or result.boxes.id is None:
            return boxes
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            tid = int(b.id[0])
            label = self.profile.id_to_name(cls)
            boxes.append(Box(x1, y1, x2, y2, conf, cls, label=label, track_id=tid))
        return boxes

    def detect_relations(self, boxes: List[Box], scenarios: List[ResolvedScenario]) -> List[SpatialRelation]:
        """端到端：从一帧检测结果直接得到空间关系列表（按场景）。"""
        return evaluate_scenarios(boxes, scenarios)


# ------------------------- 可视化辅助（通用） ---------------------------------
def draw_boxes(frame: np.ndarray, boxes: List[Box], color=(0, 255, 0)) -> np.ndarray:
    """在帧上画所有框 + 类别/置信度标签。"""
    for b in boxes:
        cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(
            frame,
            f"{b.label} {b.conf:.2f}",
            (int(b.x1), max(0, int(b.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_relation(frame: np.ndarray, relation: SpatialRelation) -> np.ndarray:
    """针对一次触发，画高亮标注：承载面蓝色、主体红色，并打 "<主体> ON <承载面>!"。"""
    r = relation
    # 承载面：蓝色
    cv2.rectangle(
        frame,
        (int(r.surface.x1), int(r.surface.y1)),
        (int(r.surface.x2), int(r.surface.y2)),
        (255, 0, 0),
        2,
    )
    # 主体：红色
    cv2.rectangle(
        frame,
        (int(r.subject.x1), int(r.subject.y1)),
        (int(r.subject.x2), int(r.subject.y2)),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"{r.subject.label} ON {r.surface.label}!",
        (int(r.subject.x1), max(0, int(r.subject.y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
