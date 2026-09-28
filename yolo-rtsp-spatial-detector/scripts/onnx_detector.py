"""ONNX + onnxruntime 轻量推理后端：不依赖 torch / ultralytics。

为什么需要它
------------
ultralytics 推理链会把 torch（CPU 版 ~200MB 下载 / >1GB 磁盘）拉进部署机。
本模块用 onnxruntime 直接加载 ONNX 权重，部署机最小依赖变为：

    opencv-python + numpy + onnxruntime（合计下载 ~70MB）

为什么不是 cv2.dnn
------------------
实测（OpenCV 4.13.0）cv2.dnn 虽能解析 YOLO26 端到端 ONNX，但 TopK /
GatherElements / Mod / Tile 等端到端头算子的 forward 数值**静默错误**
（bus conf 0.925 -> 0.083，坐标错乱）。onnxruntime 输出与 ultralytics
几乎完全一致，是唯一可靠的轻量引擎。

输出格式（与导出方式绑定）
--------------------------
只支持**端到端导出**的 ONNX（输出 shape (1, 300, 6)，每行
[x1, y1, x2, y2, conf, cls]，已含 NMS-free 的 top-300 检测）。
这是 YOLO26 用 export_onnx.py 默认参数导出的格式；传统 (1, 84, 8400)
原始输出不支持。iou_threshold 因此不参与计算（NMS 已固化在模型里），
仅保留参数位以共享 DetectConfig。

坐标还原
--------
输入经 letterbox（等比缩放 + 灰边填充到 input_size 正方形）后进模型，
输出坐标位于 letterbox 坐标系，需按 (scale, pad_x, pad_y) 还原回原图
像素坐标——与 ultralytics 的预处理约定一致，两种后端可直接对比。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from geometry import Box, SpatialRelation
from model_catalog import COCO_PROFILE, ModelProfile
from scenario import ResolvedScenario, evaluate_scenarios

logger = logging.getLogger("onnx_detector")

# letterbox 填充色（ultralytics 约定的中灰 114）
_PAD_COLOR = 114


def letterbox(frame: np.ndarray, new_size: int) -> Tuple[np.ndarray, float, int, int]:
    """等比缩放 + 居中填充到 new_size x new_size。

    返回 (画布, scale, pad_x, pad_y)。原图坐标 -> 画布坐标：乘 scale 加 pad；
    反向还原（模型输出 -> 原图）：减 pad 除 scale。
    """
    h, w = frame.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), _PAD_COLOR, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


class OnnxDetector:
    """onnxruntime 加载端到端 ONNX，接口与 YoloDetector 完全一致。"""

    def __init__(self, cfg, profile: Optional[ModelProfile] = None):
        """
        cfg : DetectConfig（与 ultralytics 后端共用；input_size 为本后端新增字段）
        """
        self.cfg = cfg
        self.profile = profile or COCO_PROFILE
        self.input_size = int(getattr(cfg, "input_size", 640))

        logger.info(
            "加载 ONNX 模型: %s (onnxruntime 后端, input_size=%d, 能力表: %s)",
            cfg.model_path, self.input_size, self.profile.name,
        )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime 未安装（ONNX 轻量后端需要它）。"
                "部署机执行: pip install onnxruntime"
            ) from exc

        providers = ["CPUExecutionProvider"]
        if "cuda" in (cfg.device or "").lower():
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(
            self._read_model_bytes(cfg.model_path), providers=providers
        )
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _read_model_bytes(path: str) -> bytes:
        """读模型文件为字节流。

        onnxruntime 的 InferenceSession 虽可接受路径，但 Windows 上路径含
        非 ASCII 字符（如中文用户名）时不可靠；统一走 Python 层读字节，
        行为与平台无关。
        """
        with open(path, "rb") as f:
            return f.read()

    # ------------------------------------------------------------------
    # 推理与解析
    # ------------------------------------------------------------------
    def infer(self, frame: np.ndarray) -> List[Box]:
        """对单帧推理，返回 Box 列表（坐标已还原到原图像素坐标系）。"""
        canvas, scale, pad_x, pad_y = letterbox(frame, self.input_size)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW

        out = self.sess.run(None, {self.input_name: blob})[0]  # (1, 300, 6)
        return self._parse(out[0], scale, pad_x, pad_y)

    def _parse(self, dets: np.ndarray, scale: float, pad_x: int, pad_y: int) -> List[Box]:
        """解析端到端输出并还原坐标。

        dets : (N, 6)，每行 [x1, y1, x2, y2, conf, cls]（letterbox 坐标系）
        """
        boxes: List[Box] = []
        if dets.ndim != 2 or dets.shape[1] != 6:
            raise ValueError(
                f"非端到端 ONNX 输出 shape={dets.shape}（期望 (N, 6)：x1,y1,x2,y2,conf,cls）。"
                "传统 (84, 8400) 原始输出请用 export_onnx.py 默认参数重新导出。"
            )
        for row in dets:
            conf = float(row[4])
            if conf < self.cfg.conf_threshold:
                continue
            x1 = (float(row[0]) - pad_x) / scale
            y1 = (float(row[1]) - pad_y) / scale
            x2 = (float(row[2]) - pad_x) / scale
            y2 = (float(row[3]) - pad_y) / scale
            cls = int(row[5])
            boxes.append(Box(x1, y1, x2, y2, conf, cls, label=""))
        return boxes

    def detect(self, frame: np.ndarray) -> List[Box]:
        """推理并回填类别标签（来自 ModelProfile 能力表，单一事实来源）。"""
        boxes = self.infer(frame)
        for b in boxes:
            b.label = self.profile.id_to_name(b.cls)
        return boxes

    def detect_relations(
        self, boxes: List[Box], scenarios: List[ResolvedScenario]
    ) -> List[SpatialRelation]:
        """端到端：从一帧检测结果直接得到空间关系列表（按场景）。"""
        return evaluate_scenarios(boxes, scenarios)
