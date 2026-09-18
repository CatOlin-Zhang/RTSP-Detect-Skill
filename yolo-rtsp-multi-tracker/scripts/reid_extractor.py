"""OSNet 外观特征提取器：把人员裁剪图转换为 512 维归一化特征向量。

设计要点
--------
* 封装 torchreid 的 OSNet 模型，对外提供统一接口：
  extract(crop_bgr) -> np.ndarray (512,)
* 延迟 import torch / torchreid，使无 GPU 环境也能导入本模块。
* 支持单张和批量提取（同一帧内多个人）。
* 特征向量已 L2 归一化，可直接用 cosine similarity（点积）比较。

模型选择
--------
torchreid 提供多个 ReID 模型，默认使用 OSNet（Omni-Scale Network）：
  * 参数量 ~2.2M，推理快（适合实时场景）
  * 在 Market1501 上 Rank-1 达 94.5%
  * 输出 512 维特征
  * 预训练权重自动从 Google Drive 下载（首次使用）

若 torchreid 不可用，回退到 torchvision 的 resnet18 + 自定义 head（精度略低）。
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

# 解决 numpy + torch 的 OpenMP 库冲突（Anaconda 环境常见问题）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger("reid_extractor")


class ReIDExtractor:
    """人员外观特征提取器。

    Parameters
    ----------
    model_name : str
        模型名称。默认 'osnet_x1_0'（OSNet 标准版）。
    device : str
        推理设备：'cuda:0', 'cpu', '' (自动选择)。
    input_size : tuple
        模型输入尺寸 (height, width)。OSNet 默认 (256, 128)。
    """

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        device: str = "",
        input_size: tuple = (256, 128),
    ):
        self.model_name = model_name
        self.device = device or ("cuda:0" if _cuda_available() else "cpu")
        self.input_size = input_size
        self._model = None
        self._backend = None  # 'torchreid' or 'torchvision'

    def _init_model(self) -> None:
        """延迟初始化模型（首次提取时加载）。"""
        if self._model is not None:
            return

        # 优先尝试 torchreid
        try:
            self._init_torchreid()
            return
        except Exception as exc:
            logger.warning("torchreid 初始化失败: %s，尝试 torchvision 回退", exc)

        # 回退到 torchvision resnet18
        try:
            self._init_torchvision()
        except Exception as exc:
            raise RuntimeError(f"无法初始化任何 ReID 模型: {exc}") from exc

    def _init_torchreid(self) -> None:
        """使用 torchreid.models.build_model 构建 OSNet（底层 API）。"""
        import torch
        import torchreid

        logger.info("加载 torchreid OSNet: %s (device=%s)", self.model_name, self.device)
        model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1,      # 不需要分类，只要特征
            loss="softmax",
            pretrained=True,    # 自动下载 ImageNet 预训练权重
            use_gpu=self.device != "cpu",
        )
        model.to(self.device)
        model.eval()
        self._model = model
        self._backend = "torchreid"
        logger.info("torchreid OSNet 加载成功 (feature_dim=%d)", model.feature_dim)

    def _init_torchvision(self) -> None:
        """回退方案：torchvision resnet18 + 全局平均池化。

        精度不如 OSNet，但在 torchreid 不可用时提供基本功能。
        """
        import torch
        import torchvision.models as models

        logger.info("加载 torchvision resnet18 回退 ReID (device=%s)", self.device)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 去掉最后的分类层，保留 512 维特征
        self._model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self._model.to(self.device)
        self._model.eval()
        self._backend = "torchvision"
        logger.info("torchvision resnet18 回退 ReID 加载成功")

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """从单张人员裁剪图提取特征向量。

        Parameters
        ----------
        crop_bgr : np.ndarray
            BGR 格式的人员裁剪图（从 YOLO 检测框裁出）。

        Returns
        -------
        np.ndarray or None
            512 维 L2 归一化特征向量，提取失败返回 None。
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        self._init_model()

        try:
            if self._backend == "torchreid":
                return self._extract_torchreid(crop_bgr)
            else:
                return self._extract_torchvision(crop_bgr)
        except Exception as exc:
            logger.warning("特征提取失败: %s", exc)
            return None

    def extract_batch(self, crops_bgr: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """批量提取特征（同一帧内多个人）。"""
        self._init_model()

        if self._backend == "torchreid":
            try:
                return self._extract_batch_torchreid(crops_bgr)
            except Exception as exc:
                logger.warning("批量特征提取失败，回退到逐张提取: %s", exc)
                return [self.extract(c) for c in crops_bgr]
        else:
            return [self.extract(c) for c in crops_bgr]

    def _extract_batch_torchreid(self, crops_bgr: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """torchreid 批量提取：把所有有效裁剪打成 batch 一次推理。"""
        import torch

        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0:
                valid_crops.append(crop)
                valid_indices.append(i)

        if not valid_crops:
            return [None] * len(crops_bgr)

        tensors = [self._preprocess(c) for c in valid_crops]
        batch = torch.cat(tensors, dim=0).to(self.device)

        with torch.no_grad():
            features = self._model(batch)

        features_np = features.cpu().detach().numpy()
        results = [None] * len(crops_bgr)
        for idx, feat in zip(valid_indices, features_np):
            feat = feat.flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            results[idx] = feat
        return results

    def _extract_torchreid(self, crop_bgr: np.ndarray) -> np.ndarray:
        """torchreid 单张提取：预处理 -> 推理 -> 归一化。"""
        import torch

        tensor = self._preprocess(crop_bgr).to(self.device)
        with torch.no_grad():
            feature = self._model(tensor)

        feature = feature.cpu().detach().numpy().flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

    def _preprocess(self, crop_bgr: np.ndarray):
        """把 BGR 裁剪图预处理为模型输入 tensor [1, 3, 256, 128]。"""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return transform(crop_rgb).unsqueeze(0)

    def _extract_torchvision(self, crop_bgr: np.ndarray) -> np.ndarray:
        """torchvision 回退单张提取。"""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self._model(tensor)

        feature = feature.cpu().detach().numpy().flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature

    @property
    def feature_dim(self) -> int:
        """特征维度。"""
        return 512 if self._backend == "torchreid" else 512

    @property
    def backend(self) -> str:
        """当前使用的后端：'torchreid' 或 'torchvision'。"""
        return self._backend or "uninitialized"


def _cuda_available() -> bool:
    """检测 CUDA 是否可用（不 import torch 的情况下）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def crop_person(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                padding: float = 0.1) -> np.ndarray:
    """从帧中裁剪人员区域，带少量 padding。

    Parameters
    ----------
    frame : BGR 帧
    x1, y1, x2, y2 : 检测框坐标
    padding : 裁剪向外扩展比例（默认 10%）

    Returns
    -------
    np.ndarray : 裁剪后的 BGR 图像
    """
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    return frame[cy1:cy2, cx1:cx2]
