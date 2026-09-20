"""多摄像头并行流管理器：每路 RTSP 独立线程读帧，主线程串行推理+关联。

设计要点
--------
* 每个摄像头一个 CameraWorker 守护线程，持续读帧到单帧缓冲区。
* 主线程按轮询方式遍历所有摄像头，取最新帧做 track + 特征提取 + 跨摄关联。
* 推理是串行的（共享 GPU），读帧是并行的（各线程独立 cv2.VideoCapture）。
* 自动重连：RTSP 断流后指数退避重试。

架构
----
CameraWorker (thread)  ──→  最新帧缓冲
CameraWorker (thread)  ──→  最新帧缓冲
CameraWorker (thread)  ──→  最新帧缓冲
                                ↓  主线程轮询
                           YOLO track() → OSNet features → Cross-camera association
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.tracking_models import CameraConfig

logger = logging.getLogger("multi_camera")


# ---------------------------------------------------------------------------
# 单路摄像头读取线程
# ---------------------------------------------------------------------------
@dataclass
class FrameBuffer:
    """单帧缓冲区（线程安全）。"""

    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_idx: int = 0
    _lock: threading.Lock = None

    def __post_init__(self):
        if self._lock is None:
            self._lock = threading.Lock()

    def update(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> None:
        with self._lock:
            self.frame = frame
            self.timestamp = timestamp
            self.frame_idx = frame_idx

    def get(self) -> Tuple[Optional[np.ndarray], float, int]:
        """返回 (frame, timestamp, frame_idx)。无帧时 frame=None。"""
        with self._lock:
            return self.frame, self.timestamp, self.frame_idx


class CameraWorker:
    """单路摄像头的读帧线程。"""

    def __init__(
        self,
        config: CameraConfig,
        buffer_size: int = 2,
        frame_skip: int = 0,
    ):
        self.config = config
        self.buffer_size = buffer_size
        self.frame_skip = frame_skip

        self._buffer = FrameBuffer()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

        # 状态统计
        self.frames_read = 0
        self.last_error = ""
        self.is_connected = False

    def start(self) -> None:
        """启动读帧线程。"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CameraWorker %s 已在运行", self.config.id)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"cam-{self.config.id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("CameraWorker %s 已启动 (source=%s)", self.config.id, self.config.source)

    def stop(self) -> None:
        """停止读帧线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_frame(self) -> Tuple[Optional[np.ndarray], float, int]:
        """获取最新帧。"""
        return self._buffer.get()

    def _read_loop(self) -> None:
        """持续读帧的主循环。"""
        cap = self._open_capture()
        frame_idx = 0

        while self._running:
            if cap is None or not cap.isOpened():
                self.is_connected = False
                logger.warning(
                    "[%s] 视频源断开，%0.1f 秒后重连...",
                    self.config.id, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
                cap = self._open_capture()
                continue

            ok, frame = cap.read()
            if not ok:
                self.last_error = "read failed"
                cap.release()
                cap = None
                continue

            # 重连成功，重置延迟
            self._reconnect_delay = 1.0
            self.is_connected = True
            frame_idx += 1
            self.frames_read += 1

            # 跳帧
            if self.frame_skip and frame_idx % (self.frame_skip + 1) != 0:
                continue

            self._buffer.update(frame, time.time(), frame_idx)

        # 清理
        if cap is not None:
            cap.release()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """打开视频源。"""
        try:
            cap = cv2.VideoCapture(self.config.source)
            if not cap.isOpened():
                self.last_error = f"cannot open {self.config.source}"
                return None
            # RTSP 流设置小缓冲区，减少延迟
            if isinstance(self.config.source, str) and self.config.source.lower().startswith("rtsp"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            return cap
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("[%s] 打开视频源失败: %s", self.config.id, exc)
            return None


# ---------------------------------------------------------------------------
# 多摄像头管理器
# ---------------------------------------------------------------------------
class MultiCameraManager:
    """管理多路摄像头流，提供统一的帧获取接口。

    Parameters
    ----------
    cameras : list of CameraConfig
        摄像头配置列表。
    buffer_size : int
        RTSP 缓冲区大小（越小延迟越低）。
    frame_skip : int
        每 N 帧处理 1 帧（0=不跳）。
    """

    def __init__(
        self,
        cameras: List[CameraConfig],
        buffer_size: int = 2,
        frame_skip: int = 0,
    ):
        self.cameras = {c.id: c for c in cameras}
        self._workers: Dict[str, CameraWorker] = {}
        self._buffer_size = buffer_size
        self._frame_skip = frame_skip

    def start_all(self) -> None:
        """启动所有摄像头的读帧线程。"""
        for cam_id, config in self.cameras.items():
            worker = CameraWorker(config, self._buffer_size, self._frame_skip)
            worker.start()
            self._workers[cam_id] = worker
        logger.info("已启动 %d 路摄像头", len(self._workers))

    def stop_all(self) -> None:
        """停止所有摄像头。"""
        for worker in self._workers.values():
            worker.stop()
        self._workers.clear()
        logger.info("所有摄像头已停止")

    def get_frame(self, cam_id: str) -> Tuple[Optional[np.ndarray], float, int]:
        """获取指定摄像头的最新帧。"""
        worker = self._workers.get(cam_id)
        if worker is None:
            return None, 0.0, 0
        return worker.get_frame()

    def get_all_frames(self) -> Dict[str, Tuple[Optional[np.ndarray], float, int]]:
        """获取所有摄像头的最新帧。"""
        return {cam_id: worker.get_frame() for cam_id, worker in self._workers.items()}

    def iter_cameras(self):
        """遍历所有摄像头，yield (cam_id, frame, timestamp, frame_idx)。

        跳过无帧的摄像头。
        """
        for cam_id, worker in self._workers.items():
            frame, ts, idx = worker.get_frame()
            if frame is not None:
                yield cam_id, frame, ts, idx

    def status(self) -> Dict[str, dict]:
        """返回所有摄像头的状态摘要。"""
        result = {}
        for cam_id, worker in self._workers.items():
            _, ts, idx = worker.get_frame()
            result[cam_id] = {
                "connected": worker.is_connected,
                "frames_read": worker.frames_read,
                "last_frame_idx": idx,
                "last_frame_age": time.time() - ts if ts > 0 else -1,
                "last_error": worker.last_error,
            }
        return result

    @property
    def camera_ids(self) -> List[str]:
        """所有摄像头 ID 列表。"""
        return list(self.cameras.keys())
