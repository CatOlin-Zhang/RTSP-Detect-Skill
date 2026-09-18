"""触发截图的队列管理：粗检命中 → 截图落盘到 pending/ → 等待 Agent 精判。

设计要点
--------
* **本模块不做任何精判。** 粗检截图 + 事件元数据落盘到 ``pending/``，
  由外部 Agent 读取后做细粒度识别，把判定结果写回 ``reviewed/``。
* **不阻塞实时循环。** 截图保存是纯文件 I/O，毫秒级完成。
* **Agent 只需要：**
  1. 读 ``pending/<id>.jpg`` + ``pending/<id>.json`` 了解发生了什么
  2. 看截图做判断
  3. 把结论写进 ``reviewed/<id>.json``（可选，HUD 会显示）

目录约定（均在 output_dir 下）::

    pending/<event_id>.jpg     触发帧（带标注）
    pending/<event_id>.json    事件元数据（场景、关系、主体、置信度等）
    reviewed/<event_id>.json   Agent 回写的判定结果
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger("vision_review")


@dataclass
class ReviewConfig:
    """截图队列配置。目前仅管理目录结构，不涉及 API 调用。"""
    pass

    @classmethod
    def from_settings(cls, settings: dict) -> "ReviewConfig":
        return cls()


class VisionReviewer:
    """触发截图的落盘队列：提交不阻塞，Agent 异步读取做精判。"""

    def __init__(self, out_dir: str, cfg: ReviewConfig | None = None):
        self.out_dir = out_dir
        self.pending_dir = os.path.join(out_dir, "pending")
        self.reviewed_dir = os.path.join(out_dir, "reviewed")
        os.makedirs(self.pending_dir, exist_ok=True)
        os.makedirs(self.reviewed_dir, exist_ok=True)

        self._last_status: str = ""
        self._last_submit_ts: float = 0.0
        self._seen_verdicts: Dict[str, str] = {}

        logger.info("截图队列就绪: pending=%s  reviewed=%s", self.pending_dir, self.reviewed_dir)

    # ---------------- 提交（非阻塞） ----------------
    def submit(self, frame: np.ndarray, meta: Dict[str, Any]) -> str:
        """把一帧 + 元数据保存到 pending/，等待 Agent 读取做精判。"""
        event_id = meta.get("event_id") or time.strftime("%Y%m%d_%H%M%S", time.localtime())
        meta = dict(meta)
        meta["event_id"] = event_id
        meta["created_at"] = time.time()

        try:
            img_path = os.path.join(self.pending_dir, f"{event_id}.jpg")
            cv2.imwrite(img_path, frame)
            with open(os.path.join(self.pending_dir, f"{event_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("粗检截图已入队 -> pending/%s", event_id)
        except Exception as exc:
            logger.error("写入截图队列失败: %s", exc)

        self._last_submit_ts = time.time()
        self._last_status = "粗检触发，等待 Agent 精判…"
        return event_id

    # ---------------- 轮询 Agent 回写的判定结果 ----------------
    def poll_status(self) -> str:
        """检查 reviewed/ 目录中 Agent 回写的判定结果，供 HUD 显示。

        Agent 将判定 JSON 写入 ``reviewed/<id>.json``，本方法读取最新的
        那个文件并返回一句话摘要。判定 JSON 格式由 Agent 自定义，本模块
        兼容以下字段（均可选）：

        - ``confirmed``: bool — 是否确认违规
        - ``verdict``: str — 判定结论文本
        - ``reason``: str — 一句话说明
        - ``confidence``: float — 判定置信度
        """
        try:
            files = [f for f in os.listdir(self.reviewed_dir) if f.endswith(".json")]
        except OSError:
            return self._last_status
        if not files:
            return self._last_status

        newest = max(files, key=lambda f: os.path.getmtime(os.path.join(self.reviewed_dir, f)))
        path = os.path.join(self.reviewed_dir, newest)
        mtime = str(os.path.getmtime(path))
        if self._seen_verdicts.get(newest) == mtime:
            return self._last_status
        if float(mtime) < self._last_submit_ts - 1.0:
            self._seen_verdicts[newest] = mtime
            return self._last_status
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            confirmed = data.get("confirmed")
            reason = (data.get("reason") or "").strip()
            conf = data.get("confidence")
            if confirmed is True:
                text = "确认违规"
            elif confirmed is False:
                text = "已排除，非违规"
            else:
                text = str(data.get("verdict", "已复核"))
            if reason:
                text += f"｜{reason[:18]}"
            if isinstance(conf, (int, float)):
                text += f" ({float(conf):.2f})"
            self._seen_verdicts[newest] = mtime
            self._last_status = text
            logger.info("Agent 精判结果[%s]: %s", newest.replace(".json", ""), text)
            return text
        except Exception as exc:
            logger.warning("读取 Agent 精判结果失败: %s", exc)
            return self._last_status
