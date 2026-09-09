"""下游推送：当「动物在桌上」被触发时，把截图交给下一阶段。

典型下一阶段是「多模态大模型二次确认」（例如判断宠物是否在偷吃饭菜）。
这里把推送抽象成 Sink 接口，开箱提供三种实现：

  * FileSink    把截图落盘到本地目录（并记录带标注的版本）。
  * HttpSink    把图片以 multipart 形式 POST 到一个 HTTP 接口（你的多模态服务）。
  * CallableSink 直接回调一个 Python 函数 —— 最灵活，可以在里面调用你自己的
                多模态模型 API / 消息队列 / 数据库写入等。

也可以把多个 Sink 组合进 CompositeSink 一起触发。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2

from geometry import SpatialRelation

logger = logging.getLogger("downstream")


@dataclass
class TriggerEvent:
    """一次触发事件的载体，沿 Sink 链传递。"""

    relation: SpatialRelation
    frame: "np.ndarray"  # 触发现场的原始帧（BGR）
    image_path: str = ""  # FileSink 落盘后回填
    timestamp: float = field(default_factory=time.time)


class Sink:
    def emit(self, event: TriggerEvent) -> None:  # pragma: no cover - 接口
        raise NotImplementedError


class FileSink(Sink):
    """把触发帧保存为 JPEG，可选保存带标注版本。"""

    def __init__(
        self,
        out_dir: str,
        save_annotated: bool = True,
        draw_fn: Optional[Callable[[np.ndarray, SpatialRelation], np.ndarray]] = None,
    ):
        self.out_dir = out_dir
        self.save_annotated = save_annotated
        self.draw_fn = draw_fn
        os.makedirs(out_dir, exist_ok=True)

    def emit(self, event: TriggerEvent) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
        raw_path = os.path.join(self.out_dir, f"capture_{ts}.jpg")
        ok = cv2.imwrite(raw_path, event.frame)
        if not ok:
            raise IOError(f"cv2.imwrite 失败，无法保存截图: {raw_path}")
        event.image_path = raw_path
        logger.info("截图已保存 -> %s", raw_path)

        if self.save_annotated and self.draw_fn is not None:
            try:
                ann = self.draw_fn(event.frame, event.relation)
                ann_path = os.path.join(self.out_dir, f"capture_{ts}_annotated.jpg")
                if not cv2.imwrite(ann_path, ann):
                    logger.warning("标注图保存失败: %s", ann_path)
            except Exception as exc:  # 标注失败不应阻断主流程
                logger.warning("标注图生成失败: %s", exc)


class HttpSink(Sink):
    """把截图 POST 到下游 HTTP 服务（如多模态模型服务）。需要 requests 库。"""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def emit(self, event: TriggerEvent) -> None:
        import requests  # 可选依赖，用到才导入

        with open(event.image_path, "rb") as f:
            files = {"image": f}
            data = {
                "timestamp": event.timestamp,
                "scenario": event.relation.scenario,
                "relationship": event.relation.relationship,
                "subject": event.relation.subject.label,
                "subject_conf": round(event.relation.subject.conf, 3),
                "surface": event.relation.surface.label,
                "matched": True,
            }
            resp = requests.post(self.url, files=files, data=data, timeout=self.timeout)
            logger.info("已推送到下游 %s -> HTTP %s", self.url, resp.status_code)


class CallableSink(Sink):
    """回调自定义函数，把事件交给你自己的业务逻辑（推荐用于接多模态模型）。"""

    def __init__(self, fn: Callable[[TriggerEvent], None]):
        self.fn = fn

    def emit(self, event: TriggerEvent) -> None:
        self.fn(event)


class CompositeSink(Sink):
    """顺序触发多个 Sink。"""

    def __init__(self, sinks: "list[Sink]" | None = None):
        self.sinks: "list[Sink]" = list(sinks or [])

    def add(self, sink: Sink) -> "CompositeSink":
        self.sinks.append(sink)
        return self

    def emit(self, event: TriggerEvent) -> None:
        for s in self.sinks:
            try:
                s.emit(event)
            except Exception as exc:  # 单个 sink 失败不牵连其他
                logger.error("Sink %s 触发失败: %s", type(s).__name__, exc)
