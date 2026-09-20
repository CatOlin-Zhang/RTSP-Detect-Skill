"""实时画面左上角的 HUD（状态面板）渲染。

为什么单独一个文件
------------------
1. **OpenCV 的 cv2.putText 不支持中文**，画出来是方块/问号。而告警语「宠物上餐桌」
   必须是中文，所以文本统一交给 PIL 渲染，再合成回 BGR 帧。
2. HUD 只认识一个纯数据类 ``HudState``，**不碰任何 YOLO / 场景判定逻辑**——
   判定依旧来自 scenario 层。所以换场景、换模型都不用动这里。
3. 为了不掉帧，面板是在一块**小画布**上画好再 alpha 合成，而不是每帧把整幅图
   转成 PIL（那样每帧要来回转换多次，代价高）。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # PIL 是 ultralytics 的间接依赖，一般都在；缺失时退化成英文 ASCII
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


# Windows 自带中文字体候选（按优先级）。其它系统可自己往这里加路径。
FONT_CANDIDATES: Tuple[str, ...] = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

_font_cache: dict = {}


def _font(size: int, bold: bool = False):
    """按 (字号, 粗细) 缓存字体对象，避免每帧重复解析 TTF。"""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    if _HAS_PIL:
        cands = FONT_CANDIDATES
        if not bold:  # 非粗体优先用常规字重
            cands = tuple(c for c in FONT_CANDIDATES if "bd" not in c) + FONT_CANDIDATES
        for path in cands:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
    _font_cache[key] = font
    return font


# ---------------------------------------------------------------------------
# 状态数据：main.py 每帧填一次，HUD 只读它
# ---------------------------------------------------------------------------
@dataclass
class HudState:
    """HUD 需要展示的全部状态（不含任何业务判定逻辑）。"""

    fps: float = 0.0
    streaming: bool = True
    scenario_count: int = 0

    alert: bool = False            # 当前帧是否命中某个场景
    scenario: str = ""             # 命中的场景名
    subject: str = ""              # 主体标签（如 dog）
    surface: str = ""              # 承载面标签（如 dining table）
    conf: float = 0.0              # 主体置信度
    alert_age: float = 0.0         # 距上次触发过去的秒数（用于闪烁/自动熄灭）

    total_triggers: int = 0        # 累计触发次数
    last_capture: str = ""         # 最近一次截图文件名
    llm_status: str = ""           # Agent 精判状态文本


def _measure(lines: Sequence[tuple]) -> Tuple[int, List[int]]:
    """返回 (面板宽度, 每行高度列表)。"""
    probe = Image.new("RGBA", (8, 8)) if _HAS_PIL else None
    draw = ImageDraw.Draw(probe) if _HAS_PIL else None
    widths, heights = [], []
    for text, size, _color, bold in lines:
        font = _font(size, bold)
        if _HAS_PIL and font is not None:
            box = draw.textbbox((0, 0), text, font=font)
            widths.append(box[2] - box[0])
            heights.append(box[3] - box[1])
        else:  # 退化估算
            widths.append(int(len(text) * size * 0.6))
            heights.append(size)
    return max(widths or [0]), heights


def _render_panel(lines: Sequence[tuple], bg: tuple, alpha: int, padding: int = 14):
    """把若干行文本画到一块带圆角底的 RGBA 小画布上。"""
    max_w, heights = _measure(lines)
    line_h = [h + 8 for h in heights]
    width = int(max_w + padding * 2)
    height = int(sum(line_h) + padding * 2)

    if _HAS_PIL:
        img = Image.new("RGBA", (width, height), tuple(bg) + (alpha,))
        draw = ImageDraw.Draw(img)
        y = padding
        for (text, size, color, bold), lh in zip(lines, line_h):
            draw.text((padding, y), text, font=_font(size, bold), fill=tuple(color) + (255,))
            y += lh
        return np.array(img)  # RGBA

    # 退化路径：直接用 OpenCV 画块 + ASCII 文本
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, :3] = bg[::-1]
    img[:, :, 3] = alpha
    y = padding
    for (text, size, color, _bold), lh in zip(lines, line_h):
        cv2.putText(
            img[:, :, :3],
            text.encode("ascii", "ignore").decode(),
            (padding, y + size),
            cv2.FONT_HERSHEY_SIMPLEX,
            size / 30.0,
            color,
            1,
            cv2.LINE_AA,
        )
        y += lh
    return img


def _overlay(frame: np.ndarray, panel: np.ndarray, pos: Tuple[int, int] = (12, 12)) -> np.ndarray:
    """把 RGBA 面板 alpha 合成到 BGR 帧上（自动裁剪，越界不报错）。"""
    fh, fw = frame.shape[:2]
    ph, pw = panel.shape[:2]
    x, y = int(pos[0]), int(pos[1])
    x2, y2 = min(fw, x + pw), min(fh, y + ph)
    if x2 <= x or y2 <= y:
        return frame
    panel = panel[: y2 - y, : x2 - x]
    roi = frame[y:y2, x:x2]
    a = panel[:, :, 3:4].astype(np.float32) / 255.0
    rgb = panel[:, :, :3][:, :, ::-1].astype(np.float32)  # RGB -> BGR
    roi[:] = (a * rgb + (1.0 - a) * roi.astype(np.float32)).astype(np.uint8)
    return frame


# 配色（BGR）
C_TEXT = (235, 235, 235)
C_DIM = (170, 170, 170)
C_GREEN = (110, 220, 120)
C_RED = (90, 90, 245)
C_YELLOW = (90, 210, 245)
C_BG_NORMAL = (28, 28, 28)
C_BG_ALERT = (25, 25, 60)


def draw_hud(frame: np.ndarray, st: HudState) -> np.ndarray:
    """在帧的**左上角**绘制状态面板；命中场景时变红并闪烁。"""
    if st.alert:
        # 0.6s 周期的闪烁，让告警在实时画面里足够醒目
        blink = (time.time() % 0.6) < 0.4
        bg = C_BG_ALERT if blink else (45, 25, 25)
        lines = [
            ("【告警】空间关系命中", 24, C_RED, True),
            (f"{st.subject} 在 {st.surface} 上   置信度 {st.conf:.2f}", 19, C_TEXT, False),
            (f"场景 {st.scenario}   已截图，等待 Agent 精判", 17, C_YELLOW, False),
        ]
        if st.last_capture:
            lines.append((f"截图 {st.last_capture}", 15, C_DIM, False))
        if st.llm_status:
            lines.append((f"Agent 精判：{st.llm_status}", 18, C_GREEN, True))
    else:
        bg = C_BG_NORMAL
        state = "监控中" if st.streaming else "连接中断，重连中…"
        lines = [
            (f"● {state}", 22, C_GREEN if st.streaming else C_YELLOW, True),
            (f"FPS {st.fps:5.1f}   场景规则 {st.scenario_count} 条", 17, C_TEXT, False),
        ]
        if st.total_triggers:
            lines.append((f"累计触发 {st.total_triggers} 次", 16, C_DIM, False))
        if st.llm_status:
            lines.append((f"Agent 精判：{st.llm_status}", 17, C_GREEN, False))

    panel = _render_panel(lines, bg, alpha=205)
    frame = _overlay(frame, panel, pos=(12, 12))

    if st.alert and (time.time() % 0.6) < 0.4:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), C_RED, 6)
    return frame
