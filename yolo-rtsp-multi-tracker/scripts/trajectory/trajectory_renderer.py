"""轨迹渲染器：在平面图上精确绘制路径、节点标记、编号和时间戳。

设计原则
--------
* 使用 PIL (Pillow) 做精确的几何绘制（路径线、箭头、节点标记）。
* 节点编号 ①②③ 直接渲染在对应位置旁。
* 时间戳标注在每个关键节点旁。
* 虚线表示推测路径（消失→重现）。
* 输出 PNG，可直接交付或叠加 VLM 语义标注。

输出物
------
* trajectory_live.png  — 实时更新（每 N 秒刷新）
* trajectory_final.png — 追踪结束后的完整轨迹图
"""
from __future__ import annotations

import io
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("trajectory_renderer")

# 延迟导入 PIL，避免无 Pillow 环境导入失败
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow 未安装，轨迹渲染不可用。pip install Pillow")


# ---------------------------------------------------------------------------
# 颜色工具
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """'#E74C3C' → (231, 76, 60)"""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ---------------------------------------------------------------------------
# 渲染配置
# ---------------------------------------------------------------------------
class RenderConfig:
    """渲染参数。"""

    def __init__(
        self,
        output_size: Tuple[int, int] = (1200, 900),
        node_radius: int = 10,
        path_width: int = 3,
        arrow_size: int = 12,
        font_size: int = 14,
        font_size_small: int = 11,
        font_size_title: int = 20,
        padding: int = 40,
        bg_color: str = "#FFFFFF",
        camera_color: str = "#333333",
        camera_icon_size: int = 8,
    ):
        self.output_size = output_size
        self.node_radius = node_radius
        self.path_width = path_width
        self.arrow_size = arrow_size
        self.font_size = font_size
        self.font_size_small = font_size_small
        self.font_size_title = font_size_title
        self.padding = padding
        self.bg_color = bg_color
        self.camera_color = camera_color
        self.camera_icon_size = camera_icon_size


# ---------------------------------------------------------------------------
# 主渲染器
# ---------------------------------------------------------------------------
class TrajectoryRenderer:
    """在平面图上绘制轨迹标注。

    Parameters
    ----------
    floor_plan_path : str
        平面图图片路径。
    config : RenderConfig
        渲染参数。
    """

    def __init__(self, floor_plan_path: str = "", config: Optional[RenderConfig] = None):
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow 未安装，无法渲染轨迹图")
        self.config = config or RenderConfig()
        self._floor_plan_path = floor_plan_path
        self._font = self._load_font(self.config.font_size)
        self._font_small = self._load_font(self.config.font_size_small)
        self._font_title = self._load_font(self.config.font_size_title)

    def _load_font(self, size: int) -> "ImageFont.FreeTypeFont":
        """加载字体（优先中文字体）。"""
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",       # 黑体
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _to_pixel(self, nx: float, ny: float, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """归一化坐标 (0~1) → 像素坐标。"""
        pad = self.config.padding
        w, h = img_size
        draw_w = w - 2 * pad
        draw_h = h - 2 * pad
        px = pad + int(nx * draw_w)
        py = pad + int(ny * draw_h)
        return (px, py)

    # -------------------------------------------------------------------
    # 主绘制入口
    # -------------------------------------------------------------------
    def render(
        self,
        annotation: dict,
        output_path: str = "trajectory.png",
    ) -> str:
        """渲染完整轨迹图。

        Parameters
        ----------
        annotation : dict
            TrajectoryAnnotator.build_annotation() 的输出。
        output_path : str
            输出 PNG 路径。

        Returns
        -------
        str : 输出文件路径。
        """
        # 加载底图
        img = self._load_base_image()
        draw = ImageDraw.Draw(img, "RGBA")
        img_size = img.size

        # 绘制摄像头位置
        self._draw_cameras(draw, annotation.get("camera_positions", {}), img_size)

        # 绘制每人的轨迹
        for person in annotation.get("persons", []):
            color = person.get("color", "#E74C3C")
            nodes = {n["id"]: n for n in person.get("nodes", [])}

            # 先画路径段
            for seg in person.get("segments", []):
                from_node = nodes.get(seg["from"])
                to_node = nodes.get(seg["to"])
                if from_node and to_node:
                    self._draw_segment(
                        draw, from_node, to_node, seg, color, img_size,
                    )

            # 再画节点（在路径上层）
            for node in person.get("nodes", []):
                self._draw_node(draw, node, color, img_size)

        # 绘制图例
        self._draw_legend(draw, img_size, annotation.get("total_persons", 0))

        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path, "PNG")
        logger.info("轨迹图已保存: %s", output_path)
        return output_path

    def render_to_bytes(self, annotation: dict) -> bytes:
        """渲染到内存（用于发送给 VLM）。"""
        img = self._load_base_image()
        draw = ImageDraw.Draw(img, "RGBA")
        img_size = img.size

        self._draw_cameras(draw, annotation.get("camera_positions", {}), img_size)

        for person in annotation.get("persons", []):
            color = person.get("color", "#E74C3C")
            nodes = {n["id"]: n for n in person.get("nodes", [])}
            for seg in person.get("segments", []):
                from_node = nodes.get(seg["from"])
                to_node = nodes.get(seg["to"])
                if from_node and to_node:
                    self._draw_segment(draw, from_node, to_node, seg, color, img_size)
            for node in person.get("nodes", []):
                self._draw_node(draw, node, color, img_size)

        self._draw_legend(draw, img_size, annotation.get("total_persons", 0))

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    # -------------------------------------------------------------------
    # 底图加载
    # -------------------------------------------------------------------
    def _load_base_image(self) -> "Image.Image":
        """加载平面图底图，缩放到输出尺寸。"""
        if self._floor_plan_path and os.path.exists(self._floor_plan_path):
            img = Image.open(self._floor_plan_path).convert("RGBA")
            img = img.resize(self.config.output_size, Image.LANCZOS)
            # 半透明白色覆盖层，让标注更清晰
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 60))
            img = Image.alpha_composite(img, overlay)
            return img
        else:
            # 无底图：空白画布
            img = Image.new("RGBA", self.config.output_size, hex_to_rgba(self.config.bg_color))
            return img

    # -------------------------------------------------------------------
    # 绘制元素
    # -------------------------------------------------------------------
    def _draw_cameras(
        self, draw: ImageDraw, cameras: dict, img_size: Tuple[int, int],
    ) -> None:
        """绘制摄像头位置图标 + 名称。"""
        cam_color = hex_to_rgb(self.config.camera_color)
        r = self.config.camera_icon_size

        for cam_id, pos in cameras.items():
            px, py = self._to_pixel(pos["x"], pos["y"], img_size)

            # 画摄像头图标（小三角 + 圆点）
            draw.ellipse([px - r, py - r, px + r, py + r], fill=cam_color, outline=cam_color)
            # FoV 方向指示小线段
            fov_dir = pos.get("fov_direction", 0)
            angle_rad = math.radians(fov_dir)
            end_x = px + int((r + 8) * math.cos(angle_rad))
            end_y = py + int((r + 8) * math.sin(angle_rad))
            draw.line([(px, py), (end_x, end_y)], fill=cam_color, width=2)

            # 摄像头名称
            draw.text((px + r + 4, py - 6), cam_id, fill=cam_color, font=self._font_small)

    def _draw_segment(
        self,
        draw: ImageDraw,
        from_node: dict,
        to_node: dict,
        seg: dict,
        color: str,
        img_size: Tuple[int, int],
    ) -> None:
        """绘制路径段（实线/虚线/点线 + 箭头）。"""
        from_pos = from_node["position"]
        to_pos = to_node["position"]
        p1 = self._to_pixel(from_pos["x"], from_pos["y"], img_size)
        p2 = self._to_pixel(to_pos["x"], to_pos["y"], img_size)

        rgb = hex_to_rgb(color)
        style = seg.get("style", "solid")

        if style == "dashed":
            self._draw_dashed_line(draw, p1, p2, rgb, dash_len=10, gap_len=6)
        elif style == "dotted":
            self._draw_dashed_line(draw, p1, p2, rgb, dash_len=3, gap_len=6)
        else:
            draw.line([p1, p2], fill=rgb, width=self.config.path_width)

        # 箭头
        self._draw_arrow(draw, p1, p2, rgb)

        # 路径标签（如 "4s", "?15s"）
        if seg.get("label"):
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2 - 12
            draw.text((mid_x, mid_y), seg["label"], fill=rgb, font=self._font_small)

    def _draw_dashed_line(
        self,
        draw: ImageDraw,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        dash_len: int = 10,
        gap_len: int = 6,
    ) -> None:
        """绘制虚线。"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return

        unit_x = dx / length
        unit_y = dy / length
        pos = 0.0
        while pos < length:
            start_x = p1[0] + unit_x * pos
            start_y = p1[1] + unit_y * pos
            end_pos = min(pos + dash_len, length)
            end_x = p1[0] + unit_x * end_pos
            end_y = p1[1] + unit_y * end_pos
            draw.line(
                [(int(start_x), int(start_y)), (int(end_x), int(end_y))],
                fill=color, width=self.config.path_width,
            )
            pos += dash_len + gap_len

    def _draw_arrow(
        self,
        draw: ImageDraw,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """在线段终点画箭头。"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 20:
            return  # 太短不画箭头

        # 箭头位置：距终点 node_radius 处
        r = self.config.node_radius + 2
        ratio = (length - r) / length
        tip_x = p1[0] + dx * ratio
        tip_y = p1[1] + dy * ratio

        angle = math.atan2(dy, dx)
        arrow_len = self.config.arrow_size
        arrow_angle = math.radians(25)

        left_x = tip_x - arrow_len * math.cos(angle - arrow_angle)
        left_y = tip_y - arrow_len * math.sin(angle - arrow_angle)
        right_x = tip_x - arrow_len * math.cos(angle + arrow_angle)
        right_y = tip_y - arrow_len * math.sin(angle + arrow_angle)

        draw.polygon(
            [(int(tip_x), int(tip_y)), (int(left_x), int(left_y)), (int(right_x), int(right_y))],
            fill=color,
        )

    def _draw_node(
        self,
        draw: ImageDraw,
        node: dict,
        color: str,
        img_size: Tuple[int, int],
    ) -> None:
        """绘制节点标记 + 编号 + 时间戳。"""
        pos = node["position"]
        px, py = self._to_pixel(pos["x"], pos["y"], img_size)
        r = self.config.node_radius
        rgb = hex_to_rgb(color)
        node_type = node["type"]
        node_id = node["id"]

        # 根据类型画不同标记
        if node_type == "appear":
            # 实心圆
            draw.ellipse([px - r, py - r, px + r, py + r], fill=rgb, outline=rgb)
        elif node_type == "linger":
            # 双圆（同心圆）
            draw.ellipse([px - r, py - r, px + r, py + r], fill=rgb, outline=rgb)
            draw.ellipse(
                [px - r - 4, py - r - 4, px + r + 4, py + r + 4],
                outline=rgb, width=2,
            )
        elif node_type == "transit":
            # 菱形
            points = [(px, py - r), (px + r, py), (px, py + r), (px - r, py)]
            draw.polygon(points, fill=rgb, outline=rgb)
        elif node_type == "pass":
            # 小圆点
            r_small = r // 2
            draw.ellipse([px - r_small, py - r_small, px + r_small, py + r_small], fill=rgb)
        elif node_type == "vanish":
            # 空心圆 + 虚线边框
            draw.ellipse([px - r, py - r, px + r, py + r], outline=rgb, width=2)
            # 问号标记
            draw.text((px + r + 2, py - r - 2), "?", fill=rgb, font=self._font)
        elif node_type == "reappear":
            # 空心圆 + 感叹号
            draw.ellipse([px - r, py - r, px + r, py + r], outline=rgb, width=2)
            draw.text((px + r + 2, py - r - 2), "!", fill=rgb, font=self._font)
        elif node_type == "exit":
            # 实心方块
            draw.rectangle([px - r, py - r, px + r, py + r], fill=rgb, outline=rgb)

        # 编号（所有节点都标）
        label_x = px - r - 18
        label_y = py - r - 4
        draw.text((label_x, label_y), node_id, fill=rgb, font=self._font)

        # 时间戳（关键节点标）
        if node.get("vlm_visible") or node_type in ("transit",):
            time_x = px + r + 6
            time_y = py + 2
            draw.text((time_x, time_y), node.get("time", ""), fill=rgb, font=self._font_small)

    def _draw_legend(
        self,
        draw: ImageDraw,
        img_size: Tuple[int, int],
        total_persons: int,
    ) -> None:
        """绘制图例（右下角）。"""
        w, h = img_size
        x0 = w - 200
        y0 = h - 140

        # 半透明背景
        draw.rectangle([x0 - 10, y0 - 10, x0 + 190, y0 + 130], fill=(255, 255, 255, 200))

        items = [
            ("●", "出现/离开", "#2ECC71"),
            ("◎", "停留", "#F39C12"),
            ("◆", "转移(静默)", "#3498DB"),
            ("○?", "消失", "#E74C3C"),
            ("○!", "重现", "#27AE60"),
            ("━━", "确认路径", "#333333"),
            ("- -", "推测路径", "#999999"),
        ]

        draw.text((x0, y0), "图例", fill=(0, 0, 0), font=self._font)
        for i, (symbol, meaning, c) in enumerate(items):
            y = y0 + 20 + i * 15
            draw.text((x0 + 5, y), f"{symbol}  {meaning}", fill=hex_to_rgb(c), font=self._font_small)
