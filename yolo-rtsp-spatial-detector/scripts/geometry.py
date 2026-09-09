"""纯 Python 的空间关系几何判定（不依赖 cv2 / numpy / torch）。

设计原则：这一层**完全不知道「动物」「桌子」**这种语义，它只处理「两个框之间
的空间关系」。所有业务语义（什么在上什么、阈值多少）都来自 scenario 层。

坐标约定：图像左上角为原点，x 向右、y 向下（与 OpenCV 一致）。

核心思路（以默认的 on_top 关系为例）
------------------------------------
YOLO 只输出各自独立的边界框，并不知道一个目标是否真的「在另一个目标上面」。
我们用几何近似推断：

  * 取「主体(subject)」边界框的**底部中心点**（脚 / 接触支撑面的点）。
  * 判断该点是否落在「承载面(surface)」边界框的**上表面区域**内——承载面区域
    = 承载框的上半部分（因为一张桌子/沙发的框还包含桌腿 / 地面空隙，主体站在
    地上时脚在图像坐标里位于承载框下方，不会进入表面区域）。

这样能排除「主体只是站在承载物旁边」（脚在地面，y 比表面区域更靠下）的误判。

关系评估器是可注册的
--------------------
on_top 只是一种关系。想加「within（中心落入框内，适合『人坐在区域里』）」、
「above（在上方但不接触）」等，只要用 register_relationship 注册一个函数即可，
无需改动本文件其它逻辑。这就是「不写死实现」的关键扩展点。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


# ---------------------------------------------------------------------------
# 边界框
# ---------------------------------------------------------------------------
@dataclass
class Box:
    """一个目标边界框，像素坐标，格式 xyxy。"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    cls: int = -1
    label: str = ""

    # ---- 几何辅助属性 -----------------------------------------------------
    @property
    def cx(self) -> float:
        """水平中心点。"""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """垂直中心点。"""
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom(self) -> float:
        """底边 y 坐标。"""
        return self.y2

    @property
    def bottom_center(self) -> "tuple[float, float]":
        """底部中心点 (x, y)，即主体脚下接触支撑面的点。"""
        return (self.cx, self.y2)

    @property
    def upper_center(self) -> "tuple[float, float]":
        """上部中心点 (x, y1 + height*0.25)，近似头部/躯干上沿位置。

        用于「探头/趴上桌」姿态：脚还在地上或椅子上，但头和上身已经
        伸到承载面上方。
        """
        return (self.cx, self.y1 + self.height * 0.25)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "Box") -> float:
        """与另一个框的交并比。"""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection_area(self, other: "Box") -> float:
        """与另一个框的交集面积（像素²）。"""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


# ---------------------------------------------------------------------------
# 可注册的关系评估器
# ---------------------------------------------------------------------------
# 每个评估器签名：fn(subject: Box, surface: Box, **thresholds) -> bool
RelationshipEvaluator = Callable[[Box, Box], bool]
RELATIONSHIP_EVALUATORS: Dict[str, RelationshipEvaluator] = {}


def register_relationship(name: str, fn: RelationshipEvaluator) -> None:
    """注册一个新的空间关系评估器，供 scenario 用 relationship 名引用。"""
    RELATIONSHIP_EVALUATORS[name] = fn


def surface_zone(
    surface: Box,
    surface_ratio: float = 0.6,
    margin_x_ratio: float = 0.0,
) -> Box:
    """返回承载面边界框的「上表面区域」子框（通用，不限于桌子）。

    surface_ratio  : 从承载框顶部往下算，占整框高度的比例，作为可用表面。
                    典型 0.5~0.7。俯视角度越大越接近 1，平视角越大越接近 0.5。
    margin_x_ratio : 左右各向内收缩比例，容忍框被估大 / 透视水平偏移。
    """
    m = margin_x_ratio * surface.width
    return Box(
        x1=surface.x1 + m,
        y1=surface.y1,
        x2=surface.x2 - m,
        y2=surface.y1 + surface_ratio * surface.height,
        conf=surface.conf,
        cls=surface.cls,
        label=surface.label,
    )


def is_on_top(
    subject: Box,
    surface: Box,
    surface_ratio: float = 0.6,
    margin_x_ratio: float = 0.0,
    **_ignored,
) -> bool:
    """判定 subject 是否「在 surface 上面」。

    条件：subject 底部中心点落在 surface 表面区域内。
        surface_x1 < subject_bottom_center_x < surface_x2
        且  surface_y1 < subject_bottom_center_y < surface_y2
    **_ignored 吸收其它关系用到的阈值（如 within 的 margin_y_ratio），保证评估器
    能共享同一份场景阈值字典而不必精确匹配键。
    """
    zone = surface_zone(surface, surface_ratio, margin_x_ratio)
    ax, ay = subject.bottom_center
    return (zone.x1 <= ax <= zone.x2) and (zone.y1 <= ay <= zone.y2)


def is_body_over(
    subject: Box,
    surface: Box,
    surface_ratio: float = 0.6,
    margin_x_ratio: float = 0.0,
    min_overlap_ratio: float = 0.25,
    **_ignored,
) -> bool:
    """判定 subject 是否「身体探入 surface 上方区域」（趴桌沿 / 探头够食姿态）。

    弥补 on_top 的盲区：宠物脚踩在椅子/地上、但头和上身已经伸到桌面上方偷吃时，
    脚点不在桌面区内，on_top 判负。这里改看两件事，满足其一即命中：

      1. 主体「上部中心点」（近似头部位置）落入表面区域；
      2. 主体框与表面区域的交集占主体面积的比例 >= min_overlap_ratio
        （默认 0.25，即身体约四分之一以上已经探到桌面正上方）。

    min_overlap_ratio 同时承担排误职责：站在桌边地上的宠物头部也许能碰到桌面
    高度，但其框与桌面区的交集占比很小，会被该阈值排除。
    """
    zone = surface_zone(surface, surface_ratio, margin_x_ratio)
    ux, uy = subject.upper_center
    if (zone.x1 <= ux <= zone.x2) and (zone.y1 <= uy <= zone.y2):
        return True
    if subject.area <= 0:
        return False
    return subject.intersection_area(zone) / subject.area >= min_overlap_ratio


def is_on_or_over(
    subject: Box,
    surface: Box,
    surface_ratio: float = 0.6,
    margin_x_ratio: float = 0.0,
    min_overlap_ratio: float = 0.25,
    **_ignored,
) -> bool:
    """on_top 或 body_over：整个爬上去，或脚在下方但身体已探到正上方，都算。"""
    return is_on_top(subject, surface, surface_ratio, margin_x_ratio) or is_body_over(
        subject, surface, surface_ratio, margin_x_ratio, min_overlap_ratio
    )


def is_within(
    subject: Box,
    surface: Box,
    margin_x_ratio: float = 0.0,
    margin_y_ratio: float = 0.0,
    **_ignored,
) -> bool:
    """判定 subject 是否「完全落在 surface 框内」（中心点在收缩后的框内）。

    适合「人坐在椅子里」「物体放进容器」这类关系——关注的是整体位置而非脚点。
    """
    m = margin_x_ratio * surface.width
    n = margin_y_ratio * surface.height
    cx, cy = subject.cx, subject.cy
    return (surface.x1 + m <= cx <= surface.x2 - m) and (
        surface.y1 + n <= cy <= surface.y2 - n
    )


# 注册内置关系
register_relationship("on_top", is_on_top)
register_relationship("within", is_within)
register_relationship("body_over", is_body_over)
register_relationship("on_or_over", is_on_or_over)


# ---------------------------------------------------------------------------
# 关系判定结果
# ---------------------------------------------------------------------------
@dataclass
class SpatialRelation:
    """一次「subject 与 surface 在某关系下」的判定结果（通用，无业务语义）。"""

    scenario: str          # 来自哪个场景规则
    relationship: str      # 关系名，如 "on_top"
    subject: Box           # 主体（被检查是否在某物上的目标）
    surface: Box           # 承载面 / 区域
    on: bool               # 是否满足该空间关系
    iou: float = 0.0       # subject 与 surface 框的 IoU（诊断用）
    surface_ratio: float = 0.6
    margin_x_ratio: float = 0.0
