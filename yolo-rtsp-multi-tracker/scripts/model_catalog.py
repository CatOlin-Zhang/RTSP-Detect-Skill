"""模型能力映射（Model Capability Catalog）。

这一层回答一个问题：**「当前这个检测模型，到底能认出哪些东西？」**

为什么需要它
------------
YOLO 只输出「类别 id + 边界框」。但用户关心的语义是「猫」「沙发」「餐桌」这种
人类可理解的词。本文件把「模型 -> 它的类别清单」做成一份**数据化的能力表**
(ModelProfile)，而不是把类别 id 硬编码进检测逻辑：

  * 内置 YOLOv8 在 COCO 上预训练的 80 类能力表（index = 类别 id）。
  * 自定义 / 其它数据集训练的模型，可以传一份自己的类别清单进来，无需改任何代码。
  * 场景规则 (scenario.py) 里用「类名」或「类 id」来指代目标，由这里统一解析成 id。

这样整个 Skill 就从「写死的动物/桌子检测」变成「基于模型能力表的通用空间关系检测」：
用户想检测「猫在沙发上」「人在椅子上」「车在车道上」，只要改配置即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# 内置能力表：COCO 预训练 80 类（YOLO26/YOLOv8 通用）。index 即 ultralytics 输出的类别 id。
# 这是「模型能认出什么」的声明数据，可根据需要整体替换为其它模型的能力表。
# ---------------------------------------------------------------------------
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class ModelProfile:
    """一个检测模型的能力表：名字 + 它的类别清单（index = 类别 id）。"""

    name: str
    class_names: List[str] = field(default_factory=list)

    def id_to_name(self, idx: int) -> str:
        """类别 id -> 类名；越界返回 'class_{idx}' 兜底。"""
        try:
            return self.class_names[idx]
        except IndexError:
            return f"class_{idx}"

    def name_to_id(self, name: str) -> Optional[int]:
        """类名 -> 类别 id（大小写不敏感）；找不到返回 None。"""
        target = name.strip().lower()
        for i, n in enumerate(self.class_names):
            if n.lower() == target:
                return i
        return None

    def list_capabilities(self) -> List[tuple]:
        """列出 (id, 类名) 能力清单，便于日志 / CLI 打印。"""
        return [(i, n) for i, n in enumerate(self.class_names)]

    def has(self, ref: "ClassRef") -> bool:
        """该能力表是否能解析给定的类引用（id 或 类名）。"""
        try:
            return len(resolve_class(ref, self)) > 0
        except (KeyError, TypeError):
            return False


# 内置模型注册表。可插拔：自定义模型用 register_model() 挂进来。
MODEL_PROFILES: Dict[str, ModelProfile] = {
    "coco": ModelProfile("coco", list(COCO_CLASSES)),
}
COCO_PROFILE = MODEL_PROFILES["coco"]


def register_model(profile: ModelProfile) -> None:
    """把一个自定义模型的能力表注册进全局注册表，供 settings 用 profile 名引用。"""
    MODEL_PROFILES[profile.name] = profile


# 类引用可以是：int(id) / str(类名) / str(数字) / 以上组成的 list
ClassRef = Union[int, str, Sequence[Union[int, str]]]


def resolve_class(ref: ClassRef, profile: ModelProfile) -> List[int]:
    """把一个（或一组）类引用解析成具体的类别 id 列表。

    接受形式：
      * 15                -> [15]
      * "cat"             -> [15]   （按 profile 能力表大小写不敏感匹配）
      * "15"              -> [15]   （纯数字字符串按 id 处理）
      * ["cat", "dog"]    -> [15, 16]
      * [15, "sofa"]      -> [15, <sofa 的 id>]
    解析失败（类名不在能力表）抛 KeyError，便于在配置阶段就暴露错误。
    """
    if isinstance(ref, (list, tuple, set)):
        ids: List[int] = []
        for r in ref:
            ids.extend(resolve_class(r, profile))
        return ids
    if isinstance(ref, bool):
        raise TypeError(f"类引用不支持布尔值: {ref!r}")
    if isinstance(ref, int):
        return [ref]
    if isinstance(ref, str):
        s = ref.strip()
        try:
            return [int(s)]
        except ValueError:
            pass
        iid = profile.name_to_id(s)
        if iid is None:
            raise KeyError(
                f"类 '{ref}' 不在模型 '{profile.name}' 的能力表中，"
                f"可用类见 list_capabilities()"
            )
        return [iid]
    raise TypeError(f"不支持的类引用类型: {type(ref).__name__}")


def build_model_profile(settings: dict) -> ModelProfile:
    """从 settings 构造 ModelProfile：

      * settings.model.class_names  -> 用自定义类别清单（list 或 {id:name} 字典）建表
      * settings.model.profile      -> 引用已注册的能力表名（默认 "coco"）
      * 都没给                      -> 默认 COCO
    """
    m = settings.get("model", {}) or {}
    custom = m.get("class_names")
    if custom:
        if isinstance(custom, dict):
            max_k = max(int(k) for k in custom)
            names = [custom.get(str(i), custom.get(i, f"class_{i}")) for i in range(max_k + 1)]
        else:
            names = [str(n) for n in custom]
        return ModelProfile(m.get("name", "custom"), names)
    profile_name = m.get("profile", "coco")
    if profile_name in MODEL_PROFILES:
        return MODEL_PROFILES[profile_name]
    # 未知 profile 名不致命：回退 COCO 并告警交由调用方日志
    return COCO_PROFILE
