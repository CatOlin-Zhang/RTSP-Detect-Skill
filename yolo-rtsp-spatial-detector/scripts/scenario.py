"""场景规则层（Scenario）：把「用户想检测什么」从代码里解耦成声明式配置。

用户关心的从来不是「动物」「桌子」这种写死的词，而是具体场景，例如：
  * 宠物是否偷吃   -> subject=cat/dog, surface=dining table, relationship=on_top
  * 猫是否上沙发   -> subject=cat,      surface=couch,      relationship=on_top
  * 人是否坐椅子   -> subject=person,   surface=chair,      relationship=within
  * 车是否在车道   -> subject=car,      surface=road_zone,  relationship=within

本层做三件事：
  1. 把 settings 里的 scenario 配置解析成 ResolvedScenario（类名 -> 模型能力表里的 id）。
  2. 对一帧的框，按每个启用的场景做匹配 + 关系评估，产出 SpatialRelation 列表。
  3. 多 subject / 多 surface 时，给每个 subject 选最合适的一对（on 优先，其次 IoU）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from geometry import Box, SpatialRelation
from model_catalog import ClassRef, ModelProfile, resolve_class

logger = logging.getLogger("scenario")


# 关系评估器需要的阈值键（不同关系用的键不同，这里给一个默认值集合）
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "surface_ratio": 0.6,
    "margin_x_ratio": 0.0,
    "margin_y_ratio": 0.0,
    "min_overlap_ratio": 0.25,  # body_over / on_or_over 用：身体探入桌面区的最小占比
}


@dataclass
class ResolvedScenario:
    """解析完成、可直接用于推理的场景规则。"""

    name: str
    subject_ids: set          # 主体类别 id 集合
    surface_ids: set          # 承载面类别 id 集合
    relationship: str = "on_top"
    thresholds: dict = field(default_factory=dict)
    min_conf: float = 0.0
    cooldown: float = 0.0
    enabled: bool = True

    def subject_label(self, profile: ModelProfile) -> str:
        ids = sorted(self.subject_ids)
        return "/".join(profile.id_to_name(i) for i in ids) if ids else "?"

    def surface_label(self, profile: ModelProfile) -> str:
        ids = sorted(self.surface_ids)
        return "/".join(profile.id_to_name(i) for i in ids) if ids else "?"


def resolve_scenario(cfg: dict, profile: ModelProfile) -> Optional[ResolvedScenario]:
    """把单个 scenario 配置解析为 ResolvedScenario。

    配置字段（全部可选，除了 name / subject / surface）：
      name          场景名（触发日志 / 冷却键 / 下游标识）
      subject       主体类引用：int / 类名 / 数字串 / 列表
      surface       承载面类引用
      relationship  关系名（默认 "on_top"），需已注册
      surface_ratio 表面区域占比（on_top 用）
      margin_x_ratio 表面区域左右内缩（on_top 用）
      margin_y_ratio 框内上下内缩（within 用）
      min_conf      主体最低置信度
      cooldown      该场景两次截图最小间隔（秒，0=用全局）
      enabled       是否启用（默认 true）
    解析失败（类名不在能力表 / 关系未注册）返回 None 并打印告警。
    """
    name = cfg.get("name")
    if not name:
        logger.warning("跳过无名场景: %s", cfg)
        return None

    try:
        subject_ids = set(resolve_class(cfg["subject"], profile))
        surface_ids = set(resolve_class(cfg["surface"], profile))
    except (KeyError, TypeError) as exc:
        logger.warning("场景 '%s' 类引用解析失败，已跳过: %s", name, exc)
        return None

    rel = cfg.get("relationship", "on_top")
    if rel not in _REL_EVAL_NAMES():
        logger.warning(
            "场景 '%s' 的关系 '%s' 未注册，已跳过（可用: %s）",
            name, rel, sorted(_REL_EVAL_NAMES()),
        )
        return None

    thresholds = {
        "surface_ratio": float(cfg.get("surface_ratio", DEFAULT_THRESHOLDS["surface_ratio"])),
        "margin_x_ratio": float(cfg.get("margin_x_ratio", DEFAULT_THRESHOLDS["margin_x_ratio"])),
        "margin_y_ratio": float(cfg.get("margin_y_ratio", DEFAULT_THRESHOLDS["margin_y_ratio"])),
        "min_overlap_ratio": float(
            cfg.get("min_overlap_ratio", DEFAULT_THRESHOLDS["min_overlap_ratio"])
        ),
    }
    return ResolvedScenario(
        name=name,
        subject_ids=subject_ids,
        surface_ids=surface_ids,
        relationship=rel,
        thresholds=thresholds,
        min_conf=float(cfg.get("min_conf", 0.0)),
        cooldown=float(cfg.get("cooldown", 0.0)),
        enabled=bool(cfg.get("enabled", True)),
    )


def _REL_EVAL_NAMES():
    # 延迟 import 避免几何 -> scenario 循环
    from geometry import RELATIONSHIP_EVALUATORS
    return RELATIONSHIP_EVALUATORS


def build_scenarios(
    settings: dict,
    profile: ModelProfile,
    default_surface_ratio: Optional[float] = None,
) -> List[ResolvedScenario]:
    """从 settings 构建全部 ResolvedScenario。

    default_surface_ratio：命令行 --surface-ratio 传入时，作为未显式设置该值的
    场景的默认值（方便一次性调所有 on_top 场景）。
    """
    raw = settings.get("scenarios") or []
    out: List[ResolvedScenario] = []
    for cfg in raw:
        sc = resolve_scenario(cfg, profile)
        if sc is None:
            continue
        if default_surface_ratio is not None and "surface_ratio" not in cfg:
            sc.thresholds["surface_ratio"] = default_surface_ratio
        out.append(sc)
    return out


def evaluate_scenarios(boxes: List[Box], scenarios: List[ResolvedScenario]) -> List[SpatialRelation]:
    """对一帧检测结果，按所有启用的场景产出空间关系列表。

    每个 subject 匹配一张最合适的 surface（on 优先，其次 IoU），结果包含 on=False
    的诊断项，供主流程按需过滤 / 下游判断。
    """
    from geometry import RELATIONSHIP_EVALUATORS, is_on_top

    results: List[SpatialRelation] = []
    for sc in scenarios:
        if not sc.enabled:
            continue
        subs = [b for b in boxes if b.cls in sc.subject_ids]
        surfs = [b for b in boxes if b.cls in sc.surface_ids]
        if not subs or not surfs:
            continue
        evaluator = RELATIONSHIP_EVALUATORS.get(sc.relationship, is_on_top)

        for s in subs:
            if s.conf < sc.min_conf:
                continue
            best: Optional[Box] = None
            best_score = -1.0
            best_on = False
            best_iou = 0.0
            for t in surfs:
                on = evaluator(s, t, **sc.thresholds)
                iou = s.iou(t)
                score = (10.0 if on else 0.0) + iou
                if score > best_score:
                    best, best_score, best_on, best_iou = t, score, on, iou
            if best is not None:
                results.append(
                    SpatialRelation(
                        scenario=sc.name,
                        relationship=sc.relationship,
                        subject=s,
                        surface=best,
                        on=best_on,
                        iou=best_iou,
                        surface_ratio=sc.thresholds.get("surface_ratio", 0.6),
                        margin_x_ratio=sc.thresholds.get("margin_x_ratio", 0.0),
                    )
                )
    return results
