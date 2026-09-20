"""VLM 语义标注：构造精简 prompt 发送给多模态模型，拿回标注叠加到轨迹图。

设计原则
--------
* Token 效率第一：只发关键节点摘要（VLM_NODE_TYPES），transit/pass 不发。
* 代码已画好 90%（路径、节点、编号、时间戳），VLM 只补语义文字。
* 每人 ~2000 token（图片 + ~150 字 prompt + ~100 字输出）。
* 多人 ≤ 3 时合并为一次 VLM 调用。

VLM 负责的事
-------------
1. 为每个编号节点生成 ≤4 字标注（如"进入""逗留""离开"）
2. 一句话总结轨迹（≤30 字）
3. （可选）直接在图上画补充标注

VLM 不负责的事（代码已做完）
-----------------------------
* 画路径线、箭头、节点标记
* 标注编号 ①②③
* 标注时间戳
* 图例
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("vlm_annotate_prompt")


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------
VLM_PROMPT_TEMPLATE = """\
这是一张人员移动轨迹图，节点已用编号标记在图上。

节点摘要:
{nodes_text}

请完成:
1. 为每个编号节点生成简短标注(≤4字，如"进入""逗留""经过""离开""消失""重现")
2. 用一句话(≤30字)总结此人的行动轨迹

严格按以下JSON格式输出:
{{"labels": {{"①": "进入", "②": "逗留", ...}}, "summary": "..."}}"""

VLM_MULTI_PERSON_TEMPLATE = """\
这是一张多人移动轨迹图，不同颜色代表不同人。

{persons_text}

请为每人完成:
1. 为每个编号节点生成简短标注(≤4字)
2. 用一句话(≤20字)总结该人轨迹

严格按以下JSON格式输出:
{{"persons": [{{"global_id": 1, "labels": {{"①": "...", ...}}, "summary": "..."}}, ...]}}"""


# ---------------------------------------------------------------------------
# Prompt 构建器
# ---------------------------------------------------------------------------
def build_vlm_prompt_single(annotation: dict, global_id: int) -> str:
    """为单人轨迹构建 VLM prompt。"""
    persons = annotation.get("persons", [])
    person = None
    for p in persons:
        if p["global_id"] == global_id:
            person = p
            break

    if not person:
        return ""

    # 只包含 VLM 可见节点
    nodes_text_lines = []
    for node in person["nodes"]:
        if not node.get("vlm_visible"):
            continue
        line = f"{node['id']} {node['time']} {node['type']}"
        if node.get("duration_sec"):
            line += f" ({node['duration_sec']:.0f}s)"
        nodes_text_lines.append(line)

    nodes_text = "\n".join(nodes_text_lines) if nodes_text_lines else "(无关键节点)"
    return VLM_PROMPT_TEMPLATE.format(nodes_text=nodes_text)


def build_vlm_prompt_multi(annotation: dict) -> str:
    """为多人轨迹构建合并 VLM prompt（≤3 人时使用）。"""
    persons_text_parts = []
    for person in annotation.get("persons", []):
        gid = person["global_id"]
        color = person.get("color", "")
        nodes_lines = []
        for node in person["nodes"]:
            if not node.get("vlm_visible"):
                continue
            line = f"  {node['id']} {node['time']} {node['type']}"
            if node.get("duration_sec"):
                line += f" ({node['duration_sec']:.0f}s)"
            nodes_lines.append(line)
        nodes_text = "\n".join(nodes_lines) if nodes_lines else "  (无关键节点)"
        persons_text_parts.append(f"[人员#{gid} 颜色={color}]\n{nodes_text}")

    persons_text = "\n\n".join(persons_text_parts)
    return VLM_MULTI_PERSON_TEMPLATE.format(persons_text=persons_text)


def build_vlm_prompt(annotation: dict) -> str:
    """根据人数自动选择单人/多人 prompt。"""
    total = annotation.get("total_persons", 0)
    if total == 0:
        return ""
    if total == 1:
        gid = annotation["persons"][0]["global_id"]
        return build_vlm_prompt_single(annotation, gid)
    return build_vlm_prompt_multi(annotation)


# ---------------------------------------------------------------------------
# VLM 响应解析
# ---------------------------------------------------------------------------
def parse_vlm_response(response_text: str) -> dict:
    """解析 VLM 返回的 JSON。

    单人格式: {"labels": {"①": "进入", ...}, "summary": "..."}
    多人格式: {"persons": [{"global_id": 1, "labels": {...}, "summary": "..."}]}

    Returns
    -------
    dict : 解析后的标注数据。解析失败返回空 dict。
    """
    text = response_text.strip()
    # 尝试提取 JSON 块
    if "```" in text:
        # 提取 ```json ... ``` 之间的内容
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            text = text[start+3:end]
            # 去掉可能的 "json" 标记
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        # 尝试找第一个 { ... } 块
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end+1])
            except json.JSONDecodeError:
                pass
        logger.warning("VLM 响应解析失败: %s", text[:200])
        return {}


# ---------------------------------------------------------------------------
# 标注叠加（将 VLM 结果写入 annotation）
# ---------------------------------------------------------------------------
def apply_vlm_labels(annotation: dict, vlm_result: dict) -> dict:
    """将 VLM 生成的标签叠加回 annotation。

    修改 person["nodes"][i]["vlm_label"] 字段。
    """
    if not vlm_result:
        return annotation

    persons = annotation.get("persons", [])

    if "labels" in vlm_result:
        # 单人格式
        if persons:
            labels = vlm_result["labels"]
            for node in persons[0]["nodes"]:
                if node["id"] in labels:
                    node["vlm_label"] = labels[node["id"]]
            if "summary" in vlm_result:
                persons[0]["vlm_summary"] = vlm_result["summary"]

    elif "persons" in vlm_result:
        # 多人格式
        vlm_persons = {p["global_id"]: p for p in vlm_result["persons"]}
        for person in persons:
            gid = person["global_id"]
            if gid in vlm_persons:
                vlm_p = vlm_persons[gid]
                labels = vlm_p.get("labels", {})
                for node in person["nodes"]:
                    if node["id"] in labels:
                        node["vlm_label"] = labels[node["id"]]
                if "summary" in vlm_p:
                    person["vlm_summary"] = vlm_p["summary"]

    return annotation


# ---------------------------------------------------------------------------
# VLM 调用接口（需要外部提供实际的 VLM 调用函数）
# ---------------------------------------------------------------------------
class VLMAnnotator:
    """VLM 语义标注器。

    Parameters
    ----------
    call_fn : callable
        VLM 调用函数，签名: call_fn(prompt: str, image_bytes: bytes) -> str
        返回 VLM 的文本响应。
    max_persons_per_call : int
        单次 VLM 调用最多处理的人数（≤3 合并，>3 分批）。
    """

    def __init__(self, call_fn=None, max_persons_per_call: int = 3):
        self.call_fn = call_fn
        self.max_persons = max_persons_per_call

    def annotate(
        self,
        annotation: dict,
        image_bytes: bytes,
    ) -> dict:
        """发送标注请求并返回更新后的 annotation。

        Parameters
        ----------
        annotation : dict
            TrajectoryAnnotator.build_annotation() 的输出。
        image_bytes : bytes
            TrajectoryRenderer.render_to_bytes() 的输出（已画好路径+节点的底图）。

        Returns
        -------
        dict : 叠加了 VLM 标签的 annotation。
        """
        if not self.call_fn:
            logger.info("未配置 VLM 调用函数，跳过语义标注")
            return annotation

        total = annotation.get("total_persons", 0)
        if total == 0:
            return annotation

        if total <= self.max_persons:
            # 合并为一次调用
            prompt = build_vlm_prompt(annotation)
            logger.info("VLM 标注: %d 人合并调用 (prompt ~%d 字)", total, len(prompt))
            try:
                response = self.call_fn(prompt, image_bytes)
                result = parse_vlm_response(response)
                annotation = apply_vlm_labels(annotation, result)
            except Exception as exc:
                logger.error("VLM 标注调用失败: %s", exc)
        else:
            # 分批调用
            for i in range(0, total, self.max_persons):
                batch_ids = [
                    p["global_id"]
                    for p in annotation["persons"][i:i + self.max_persons]
                ]
                prompt = self._build_batch_prompt(annotation, batch_ids)
                logger.info("VLM 标注: 批次 %d~%d", i, i + len(batch_ids))
                try:
                    response = self.call_fn(prompt, image_bytes)
                    result = parse_vlm_response(response)
                    annotation = apply_vlm_labels(annotation, result)
                except Exception as exc:
                    logger.error("VLM 标注调用失败: %s", exc)

        return annotation

    def _build_batch_prompt(self, annotation: dict, global_ids: List[int]) -> str:
        """为一批人构建 prompt。"""
        # 筛选出目标人员
        filtered_persons = [
            p for p in annotation["persons"] if p["global_id"] in global_ids
        ]
        batch_annotation = dict(annotation)
        batch_annotation["persons"] = filtered_persons
        batch_annotation["total_persons"] = len(filtered_persons)
        return build_vlm_prompt(batch_annotation)


# ---------------------------------------------------------------------------
# 便利函数：生成完整的标注 + 渲染流程
# ---------------------------------------------------------------------------
def annotate_and_render(
    annotation: dict,
    renderer,
    output_dir: str,
    vlm_annotator: Optional[VLMAnnotator] = None,
) -> Tuple[str, dict]:
    """完整的标注+渲染流程。

    1. 渲染底图（代码画路径+节点+编号）
    2. VLM 语义标注（如果配置了）
    3. 重新渲染（叠加 VLM 标签）
    4. 保存最终图

    Returns
    -------
    (output_path, annotated_annotation)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # Step 1: 渲染底图
    base_path = os.path.join(output_dir, f"trajectory_{timestamp_str}_base.png")
    renderer.render(annotation, base_path)

    # Step 2: VLM 语义标注
    if vlm_annotator and vlm_annotator.call_fn:
        image_bytes = renderer.render_to_bytes(annotation)
        annotation = vlm_annotator.annotate(annotation, image_bytes)

    # Step 3: 最终渲染（如果有 VLM 标签，叠加上去）
    final_path = os.path.join(output_dir, f"trajectory_{timestamp_str}_final.png")
    renderer.render(annotation, final_path)

    # 保存 annotation JSON
    json_path = os.path.join(output_dir, f"trajectory_{timestamp_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)

    logger.info("轨迹标注完成: %s", final_path)
    return final_path, annotation
