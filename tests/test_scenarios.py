"""能力映射 + 场景规则单元测试：验证「模型能认什么」与「按场景判定」不写死。

运行：
    python test_scenarios.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from geometry import Box
from model_catalog import (
    COCO_PROFILE,
    ModelProfile,
    build_model_profile,
    resolve_class,
)
from scenario import build_scenarios, evaluate_scenarios, resolve_scenario


def test_coco_capability_map():
    # COCO 能力表应包含常见类，且能用类名反查 id
    assert COCO_PROFILE.name_to_id("cat") == 15
    assert COCO_PROFILE.name_to_id("dog") == 16
    assert COCO_PROFILE.name_to_id("dining table") == 60
    assert COCO_PROFILE.name_to_id("couch") == 57
    assert COCO_PROFILE.id_to_name(0) == "person"
    assert len(COCO_PROFILE.class_names) == 80
    print("PASS test_coco_capability_map")


def test_resolve_class_mixed():
    # 类名 / id / 列表混合引用都能解析
    assert resolve_class(["cat", "dog"], COCO_PROFILE) == [15, 16]
    assert resolve_class("dining table", COCO_PROFILE) == [60]
    assert resolve_class(15, COCO_PROFILE) == [15]
    assert resolve_class("15", COCO_PROFILE) == [15]
    print("PASS test_resolve_class_mixed")


def test_custom_model_profile():
    # 自定义模型能力表：完全不依赖 COCO，验证可插拔
    custom = ModelProfile("my_model", ["background", "table", "cat", "dog"])
    assert custom.name_to_id("cat") == 2
    assert resolve_class(["cat", "table"], custom) == [2, 1]
    print("PASS test_custom_model_profile")


def test_build_model_profile_from_settings():
    settings = {"model": {"class_names": ["person", "cat", "sofa"]}}
    prof = build_model_profile(settings)
    assert prof.name_to_id("sofa") == 2
    # 默认回退 coco
    assert build_model_profile({}).name_to_id("cat") == 15
    print("PASS test_build_model_profile_from_settings")


def test_build_scenarios_mixed():
    settings = {
        "scenarios": [
            {"name": "a", "subject": ["cat", "dog"], "surface": "dining table", "relationship": "on_top"},
            {"name": "b", "subject": "person", "surface": "chair", "relationship": "within"},
            {"name": "bad", "subject": "unicorn", "surface": "table", "relationship": "on_top"},
        ]
    }
    scs = build_scenarios(settings, COCO_PROFILE)
    names = {s.name for s in scs}
    # unicorn 不在能力表 -> 被跳过
    assert names == {"a", "b"}, names
    a = next(s for s in scs if s.name == "a")
    assert a.subject_ids == {15, 16} and a.surface_ids == {60}
    print("PASS test_build_scenarios_mixed")


def test_evaluate_multi_scenarios():
    # 用 COCO 能力表解析：cat=15, dog=16, dining table=60, couch=57, person=0, chair=56
    settings = {
        "scenarios": [
            {"name": "pet_on_table", "subject": ["cat", "dog"], "surface": "dining table", "relationship": "on_top"},
            {"name": "cat_on_sofa", "subject": "cat", "surface": "couch", "relationship": "on_top"},
        ]
    }
    prof = COCO_PROFILE
    scs = build_scenarios(settings, prof)

    table = Box(100, 100, 400, 400, 0.9, 60, "dining table")
    cat_on = Box(200, 120, 300, 250, 0.8, 15, "cat")
    dog_floor = Box(200, 450, 300, 600, 0.8, 16, "dog")
    sofa = Box(500, 100, 800, 400, 0.9, 57, "couch")
    # 沙发上的猫：底部中心点 (650,260) 落在沙发表面区 x[500,800] y[100,280] 内
    cat_on_sofa = Box(600, 150, 700, 260, 0.8, 15, "cat")

    boxes = [table, cat_on, dog_floor, sofa, cat_on_sofa]
    rels = evaluate_scenarios(boxes, scs)

    # 每个主体产出一条关系；按场景聚合成列表判断是否「存在」命中
    by_sc: "dict[str, list]" = {}
    for r in rels:
        by_sc.setdefault(r.scenario, []).append(r)

    # pet_on_table: 至少一条 cat/dog 在桌上(on=True)，且主体标签是 cat
    pet_hits = [r for r in by_sc["pet_on_table"] if r.on]
    assert pet_hits and pet_hits[0].subject.label == "cat", by_sc["pet_on_table"]
    # dog 不在桌上 -> 存在 on=False 的诊断项
    assert any(not r.on for r in by_sc["pet_on_table"])

    # cat_on_sofa: 存在猫在沙发上(on=True)，承载面标签为 COCO 类名 "couch"
    sofa_hits = [r for r in by_sc["cat_on_sofa"] if r.on]
    assert sofa_hits and sofa_hits[0].surface.label == "couch", by_sc["cat_on_sofa"]
    print("PASS test_evaluate_multi_scenarios")


def test_scenario_unknown_relationship_skipped():
    settings = {"scenarios": [
        {"name": "x", "subject": "cat", "surface": "table", "relationship": "floating"}
    ]}
    scs = build_scenarios(settings, COCO_PROFILE)
    assert scs == []
    print("PASS test_scenario_unknown_relationship_skipped")


if __name__ == "__main__":
    test_coco_capability_map()
    test_resolve_class_mixed()
    test_custom_model_profile()
    test_build_model_profile_from_settings()
    test_build_scenarios_mixed()
    test_evaluate_multi_scenarios()
    test_scenario_unknown_relationship_skipped()
    print("\nALL SCENARIO/CAPABILITY TESTS PASSED ✅")
