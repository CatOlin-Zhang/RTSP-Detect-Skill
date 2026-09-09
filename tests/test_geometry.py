"""纯几何逻辑单元测试：不依赖 cv2 / numpy / torch，用合成边界框验证判定。

运行：
    python test_geometry.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from geometry import Box, is_on_top, surface_zone, is_within, SpatialRelation


def _surface():
    # 承载面框: x[100,400], y[100,400] -> 高 300
    return Box(100, 100, 400, 400, 0.9, 60, "dining table")


def test_surface_zone():
    t = _surface()
    z = surface_zone(t, surface_ratio=0.6)
    assert z.y1 == 100 and z.y2 == 280, (z.y1, z.y2)
    z2 = surface_zone(t, surface_ratio=0.6, margin_x_ratio=0.1)
    assert z2.x1 == 130 and z2.x2 == 370, (z2.x1, z2.x2)
    print("PASS test_surface_zone")


def test_on_top_true():
    s = _surface()
    sub_on = Box(200, 120, 300, 250, 0.8, 15, "cat")
    assert is_on_top(sub_on, s) is True
    print("PASS test_on_top_true")


def test_next_to_false():
    s = _surface()
    sub_floor = Box(200, 450, 300, 600, 0.8, 15, "cat")
    assert is_on_top(sub_floor, s) is False
    print("PASS test_next_to_false")


def test_x_outside_false():
    s = _surface()
    sub_off = Box(500, 100, 600, 250, 0.8, 15, "cat")
    assert is_on_top(sub_off, s) is False
    print("PASS test_x_outside_false")


def test_y_too_high_false():
    s = _surface()
    sub_high = Box(200, 0, 300, 80, 0.8, 15, "cat")
    assert is_on_top(sub_high, s) is False
    print("PASS test_y_too_high_false")


def test_within_true_false():
    s = _surface()
    inside = Box(200, 200, 300, 300, 0.8, 0, "person")
    assert is_within(inside, s, margin_x_ratio=0.2, margin_y_ratio=0.2) is True
    outside = Box(50, 50, 90, 90, 0.8, 0, "person")
    assert is_within(outside, s) is False
    print("PASS test_within_true_false")


def test_relation_dataclass():
    s = _surface()
    sub = Box(200, 120, 300, 250, 0.8, 15, "cat")
    r = SpatialRelation("pet_on_table", "on_top", sub, s, True, 0.1, 0.6, 0.0)
    assert r.scenario == "pet_on_table" and r.on is True
    print("PASS test_relation_dataclass")


if __name__ == "__main__":
    test_surface_zone()
    test_on_top_true()
    test_next_to_false()
    test_x_outside_false()
    test_y_too_high_false()
    test_within_true_false()
    test_relation_dataclass()
    print("\nALL GEOMETRY TESTS PASSED ✅")
