"""ONNX 轻量后端单元测试：letterbox / 输出解析 / 后端选择逻辑。

不依赖 torch：onnx_detector.py 只用 cv2 + numpy（与部署机最小依赖一致）。
推理部分（需要真实 yolo26n.onnx）在端到端验证中覆盖，见 test 末尾说明。

运行：
    python test_onnx_detector.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "yolo-rtsp-spatial-detector", "scripts"))

from onnx_detector import letterbox


class _FakeCfg:
    """与 DetectConfig 字段兼容的最小配置替身。"""

    conf_threshold = 0.5
    iou_threshold = 0.45
    device = ""
    input_size = 640


def test_letterbox_square_no_pad():
    # 正方形输入：无 padding，scale=1
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas, scale, px, py = letterbox(img, 640)
    assert scale == 1.0 and px == 0 and py == 0
    assert canvas.shape == (640, 640, 3)
    print("PASS test_letterbox_square_no_pad")


def test_letterbox_wide():
    # 1280x640 宽图 -> 640x640 画布：scale=0.5，左右各填 (640-640)//2=0，上下各填 (640-320)//2=160
    img = np.zeros((640, 1280, 3), dtype=np.uint8)
    canvas, scale, px, py = letterbox(img, 640)
    assert abs(scale - 0.5) < 1e-9
    assert px == 0 and py == 160
    assert canvas.shape == (640, 640, 3)
    # 顶部 padding 区应为填充色 114
    assert (canvas[:160] == 114).all()
    print("PASS test_letterbox_wide")


def test_letterbox_tall():
    # 480x960 高图 -> scale=640/960，左右各填 (640-320)//2=160，上下填 0
    img = np.zeros((960, 480, 3), dtype=np.uint8)
    canvas, scale, px, py = letterbox(img, 640)
    assert abs(scale - 2 / 3) < 1e-9
    assert px == 160 and py == 0
    assert (canvas[:, :160] == 114).all() and (canvas[:, 480:] == 114).all()
    print("PASS test_letterbox_tall")


def test_parse_e2e_output():
    # 端到端输出 (N, 6) 解析：坐标还原 + conf 过滤 + 标签回填
    from onnx_detector import OnnxDetector

    cfg = _FakeCfg()
    det = OnnxDetector.__new__(OnnxDetector)  # 不加载模型，只测 _parse
    det.cfg = cfg
    det.profile = _MiniProfile()

    # 模拟 letterbox 参数：scale=0.5, pad=(0, 160)
    # 原图 1280x640 -> 画布 640x640。原图 (100,200)-(300,400)
    # -> 画布 (50, 100+160=260)-(150, 200+160=360)
    dets = np.array([
        [50, 260, 150, 360, 0.9, 16],   # dog 0.9 -> 保留
        [60, 270, 140, 350, 0.3, 15],   # cat 0.3 -> conf < 0.5 被过滤
        [10, 10, 20, 20, 0.8, 0],       # person 0.8 -> 保留
    ])
    boxes = det._parse(dets, scale=0.5, pad_x=0, pad_y=160)
    assert len(boxes) == 2
    b = boxes[0]
    assert (b.cls, round(b.conf, 2)) == (16, 0.9)
    assert (round(b.x1), round(b.y1), round(b.x2), round(b.y2)) == (100, 200, 300, 400)

    # detect() = infer + 标签回填：stub 掉 infer 验证标签来自能力表
    det.infer = lambda frame: boxes
    labeled = det.detect(np.zeros((640, 1280, 3), dtype=np.uint8))
    assert labeled[0].label == "dog" and labeled[1].label == "person"
    print("PASS test_parse_e2e_output")


def test_parse_rejects_raw_output():
    # 传统 (1, 84, 8400) 原始输出应报错并提示重新导出
    from onnx_detector import OnnxDetector

    det = OnnxDetector.__new__(OnnxDetector)
    det.cfg = _FakeCfg()
    raw = np.zeros((84, 8400))
    try:
        det._parse(raw, 1.0, 0, 0)
    except ValueError as e:
        assert "export_onnx" in str(e)
        print("PASS test_parse_rejects_raw_output")
    else:
        raise AssertionError("raw (84,8400) output should raise ValueError")


def test_build_detector_routing():
    # main.build_detector 的后端路由：auto+.onnx -> OnnxDetector；显式 onnx 同样。
    # 路由测试不真加载模型文件：临时把 OnnxDetector.__init__ 替换为 no-op。
    import main as m
    from onnx_detector import OnnxDetector

    cfg = _FakeCfg()
    prof = _MiniProfile()
    orig_init = OnnxDetector.__init__
    OnnxDetector.__init__ = lambda self, cfg, profile=None: None
    try:
        cfg.model_path = "yolo26n.onnx"
        assert type(m.build_detector({"backend": "auto"}, cfg, prof)).__name__ == "OnnxDetector"
        cfg.model_path = "yolo26n.pt"
        assert type(m.build_detector({"backend": "onnx"}, cfg, prof)).__name__ == "OnnxDetector"
    finally:
        OnnxDetector.__init__ = orig_init
    print("PASS test_build_detector_routing (onnx routes)")


class _MiniProfile:
    """只提供 id_to_name 的最小能力表替身。"""

    name = "mini"

    def id_to_name(self, idx):
        return {0: "person", 15: "cat", 16: "dog"}.get(idx, f"class_{idx}")


if __name__ == "__main__":
    test_letterbox_square_no_pad()
    test_letterbox_wide()
    test_letterbox_tall()
    test_parse_e2e_output()
    test_parse_rejects_raw_output()
    test_build_detector_routing()
    print("\nALL ONNX DETECTOR TESTS PASSED")
