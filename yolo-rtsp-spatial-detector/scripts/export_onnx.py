"""把 .pt 权重导出为 ONNX —— 一次性操作，只需在有 torch/ultralytics 的开发机上执行。

为什么需要它
------------
本 Skill 默认走 ultralytics 推理（依赖 torch ~200MB 下载 / >1GB 磁盘）。
把模型导出成 ONNX 后，部署机可以用 onnxruntime 直接加载推理：

  * 部署机只需要 opencv-python + numpy + onnxruntime（合计下载约 70MB），
    不再需要 torch/ultralytics。
  * 导出时固定输入尺寸（默认 640x640），部署端预处理最简单。

注意：不要试图用 cv2.dnn 跑这个产物——YOLO26 端到端头里的 TopK /
GatherElements 等算子在 cv2.dnn 里 forward 数值静默错误（实测）。

用法
----
  python export_onnx.py                     # yolo26n.pt -> yolo26n.onnx
  python export_onnx.py yolo26n.pt          # 指定权重
  python export_onnx.py --imgsz 480         # 更小的输入尺寸（更快，精度略降）
  python export_onnx.py --opset 12          # 兼容老版推理引擎（默认 12）

导出完成后，把 .onnx 文件随 Skill 一起分发，settings.json 里
model_path 指向 .onnx（或 backend 设为 "onnx"）即可。
"""
from __future__ import annotations

import argparse
import os

# 必须在 import torch 之前设置：Anaconda 环境 numpy 与 torch 各带一份 OpenMP
# 运行时（libiomp5md.dll），不设会在加载时 OMP Error #15 崩溃。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def export(pt_path: str, imgsz: int, opset: int) -> str:
    """执行导出，返回产物路径。"""
    from ultralytics import YOLO  # 仅开发机需要；部署机不 import 本模块

    model = YOLO(pt_path)
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,   # onnxslim 图简化，产物更小更快
        dynamic=False,   # 固定输入尺寸：部署端 cv2.dnn 加载最稳
        half=False,      # cv2.dnn 对 fp16 支持有限，保持 fp32
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"ONNX 已导出: {path} ({size_mb:.1f} MB, imgsz={imgsz}, opset={opset})")
    return path


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="把 .pt 权重导出为 ONNX（开发机一次性操作）")
    p.add_argument("model", nargs="?", default=os.path.join(here, "yolo26n.pt"),
                   help="输入 .pt 权重路径（默认 yolo26n.pt）")
    p.add_argument("--imgsz", type=int, default=640, help="固定输入尺寸（默认 640）")
    p.add_argument("--opset", type=int, default=12,
                   help="ONNX opset 版本（默认 12，兼容大多数 cv2.dnn）")
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"找不到权重文件: {args.model}")
        return 2
    export(args.model, args.imgsz, args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
