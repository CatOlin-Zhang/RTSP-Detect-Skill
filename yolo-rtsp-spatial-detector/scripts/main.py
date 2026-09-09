"""主流程 / CLI：接入 RTSP -> 逐帧推理 -> 按场景判定空间关系 -> 触发截图推下游。

用法示例
--------
  # 用 settings.json 里的 rtsp_url 跑
  python main.py

  # 临时覆盖 RTSP 地址
  python main.py --rtsp "rtsp://user:pass@192.168.1.64:554/stream"

  # 用本地摄像头(索引0)或视频文件调试
  python main.py --source 0
  python main.py --source demo.mp4

  # 跳帧（每 3 帧处理 1 帧，提升实时性）
  python main.py --frame-skip 2

  # 显示预览窗口
  python main.py --show

  # 打印当前模型的能力表（能认出哪些类），便于配置 scenario
  python main.py --list-classes

  # 临时把所有 on_top 场景的表面区域比例调成 0.5
  python main.py --surface-ratio 0.5

可调参数都写在 settings.json 里；命令行参数会覆盖对应的 settings 项。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from spatial_detector import DetectConfig, YoloDetector, draw_boxes, draw_relation
from downstream import (
    TriggerEvent,
    FileSink,
    HttpSink,
    CallableSink,
    CompositeSink,
)
from model_catalog import build_model_profile
from scenario import build_scenarios
from hud import HudState, draw_hud
from vision_review import VisionReviewer, ReviewConfig
from preview import MjpegServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS = os.path.join(HERE, "settings.json")


def load_settings(path: str) -> dict:
    """读取 settings.json；失败返回空 dict（命令行/默认值兜底）。"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def open_capture(source, buffer_size: int = 2):
    """打开视频源。RTSP 建议把 CAP_PROP_BUFFERSIZE 调小，避免帧堆积导致高延迟。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def run(settings: dict, show: bool = False, mjpeg_port: int = 0) -> None:
    source = settings.get("rtsp_url", "")
    frame_skip = int(settings.get("frame_skip", 0))
    save_cooldown = float(settings.get("save_cooldown", 10.0))
    buffer_size = int(settings.get("buffer_size", 2))
    output_dir = settings.get("output_dir", os.path.join(HERE, "captures"))
    default_surface_ratio = settings.get("surface_ratio")  # settings 顶层也可给默认
    alert_hold = float(settings.get("alert_hold", 3.0))  # 触发后告警在 HUD 上保留的秒数

    profile = build_model_profile(settings)
    scenarios = build_scenarios(settings, profile, default_surface_ratio=default_surface_ratio)
    if not scenarios:
        logger.error("没有任何可用的 scenario，请检查 settings.json 的 scenarios 配置。")
        return

    logger.info("模型能力表: %s（共 %d 类）", profile.name, len(profile.class_names))
    for sc in scenarios:
        logger.info(
            "已加载场景 [%s]: %s -> %s (%s)",
            sc.name, sc.subject_label(profile), sc.surface_label(profile), sc.relationship,
        )

    cfg = DetectConfig(
        model_path=settings.get("model_path", "yolov8n.pt"),
        conf_threshold=float(settings.get("conf_threshold", 0.30)),
        iou_threshold=float(settings.get("iou_threshold", 0.45)),
        device=settings.get("device", ""),
    )
    detector = YoloDetector(cfg, profile=profile)

    # 预热：用一张空帧跑一次，避免首帧推理拖慢实时流
    try:
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    except Exception as exc:
        logger.warning("模型预热跳过: %s", exc)

    # 组装下游 Sink：文件落盘（带标注）始终开启；若配置了 HTTP 地址则追加推送
    composite = CompositeSink()
    composite.add(FileSink(output_dir, save_annotated=True, draw_fn=draw_relation))
    http_url = settings.get("downstream_http_url", "")
    if http_url:
        composite.add(HttpSink(http_url))

    # 截图队列：粗检触发即落盘到 pending/，等 Agent 读取做精判
    reviewer = VisionReviewer(output_dir, ReviewConfig.from_settings(settings))

    # 可选的 MJPEG 直播（浏览器 / VLC 都能看带标注的画面）
    mjpeg = None
    if mjpeg_port:
        mjpeg = MjpegServer(port=mjpeg_port)
        logger.info("MJPEG 预览地址（带 HUD 的检测画面）: %s", mjpeg.start())

    cap = open_capture(source, buffer_size)
    last_save: "dict[str, float]" = defaultdict(float)
    frame_idx = 0
    reconnect_delay = 1.0

    hud = HudState(scenario_count=len(scenarios))
    fps_t0, fps_count = time.time(), 0
    last_alert_at = 0.0
    total_triggers = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("读取失败，尝试重连 RTSP ...")
                hud.streaming = False
                cap.release()
                time.sleep(reconnect_delay)
                cap = open_capture(source, buffer_size)
                hud.streaming = True
                continue

            frame_idx += 1
            fps_count += 1
            if frame_skip and frame_idx % (frame_skip + 1) != 0:
                continue

            now = time.time()
            if now - fps_t0 >= 0.5:
                hud.fps = fps_count / (now - fps_t0)
                fps_t0, fps_count = now, 0

            boxes = detector.detect(frame)
            relations = detector.detect_relations(boxes, scenarios)
            triggered = [r for r in relations if r.on]

            # 每个场景取置信度最高的触发主体，并按该场景冷却去抖
            best_per_scenario: "dict[str, SpatialRelation]" = {}
            for r in triggered:
                cur = best_per_scenario.get(r.scenario)
                if cur is None or r.subject.conf > cur.subject.conf:
                    best_per_scenario[r.scenario] = r

            for sc_name, rel in best_per_scenario.items():
                cd = _cooldown_for(sc_name, scenarios, save_cooldown)
                if now - last_save[sc_name] >= cd:
                    last_save[sc_name] = now
                    total_triggers += 1
                    event = TriggerEvent(relation=rel, frame=frame.copy())
                    composite.emit(event)
                    logger.info(
                        "触发[%s]: %s 在 %s 上 (conf=%.2f)",
                        sc_name, rel.subject.label, rel.surface.label, rel.subject.conf,
                    )

                    # 带标注的帧（红框主体 / 蓝框承载面）存入 pending/ 供 Agent 精判
                    try:
                        ann = draw_boxes(draw_relation(frame.copy(), rel), boxes)
                    except Exception:
                        ann = frame.copy()
                    reviewer.submit(
                        ann,
                        {
                            "scenario": sc_name,
                            "relationship": rel.relationship,
                            "subject": rel.subject.label,
                            "subject_conf": round(rel.subject.conf, 3),
                            "surface": rel.surface.label,
                            "suspicious": True,
                        },
                    )
                    hud.last_capture = os.path.basename(event.image_path or "")

            # 命中后告警在 HUD 上保留 alert_hold 秒，避免一闪而过看不清
            if triggered:
                last_alert_at = now
                top = max(best_per_scenario.values(), key=lambda r: r.subject.conf)
                hud.alert = True
                hud.scenario = top.scenario
                hud.subject = top.subject.label
                hud.surface = top.surface.label
                hud.conf = top.subject.conf
            hud.alert = (time.time() - last_alert_at) < alert_hold
            hud.total_triggers = total_triggers
            hud.llm_status = reviewer.poll_status()

            if show or mjpeg:
                vis = draw_boxes(frame, boxes)
                for r in triggered:
                    vis = draw_relation(vis, r)
                vis = draw_hud(vis, hud)  # 左上角状态提示
                if mjpeg:
                    mjpeg.update(vis)
                if show:
                    try:
                        cv2.imshow("spatial_detector", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("用户中断")
                            break
                    except cv2.error as exc:
                        # 无桌面会话（服务/远程上下文）时 imshow 会抛错：降级只走 MJPEG
                        logger.warning("无法打开预览窗口，已自动改用 MJPEG 输出: %s", exc)
                        show = False
    finally:
        cap.release()
        if mjpeg:
            mjpeg.stop()
        if show:
            cv2.destroyAllWindows()


def _cooldown_for(scenario_name: str, scenarios, global_cd: float) -> float:
    for sc in scenarios:
        if sc.name == scenario_name:
            return sc.cooldown if sc.cooldown > 0 else global_cd
    return global_cd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RTSP 通用空间关系检测（基于模型能力表 + 场景规则）")
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="settings.json 路径")
    p.add_argument("--source", default=None, help="覆盖视频源: rtsp://... | 摄像头索引 | 视频文件")
    p.add_argument("--rtsp", default=None, help="覆盖 RTSP 地址")
    p.add_argument("--frame-skip", type=int, default=None, help="每 N 帧处理 1 帧 (0=不跳)")
    p.add_argument("--model", default=None, help="覆盖模型路径，如 yolo26n.pt")
    p.add_argument("--surface-ratio", type=float, default=None, help="on_top 场景表面区域占比默认值 0~1")
    p.add_argument("--conf", type=float, default=None, help="置信度阈值")
    p.add_argument("--cooldown", type=float, default=None, help="两次截图/送检最小间隔(秒)，留空则用 settings.json 的 save_cooldown(默认10秒)")
    p.add_argument("--list-classes", action="store_true", help="打印当前模型能力表后退出")
    p.add_argument("--show", action="store_true", help="显示 OpenCV 预览窗口（左上角带 HUD）")
    p.add_argument("--mjpeg-port", type=int, default=None,
                   help="MJPEG 直播端口(0/不传=关闭)；浏览器或 VLC 打开 http://127.0.0.1:PORT")
    p.add_argument("--alert-hold", type=float, default=None, help="触发后告警在 HUD 上保留的秒数")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.settings)

    # 仅打印能力表
    if args.list_classes:
        profile = build_model_profile(settings)
        print(f"模型能力表: {profile.name}（共 {len(profile.class_names)} 类）")
        for i, name in enumerate(profile.class_names):
            print(f"  {i:>2}  {name}")
        return 0

    # 命令行覆盖 settings
    if args.source is not None:
        settings["rtsp_url"] = args.source
    if args.rtsp is not None:
        settings["rtsp_url"] = args.rtsp
    if args.frame_skip is not None:
        settings["frame_skip"] = args.frame_skip
    if args.model is not None:
        settings["model_path"] = args.model
    if args.surface_ratio is not None:
        settings["surface_ratio"] = args.surface_ratio
    if args.conf is not None:
        settings["conf_threshold"] = args.conf
    if args.cooldown is not None:
        settings["save_cooldown"] = args.cooldown
    if args.alert_hold is not None:
        settings["alert_hold"] = args.alert_hold

    if not settings.get("rtsp_url"):
        logger.error("未配置视频源：请在 settings.json 设置 rtsp_url，或用 --source/--rtsp 指定。")
        return 2

    logger.info("视频源: %s | 模型: %s", settings.get("rtsp_url"), settings.get("model_path", "yolov8n.pt"))
    try:
        run(settings, show=args.show, mjpeg_port=args.mjpeg_port or 0)
    except KeyboardInterrupt:
        logger.info("已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
