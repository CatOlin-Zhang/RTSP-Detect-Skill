"""主流程 / CLI：支持两种模式。

模式 1 — spatial（空间关系检测，原版功能）
  python main.py --mode spatial --rtsp "rtsp://..."

模式 2 — tracking（多摄像头人员追踪）
  python main.py --mode tracking

通用用法
--------
  python main.py                          # 用 settings.json 里的配置跑
  python main.py --source 0               # 本地摄像头调试
  python main.py --source demo.mp4        # 视频文件调试
  python main.py --show                   # 显示预览窗口
  python main.py --list-classes           # 打印模型能力表
  python main.py --frame-skip 2           # 跳帧

可调参数写在 settings.json 里；命令行参数会覆盖对应的 settings 项。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict

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
from model_catalog import build_model_profile, COCO_PROFILE
from scenario import build_scenarios
from hud import HudState, draw_hud
from vision_review import VisionReviewer, ReviewConfig
from preview import MjpegServer

# Tracking mode imports
from tracking_models import CameraConfig, CameraTrack, Topology, TopologyLink
from reid_extractor import ReIDExtractor, crop_person
from cross_camera import CrossCameraAssociator, AssociationConfig
from multi_camera import MultiCameraManager

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


# ============================================================================
# Tracking 模式：多摄像头人员追踪
# ============================================================================
def _parse_topology(settings: dict) -> Topology:
    """从 settings.json 解析摄像头配置和拓扑关系。"""
    cameras = {}
    for c in settings.get("cameras", []):
        cam = CameraConfig(
            id=c["id"],
            name=c.get("name", c["id"]),
            source=c["source"],
            description=c.get("description", ""),
        )
        cameras[cam.id] = cam

    links = []
    for l in settings.get("topology", {}).get("links", []):
        link = TopologyLink(
            from_cam=l["from"],
            to_cam=l["to"],
            transit_sec=tuple(l.get("transit_sec", [0, 30])),
            overlap=l.get("overlap", False),
            bidirectional=l.get("bidirectional", True),
        )
        links.append(link)

    return Topology(cameras=cameras, links=links)


def run_tracking(settings: dict, show: bool = False, mjpeg_port: int = 0) -> None:
    """多摄像头人员追踪主循环。

    流程：多路并行读帧 -> YOLO track() -> OSNet 特征提取 -> 跨摄关联 -> 轨迹输出。
    """
    # 解析配置
    topology = _parse_topology(settings)
    if not topology.cameras:
        logger.error("未配置摄像头，请在 settings.json 的 cameras 数组中配置。")
        return

    tracking_cfg = settings.get("tracking", {})
    track_classes_names = tracking_cfg.get("track_classes", ["person"])
    # 把类名转为 class id（基于 COCO 能力表）
    profile = build_model_profile(settings)
    track_class_ids = set()
    for name in track_classes_names:
        try:
            from model_catalog import resolve_class
            track_class_ids.update(resolve_class(name, profile))
        except Exception:
            logger.warning("无法解析跟踪类别: %s", name)
    if not track_class_ids:
        logger.error("没有有效的跟踪类别。")
        return

    logger.info("跟踪类别: %s (ids=%s)", track_classes_names, track_class_ids)
    logger.info("摄像头数: %d | 拓扑链接数: %d", len(topology.cameras), len(topology.links))
    for cam in topology.cameras.values():
        logger.info("  [%s] %s -> %s", cam.id, cam.name, cam.source)

    # 初始化组件
    det_cfg = DetectConfig(
        model_path=settings.get("model_path", "yolo26n.pt"),
        conf_threshold=float(settings.get("conf_threshold", 0.25)),
        iou_threshold=float(settings.get("iou_threshold", 0.45)),
        device=settings.get("device", ""),
        tracker=tracking_cfg.get("tracker", "botsort.yaml"),
    )
    detector = YoloDetector(det_cfg, profile=profile)

    reid = ReIDExtractor(
        model_name="osnet_x1_0",
        device=settings.get("device", ""),
    )

    assoc_cfg = AssociationConfig(
        reid_threshold=float(tracking_cfg.get("reid_threshold", 0.45)),
        gallery_max_age_sec=float(tracking_cfg.get("gallery_max_age_sec", 120)),
    )
    associator = CrossCameraAssociator(topology, assoc_cfg)

    cam_mgr = MultiCameraManager(
        cameras=list(topology.cameras.values()),
        buffer_size=int(settings.get("buffer_size", 2)),
        frame_skip=int(settings.get("frame_skip", 0)),
    )

    # 预热模型
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.track(dummy)
        reid.extract(dummy[100:300, 200:280])  # 模拟人员裁剪
        logger.info("模型预热完成")
    except Exception as exc:
        logger.warning("模型预热跳过: %s", exc)

    # 可选 MJPEG 预览
    mjpeg = None
    if mjpeg_port:
        mjpeg = MjpegServer(port=mjpeg_port)
        logger.info("MJPEG 预览: %s", mjpeg.start())

    # 每摄像头的轨迹跟踪状态: {cam_id: {track_id: CameraTrack}}
    active_tracks: Dict[str, Dict[int, CameraTrack]] = {
        cam_id: {} for cam_id in topology.cameras
    }
    # 每摄像头最后处理的帧号（避免重复处理）
    last_frame_idx: Dict[str, int] = {cam_id: 0 for cam_id in topology.cameras}
    # 轨迹消失判定: 连续 N 帧未出现则认为消失
    LOST_THRESHOLD_SEC = 3.0

    cam_mgr.start_all()
    fps_t0, fps_count = time.time(), 0

    try:
        while True:
            now = time.time()
            processed_any = False

            for cam_id, frame, ts, frame_idx in cam_mgr.iter_cameras():
                # 跳过已处理的帧
                if frame_idx <= last_frame_idx[cam_id]:
                    continue
                last_frame_idx[cam_id] = frame_idx
                processed_any = True

                # FPS 计算
                fps_count += 1
                if now - fps_t0 >= 1.0:
                    fps = fps_count / (now - fps_t0)
                    fps_t0, fps_count = now, 0
                    active_persons = associator.get_active_persons()
                    logger.info(
                        "FPS=%.1f | persons=%d | gallery=%d | %s",
                        fps, len(active_persons), associator._gallery.size, cam_id,
                    )

                # 1. YOLO 跟踪
                tracked_boxes = detector.track(frame)

                # 2. 过滤目标类别
                person_boxes = [b for b in tracked_boxes if b.cls in track_class_ids]

                # 3. 检测消失的轨迹
                current_track_ids = {b.track_id for b in person_boxes}
                lost_ids = set(active_tracks[cam_id].keys()) - current_track_ids
                for tid in lost_ids:
                    lost_track = active_tracks[cam_id].pop(tid)
                    if (now - lost_track.last_seen) >= LOST_THRESHOLD_SEC:
                        associator.on_track_lost(lost_track)
                        logger.debug(
                            "轨迹消失: cam=%s tid=%d gid=%d",
                            cam_id, tid, lost_track.global_id,
                        )

                # 4. 裁剪人员 + 提取特征
                crops = []
                for b in person_boxes:
                    crop = crop_person(frame, b.x1, b.y1, b.x2, b.y2)
                    crops.append(crop)

                features = reid.extract_batch(crops) if crops else []

                # 5. 更新轨迹 + 跨摄关联
                # 收集其他摄像头的活跃特征（用于重叠视野匹配）
                other_active = {}
                for other_cam, other_tracks in active_tracks.items():
                    if other_cam == cam_id:
                        continue
                    other_feats = []
                    for otid, ot in other_tracks.items():
                        if ot.feature is not None:
                            other_feats.append((otid, ot.feature))
                    if other_feats:
                        other_active[other_cam] = other_feats

                for box, feat in zip(person_boxes, features):
                    if feat is None:
                        continue

                    # 获取或创建轨迹
                    if box.track_id not in active_tracks[cam_id]:
                        active_tracks[cam_id][box.track_id] = CameraTrack(
                            track_id=box.track_id,
                            camera_id=cam_id,
                            first_seen=now,
                            last_seen=now,
                        )
                    track = active_tracks[cam_id][box.track_id]
                    track.update((box.x1, box.y1, box.x2, box.y2), now)

                    # 跨摄关联
                    result = associator.associate(
                        track, feat, active_tracks_in_other_cams=other_active,
                    )
                    if result.matched and result.match_source == "new":
                        logger.info(
                            "新人员 global_id=%d 首次出现在 %s",
                            result.global_id, cam_id,
                        )
                    elif result.matched and result.match_source in ("gallery", "neighbor", "overlap"):
                        logger.info(
                            "跨摄匹配: cam=%s tid=%d -> gid=%d (%s, sim=%.3f)",
                            cam_id, box.track_id, result.global_id,
                            result.match_source, result.similarity,
                        )

                # 6. 可视化（画跟踪框 + global_id）
                if show or mjpeg:
                    vis = frame.copy()
                    for box in person_boxes:
                        track = active_tracks[cam_id].get(box.track_id)
                        gid = track.global_id if track else -1
                        color = (0, 255, 0) if gid >= 0 else (0, 255, 255)
                        cv2.rectangle(vis, (int(box.x1), int(box.y1)),
                                      (int(box.x2), int(box.y2)), color, 2)
                        label = f"G{gid} T{box.track_id}" if gid >= 0 else f"T{box.track_id}"
                        cv2.putText(vis, label, (int(box.x1), max(0, int(box.y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    # 摄像头名称 + FPS
                    cv2.putText(vis, f"{cam_id} | FPS={fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    if mjpeg:
                        mjpeg.update(vis)
                    if show:
                        try:
                            cv2.imshow(f"tracker-{cam_id}", vis)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                logger.info("用户中断")
                                return
                        except cv2.error:
                            show = False

            # 没有新帧时小睡一下，避免空转吃 CPU
            if not processed_any:
                time.sleep(0.01)

    finally:
        cam_mgr.stop_all()
        if mjpeg:
            mjpeg.stop()
        if show:
            cv2.destroyAllWindows()
        # 输出最终轨迹报告
        logger.info("\n%s", associator.trajectory_report())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RTSP 空间检测 / 多摄像头人员追踪")
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="settings.json 路径")
    p.add_argument("--mode", default=None, help="运行模式: spatial(空间检测) | tracking(人员追踪)")
    p.add_argument("--source", default=None, help="覆盖视频源: rtsp://... | 摄像头索引 | 视频文件")
    p.add_argument("--rtsp", default=None, help="覆盖 RTSP 地址（仅 spatial 模式）")
    p.add_argument("--frame-skip", type=int, default=None, help="每 N 帧处理 1 帧 (0=不跳)")
    p.add_argument("--model", default=None, help="覆盖模型路径，如 yolo26n.pt")
    p.add_argument("--surface-ratio", type=float, default=None, help="on_top 场景表面区域占比默认值 0~1（仅 spatial）")
    p.add_argument("--conf", type=float, default=None, help="置信度阈值")
    p.add_argument("--cooldown", type=float, default=None, help="两次截图最小间隔(秒)（仅 spatial）")
    p.add_argument("--list-classes", action="store_true", help="打印当前模型能力表后退出")
    p.add_argument("--show", action="store_true", help="显示 OpenCV 预览窗口")
    p.add_argument("--mjpeg-port", type=int, default=None,
                   help="MJPEG 直播端口(0/不传=关闭)")
    p.add_argument("--alert-hold", type=float, default=None, help="触发后告警 HUD 保留秒数（仅 spatial）")
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
    if args.mode is not None:
        settings["mode"] = args.mode
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

    mode = settings.get("mode", "spatial")

    if mode == "tracking":
        logger.info("=== 多摄像头人员追踪模式 ===")
        try:
            run_tracking(settings, show=args.show, mjpeg_port=args.mjpeg_port or 0)
        except KeyboardInterrupt:
            logger.info("已停止")
        return 0

    # spatial 模式（原版）
    if not settings.get("rtsp_url"):
        logger.error("未配置视频源：请在 settings.json 设置 rtsp_url，或用 --source/--rtsp 指定。")
        return 2

    logger.info("=== 空间关系检测模式 ===")
    logger.info("视频源: %s | 模型: %s", settings.get("rtsp_url"), settings.get("model_path", "yolov8n.pt"))
    try:
        run(settings, show=args.show, mjpeg_port=args.mjpeg_port or 0)
    except KeyboardInterrupt:
        logger.info("已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
