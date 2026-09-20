"""把「带标注的检测画面」以 MJPEG 方式在浏览器 / VLC 里直播。

为什么需要它
------------
cv2.imshow 依赖本机桌面会话（远程桌面、服务会话、无显示器环境都可能弹不出窗口）。
这个内置的小 HTTP 服务不依赖 GUI，任何能播 MJPEG 的东西都能看：

    浏览器  http://127.0.0.1:8090
    VLC     http://127.0.0.1:8090

它只是把主循环渲染好的帧原样转发出去，自己不做任何检测或绘制。
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class _Handler(BaseHTTPRequestHandler):
    """multipart/x-mixed-replace 输出，即 MJPEG。"""

    def do_GET(self):  # noqa: N802
        if self.path not in ("/", "/stream", "/stream.mjpg"):
            self.send_error(404)
            return
        srv = getattr(self.server, "mjpeg", None)
        if srv is None:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--jpgboundary")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                frame = srv.get_frame()
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        data = buf.tobytes()
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data + b"\r\n")
                time.sleep(1.0 / max(1.0, srv.fps))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):  # 关掉默认访问日志，避免刷屏
        pass


class _Server(HTTPServer):
    """让 handler 能拿到 MjpegServer 实例（self.server.mjpeg）。"""

    def __init__(self, addr, handler, mjpeg: "MjpegServer"):
        super().__init__(addr, handler)
        self.mjpeg = mjpeg


class MjpegServer:
    """极简 MJPEG 直播服务（单文件、无第三方依赖）。"""

    def __init__(self, port: int = 8090, fps: float = 15.0):
        self.port = port
        self.fps = fps
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def start(self) -> str:
        self._httpd = _Server(("0.0.0.0", self.port), _Handler, self)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
