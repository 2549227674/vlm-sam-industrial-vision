"""MJPEG HTTP Server for streaming video frames to browser."""

import threading
import time
from typing import Optional, Callable
from flask import Flask, Response
import logging

# 禁用 Flask 默认日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 全局变量
_mjpeg_server_instance: Optional['MJPEGServer'] = None
_server_thread: Optional[threading.Thread] = None


class MJPEGServer:
    """MJPEG HTTP Server for browser video streaming."""

    def __init__(self, port: int = 8889):
        self.port = port
        self.app = Flask(__name__)
        self.running = False
        self._frame_getter: Optional[Callable[[], Optional[bytes]]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

        # 设置路由
        self.app.add_url_rule('/video_feed', 'video_feed', self._video_feed)
        self.app.add_url_rule('/health', 'health', self._health)
        self.app.add_url_rule('/', 'index', self._index)

    def set_frame_getter(self, getter: Callable[[], Optional[bytes]]):
        """设置帧获取函数"""
        self._frame_getter = getter

    def update_frame(self, jpeg_data: bytes):
        """更新最新帧（线程安全）"""
        with self._lock:
            self._latest_jpeg = jpeg_data

    def _get_frame(self) -> Optional[bytes]:
        """获取最新帧"""
        if self._frame_getter:
            return self._frame_getter()
        with self._lock:
            return self._latest_jpeg

    def _generate_frames(self):
        """生成 MJPEG 流"""
        while self.running:
            frame = self._get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    def _video_feed(self):
        """MJPEG 视频流端点"""
        return Response(
            self._generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _health(self):
        """健康检查端点"""
        return {'status': 'ok', 'running': self.running}

    def _index(self):
        """简单的测试页面"""
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>MJPEG Stream</title></head>
        <body style="margin:0; background:#000;">
            <img src="/video_feed" style="width:100%; height:100vh; object-fit:contain;">
        </body>
        </html>
        '''

    def start(self):
        """启动服务器"""
        self.running = True
        try:
            self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"[MJPEGServer] Error: {e}")
            self.running = False

    def stop(self):
        """停止服务器"""
        self.running = False


def start_mjpeg_server(port: int = 8889) -> MJPEGServer:
    """启动 MJPEG 服务器（后台线程）"""
    global _mjpeg_server_instance, _server_thread

    if _mjpeg_server_instance and _mjpeg_server_instance.running:
        return _mjpeg_server_instance

    _mjpeg_server_instance = MJPEGServer(port)
    _server_thread = threading.Thread(target=_mjpeg_server_instance.start, daemon=True)
    _server_thread.start()

    # 等待服务器启动
    time.sleep(0.5)
    print(f"[MJPEGServer] Started on port {port}")

    return _mjpeg_server_instance


def get_mjpeg_server() -> Optional[MJPEGServer]:
    """获取 MJPEG 服务器实例"""
    return _mjpeg_server_instance


def stop_mjpeg_server():
    """停止 MJPEG 服务器"""
    global _mjpeg_server_instance
    if _mjpeg_server_instance:
        _mjpeg_server_instance.stop()
        _mjpeg_server_instance = None

