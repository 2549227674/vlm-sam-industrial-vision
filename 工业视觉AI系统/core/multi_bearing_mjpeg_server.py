"""MJPEG HTTP Server for multi-bearing production line monitoring."""

import threading
import time
from typing import Optional, Dict, Callable
from flask import Flask, Response
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 禁用 Flask 默认日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 全局变量
_multi_bearing_mjpeg_server: Optional['MultiBearingMJPEGServer'] = None
_server_thread: Optional[threading.Thread] = None

# 尝试加载中文字体
def get_chinese_font(size=20):
    """获取支持中文的字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    return ImageFont.load_default()


class MultiBearingMJPEGServer:
    """Multi-Bearing MJPEG HTTP Server - 支持多路视频流合并显示"""

    def __init__(self, port: int = 8890):
        self.port = port
        self.app = Flask(__name__)
        self.running = False
        self._frame_getter: Optional[Callable[[], Dict]] = None
        self._latest_frames: Dict[str, np.ndarray] = {}
        self._combined_frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._target_fps = 25
        self._frame_interval = 1.0 / self._target_fps

        # 设置路由
        self.app.add_url_rule('/video_feed', 'video_feed', self._video_feed)
        self.app.add_url_rule('/video_feed/<int:line_id>', 'video_feed_line', self._video_feed_line)
        self.app.add_url_rule('/combined_feed', 'combined_feed', self._combined_feed)
        self.app.add_url_rule('/health', 'health', self._health)
        self.app.add_url_rule('/', 'index', self._index)

    def set_frame_getter(self, getter: Callable[[], Dict]):
        """设置帧获取函数 - 返回 {line_id: {'frame': ndarray, 'name': str}}"""
        self._frame_getter = getter

    def update_frames(self, frames_dict: Dict[str, Dict]):
        """更新所有生产线的最新帧（线程安全）"""
        with self._lock:
            for line_id, frame_data in frames_dict.items():
                if 'frame' in frame_data:
                    self._latest_frames[line_id] = frame_data

    def _get_frames(self) -> Dict:
        """获取最新帧"""
        if self._frame_getter:
            try:
                frames = self._frame_getter()
                if frames:
                    # 同时更新本地缓存
                    with self._lock:
                        for line_id, frame_data in frames.items():
                            if 'frame' in frame_data:
                                self._latest_frames[line_id] = frame_data
                    return frames
            except Exception as e:
                print(f"[MJPEG] frame_getter error: {e}")

        # 返回缓存的帧
        with self._lock:
            return self._latest_frames.copy()

    def _create_combined_frame(self, frames_dict: Dict) -> Optional[bytes]:
        """合并多路视频为一帧（网格布局）"""
        if not frames_dict:
            return None

        frames = []
        names = []
        for line_id, frame_data in sorted(frames_dict.items()):
            if 'frame' in frame_data:
                frame = frame_data['frame']
                name = frame_data.get('name', f'Line {line_id}')
                frames.append(frame)
                names.append(name)

        if not frames:
            return None

        # 统一尺寸
        target_h, target_w = 360, 480
        resized_frames = []

        # 获取中文字体
        font = get_chinese_font(24)

        for frame, name in zip(frames, names):
            resized = cv2.resize(frame, (target_w, target_h))

            # 使用 PIL 绘制中文标签
            # 转换 BGR -> RGB
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 绘制半透明背景
            text_bbox = draw.textbbox((10, 5), name, font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2],
                          fill=(0, 0, 0, 180))
            # 绘制文字
            draw.text((10, 5), name, font=font, fill=(0, 255, 136))

            # 转回 BGR
            resized = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            resized_frames.append(resized)

        # 计算网格布局
        n = len(resized_frames)
        if n == 1:
            cols, rows = 1, 1
        elif n == 2:
            cols, rows = 2, 1
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3

        # 补齐空白帧
        while len(resized_frames) < cols * rows:
            blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            resized_frames.append(blank)

        # 构建网格
        grid_rows = []
        for r in range(rows):
            row_frames = resized_frames[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_frames))
        combined = np.vstack(grid_rows)

        # 编码为 JPEG
        _, jpeg = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def _create_waiting_frame(self):
        """创建等待画面"""
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:] = (30, 30, 40)  # 深色背景

        # 使用 PIL 绘制中文
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = get_chinese_font(24)

        text = "等待视频流..."
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x = (480 - text_w) // 2
        y = (360 - text_h) // 2
        draw.text((x, y), text, font=font, fill=(100, 255, 136))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def _generate_combined_frames(self):
        """生成合并的 MJPEG 流"""
        waiting_frame = self._create_waiting_frame()
        frame_count = 0

        while self.running:
            start_time = time.time()
            frames = self._get_frames()

            if frames:
                jpeg = self._create_combined_frame(frames)
                if jpeg:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                    frame_count += 1
            else:
                # 没有帧时显示等待画面
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + waiting_frame + b'\r\n')

            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0.01, self._frame_interval - elapsed)
            time.sleep(sleep_time)

    def _generate_single_line_frames(self, line_id: int):
        """生成单条生产线的 MJPEG 流"""
        font = get_chinese_font(24)

        while self.running:
            start_time = time.time()
            frames = self._get_frames()
            line_key = str(line_id)

            if frames and line_key in frames:
                frame_data = frames[line_key]
                if 'frame' in frame_data:
                    frame = frame_data['frame'].copy()
                    name = frame_data.get('name', f'Line {line_id}')

                    # 使用 PIL 绘制中文标签
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)

                    # 绘制背景和文字
                    text_bbox = draw.textbbox((10, 5), name, font=font)
                    draw.rectangle([text_bbox[0]-5, text_bbox[1]-2, text_bbox[2]+5, text_bbox[3]+2],
                                  fill=(0, 0, 0, 180))
                    draw.text((10, 5), name, font=font, fill=(0, 255, 136))

                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, self._frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _video_feed(self):
        """默认视频流（合并显示）"""
        return Response(
            self._generate_combined_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _video_feed_line(self, line_id: int):
        """单条生产线视频流"""
        return Response(
            self._generate_single_line_frames(line_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _combined_feed(self):
        """合并视频流端点"""
        return Response(
            self._generate_combined_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _health(self):
        """健康检查端点"""
        return {'status': 'ok', 'running': self.running, 'port': self.port}

    def _index(self):
        """全屏监控页面"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>🏭 多轴承生产线实时监控</title>
            <meta charset="utf-8">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    font-family: 'Segoe UI', Arial, sans-serif;
                    min-height: 100vh;
                    color: #fff;
                }
                .header {
                    background: rgba(0, 255, 136, 0.1);
                    border-bottom: 2px solid #00ff88;
                    padding: 15px 30px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .header h1 {
                    color: #00ff88;
                    font-size: 24px;
                    text-shadow: 0 0 10px rgba(0,255,136,0.5);
                }
                .status {
                    display: flex;
                    gap: 20px;
                    align-items: center;
                }
                .status-item {
                    background: rgba(255,255,255,0.1);
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 14px;
                }
                .status-item.online {
                    background: rgba(0, 255, 136, 0.2);
                    color: #00ff88;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                    min-height: calc(100vh - 70px);
                }
                .video-wrapper {
                    border: 3px solid #00ff88;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 0 30px rgba(0,255,136,0.3);
                    max-width: 95vw;
                    max-height: 85vh;
                }
                .video-wrapper img {
                    display: block;
                    width: 100%;
                    height: auto;
                    max-height: 85vh;
                    object-fit: contain;
                }
                .footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    background: rgba(0,0,0,0.8);
                    padding: 10px;
                    text-align: center;
                    font-size: 12px;
                    color: #888;
                }
                .time {
                    font-family: 'Courier New', monospace;
                    color: #00ff88;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏭 多轴承生产线实时监控系统</h1>
                <div class="status">
                    <span class="status-item online">● 运行中</span>
                    <span class="status-item">YOLO 实时检测</span>
                    <span class="status-item time" id="clock">--:--:--</span>
                </div>
            </div>
            <div class="container">
                <div class="video-wrapper">
                    <img src="/combined_feed" alt="实时监控画面">
                </div>
            </div>
            <div class="footer">
                Multi-Bearing Production Line Monitor | MJPEG Stream @ 25 FPS | Press F11 for fullscreen
            </div>
            <script>
                function updateClock() {
                    const now = new Date();
                    document.getElementById('clock').textContent = now.toLocaleTimeString('zh-CN');
                }
                setInterval(updateClock, 1000);
                updateClock();
            </script>
        </body>
        </html>
        '''

    def start(self):
        """启动服务器"""
        self.running = True
        try:
            self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"[MultiBearingMJPEGServer] Error: {e}")
            self.running = False

    def stop(self):
        """停止服务器"""
        self.running = False


def start_multi_bearing_mjpeg_server(port: int = 8890) -> MultiBearingMJPEGServer:
    """启动多轴承 MJPEG 服务器（后台线程）"""
    global _multi_bearing_mjpeg_server, _server_thread

    if _multi_bearing_mjpeg_server and _multi_bearing_mjpeg_server.running:
        return _multi_bearing_mjpeg_server

    _multi_bearing_mjpeg_server = MultiBearingMJPEGServer(port)
    _server_thread = threading.Thread(target=_multi_bearing_mjpeg_server.start, daemon=True)
    _server_thread.start()

    # 等待服务器启动
    time.sleep(0.5)
    print(f"[MultiBearingMJPEGServer] Started on port {port}")

    return _multi_bearing_mjpeg_server


def get_multi_bearing_mjpeg_server() -> Optional[MultiBearingMJPEGServer]:
    """获取多轴承 MJPEG 服务器实例"""
    return _multi_bearing_mjpeg_server


def stop_multi_bearing_mjpeg_server():
    """停止多轴承 MJPEG 服务器"""
    global _multi_bearing_mjpeg_server
    if _multi_bearing_mjpeg_server:
        _multi_bearing_mjpeg_server.stop()
        _multi_bearing_mjpeg_server = None

