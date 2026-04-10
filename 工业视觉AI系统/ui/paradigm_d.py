"""Paradigm D: Real-time YOLO Bearing Defect Detection with Edge Device Stream."""

import io
import time
import threading
from typing import Optional, Dict, List
from collections import deque
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import cv2
import pandas as pd

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

from core.socket_server import start_server, get_server
from core.mjpeg_server import start_mjpeg_server, get_mjpeg_server

# 默认模型路径（轴承缺陷检测权重）
DEFAULT_MODEL_PATH = r"C:\Users\Han\PycharmProjects\PythonProject8\runs\train_bearing\bearing_line_20260108_164005\weights\best.pt"

# 全局detector实例（确保跨rerun保持同一实例）
_global_detector: Optional['YOLOStreamDetector'] = None

def get_global_detector() -> Optional['YOLOStreamDetector']:
    """获取全局detector实例"""
    global _global_detector
    return _global_detector

def set_global_detector(detector: 'YOLOStreamDetector'):
    """设置全局detector实例"""
    global _global_detector
    _global_detector = detector

# 轴承缺陷类别（8类）
DEFECT_CLASSES = {
    0: "Casting_burr",      # 铸造毛刺
    1: "crack",             # 裂纹
    2: "scratch",           # 划痕
    3: "pit",               # 凹坑
    4: "Polished_casting",  # 抛光铸件
    5: "strain",            # 应变
    6: "unpolished_casting",# 未抛光铸件
    7: "burr"               # 毛刺
}

# 缺陷类别中文名
DEFECT_CLASSES_CN = {
    0: "铸造毛刺",
    1: "裂纹",
    2: "划痕",
    3: "凹坑",
    4: "抛光铸件",
    5: "应变",
    6: "未抛光铸件",
    7: "毛刺"
}

# 每种缺陷的颜色（BGR格式）
DEFECT_COLORS = [
    (0, 0, 255),     # 红色 - Casting_burr
    (0, 255, 0),     # 绿色 - crack
    (255, 0, 0),     # 蓝色 - scratch
    (0, 255, 255),   # 黄色 - pit
    (255, 0, 255),   # 紫色 - Polished_casting
    (255, 255, 0),   # 青色 - strain
    (128, 0, 255),   # 粉色 - unpolished_casting
    (0, 128, 255),   # 橙色 - burr
]


class YOLOStreamDetector:
    """YOLO实时流检测器"""

    def __init__(self, model_path: str, device: str = 'cuda:0', conf_threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = None

        # 统计数据
        self.total_frames = 0
        self.total_defects = 0
        self.defect_counts = {i: 0 for i in range(8)}
        self.inference_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)
        self.defect_history = deque(maxlen=1000)

        # 最新检测结果
        self.latest_result = None
        self.latest_annotated_frame = None
        self.latest_jpeg = None  # 用于MJPEG流
        self._lock = threading.Lock()

        # 运行状态
        self.running = False
        self.start_time = None
        self._detection_thread = None

    def load_model(self):
        """加载YOLO模型"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        else:
            self.class_names = DEFECT_CLASSES

        return True

    def start_detection_loop(self, frame_getter):
        """启动检测循环（后台线程）"""
        self.running = True
        self.start_time = time.time()
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(frame_getter,),
            daemon=True
        )
        self._detection_thread.start()

    def stop_detection_loop(self):
        """停止检测循环"""
        self.running = False
        if self._detection_thread:
            self._detection_thread.join(timeout=2.0)

    def _detection_loop(self, frame_getter):
        """检测循环主函数"""
        print("[YOLODetector] Detection loop started")
        frame_count = 0
        while self.running:
            try:
                frame_data = frame_getter()
                if frame_data and hasattr(frame_data, 'jpeg_data') and frame_data.jpeg_data:
                    result = self.detect_jpeg(frame_data.jpeg_data)
                    frame_count += 1
                    if frame_count % 30 == 0:  # 每30帧打印一次
                        print(f"[YOLODetector] Processed {frame_count} frames, total_defects={self.total_defects}")
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[YOLODetector] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def detect_jpeg(self, jpeg_data: bytes) -> Optional[bytes]:
        """检测JPEG数据，返回带标注的JPEG"""
        if self.model is None:
            print("[YOLODetector] Model is None!")
            return None

        try:
            # 解码JPEG
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("[YOLODetector] Failed to decode JPEG")
                return None

            # 执行推理
            start_time = time.time()
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            inference_time = (time.time() - start_time) * 1000

            # 解析结果
            result = results[0]
            detections = []
            frame_defect_counts = {i: 0 for i in range(8)}

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().item())
                    conf = float(box.conf[0].cpu().item())

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, f"Unknown-{cls_id}"),
                        'class_name_cn': DEFECT_CLASSES_CN.get(cls_id, f"未知-{cls_id}"),
                        'confidence': conf
                    })

                    if cls_id in frame_defect_counts:
                        frame_defect_counts[cls_id] += 1

            # 绘制检测结果
            annotated_frame = self._draw_detections(frame, detections)

            # 编码为JPEG
            _, jpeg_out = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_jpeg = jpeg_out.tobytes()

            # 更新统计（线程安全）
            with self._lock:
                self.total_frames += 1
                self.total_defects += len(detections)
                for cls_id, count in frame_defect_counts.items():
                    self.defect_counts[cls_id] += count
                self.inference_times.append(inference_time)

                self.defect_history.append({
                    'timestamp': time.time(),
                    'defect_count': len(detections),
                    'defect_types': frame_defect_counts.copy()
                })

                self.latest_result = {
                    'detections': detections,
                    'inference_time_ms': inference_time,
                    'defect_count': len(detections)
                }
                self.latest_annotated_frame = annotated_frame
                self.latest_jpeg = annotated_jpeg

            return annotated_jpeg

        except Exception as e:
            print(f"[YOLODetector] Detection error: {e}")
            return None

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """在帧上绘制检测结果（英文显示）"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']

            color = DEFECT_COLORS[cls_id % len(DEFECT_COLORS)]

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 使用英文标签
            label = f"{det['class_name']}: {conf:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # 绘制标签背景
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)

            # 绘制标签文字
            cv2.putText(annotated, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        # 左上角统计信息
        with self._lock:
            info_text = f"Frames: {self.total_frames} | Defects: {self.total_defects}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated

    def get_latest_jpeg(self) -> Optional[bytes]:
        """获取最新的标注JPEG（用于MJPEG流）"""
        with self._lock:
            return self.latest_jpeg

    def get_stats(self) -> Dict:
        """获取统计数据"""
        with self._lock:
            avg_inference = np.mean(self.inference_times) if self.inference_times else 0
            fps = 1000 / avg_inference if avg_inference > 0 else 0
            defect_rate = (self.total_defects / self.total_frames * 100) if self.total_frames > 0 else 0

            return {
                'total_frames': self.total_frames,
                'total_defects': self.total_defects,
                'defect_counts': self.defect_counts.copy(),
                'avg_inference_ms': avg_inference,
                'fps': fps,
                'defect_rate': defect_rate,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0,
                'latest_result': self.latest_result
            }

    def reset_stats(self):
        """重置统计数据"""
        with self._lock:
            self.total_frames = 0
            self.total_defects = 0
            self.defect_counts = {i: 0 for i in range(8)}
            self.inference_times.clear()
            self.fps_history.clear()
            self.defect_history.clear()
            self.start_time = time.time()


def render(device: str, sam_proc=None, sam_model=None, sam_dtype=None):
    """Render Paradigm D UI - YOLO实时流检测（MJPEG方案）"""

    st.header("📹 范式 D：实时流工业检测")
    st.caption("边缘端视频流 + YOLO轴承缺陷实时检测 + MJPEG流畅预览")

    if not YOLO_AVAILABLE:
        st.error("❌ YOLO模型不可用，请安装 ultralytics")
        return

    # ========== 初始化 Session State ==========
    if "socket_server_started" not in st.session_state:
        st.session_state.socket_server_started = False
    if "mjpeg_server_started" not in st.session_state:
        st.session_state.mjpeg_server_started = False
    if "yolo_detector" not in st.session_state:
        st.session_state.yolo_detector = None
    if "yolo_model_loaded" not in st.session_state:
        st.session_state.yolo_model_loaded = False
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False

    # ========== 侧边栏配置 ==========
    with st.sidebar:
        st.subheader("🔧 YOLO检测配置")

        model_path = st.text_input(
            "YOLO权重路径",
            value=DEFAULT_MODEL_PATH,
            key="yolo_model_path"
        )

        model_exists = Path(model_path).exists()
        if model_exists:
            st.success("✅ 模型文件存在")
        else:
            st.error("❌ 模型文件不存在")

        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="yolo_conf_threshold"
        )

        device_option = st.selectbox(
            "推理设备",
            options=["cuda:0", "cpu"],
            index=0 if device == "cuda" else 1,
            key="yolo_device"
        )

        st.divider()

        # 加载模型按钮
        if not st.session_state.yolo_model_loaded:
            if st.button("📦 加载YOLO模型", use_container_width=True, disabled=not model_exists):
                with st.spinner("正在加载YOLO模型..."):
                    try:
                        detector = YOLOStreamDetector(
                            model_path=model_path,
                            device=device_option,
                            conf_threshold=conf_threshold
                        )
                        detector.load_model()
                        # 使用全局变量和session_state双重保存
                        set_global_detector(detector)
                        st.session_state.yolo_detector = detector
                        st.session_state.yolo_model_loaded = True
                        st.success("✅ 模型加载成功！")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"模型加载失败：{e}")
        else:
            st.success("✅ YOLO模型已加载")

            if st.session_state.yolo_detector:
                st.session_state.yolo_detector.conf_threshold = conf_threshold

            if st.button("🔄 重置统计", use_container_width=True):
                if st.session_state.yolo_detector:
                    st.session_state.yolo_detector.reset_stats()
                    st.success("统计数据已重置")
                    st.rerun()

    # ========== 主界面 ==========

    # 服务器控制区
    st.subheader("🌐 服务器配置")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        server_host = st.text_input("服务器地址", value="0.0.0.0", key="server_host")

    with col2:
        socket_port = st.number_input("Socket端口", value=8888, min_value=1024, max_value=65535, key="socket_port")

    with col3:
        mjpeg_port = st.number_input("MJPEG端口", value=8889, min_value=1024, max_value=65535, key="mjpeg_port")

    with col4:
        st.write("")
        st.write("")

        # 一键启动所有服务
        all_started = (st.session_state.socket_server_started and
                      st.session_state.mjpeg_server_started and
                      st.session_state.detection_running)

        if not all_started:
            if st.button("🚀 启动全部", use_container_width=True,
                        disabled=not st.session_state.yolo_model_loaded):
                try:
                    # 1. 启动Socket服务器
                    if not st.session_state.socket_server_started:
                        start_server(server_host, int(socket_port))
                        time.sleep(0.5)  # 等待服务器启动
                        socket_server = get_server()
                        if socket_server and socket_server.running:
                            st.session_state.socket_server_started = True
                            print("[ParadigmD] Socket server started")
                        else:
                            st.error("Socket服务器启动失败，端口可能被占用")
                            return

                    # 2. 启动MJPEG服务器
                    if not st.session_state.mjpeg_server_started:
                        mjpeg_server = start_mjpeg_server(int(mjpeg_port))
                        st.session_state.mjpeg_server_started = True
                        print("[ParadigmD] MJPEG server started")

                    # 3. 启动检测循环
                    if not st.session_state.detection_running:
                        detector = get_global_detector() or st.session_state.yolo_detector
                        socket_server = get_server()
                        mjpeg_server = get_mjpeg_server()

                        if detector and socket_server and mjpeg_server:
                            # 设置MJPEG帧获取器
                            mjpeg_server.set_frame_getter(detector.get_latest_jpeg)

                            # 启动检测循环
                            detector.start_detection_loop(socket_server.get_latest_frame)
                            st.session_state.detection_running = True
                            print("[ParadigmD] Detection loop started")
                        else:
                            st.error("无法启动检测：缺少必要组件")
                            return

                    st.success("✅ 所有服务已启动")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败：{e}")
                    import traceback
                    traceback.print_exc()
        else:
            st.success("✅ 运行中")

    # 状态显示
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.socket_server_started:
            st.success(f"✅ Socket: {socket_port}")
        else:
            st.warning("⏳ Socket: 未启动")
    with col2:
        if st.session_state.mjpeg_server_started:
            st.success(f"✅ MJPEG: {mjpeg_port}")
        else:
            st.warning("⏳ MJPEG: 未启动")
    with col3:
        if st.session_state.detection_running:
            st.success("✅ 检测: 运行中")
        else:
            st.warning("⏳ 检测: 未启动")

    if not st.session_state.yolo_model_loaded:
        st.warning("👈 请先在左侧面板加载YOLO模型")
        return

    if not all_started:
        st.info("""
        **使用说明**：
        1. 点击 **"🚀 启动全部"** 启动所有服务
        2. 在边缘端运行: `./edge_device /dev/video0 <服务器IP> 8888`
        3. 视频流将在下方实时显示
        """)
        return

    st.divider()

    # ========== 实时视频 + 统计面板 ==========
    col_video, col_stats = st.columns([2, 1])

    with col_video:
        st.subheader("🎬 实时检测预览")

        # 使用 iframe 嵌入 MJPEG 流 - 无闪烁！
        mjpeg_url = f"http://127.0.0.1:{mjpeg_port}/video_feed"

        video_html = f"""
        <div style="background:#000; border-radius:8px; overflow:hidden;">
            <img src="{mjpeg_url}" 
                 style="width:100%; height:auto; display:block;"
                 onerror="this.style.display='none'; document.getElementById('error-msg').style.display='block';">
            <div id="error-msg" style="display:none; color:#fff; padding:20px; text-align:center;">
                ⏳ 等待视频流连接...
            </div>
        </div>
        """
        components.html(video_html, height=400)

        st.caption(f"📡 MJPEG流地址: `{mjpeg_url}`")

    with col_stats:
        st.subheader("📊 实时统计")

        # 优先使用全局detector实例获取统计
        detector = get_global_detector() or st.session_state.yolo_detector
        stats = detector.get_stats()

        st.metric("总检测帧数", f"{stats['total_frames']:,}")
        st.metric("检出缺陷总数", stats['total_defects'])
        st.metric("缺陷率", f"{stats['defect_rate']:.2f}%")
        st.metric("平均推理时间", f"{stats['avg_inference_ms']:.1f}ms")
        st.metric("检测FPS", f"{stats['fps']:.1f}")
        st.metric("运行时长", f"{stats['elapsed_time']:.1f}s")

    st.divider()

    # ========== 缺陷类型分布 ==========
    st.subheader("📈 缺陷类型分布")

    defect_counts = stats['defect_counts']
    has_defects = any(count > 0 for count in defect_counts.values())

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**缺陷类型统计（柱状图）**")
        if has_defects:
            chart_data = []
            for cls_id, count in defect_counts.items():
                chart_data.append({
                    '缺陷类型': DEFECT_CLASSES_CN.get(cls_id, f"类型{cls_id}"),
                    '数量': count
                })
            df = pd.DataFrame(chart_data)
            df = df.set_index('缺陷类型')
            st.bar_chart(df, height=250)
        else:
            st.info("暂未检出缺陷")

    with col_chart2:
        st.markdown("**缺陷类型占比**")
        if has_defects:
            table_data = []
            total = sum(defect_counts.values())
            for cls_id, count in defect_counts.items():
                if count > 0:
                    pct = count / total * 100 if total > 0 else 0
                    table_data.append({
                        'ID': cls_id,
                        '类型': DEFECT_CLASSES_CN.get(cls_id, f"类型{cls_id}"),
                        '数量': count,
                        '占比': f"{pct:.1f}%"
                    })
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        else:
            st.info("暂未检出缺陷")

    # ========== 最近检测详情 ==========
    with st.expander("🔍 最近检测详情", expanded=False):
        latest = stats.get('latest_result')
        if latest and latest.get('detections'):
            detections = latest['detections']
            st.write(f"最新帧检出 **{len(detections)}** 个缺陷")
            for i, det in enumerate(detections[:5]):
                st.write(f"  • {det['class_name_cn']} ({det['class_name']}): {det['confidence']:.2f}")
        else:
            st.info("暂无检测结果")

    # ========== 系统信息 ==========
    with st.expander("ℹ️ 系统信息"):
        st.markdown(f"""
        | 项目 | 值 |
        |------|-----|
        | 模型路径 | `{detector.model_path}` |
        | 推理设备 | `{detector.device}` |
        | 置信度阈值 | `{detector.conf_threshold}` |
        | Socket端口 | `{socket_port}` |
        | MJPEG端口 | `{mjpeg_port}` |
        """)

    # ========== 低频刷新统计数据 ==========
    time.sleep(1)  # 1秒刷新一次统计（视频流不受影响）
    st.rerun()

