"""Industrial monitoring dashboard with real image upload and VLM inference + Multi-bearing YOLO monitoring."""

import time
import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import cv2
import os
import tempfile
import numpy as np
import pandas as pd
import threading

# 原有VLM监控功能的导入（相对导入）
try:
    from core.thread_pool import get_thread_pool, Task, TaskStatus
    from core.simulator import get_simulator
    from core.vlm_model_registry import list_models
    VLM_MONITORING_AVAILABLE = True
except ImportError as e:
    VLM_MONITORING_AVAILABLE = False
    get_thread_pool = get_simulator = list_models = None
    Task = TaskStatus = None

# 导入多轴承 MJPEG 服务器
try:
    from core.multi_bearing_mjpeg_server import (
        start_multi_bearing_mjpeg_server,
        get_multi_bearing_mjpeg_server,
        stop_multi_bearing_mjpeg_server
    )
    MJPEG_SERVER_AVAILABLE = True
except ImportError:
    MJPEG_SERVER_AVAILABLE = False
    start_multi_bearing_mjpeg_server = None
    get_multi_bearing_mjpeg_server = None
    stop_multi_bearing_mjpeg_server = None

# 添加项目根目录到路径，以便导入多轴承监控系统
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 强制清除缓存的模块（解决 Streamlit 缓存问题）
if 'core.multi_bearing_monitor' in sys.modules:
    del sys.modules['core.multi_bearing_monitor']
if 'core' in sys.modules:
    # 不要删除 core，因为可能有其他模块在使用
    pass

# 调试信息
import_error_msg = None
import_traceback = None
try:
    # 添加根目录到路径以访问 bearing_core
    import sys
    import os
    _root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root_dir not in sys.path:
        sys.path.insert(0, _root_dir)
    from bearing_core.multi_bearing_monitor import MultiBearingMonitor
    MULTI_BEARING_AVAILABLE = True
except ImportError as e:
    MULTI_BEARING_AVAILABLE = False
    MultiBearingMonitor = None
    import_error_msg = str(e)
    import traceback
    import_traceback = traceback.format_exc()
except Exception as e:
    MULTI_BEARING_AVAILABLE = False
    MultiBearingMonitor = None
    import_error_msg = f"意外错误: {str(e)}"
    import traceback
    import_traceback = traceback.format_exc()


def render(device=None, sam_proc=None, sam_model=None, sam_dtype=None):
    """Render monitoring dashboard."""

    st.header("📊 工业监控看板")

    # 监控模式选择
    monitoring_mode = st.radio(
        "选择监控模式",
        ["📋 VLM多线程推理监控", "🏭 多轴承生产线YOLO并发监控"],
        horizontal=True,
        key="monitoring_mode"
    )

    if monitoring_mode == "🏭 多轴承生产线YOLO并发监控":
        render_multi_bearing_monitoring(device)
        return

    # 以下是原有的VLM监控逻辑
    st.caption("多生产线并发处理 + 范式 C VLM 推理")

    # 检查VLM监控是否可用
    if not VLM_MONITORING_AVAILABLE:
        st.error("❌ VLM监控系统不可用，缺少必要模块")
        st.info("请确保 `core/thread_pool.py` 和 `core/simulator.py` 存在")
        return

    # Initialize systems
    if "monitoring_started" not in st.session_state:
        st.session_state.monitoring_started = False
        st.session_state.simulator_started = False

    # ========== Left Sidebar: System Configuration ==========
    with st.sidebar:
        st.subheader("⚙️ 系统控制台")

        # Thread pool configuration
        st.markdown("**线程池配置**")
        max_workers = st.number_input("线程池大小", min_value=2, max_value=16, value=4, key="max_workers")

        st.divider()

        # VLM configuration
        st.markdown("**VLM 配置**")
        vlm_models = list_models(require='bbox')
        vlm_model = st.selectbox(
            "VLM 模型",
            options=vlm_models,
            index=vlm_models.index("qwen-vl-max") if "qwen-vl-max" in vlm_models else 0,
            key="monitoring_vlm_model"
        )

        max_boxes = st.slider("最大检测框数", min_value=1, max_value=10, value=3, key="monitoring_max_boxes")

        st.divider()

        # SAM configuration
        st.markdown("**SAM 配置**")
        enable_sam = st.checkbox("启用 SAM 精细分割", value=False, key="monitoring_enable_sam")

        if enable_sam:
            sam_threshold = st.slider("SAM 阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="monitoring_sam_threshold")
            mask_threshold = st.slider("Mask 阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="monitoring_mask_threshold")

        st.divider()

        # DashScope API key (可选，从环境变量读取兜底)
        # 优先从 session_state 读取，其次尝试 secrets（如果存在），最后从环境变量
        import os
        dashscope_key_default = st.session_state.get("dashscope_api_key_cached")
        if not dashscope_key_default:
            try:
                dashscope_key_default = st.secrets.get("dashscope_api_key", "")
            except Exception:
                dashscope_key_default = ""
        if not dashscope_key_default:
            dashscope_key_default = os.getenv("DASHSCOPE_API_KEY", "")

        dashscope_api_key = st.text_input("DashScope API Key", value=dashscope_key_default, type="password", help="仅用于本次会话内提交任务")
        st.session_state["dashscope_api_key_cached"] = dashscope_api_key

        st.divider()

        # Start/Stop controls
        if not st.session_state.monitoring_started:
            if st.button("🚀 启动监控", use_container_width=True):
                thread_pool = get_thread_pool(max_workers=max_workers)
                st.session_state.monitoring_started = True
                st.success("监控系统已启动")
                time.sleep(0.5)
                st.rerun()
        else:
            st.success("✅ 监控运行中")

    # ========== Main Area ==========
    if not st.session_state.monitoring_started:
        st.info("请先在左侧控制台启动监控系统")
        return

    # Production line image upload
    st.subheader("生产线图片上传")

    simulator = get_simulator()
    lines = simulator.get_lines()

    for line in lines:
        with st.expander(f"📹 {line.name} (FPS: {line.fps})", expanded=False):
            uploaded_files = st.file_uploader(
                f"上传 {line.name} 的工业图片",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"upload_{line.line_id}"
            )

            if uploaded_files:
                images = []
                for file in uploaded_files:
                    image = Image.open(file).convert("RGB")
                    images.append(image)

                simulator.upload_images(line.line_id, images)
                st.success(f"已上传 {len(images)} 张图片到 {line.name}")

                # Show preview
                cols = st.columns(min(5, len(images)))
                for idx, img in enumerate(images[:5]):
                    with cols[idx]:
                        st.image(img, use_container_width=True)
                if len(images) > 5:
                    st.caption(f"... 还有 {len(images) - 5} 张图片")

    st.divider()

    # Start production lines
    if not st.session_state.simulator_started:
        if st.button("▶️ 启动生产线", use_container_width=True):
            thread_pool = get_thread_pool()

            # Get configuration from session state
            vlm_model = st.session_state.get("monitoring_vlm_model", "qwen-vl-max")
            max_boxes = st.session_state.get("monitoring_max_boxes", 3)
            enable_sam = st.session_state.get("monitoring_enable_sam", False)
            sam_threshold = st.session_state.get("monitoring_sam_threshold", 0.5)
            mask_threshold = st.session_state.get("monitoring_mask_threshold", 0.5)

            # Set callback to submit tasks
            def on_frame(line_id, image):
                task_id = f"{line_id}_{int(time.time() * 1000)}"
                task = Task(
                    task_id=task_id,
                    line_id=line_id,
                    image_data=image,
                    prompt={
                        "vlm_model": vlm_model,
                        "max_boxes": max_boxes,
                        "enable_sam": enable_sam,
                        "sam_threshold": sam_threshold,
                        "mask_threshold": mask_threshold
                    },
                    context={
                        "device": device or "cpu",
                        "sam_proc": sam_proc,
                        "sam_model": sam_model,
                        "sam_dtype": sam_dtype,
                        "dashscope_api_key": dashscope_api_key,
                    },
                )
                try:
                    thread_pool.submit_task(task)
                except Exception as submit_err:
                    st.error(f"提交失败: {submit_err}")

                # Update defect count after inference
                # (will be updated when task completes)

            simulator.set_frame_callback(on_frame)
            simulator.start()
            st.session_state.simulator_started = True
            st.success("生产线已启动")
            time.sleep(0.5)
            st.rerun()
    else:
        st.success("✅ 生产线运行中")

        if st.button("⏸️ 停止生产线"):
            simulator.stop()
            st.session_state.simulator_started = False
            st.rerun()

    st.divider()

    # Get metrics
    thread_pool = get_thread_pool()
    metrics = thread_pool.get_metrics()

    # Thread pool metrics
    st.subheader("线程池状态")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("排队任务", metrics["queue_size"])

    with col2:
        st.metric("活跃线程", f"{metrics['active_tasks']}/{metrics['max_workers']}")

    with col3:
        st.metric("线程利用率", f"{metrics['utilization']:.1f}%")
        st.progress(min(metrics['utilization'] / 100, 1.0))  # Clamp to max 1.0

    with col4:
        st.metric("平均处理时间", f"{metrics['avg_processing_time']:.2f}s")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("已提交", metrics["total_submitted"])
    with col2:
        st.metric("已完成", metrics["total_completed"])
    with col3:
        st.metric("失败", metrics["total_failed"])
    with col4:
        st.metric("超时", metrics.get("total_timeouts", 0))

    st.caption(f"速率限制: {metrics.get('rate_limit_per_sec', 0):.1f} req/s")
    if metrics.get("last_error"):
        st.warning(f"最近错误: {metrics['last_error']}")

    st.divider()

    # Production line statistics
    st.subheader("生产线统计")

    all_stats = simulator.get_all_stats()

    for line in lines:
        stats = all_stats.get(line.line_id, {})

        with st.expander(f"📹 {line.name}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("图片数", stats.get("image_count", 0))

            with col2:
                st.metric("已处理", stats.get("frames", 0))

            with col3:
                st.metric("检出缺陷", stats.get("defects", 0))

            with col4:
                defect_rate = stats.get("defect_rate", 0.0)
                st.metric("缺陷率", f"{defect_rate:.1f}%")

    st.divider()

    # Recent completed tasks with results
    st.subheader("最近完成任务")

    completed_tasks = thread_pool.get_completed_tasks(limit=5)

    if completed_tasks:
        for task in reversed(completed_tasks):
            with st.expander(f"{task.line_id} - {task.task_id}", expanded=False):
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Show processed image with bboxes
                    if task.result and task.result.get('processed_image'):
                        st.image(task.result['processed_image'], caption="检测结果", use_container_width=True)
                    elif task.image_data:
                        st.image(task.image_data, caption="原始图片", use_container_width=True)

                with col2:
                    processing_time = task.end_time - task.start_time
                    st.write(f"**耗时**: {processing_time:.2f}s")

                    if task.status == TaskStatus.COMPLETED and task.result:
                        st.write(f"**状态**: ✅ 完成")
                        st.write(f"**缺陷检测**: {'🔴 有缺陷' if task.result.get('has_defect') else '🟢 正常'}")

                        # Show detection details
                        detections = task.result.get('detections', [])
                        if detections:
                            st.write(f"**检出数量**: {len(detections)}")
                            for i, det in enumerate(detections[:3]):  # Show first 3
                                st.write(f"  - 缺陷 {i+1}: {det.get('label', 'unknown')}")

                        # Show decision and evidence
                        decision = task.result.get('decision', '')
                        evidence = task.result.get('evidence', '')
                        if decision:
                            st.write(f"**判断**: {decision}")
                        if evidence:
                            st.write(f"**证据**: {evidence}")

                        # Update defect count
                        if task.result.get('has_defect'):
                            simulator.update_defect_count(task.line_id, True)
                    else:
                        st.write(f"**状态**: ❌ 失败")
                        st.write(f"**错误**: {task.error}")
    else:
        st.info("暂无完成任务")

    # Auto-refresh (only when production lines are running)
    if st.session_state.simulator_started:
        time.sleep(2)  # 2 second refresh
        st.rerun()


# ==================== 多轴承生产线YOLO监控 ====================

def render_multi_bearing_monitoring(device=None):
    """渲染多轴承生产线YOLO并发监控界面"""

    if not MULTI_BEARING_AVAILABLE:
        st.error("❌ 多轴承监控系统不可用")
        st.info("请确保以下文件存在：\n- `core/multi_bearing_monitor.py`\n- 配置文件在 `configs/multi_bearing/`")

        # 显示详细错误信息
        if import_error_msg:
            st.warning(f"**导入错误详情**: {import_error_msg}")

        # 显示路径信息
        with st.expander("🔍 调试信息"):
            st.code(f"""
项目根目录: {project_root}
项目根目录存在: {project_root.exists()}
core目录: {project_root / 'core'}
core目录存在: {(project_root / 'core').exists()}
multi_bearing_monitor.py: {(project_root / 'core' / 'multi_bearing_monitor.py').exists()}
__init__.py: {(project_root / 'core' / '__init__.py').exists()}

sys.path[0]: {sys.path[0] if sys.path else 'N/A'}
            """)

            if import_traceback:
                st.error("**完整错误信息**:")
                st.code(import_traceback)

        st.warning("""
        **解决方案**:
        1. 按 **C** 键清除 Streamlit 缓存
        2. 或完全重启 Streamlit (Ctrl+C 然后重新运行)
        3. 如果还不行，运行测试页面: `streamlit run test_import.py`
        """)
        return

    st.caption("基于YOLOv8的多轴承生产线并发缺陷检测 - 单模型共享架构")

    # 初始化session state
    if "bearing_monitor" not in st.session_state:
        st.session_state.bearing_monitor = None
        st.session_state.bearing_monitor_running = False
    
    # MJPEG 服务器状态
    if "bearing_mjpeg_server_started" not in st.session_state:
        st.session_state.bearing_mjpeg_server_started = False
        st.session_state.bearing_mjpeg_port = 8890

    # 新增: 历史数据追踪
    if "monitoring_history" not in st.session_state:
        st.session_state.monitoring_history = {
            "timestamps": [],
            "total_frames": [],
            "total_defects": [],
            "fps": [],
            "inference_times": [],
            "line_defects": {}
        }
    
    # 新增: 自定义视频路径
    if "custom_video_path" not in st.session_state:
        st.session_state.custom_video_path = None
        st.session_state.custom_keyframe_interval = 5

    # ========== 左侧配置面板 ==========
    with st.sidebar:
        st.subheader("🔧 YOLO监控配置")

        # 配置文件选择
        st.markdown("**生产线配置**")

        config_options = {
            "1条生产线（基准）": "configs/multi_bearing/bearing_1_line.yaml",
            "2条生产线（并发）": "configs/multi_bearing/bearing_2_lines.yaml",
            "3条生产线（极限）": "configs/multi_bearing/bearing_3_lines.yaml",
            "6条生产线（满载压力测试）": "configs/multi_bearing/bearing_6_lines.yaml"
        }

        selected_config = st.selectbox(
            "选择配置",
            options=list(config_options.keys()),
            key="bearing_config_select"
        )

        config_path = config_options[selected_config]

        # 检查配置文件是否存在
        full_config_path = project_root / config_path
        config_exists = full_config_path.exists()

        if not config_exists:
            st.error(f"❌ 配置文件不存在: {config_path}")
            st.info("请运行项目根目录下的脚本生成配置文件和视频")

        st.divider()
        
        # ========== 新增: 自定义视频上传 ==========
        st.markdown("**📹 自定义视频源**")
        
        use_custom_video = st.checkbox(
            "使用自定义视频",
            value=False,
            key="use_custom_video",
            help="上传自己的视频进行检测"
        )
        
        if use_custom_video:
            uploaded_video = st.file_uploader(
                "上传视频文件",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader",
                help="支持 MP4, AVI, MOV, MKV 格式"
            )
            
            if uploaded_video is not None:
                # 保存上传的视频到临时文件
                temp_dir = tempfile.gettempdir()
                temp_video_path = os.path.join(temp_dir, f"custom_video_{int(time.time())}.mp4")
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                st.session_state.custom_video_path = temp_video_path
                st.success(f"✅ 视频已上传: {uploaded_video.name}")
                
                # 显示视频信息
                cap = cv2.VideoCapture(temp_video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    st.info(f"📊 视频信息:\n"
                           f"- 分辨率: {width}x{height}\n"
                           f"- 帧率: {fps:.1f} FPS\n"
                           f"- 时长: {duration:.1f}s\n"
                           f"- 总帧数: {frame_count}")
            elif st.session_state.custom_video_path:
                st.info(f"📁 当前视频: {os.path.basename(st.session_state.custom_video_path)}")
        else:
            st.session_state.custom_video_path = None

        st.divider()
        
        # ========== 新增: 检测频率配置 ==========
        st.markdown("**⚡ 检测频率设置**")
        
        keyframe_interval = st.slider(
            "检测间隔（帧）",
            min_value=1,
            max_value=30,
            value=st.session_state.custom_keyframe_interval,
            key="keyframe_slider",
            help="每隔多少帧执行一次检测。值越小，检测越频繁，但GPU负载越高"
        )
        st.session_state.custom_keyframe_interval = keyframe_interval
        
        # 显示检测频率说明
        detection_rate = 30 / keyframe_interval  # 假设30fps视频
        st.caption(f"📊 约 {detection_rate:.1f} 次检测/秒 (30fps视频)")
        
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="conf_slider",
            help="只显示置信度高于此值的检测结果"
        )

        st.divider()

        # ========== MJPEG 流配置 ==========
        st.markdown("**🎥 视频流配置**")

        if MJPEG_SERVER_AVAILABLE:
            mjpeg_port = st.number_input(
                "MJPEG 端口",
                min_value=8000,
                max_value=9999,
                value=st.session_state.bearing_mjpeg_port,
                key="mjpeg_port_input",
                help="MJPEG视频流服务端口（修改后需重启监控）"
            )
            st.session_state.bearing_mjpeg_port = mjpeg_port

            if st.session_state.bearing_mjpeg_server_started:
                st.success(f"✅ 流服务运行中: http://localhost:{mjpeg_port}")
                st.caption("点击链接可在浏览器中打开全屏视图")
            else:
                st.info("启动监控后，MJPEG流服务将自动启动")
        else:
            st.warning("⚠️ MJPEG服务不可用，将使用逐帧刷新模式")

        st.divider()

        # 显示配置信息
        with st.expander("📄 配置详情", expanded=False):
            if config_exists:
                import yaml
                try:
                    with open(full_config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    st.json({
                        "模型路径": config_data.get('global', {}).get('model_path', 'N/A'),
                        "设备": config_data.get('global', {}).get('device', 'N/A'),
                        "模型共享": config_data.get('global', {}).get('shared_model', False),
                        "生产线数量": len(config_data.get('lines', {}))
                    })

                    st.markdown("**生产线列表**:")
                    for line_key, line_config in config_data.get('lines', {}).items():
                        st.text(f"• {line_config.get('name', line_key)}")
                        if use_custom_video and st.session_state.custom_video_path:
                            st.text(f"  视频: 自定义上传")
                        else:
                            st.text(f"  视频: {line_config.get('video', 'N/A')}")
                        st.text(f"  检测间隔: 每{keyframe_interval}帧")

                except Exception as e:
                    st.error(f"配置读取失败: {e}")
            else:
                st.warning("配置文件不存在，无法显示详情")

        st.divider()

        # 启动/停止按钮
        if not st.session_state.bearing_monitor_running:
            if st.button("🚀 启动YOLO监控", use_container_width=True, disabled=not config_exists):
                with st.spinner("正在初始化监控系统..."):
                    try:
                        # 如果有自定义视频，修改配置
                        import yaml

                        # 读取原配置
                        with open(full_config_path, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)

                        # 将模型路径转换为绝对路径
                        model_path = config_data['global'].get('model_path', '')
                        if model_path and not os.path.isabs(model_path):
                            config_data['global']['model_path'] = str(project_root / model_path)

                        # 修改所有生产线的配置
                        for line_key in config_data.get('lines', {}):
                            # 更新检测间隔
                            config_data['lines'][line_key]['keyframe_interval'] = keyframe_interval

                            # 处理视频路径
                            if use_custom_video and st.session_state.custom_video_path:
                                # 使用自定义视频（已经是绝对路径）
                                config_data['lines'][line_key]['video'] = st.session_state.custom_video_path
                            else:
                                # 将视频路径转换为绝对路径
                                video_path = config_data['lines'][line_key].get('video', '')
                                if video_path and not os.path.isabs(video_path):
                                    config_data['lines'][line_key]['video'] = str(project_root / video_path)

                        # 修改置信度阈值
                        config_data['global']['conf_threshold'] = conf_threshold

                        # 保存到临时配置文件
                        temp_config_path = os.path.join(tempfile.gettempdir(), "temp_bearing_config.yaml")
                        with open(temp_config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(config_data, f, allow_unicode=True)

                        monitor = MultiBearingMonitor(temp_config_path)

                        monitor.start_all()
                        st.session_state.bearing_monitor = monitor
                        st.session_state.bearing_monitor_running = True
                        
                        # 清空历史数据
                        st.session_state.monitoring_history = {
                            "timestamps": [],
                            "total_frames": [],
                            "total_defects": [],
                            "fps": [],
                            "inference_times": [],
                            "line_defects": {}
                        }
                        
                        st.success("✅ 监控系统已启动")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"启动失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.success("✅ 监控运行中")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏸️ 停止监控", use_container_width=True, type="primary"):
                    if st.session_state.bearing_monitor:
                        st.session_state.bearing_monitor.stop_all()
                    st.session_state.bearing_monitor = None
                    st.session_state.bearing_monitor_running = False
                    # 停止 MJPEG 服务器
                    if MJPEG_SERVER_AVAILABLE and st.session_state.bearing_mjpeg_server_started:
                        try:
                            stop_multi_bearing_mjpeg_server()
                        except:
                            pass
                        st.session_state.bearing_mjpeg_server_started = False
                    st.success("监控已停止")
                    time.sleep(0.5)
                    st.rerun()
            
            with col2:
                if st.button("📊 导出报告", use_container_width=True):
                    # 导出统计报告
                    if st.session_state.monitoring_history["timestamps"]:
                        df = pd.DataFrame({
                            "时间戳": st.session_state.monitoring_history["timestamps"],
                            "总帧数": st.session_state.monitoring_history["total_frames"],
                            "总缺陷": st.session_state.monitoring_history["total_defects"],
                            "FPS": st.session_state.monitoring_history["fps"],
                            "推理时间(ms)": st.session_state.monitoring_history["inference_times"]
                        })
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 下载CSV",
                            csv,
                            f"monitoring_report_{int(time.time())}.csv",
                            "text/csv",
                            use_container_width=True
                        )

    # ========== 主显示区域 ==========

    if not st.session_state.bearing_monitor_running:
        st.info("👈 请在左侧配置面板选择配置并启动监控系统")

        # 显示方案说明
        st.markdown("### 🎯 多轴承生产线方案优势")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **💾 资源高效**
            - 单模型共享
            - 节省50-60%显存
            - 300-500MB总占用
            """)

        with col2:
            st.markdown("""
            **🏭 真实场景**
            - 专业轴承工厂
            - 多条平行生产线
            - 统一检测标准
            """)

        with col3:
            st.markdown("""
            **⚡ 高性能**
            - 3条线@25-30FPS
            - 推理延迟<30ms
            - 自动负载均衡
            """)

        st.divider()
        
        # ========== 新增: 功能特性介绍 ==========
        st.markdown("### ✨ 新增功能")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📹 自定义视频上传**
            - 支持 MP4/AVI/MOV/MKV
            - 自动检测视频参数
            - 循环播放支持
            """)
            
        with col2:
            st.markdown("""
            **⚡ 检测频率自定义**
            - 1-30帧间隔可调
            - 平衡精度与性能
            - 实时生效
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 实时数据统计**
            - 历史趋势图表
            - 缺陷分布饼图
            - 性能监控曲线
            """)
            
        with col2:
            st.markdown("""
            **📥 数据导出**
            - CSV格式报告
            - 完整检测日志
            - 一键下载
            """)

        st.divider()

        st.markdown("### 📋 快速开始指南")

        with st.expander("1️⃣ 准备工作", expanded=True):
            st.markdown("""
            **检查清单**:
            - ✅ 轴承检测模型已训练: `runs/train_bearing/.../best.pt`
            - ✅ 视频文件已生成: `videos/bearing_conveyor_*.mp4`
            - ✅ 配置文件已创建: `configs/multi_bearing/*.yaml`
            
            **如果文件缺失**，请在项目根目录运行:
            ```bash
            python start_multi_bearing_monitor.py
            ```
            """)

        with st.expander("2️⃣ 选择配置"):
            st.markdown("""
            - **1条生产线**: 性能基准测试，适合GPU调试
            - **2条生产线**: 并发能力测试，验证模型共享
            - **3条生产线**: 极限性能测试，模拟真实工厂
            """)

        with st.expander("3️⃣ 启动监控"):
            st.markdown("""
            点击左侧 **"🚀 启动YOLO监控"** 按钮，系统将:
            1. 加载共享YOLO模型
            2. 启动各生产线独立线程
            3. 开始实时视频检测
            4. 显示统计数据和性能指标
            """)

        return

    # 监控运行中 - 显示实时数据
    monitor = st.session_state.bearing_monitor

    if not monitor:
        st.error("监控对象丢失，请重新启动")
        return

    # 获取实时统计
    try:
        stats = monitor.get_aggregated_stats()
    except Exception as e:
        st.error(f"获取统计失败: {e}")
        return
    
    # ========== 新增: 更新历史数据 ==========
    history = st.session_state.monitoring_history
    current_time = time.strftime("%H:%M:%S")
    history["timestamps"].append(current_time)
    history["total_frames"].append(stats["total_frames"])
    history["total_defects"].append(stats["total_defects"])
    history["fps"].append(stats["avg_fps"])
    history["inference_times"].append(stats["avg_inference_ms"])
    
    # 限制历史数据长度（保留最近100个点）
    max_history = 100
    for key in ["timestamps", "total_frames", "total_defects", "fps", "inference_times"]:
        if len(history[key]) > max_history:
            history[key] = history[key][-max_history:]

    # ========== 总体统计卡片 ==========
    st.markdown("### 📊 系统总体统计")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "运行时长",
            f"{stats['elapsed_time']:.1f}s",
            help="监控系统已运行时间"
        )

    with col2:
        st.metric(
            "总检测帧数",
            f"{stats['total_frames']:,}",
            help="所有生产线累计处理的帧数"
        )

    with col3:
        st.metric(
            "检出缺陷",
            stats['total_defects'],
            delta=f"{stats['overall_defect_rate']:.2f}% 缺陷率",
            delta_color="inverse",
            help="所有生产线检出的缺陷总数"
        )

    with col4:
        st.metric(
            "平均吞吐量",
            f"{stats['avg_fps']:.1f} FPS",
            help="所有生产线的平均处理速度"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "生产线数",
            stats['total_lines'],
            help="当前运行的生产线数量"
        )

    with col2:
        st.metric(
            "平均推理时间",
            f"{stats['avg_inference_ms']:.2f}ms",
            help="单次YOLO推理的平均耗时"
        )

    with col3:
        # 使用真实GPU利用率数据
        gpu_stats = stats.get('gpu_stats', {})
        if gpu_stats.get('is_real_data', False):
            gpu_util = gpu_stats.get('gpu_util_percent', 0)
            st.metric(
                "GPU利用率",
                f"{gpu_util:.1f}%",
                help=f"真实GPU负载 ({gpu_stats.get('device_name', 'GPU')})"
            )
        else:
            gpu_util_estimate = min(stats['avg_inference_ms'] * stats['avg_fps'] / 1000 * 100, 100)
            st.metric(
                "GPU利用率（估算）",
                f"{gpu_util_estimate:.1f}%",
                help="基于推理时间和FPS估算的GPU负载"
            )

    with col4:
        # 使用真实显存数据
        gpu_stats = stats.get('gpu_stats', {})
        if gpu_stats.get('is_real_data', False) or gpu_stats.get('used_memory_mb', 0) > 0:
            used_mem = gpu_stats.get('used_memory_mb', 0)
            total_mem = gpu_stats.get('total_memory_mb', 0)
            mem_percent = gpu_stats.get('memory_percent', 0)
            st.metric(
                "显存占用",
                f"{used_mem:.0f}MB",
                delta=f"{mem_percent:.1f}% / {total_mem:.0f}MB",
                help="真实显存使用情况"
            )
        else:
            memory_estimate = 300 + (stats['total_lines'] - 1) * 50
            st.metric(
                "显存占用（估算）",
                f"~{memory_estimate}MB",
                help="基于生产线数量估算的显存占用"
            )

    st.divider()

    # ========== 实时视频预览 (MJPEG 流式显示) ==========
    st.markdown("### 🎬 实时检测预览")

    # 启动 MJPEG 服务器（如果尚未启动）
    mjpeg_port = st.session_state.bearing_mjpeg_port

    if MJPEG_SERVER_AVAILABLE and not st.session_state.bearing_mjpeg_server_started:
        try:
            mjpeg_server = start_multi_bearing_mjpeg_server(port=mjpeg_port)
            # 设置 frame_getter 让服务器自主获取帧
            mjpeg_server.set_frame_getter(monitor.get_all_latest_frames)
            st.session_state.bearing_mjpeg_server_started = True
        except Exception as e:
            st.warning(f"MJPEG服务器启动失败: {e}")
            mjpeg_server = None
    else:
        mjpeg_server = get_multi_bearing_mjpeg_server() if MJPEG_SERVER_AVAILABLE else None
        # 确保 frame_getter 已设置
        if mjpeg_server and not mjpeg_server._frame_getter:
            mjpeg_server.set_frame_getter(monitor.get_all_latest_frames)

    # 获取最新帧用于显示快照（不影响 MJPEG 流）
    try:
        latest_frames = monitor.get_all_latest_frames()
    except Exception as e:
        st.warning(f"获取视频帧失败: {e}")
        latest_frames = {}

    # 显示方式选择
    display_mode = st.radio(
        "显示模式",
        ["🎥 独立窗口（最流畅）", "🖼️ 页面内嵌入", "📸 静态快照"],
        horizontal=True,
        key="video_display_mode",
        help="独立窗口模式最流畅，不受页面刷新影响"
    )

    if display_mode == "🎥 独立窗口（最流畅）" and MJPEG_SERVER_AVAILABLE and mjpeg_server:
        # 独立窗口模式 - 完全不受 Streamlit 刷新影响
        st.success(f"📡 MJPEG 流服务运行中")

        col1, col2, col3 = st.columns(3)
        with col1:
            # 使用 HTML 链接打开新窗口
            st.markdown(f'''
            <a href="http://localhost:{mjpeg_port}/" target="_blank" style="
                display: inline-block;
                background: linear-gradient(90deg, #00C851, #007E33);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
            ">🖥️ 打开全屏监控窗口</a>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown(f'''
            <a href="http://localhost:{mjpeg_port}/combined_feed" target="_blank" style="
                display: inline-block;
                background: linear-gradient(90deg, #33B5E5, #0099CC);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
            ">📺 多路合并视图</a>
            ''', unsafe_allow_html=True)

        with col3:
            if latest_frames:
                line_links = " | ".join([
                    f'<a href="http://localhost:{mjpeg_port}/video_feed/{line_id}" target="_blank">{frame_data.get("name", f"线{line_id}")}</a>'
                    for line_id, frame_data in latest_frames.items()
                ])
                st.markdown(f"单独查看: {line_links}", unsafe_allow_html=True)

        st.info("💡 提示: 点击上方按钮在新窗口中查看流畅的实时视频流，本页面仅显示统计数据")

        # 显示静态缩略图（不频繁刷新）
        if latest_frames and len(latest_frames) > 0:
            st.markdown("**📷 当前快照** (每2秒更新)")
            num_lines = len(latest_frames)
            cols = st.columns(min(num_lines, 3))
            for idx, (line_id, frame_data) in enumerate(latest_frames.items()):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.caption(f"{frame_data.get('name', f'Line {line_id}')}")
                    frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                    # 缩小图片减少刷新卡顿
                    small_frame = cv2.resize(frame_rgb, (320, 240))
                    st.image(small_frame, use_container_width=True)

    elif display_mode == "🖼️ 页面内嵌入" and MJPEG_SERVER_AVAILABLE and mjpeg_server:
        # iframe 嵌入模式 - 会受刷新影响但集成度高
        st.warning("⚠️ 此模式受页面刷新影响，可能有轻微卡顿")

        num_lines = len(latest_frames) if latest_frames else 1
        iframe_height = 400 if num_lines <= 2 else 760

        # 嵌入合并视频流
        st.markdown("**📺 多路合并视图**")
        components.iframe(
            f"http://localhost:{mjpeg_port}/combined_feed",
            height=iframe_height,
            scrolling=False
        )
    else:
        # 静态快照模式 - 最稳定但不实时
        # 原有的逐帧刷新显示方式
        if latest_frames:
            # 根据生产线数量动态调整列数
            num_lines = len(latest_frames)
            cols = st.columns(min(num_lines, 3))  # 最多3列

            for idx, (line_id, frame_data) in enumerate(latest_frames.items()):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"**{frame_data['name']}**")
                    # 将BGR转换为RGB用于显示
                    frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, use_container_width=True, channels="RGB")
        else:
            st.info("⏳ 等待视频帧加载...")

    st.divider()

    # ========== 新增: 数据统计图表 ==========
    st.markdown("### 📈 实时数据图表")
    
    chart_tabs = st.tabs(["📊 系统总览趋势", "🏭 各生产线趋势", "🥧 缺陷分布", "📉 检测统计"])

    with chart_tabs[0]:
        # 系统总体性能趋势图表
        if len(history["timestamps"]) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**系统总FPS趋势**")
                fps_df = pd.DataFrame({
                    "时间": history["timestamps"],
                    "FPS": history["fps"]
                })
                st.line_chart(fps_df.set_index("时间"), height=200)
            
            with col2:
                st.markdown("**平均推理时间趋势 (ms)**")
                inference_df = pd.DataFrame({
                    "时间": history["timestamps"],
                    "推理时间": history["inference_times"]
                })
                st.line_chart(inference_df.set_index("时间"), height=200)
            
            # 检测帧数和缺陷数趋势
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**累计检测帧数**")
                frames_df = pd.DataFrame({
                    "时间": history["timestamps"],
                    "帧数": history["total_frames"]
                })
                st.area_chart(frames_df.set_index("时间"), height=200)
            
            with col2:
                st.markdown("**累计缺陷数**")
                defects_df = pd.DataFrame({
                    "时间": history["timestamps"],
                    "缺陷数": history["total_defects"]
                })
                st.area_chart(defects_df.set_index("时间"), height=200)

            # GPU监控趋势（如果有真实数据）
            gpu_stats = stats.get('gpu_stats', {})
            if gpu_stats.get('is_real_data', False):
                st.markdown("**🎮 GPU实时监控**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("GPU利用率", f"{gpu_stats.get('gpu_util_percent', 0):.1f}%")
                with col2:
                    st.metric("显存使用", f"{gpu_stats.get('used_memory_mb', 0):.0f}MB")
                with col3:
                    temp = gpu_stats.get('temperature', 0)
                    st.metric("温度", f"{temp}°C" if temp > 0 else "N/A")
                with col4:
                    power = gpu_stats.get('power_usage', 0)
                    st.metric("功耗", f"{power:.1f}W" if power > 0 else "N/A")
        else:
            st.info("数据收集中，请稍候...")
    
    with chart_tabs[1]:
        # 各生产线独立性能趋势
        st.markdown("**各生产线性能趋势监控**")

        try:
            line_histories = monitor.get_all_lines_performance_history()

            if line_histories:
                # 选择要查看的生产线
                line_names = [h['name'] for h in line_histories]
                selected_lines = st.multiselect(
                    "选择要监控的生产线",
                    line_names,
                    default=line_names[:3] if len(line_names) > 3 else line_names,
                    key="selected_lines_chart"
                )

                if selected_lines:
                    # 筛选选中的生产线数据
                    selected_histories = [h for h in line_histories if h['name'] in selected_lines]

                    # FPS对比图
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**各生产线FPS对比**")
                        fps_data = {}
                        max_len = 0
                        for h in selected_histories:
                            hist = h['history']
                            if hist.get('timestamps') and hist.get('fps'):
                                fps_data[h['name']] = hist['fps']
                                max_len = max(max_len, len(hist['timestamps']))

                        if fps_data and max_len > 1:
                            # 统一时间轴
                            ref_history = selected_histories[0]['history']
                            timestamps = ref_history.get('timestamps', [])[-max_len:]

                            fps_df_data = {"时间": timestamps}
                            for name, fps_list in fps_data.items():
                                # 补齐长度
                                padded = [0] * (max_len - len(fps_list)) + fps_list[-max_len:]
                                fps_df_data[name] = padded[:len(timestamps)]

                            if len(timestamps) > 0:
                                fps_df = pd.DataFrame(fps_df_data)
                                st.line_chart(fps_df.set_index("时间"), height=250)
                        else:
                            st.info("数据收集中...")

                    with col2:
                        st.markdown("**各生产线推理时间对比 (ms)**")
                        inference_data = {}
                        for h in selected_histories:
                            hist = h['history']
                            if hist.get('inference_times'):
                                inference_data[h['name']] = hist['inference_times']

                        if inference_data and max_len > 1:
                            ref_history = selected_histories[0]['history']
                            timestamps = ref_history.get('timestamps', [])[-max_len:]

                            inf_df_data = {"时间": timestamps}
                            for name, inf_list in inference_data.items():
                                padded = [0] * (max_len - len(inf_list)) + inf_list[-max_len:]
                                inf_df_data[name] = padded[:len(timestamps)]

                            if len(timestamps) > 0:
                                inf_df = pd.DataFrame(inf_df_data)
                                st.line_chart(inf_df.set_index("时间"), height=250)
                        else:
                            st.info("数据收集中...")

                    # 缺陷检测对比
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**各生产线缺陷数趋势**")
                        defect_data = {}
                        for h in selected_histories:
                            hist = h['history']
                            if hist.get('defect_counts'):
                                defect_data[h['name']] = hist['defect_counts']

                        if defect_data and max_len > 1:
                            ref_history = selected_histories[0]['history']
                            timestamps = ref_history.get('timestamps', [])[-max_len:]

                            def_df_data = {"时间": timestamps}
                            for name, def_list in defect_data.items():
                                padded = [0] * (max_len - len(def_list)) + def_list[-max_len:]
                                def_df_data[name] = padded[:len(timestamps)]

                            if len(timestamps) > 0:
                                def_df = pd.DataFrame(def_df_data)
                                st.area_chart(def_df.set_index("时间"), height=250)
                        else:
                            st.info("数据收集中...")

                    with col2:
                        st.markdown("**各生产线帧数趋势**")
                        frame_data = {}
                        for h in selected_histories:
                            hist = h['history']
                            if hist.get('frame_counts'):
                                frame_data[h['name']] = hist['frame_counts']

                        if frame_data and max_len > 1:
                            ref_history = selected_histories[0]['history']
                            timestamps = ref_history.get('timestamps', [])[-max_len:]

                            frm_df_data = {"时间": timestamps}
                            for name, frm_list in frame_data.items():
                                padded = [0] * (max_len - len(frm_list)) + frm_list[-max_len:]
                                frm_df_data[name] = padded[:len(timestamps)]

                            if len(timestamps) > 0:
                                frm_df = pd.DataFrame(frm_df_data)
                                st.area_chart(frm_df.set_index("时间"), height=250)
                        else:
                            st.info("数据收集中...")
                else:
                    st.info("请选择至少一条生产线进行监控")
            else:
                st.info("暂无生产线历史数据")
        except Exception as e:
            st.warning(f"获取生产线历史数据失败: {e}")

    with chart_tabs[2]:
        # 缺陷分布饼图
        defect_types = stats.get('defect_types', {})
        total_defects = sum(defect_types.values())
        
        if total_defects > 0:
            # 准备数据
            defect_labels = []
            defect_counts = []
            
            for cls_id, count in sorted(defect_types.items()):
                if count > 0:
                    defect_labels.append(f"类型{cls_id}")
                    defect_counts.append(count)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**缺陷类型分布**")
                pie_df = pd.DataFrame({
                    "缺陷类型": defect_labels,
                    "数量": defect_counts
                })
                st.bar_chart(pie_df.set_index("缺陷类型"), height=300)
            
            with col2:
                st.markdown("**缺陷详情**")
                detail_df = pd.DataFrame({
                    "缺陷类型": defect_labels,
                    "数量": defect_counts,
                    "占比": [f"{c/total_defects*100:.1f}%" for c in defect_counts]
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂未检出缺陷")
    
    with chart_tabs[3]:
        # 各生产线检测统计
        st.markdown("**各生产线检测统计**")
        
        line_data = []
        for line_stat in stats['lines']:
            line_data.append({
                "生产线": line_stat['name'],
                "检测帧数": line_stat['total_frames'],
                "缺陷数量": line_stat['detected_defects'],
                "缺陷率": f"{line_stat['defect_rate']:.2f}%",
                "推理时间(ms)": f"{line_stat['avg_inference_time_ms']:.2f}",
                "状态": "🟢 运行中" if line_stat['is_running'] else "🔴 已停止"
            })
        
        if line_data:
            st.dataframe(pd.DataFrame(line_data), use_container_width=True, hide_index=True)
            
            # 生产线对比柱状图
            st.markdown("**生产线缺陷对比**")
            compare_df = pd.DataFrame({
                "生产线": [d["生产线"] for d in line_data],
                "缺陷数量": [d["缺陷数量"] for d in line_data]
            })
            st.bar_chart(compare_df.set_index("生产线"), height=200)

    st.divider()

    # ========== 各生产线详情 ==========
    st.markdown("### 🏭 各生产线详细数据")

    for line_stat in stats['lines']:
        with st.expander(f"📹 {line_stat['name']}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("检测帧数", f"{line_stat['total_frames']:,}")

            with col2:
                st.metric("缺陷数量", line_stat['detected_defects'])

            with col3:
                st.metric("缺陷率", f"{line_stat['defect_rate']:.2f}%")

            with col4:
                st.metric("推理时间", f"{line_stat['avg_inference_time_ms']:.2f}ms")

            # 缺陷类型分布
            defect_types = line_stat['defect_types']
            has_defects = any(count > 0 for count in defect_types.values())

            if has_defects:
                st.markdown("**缺陷类型分布**")

                # 创建柱状图数据
                defect_labels = []
                defect_counts = []

                for cls_id, count in sorted(defect_types.items()):
                    if count > 0:
                        defect_labels.append(f"类型 {cls_id}")
                        defect_counts.append(count)

                # 使用st.bar_chart (简单但有效)
                df = pd.DataFrame({
                    '数量': defect_counts
                }, index=defect_labels)

                st.bar_chart(df, height=200)
            else:
                st.info("暂未检出缺陷")

    st.divider()

    # ========== 性能监控图表 ==========
    st.markdown("### 📈 性能趋势监控")

    tab1, tab2, tab3 = st.tabs(["📊 实时快照", "📋 配置信息", "📥 数据导出"])

    with tab1:
        st.markdown("**当前性能快照**")

        # 使用进度条显示各项指标
        st.markdown(f"**吞吐量**: {stats['avg_fps']:.1f} / 90 FPS (目标)")
        st.progress(min(stats['avg_fps'] / 90, 1.0))

        st.markdown(f"**推理延迟**: {stats['avg_inference_ms']:.2f} / 30 ms (目标)")
        st.progress(min(stats['avg_inference_ms'] / 30, 1.0))

        st.markdown(f"**缺陷检出率**: {stats['overall_defect_rate']:.2f}%")
        st.progress(stats['overall_defect_rate'] / 100)

    with tab2:
        st.markdown("**当前配置信息**")
        st.json({
            "配置文件": selected_config,
            "生产线数": stats['total_lines'],
            "运行状态": "✅ 运行中" if all(line['is_running'] for line in stats['lines']) else "⚠️ 部分异常",
            "模型共享": "启用",
            "检测间隔": f"每{keyframe_interval}帧",
            "置信度阈值": conf_threshold,
            "自定义视频": "是" if use_custom_video and st.session_state.custom_video_path else "否",
            "设备": device or "cuda:0"
        })
    
    with tab3:
        st.markdown("**导出监控数据**")
        
        if history["timestamps"]:
            # 创建完整报告数据
            report_data = {
                "运行时长(s)": stats['elapsed_time'],
                "总检测帧数": stats['total_frames'],
                "总缺陷数": stats['total_defects'],
                "缺陷率(%)": stats['overall_defect_rate'],
                "平均FPS": stats['avg_fps'],
                "平均推理时间(ms)": stats['avg_inference_ms'],
                "生产线数": stats['total_lines']
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 导出汇总报告
                summary_df = pd.DataFrame([report_data])
                csv_summary = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 下载汇总报告",
                    csv_summary,
                    f"summary_report_{int(time.time())}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # 导出历史数据
                history_df = pd.DataFrame({
                    "时间戳": history["timestamps"],
                    "总帧数": history["total_frames"],
                    "总缺陷": history["total_defects"],
                    "FPS": history["fps"],
                    "推理时间(ms)": history["inference_times"]
                })
                csv_history = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 下载历史数据",
                    csv_history,
                    f"history_data_{int(time.time())}.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.info("暂无数据可导出")

    # 自动刷新 - 根据显示模式调整刷新频率
    display_mode_current = st.session_state.get("video_display_mode", "🎥 独立窗口（最流畅）")
    if display_mode_current == "🎥 独立窗口（最流畅）":
        time.sleep(2)  # 独立窗口模式：2秒刷新统计数据（视频在独立窗口流畅播放）
    elif display_mode_current == "🖼️ 页面内嵌入":
        time.sleep(3)  # iframe模式：3秒刷新减少卡顿
    else:
        time.sleep(1)  # 静态快照模式：1秒刷新
    st.rerun()


