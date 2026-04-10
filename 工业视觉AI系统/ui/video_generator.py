"""视频生成器 UI - 批量推理 + 传送带视频生成

模式 1：数据集标注模式
- 上传原始数据集图片
- VLM 批量推理（返回 bbox）
- 生成带检测结果的传送带视频
"""

import os
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np
from PIL import Image

from core.vlm_batch_infer import batch_infer_images
from core.video_generator_core import generate_conveyor_video_with_detections


def render(device=None, sam_proc=None, sam_model=None, sam_dtype=None):
    """渲染视频生成器 UI"""

    st.header("🎬 传送带视频生成工具")
    st.caption("数据集批量推理 + 生成带检测结果的工业演示视频")

    # 初始化状态
    if "vg_step" not in st.session_state:
        st.session_state.vg_step = 1
        st.session_state.vg_images = []
        st.session_state.vg_image_names = []
        st.session_state.vg_detections = None
        st.session_state.vg_video_path = None
        st.session_state.vg_inference_log = []

    # 进度指示器
    step_labels = ["📂 上传数据集", "🔍 VLM推理", "📊 结果预览", "🎬 生成视频", "💾 下载"]

    # 创建进度条
    cols = st.columns(5)
    for i, label in enumerate(step_labels):
        with cols[i]:
            if i + 1 < st.session_state.vg_step:
                st.success(f"✅ {label}")
            elif i + 1 == st.session_state.vg_step:
                st.info(f"▶️ {label}")
            else:
                st.text(f"⏸️ {label}")

    progress = st.session_state.vg_step / 5
    st.progress(progress)

    st.divider()

    # 根据步骤渲染不同内容
    if st.session_state.vg_step == 1:
        render_step1_upload()
    elif st.session_state.vg_step == 2:
        render_step2_inference(device)
    elif st.session_state.vg_step == 3:
        render_step3_preview()
    elif st.session_state.vg_step == 4:
        render_step4_generate()
    elif st.session_state.vg_step == 5:
        render_step5_download()


def render_step1_upload():
    """步骤 1：数据集上传"""
    st.subheader("📂 步骤 1：上传数据集图片")

    st.info("💡 **提示**：上传工业产品的原始图片（无标注、无编号），系统将自动检测缺陷并生成带标注的传送带视频。")

    uploaded_files = st.file_uploader(
        "选择图片文件（支持批量选择）",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="vg_upload_files"
    )

    if uploaded_files:
        # 加载图片
        images = []
        image_names = []

        with st.spinner("正在加载图片..."):
            for file in uploaded_files:
                try:
                    img = Image.open(file).convert("RGB")
                    images.append(img)
                    image_names.append(file.name)
                except Exception as e:
                    st.warning(f"⚠️ 加载失败: {file.name} - {e}")

        if images:
            st.session_state.vg_images = images
            st.session_state.vg_image_names = image_names

            st.success(f"✅ 已加载 {len(images)} 张图片")

            # 统计信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("图片数量", len(images))
            with col2:
                avg_size = np.mean([img.size[0] * img.size[1] for img in images])
                st.metric("平均分辨率", f"{int(avg_size**0.5)}x{int(avg_size**0.5)}")
            with col3:
                st.metric("预计耗时", f"{len(images) * 2}~{len(images) * 5}秒")

            st.divider()

            # 预览前 5 张
            st.markdown("**📸 图片预览（前5张）**")
            cols = st.columns(min(5, len(images)))
            for i, img in enumerate(images[:5]):
                with cols[i]:
                    st.image(img, caption=f"#{i+1}\n{image_names[i][:15]}...", use_container_width=True)

            if len(images) > 5:
                st.caption(f"... 还有 {len(images) - 5} 张图片")

            st.divider()

            # 下一步按钮
            if st.button("➡️ 下一步：开始 VLM 推理", type="primary", use_container_width=True):
                st.session_state.vg_step = 2
                st.rerun()
    else:
        st.info("👆 请上传图片文件开始")

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 数据集要求
            - **格式**：JPG、PNG、BMP
            - **数量**：建议 10-100 张
            - **内容**：工业产品图片（螺丝、零件、电路板等）
            
            ### 推理过程
            1. 系统将调用 VLM 批量检测每张图片的缺陷
            2. 返回缺陷类型和 bbox 坐标
            3. 生成带标注的传送带演示视频
            
            ### 预计成本
            - 50 张图片：约 5 次 API 调用（批次大小 10）
            - 预计成本：¥0.25 - ¥0.50
            """)


def render_step2_inference(device):
    """步骤 2：VLM 批量推理"""
    st.subheader("🔍 步骤 2：VLM 批量推理")

    st.info(f"📊 共 {len(st.session_state.vg_images)} 张图片待推理")

    # 推理配置
    col1, col2, col3 = st.columns(3)

    with col1:
        from core.vlm_model_registry import list_models
        vlm_models = list_models(require='bbox')
        model = st.selectbox(
            "VLM 模型",
            options=vlm_models,
            index=vlm_models.index("qwen-vl-max") if "qwen-vl-max" in vlm_models else 0,
            key="vg_vlm_model"
        )

    with col2:
        batch_size = st.slider("批次大小", 4, 20, 10, key="vg_batch_size",
                               help="每批推理的图片数量，越大速度越快但单次调用时间越长")

    with col3:
        max_boxes = st.slider("最大检测框", 1, 10, 5, key="vg_max_boxes",
                             help="每张图片最多检测的缺陷数量")

    st.divider()

    # API Key 配置
    import os
    dashscope_key = st.session_state.get("dashscope_api_key_cached")
    if not dashscope_key:
        try:
            dashscope_key = st.secrets.get("dashscope_api_key", "")
        except:
            dashscope_key = ""
    if not dashscope_key:
        dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")

    api_key_input = st.text_input(
        "DashScope API Key",
        value=dashscope_key,
        type="password",
        key="vg_api_key",
        help="用于调用 VLM API"
    )
    st.session_state["dashscope_api_key_cached"] = api_key_input

    if not api_key_input:
        st.warning("⚠️ 请输入 DashScope API Key")
        return

    st.divider()

    # 推理控制
    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        if st.button("⬅️ 返回上一步", use_container_width=True):
            st.session_state.vg_step = 1
            st.rerun()

    with col_btn2:
        if st.button("🚀 开始批量推理", type="primary", use_container_width=True):
            # 执行推理
            with st.spinner("正在调用 VLM API 进行批量推理..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()

                try:
                    # 调用批量推理
                    detections = batch_infer_images(
                        images=st.session_state.vg_images,
                        model=model,
                        batch_size=batch_size,
                        max_boxes=max_boxes,
                        api_key=api_key_input,
                        progress_callback=lambda p, msg: (
                            progress_bar.progress(p),
                            status_text.text(msg),
                            log_container.text(msg)
                        )
                    )

                    st.session_state.vg_detections = detections
                    st.session_state.vg_step = 3

                    st.success("✅ 推理完成！")
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 推理失败: {e}")
                    st.exception(e)


def render_step3_preview():
    """步骤 3：结果预览"""
    st.subheader("📊 步骤 3：检测结果预览")

    detections = st.session_state.vg_detections
    images = st.session_state.vg_images

    if not detections:
        st.error("❌ 未找到检测结果")
        return

    # 统计信息
    total = len(detections)
    defects = sum(1 for d in detections.values() if d.get('has_defect'))
    defect_rate = defects / total * 100 if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总图片", total)
    with col2:
        st.metric("检出缺陷", defects, delta=f"{defect_rate:.1f}%", delta_color="inverse")
    with col3:
        st.metric("正常图片", total - defects)

    st.divider()

    # 缺陷类型统计
    defect_types = {}
    for det in detections.values():
        if det.get('has_defect'):
            for detection in det.get('detections', []):
                dtype = detection.get('type', 'unknown')
                defect_types[dtype] = defect_types.get(dtype, 0) + 1

    if defect_types:
        st.markdown("**🔍 缺陷类型分布**")
        cols = st.columns(len(defect_types))
        for i, (dtype, count) in enumerate(defect_types.items()):
            with cols[i]:
                st.metric(dtype.upper(), count)

    st.divider()

    # 详细结果
    tab1, tab2 = st.tabs(["🔴 检出缺陷的图片", "✅ 正常图片"])

    with tab1:
        defect_images = [(idx, det) for idx, det in detections.items() if det.get('has_defect')]

        if defect_images:
            for idx, det in defect_images:
                with st.expander(f"图片 #{idx+1}: {st.session_state.vg_image_names[idx]}", expanded=False):
                    col_img, col_info = st.columns([1, 1])

                    with col_img:
                        st.image(images[idx], use_container_width=True)

                    with col_info:
                        st.markdown(f"**检测结果**")
                        for i, detection in enumerate(det.get('detections', [])):
                            st.markdown(f"**缺陷 {i+1}**")
                            st.write(f"- 类型: `{detection.get('type', 'unknown')}`")
                            st.write(f"- 置信度: `{detection.get('confidence', 0):.2f}`")
                            bbox = detection.get('bbox', [])
                            if bbox:
                                st.write(f"- 位置: `{bbox}`")
                            desc = detection.get('description', '')
                            if desc:
                                st.write(f"- 描述: {desc}")
        else:
            st.info("🎉 没有检测到缺陷图片")

    with tab2:
        normal_images = [(idx, det) for idx, det in detections.items() if not det.get('has_defect')]

        if normal_images:
            cols = st.columns(4)
            for i, (idx, det) in enumerate(normal_images[:16]):
                with cols[i % 4]:
                    st.image(images[idx], caption=f"#{idx+1}", use_container_width=True)

            if len(normal_images) > 16:
                st.caption(f"... 还有 {len(normal_images) - 16} 张正常图片")
        else:
            st.info("所有图片都检出缺陷")

    st.divider()

    # 导出功能
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # 导出 JSON
        json_data = {
            'metadata': {
                'total_images': total,
                'defect_count': defects,
                'defect_rate': defect_rate,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'detections': {str(k): v for k, v in detections.items()}
        }
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 下载检测结果 (JSON)",
            json_str,
            "detections.json",
            "application/json",
            use_container_width=True
        )

    with col_export2:
        # 导出 CSV
        import io
        csv_buffer = io.StringIO()
        csv_buffer.write("图片编号,文件名,缺陷状态,缺陷类型,置信度\n")
        for idx, det in detections.items():
            status = "有缺陷" if det.get('has_defect') else "正常"
            if det.get('has_defect'):
                for detection in det.get('detections', []):
                    csv_buffer.write(f"{idx+1},{st.session_state.vg_image_names[idx]},{status},"
                                   f"{detection.get('type', '')},{detection.get('confidence', 0):.2f}\n")
            else:
                csv_buffer.write(f"{idx+1},{st.session_state.vg_image_names[idx]},{status},,\n")

        st.download_button(
            "📥 下载统计报告 (CSV)",
            csv_buffer.getvalue(),
            "report.csv",
            "text/csv",
            use_container_width=True
        )

    st.divider()

    # 导航按钮
    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        if st.button("⬅️ 返回上一步", use_container_width=True):
            st.session_state.vg_step = 2
            st.rerun()

    with col_btn2:
        if st.button("➡️ 下一步：生成传送带视频", type="primary", use_container_width=True):
            st.session_state.vg_step = 4
            st.rerun()


def render_step4_generate():
    """步骤 4：生成传送带视频"""
    st.subheader("🎬 步骤 4：生成传送带视频")

    st.info("💡 系统将生成一个模拟工业传送带的视频，展示所有产品并标注检测到的缺陷。")

    # 视频参数配置
    col1, col2, col3 = st.columns(3)

    with col1:
        width = st.selectbox("视频宽度", [1280, 1920, 2560], index=0, key="vg_video_width")

    with col2:
        height = st.selectbox("视频高度", [720, 1080, 1440], index=0, key="vg_video_height")

    with col3:
        fps = st.slider("帧率 (FPS)", 15, 60, 30, key="vg_video_fps")

    col4, col5 = st.columns(2)

    with col4:
        speed = st.slider("传送带速度 (px/frame)", 5, 20, 10, key="vg_video_speed")

    with col5:
        vibration = st.checkbox("启用振动效果", value=True, key="vg_video_vibration")

    st.divider()

    # 背景配置
    st.markdown("**🎨 自定义背景（可选）**")
    bg_option = st.radio(
        "选择背景",
        ["默认传送带纹理", "上传自定义图片"],
        key="vg_bg_option"
    )

    background_image = None
    if bg_option == "上传自定义图片":
        bg_file = st.file_uploader(
            "选择背景图片",
            type=["jpg", "jpeg", "png"],
            key="vg_bg_upload"
        )
        if bg_file:
            background_image = Image.open(bg_file)
            st.image(background_image, caption="背景预览", use_container_width=True)

    st.divider()

    # 生成按钮
    col_btn1, col_btn2 = st.columns([1, 4])

    with col_btn1:
        if st.button("⬅️ 返回上一步", use_container_width=True):
            st.session_state.vg_step = 3
            st.rerun()

    with col_btn2:
        if st.button("🎬 开始生成视频", type="primary", use_container_width=True):
            with st.spinner("正在生成视频..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # 生成临时输出路径
                    output_dir = Path(tempfile.gettempdir()) / "conveyor_videos"
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"conveyor_{int(time.time())}.mp4"

                    # 调用视频生成
                    generate_conveyor_video_with_detections(
                        images=st.session_state.vg_images,
                        detections=st.session_state.vg_detections,
                        output_file=str(output_path),
                        width=width,
                        height=height,
                        fps=fps,
                        speed=speed,
                        vibration=vibration,
                        background_image=background_image,
                        progress_callback=lambda p, msg: (
                            progress_bar.progress(p),
                            status_text.text(msg)
                        )
                    )

                    st.session_state.vg_video_path = str(output_path)
                    st.session_state.vg_step = 5

                    st.success("✅ 视频生成完成！")
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 视频生成失败: {e}")
                    st.exception(e)


def render_step5_download():
    """步骤 5：下载视频"""
    st.subheader("💾 步骤 5：视频已生成")

    video_path = st.session_state.vg_video_path

    if not video_path or not os.path.exists(video_path):
        st.error("❌ 视频文件未找到")
        return

    st.success("✅ 传送带视频生成完成！")

    # 视频预览
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()

    st.video(video_bytes)

    st.divider()

    # 视频信息
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("文件大小", f"{file_size:.1f} MB")
    with col2:
        st.metric("产品数量", len(st.session_state.vg_images))
    with col3:
        defects = sum(1 for d in st.session_state.vg_detections.values() if d.get('has_defect'))
        st.metric("检出缺陷", defects)

    st.divider()

    # 下载和操作按钮
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        st.download_button(
            "💾 下载视频 (MP4)",
            video_bytes,
            f"conveyor_video_{int(time.time())}.mp4",
            "video/mp4",
            use_container_width=True
        )

    with col_btn2:
        if st.button("🔄 重新生成视频", use_container_width=True):
            st.session_state.vg_step = 4
            st.rerun()

    with col_btn3:
        if st.button("🆕 开始新任务", use_container_width=True):
            # 清空状态
            st.session_state.vg_step = 1
            st.session_state.vg_images = []
            st.session_state.vg_image_names = []
            st.session_state.vg_detections = None
            st.session_state.vg_video_path = None
            st.rerun()

