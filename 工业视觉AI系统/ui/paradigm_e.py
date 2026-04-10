"""范式E：SAM3直接视频检测与跟踪

核心特性：
- 无需VLM推理，直接使用SAM3文本prompt检测
- 本地推理，速度快（<200ms/帧）
- 零边际成本（无需云端API）
- 支持多prompt并行检测（scratch、dent、crack等）
- 批量视频处理能力
"""

import streamlit as st
import time
from pathlib import Path
from datetime import datetime
import tempfile
import json

from PIL import Image

from core.sam3_video_detector import SAM3VideoDefectDetector, get_preset_prompts


def render(*, device: str, sam_video_model=None, sam_video_processor=None):
    """渲染范式E界面"""

    st.markdown("### 📹 范式E：SAM3直接视频检测与跟踪")
    st.caption("💡 跳过VLM推理 | 本地SAM3文本prompt | 速度快11倍 | 零边际成本")

    # 显示核心优势
    with st.expander("✨ 核心特性与优势", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **技术特性**：
            - ✅ 本地SAM3推理（无需VLM）
            - ✅ 文本prompt驱动检测
            - ✅ 自动跨帧跟踪
            - ✅ 支持多prompt并行
            - ✅ 完全离线运行
            """)
        with col2:
            st.markdown("""
            **性能优势**：
            - ⚡ 推理速度快11倍（vs 范式C）
            - 💰 零边际成本（无API费用）
            - 📊 批量处理不受限流
            - 🔒 数据本地化（安全）
            - 🌐 离线环境可用
            """)

    # 侧边栏配置
    with st.sidebar:
        st.markdown("## ⚙️ 检测配置")

        # Prompt配置
        with st.expander("📝 Prompt配置", expanded=True):
            preset_templates = get_preset_prompts()
            preset_name = st.selectbox(
                "选择预设模板",
                list(preset_templates.keys()),
                help="根据不同产线场景选择合适的prompt模板"
            )

            default_prompts = preset_templates[preset_name]
            selected_prompts = st.multiselect(
                "检测目标（可多选）",
                default_prompts,
                default=default_prompts[:2],
                help="选择要检测的缺陷类型，可以同时检测多种类型"
            )

            custom_prompt = st.text_input(
                "自定义prompt（英文）",
                placeholder="例如: scratch on metal surface",
                help="如果预设不满足需求，可以自定义英文描述"
            )

            if custom_prompt:
                selected_prompts.append(custom_prompt)

            if not selected_prompts:
                st.warning("⚠️ 请至少选择一个prompt")

        # 检测参数
        with st.expander("🎚️ 检测参数", expanded=True):
            threshold = st.slider(
                "置信度阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="低于此阈值的检测将被过滤"
            )

            max_frames = st.number_input(
                "最大处理帧数",
                min_value=10,
                max_value=10000,
                value=500,
                step=50,
                help="限制处理的帧数，0表示处理全部"
            )

            if max_frames == 0:
                max_frames = None

    # 主界面：标签页
    tab_single, tab_batch = st.tabs(["🎬 单视频检测", "📦 批量处理（未来）"])

    with tab_single:
        render_single_video_mode(
            device=device,
            sam_video_model=sam_video_model,
            sam_video_processor=sam_video_processor,
            selected_prompts=selected_prompts,
            threshold=threshold,
            max_frames=max_frames,
        )

    with tab_batch:
        st.info("📌 批量处理功能正在开发中，敬请期待...")
        st.markdown("""
        **计划功能**：
        - 📂 支持批量上传多个视频
        - 🔄 并行处理多个视频
        - 📊 批量统计报告
        - 💾 批量导出结果
        """)


def render_single_video_mode(
    *,
    device: str,
    sam_video_model,
    sam_video_processor,
    selected_prompts: list[str],
    threshold: float,
    max_frames: int | None,
):
    """渲染单视频检测模式"""

    # 步骤1：视频上传
    st.markdown("#### 📤 步骤1：上传视频")

    video_file = st.file_uploader(
        "选择视频文件",
        type=["mp4", "avi", "mov", "mkv"],
        help="支持常见视频格式，建议<500MB"
    )

    if video_file is not None:
        # 显示视频预览
        st.video(video_file)

        # 加载视频帧
        if st.button("🔄 加载视频帧", type="secondary"):
            with st.spinner("📥 正在加载视频帧..."):
                try:
                    video_frames, fps = load_video_from_uploaded_file(video_file)

                    st.session_state["e_video_frames"] = video_frames
                    st.session_state["e_video_fps"] = fps
                    st.session_state["e_video_filename"] = video_file.name

                    st.success(f"✅ 视频加载成功：{len(video_frames)}帧，{fps} FPS")

                    # 显示视频信息
                    col1, col2, col3 = st.columns(3)
                    col1.metric("总帧数", len(video_frames))
                    col2.metric("帧率", f"{fps} FPS")
                    col3.metric("时长", f"{len(video_frames)/fps:.1f}秒")

                except Exception as e:
                    st.error(f"❌ 视频加载失败：{e}")

    # 步骤2：运行检测
    if st.session_state.get("e_video_frames") is not None:
        st.markdown("---")
        st.markdown("#### 🎯 步骤2：运行检测")

        # 显示当前配置
        st.info(
            f"📋 **检测配置**\n\n"
            f"- **Prompt**: {', '.join(selected_prompts) if selected_prompts else '（未选择）'}\n"
            f"- **置信度阈值**: {threshold}\n"
            f"- **最大处理帧数**: {max_frames if max_frames else '全部'}"
        )

        if not selected_prompts:
            st.warning("⚠️ 请在左侧配置至少一个检测prompt")
        else:
            if st.button("🚀 开始检测", type="primary", use_container_width=True):
                run_video_detection(
                    device=device,
                    sam_video_model=sam_video_model,
                    sam_video_processor=sam_video_processor,
                    selected_prompts=selected_prompts,
                    threshold=threshold,
                    max_frames=max_frames,
                )

    # 步骤3：结果展示
    if st.session_state.get("e_results") is not None:
        st.markdown("---")
        st.markdown("#### 📊 步骤3：检测结果")

        display_detection_results(st.session_state["e_results"])

        # 导出功能
        st.markdown("---")
        st.markdown("#### 💾 步骤4：导出结果")
        export_results_section(st.session_state["e_results"])


def load_video_from_uploaded_file(video_file) -> tuple[list[Image.Image], float]:
    """从上传的文件加载视频帧

    Returns:
        (video_frames, fps)
    """
    import cv2

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name

    try:
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR转RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转为PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

        cap.release()

        return frames, fps

    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


def run_video_detection(
    *,
    device: str,
    sam_video_model,
    sam_video_processor,
    selected_prompts: list[str],
    threshold: float,
    max_frames: int | None,
):
    """运行视频检测"""

    video_frames = st.session_state["e_video_frames"]

    # 创建检测器
    detector = SAM3VideoDefectDetector(device=device)
    detector.load_models(sam_video_model, sam_video_processor)

    # 显示进度
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    try:
        status_text.text("🔍 正在初始化SAM3视频检测器...")
        progress_bar.progress(10)

        status_text.text(f"🎬 正在检测 {len(selected_prompts)} 个prompt...")
        progress_bar.progress(30)

        # 运行检测
        results = detector.detect_defects_in_video(
            video_frames=video_frames,
            prompts=selected_prompts,
            threshold=threshold,
            max_frames=max_frames,
        )

        progress_bar.progress(90)
        status_text.text("📊 正在生成统计报告...")

        # 保存结果
        st.session_state["e_results"] = results
        st.session_state["e_detection_time"] = time.time() - start_time

        progress_bar.progress(100)
        status_text.empty()

        st.success(
            f"✅ 检测完成！\n\n"
            f"- 耗时：{results['statistics']['inference_time_sec']:.2f}秒\n"
            f"- 处理速度：{results['statistics']['fps']:.2f} FPS\n"
            f"- 检出实例：{results['statistics']['total_unique_instances']}个"
        )

        # 自动滚动到结果区域
        st.rerun()

    except Exception as e:
        st.error(f"❌ 检测失败：{e}")
        import traceback
        st.code(traceback.format_exc())


def display_detection_results(results: dict):
    """显示检测结果"""

    stats = results["statistics"]

    # 总体统计
    st.markdown("##### 📈 总体统计")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("唯一实例数", stats["total_unique_instances"])
    col2.metric("总检测次数", stats["total_detections"])
    col3.metric("处理帧数", stats["frames_processed"])
    col4.metric("推理速度", f"{stats['fps']:.1f} FPS")

    col1, col2 = st.columns(2)
    col1.metric("总耗时", f"{stats['inference_time_sec']:.2f}秒")
    col2.metric("平均每帧", f"{stats['avg_time_per_frame_ms']:.1f}ms")

    # 各prompt详情
    st.markdown("##### 🔍 各Prompt检测详情")

    for prompt, data in results["prompt_results"].items():
        if "error" in data:
            with st.expander(f"❌ {prompt} - 检测失败"):
                st.error(f"错误：{data['error']}")
            continue

        total_instances = data["total_instances"]
        total_detections = data["total_detections"]
        num_frames_with_detection = len(data["frames"])

        with st.expander(
            f"🎯 **{prompt}** - {total_instances}个实例 | {total_detections}次检测 | {num_frames_with_detection}帧",
            expanded=(total_instances > 0)
        ):
            if total_instances == 0:
                st.info("✅ 未检测到此类型的缺陷")
            else:
                # 统计信息
                col1, col2, col3 = st.columns(3)
                col1.metric("唯一实例", total_instances)
                col2.metric("总检测次数", total_detections)
                col3.metric("检出帧数", num_frames_with_detection)

                # 按帧展示（采样显示前10帧）
                st.markdown("**部分检测帧预览**")
                frame_indices = sorted(data["frames"].keys())

                for frame_idx in frame_indices[:10]:
                    frame_data = data["frames"][frame_idx]

                    st.markdown(f"- **帧 {frame_idx}**: {frame_data['num_instances']}个实例")

                    # 显示详细信息（可折叠）
                    details = []
                    for obj_id, score, box in zip(
                        frame_data["object_ids"],
                        frame_data["scores"],
                        frame_data["boxes"]
                    ):
                        details.append(
                            f"  - ID={obj_id}, 置信度={score:.3f}, "
                            f"Bbox=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
                        )

                    st.code("\n".join(details), language="")

                if len(frame_indices) > 10:
                    st.info(f"（仅显示前10帧，共{len(frame_indices)}帧有检测）")

                # 绘制时间轴
                st.markdown("**缺陷出现时间轴**")
                plot_detection_timeline(frame_indices, stats["frames_processed"])


def plot_detection_timeline(detection_frames: list[int], total_frames: int):
    """绘制缺陷检测时间轴"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots(figsize=(10, 2))

    # 绘制时间轴
    ax.broken_barh([(f, 1) for f in detection_frames], (0, 1), facecolors='red', alpha=0.6)
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 1)
    ax.set_xlabel('帧索引')
    ax.set_yticks([])
    ax.set_title(f'缺陷出现位置（红色=检出，共{len(detection_frames)}帧）')
    ax.grid(True, axis='x', alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


def export_results_section(results: dict):
    """导出结果区域"""

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 导出JSON", use_container_width=True):
            export_json_results(results)

    with col2:
        if st.button("📊 导出CSV", use_container_width=True):
            export_csv_results(results)


def export_json_results(results: dict):
    """导出JSON格式结果"""

    # 移除不可序列化的对象
    export_data = {
        "prompt_results": {},
        "statistics": results["statistics"],
        "export_time": datetime.now().isoformat(),
    }

    for prompt, data in results["prompt_results"].items():
        if "error" in data:
            continue

        export_data["prompt_results"][prompt] = {
            "total_instances": data["total_instances"],
            "total_detections": data["total_detections"],
            "unique_instance_ids": data.get("unique_instance_ids", []),
            "frames": {
                str(frame_idx): {
                    "object_ids": frame_data["object_ids"],
                    "scores": frame_data["scores"],
                    "boxes": frame_data["boxes"],
                    "num_instances": frame_data["num_instances"],
                }
                for frame_idx, frame_data in data["frames"].items()
            }
        }

    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

    filename = f"paradigm_e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    st.download_button(
        label="💾 下载JSON文件",
        data=json_str,
        file_name=filename,
        mime="application/json",
    )

    st.success(f"✅ JSON文件已准备好下载: {filename}")


def export_csv_results(results: dict):
    """导出CSV格式结果"""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # 写入表头
    writer.writerow(["帧索引", "Prompt类型", "对象ID", "置信度", "边界框X1", "边界框Y1", "边界框X2", "边界框Y2"])

    # 写入数据
    for prompt, data in results["prompt_results"].items():
        if "error" in data:
            continue

        for frame_idx, frame_data in data["frames"].items():
            for obj_id, score, box in zip(
                frame_data["object_ids"],
                frame_data["scores"],
                frame_data["boxes"]
            ):
                writer.writerow([
                    frame_idx,
                    prompt,
                    obj_id,
                    f"{score:.4f}",
                    f"{box[0]:.1f}",
                    f"{box[1]:.1f}",
                    f"{box[2]:.1f}",
                    f"{box[3]:.1f}",
                ])

    csv_str = output.getvalue()
    filename = f"paradigm_e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    st.download_button(
        label="💾 下载CSV文件",
        data=csv_str.encode('utf-8-sig'),  # BOM for Excel
        file_name=filename,
        mime="text/csv",
    )

    st.success(f"✅ CSV文件已准备好下载: {filename}")

