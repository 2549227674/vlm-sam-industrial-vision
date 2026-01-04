"""Paradigm A UI rendering."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from core.cv_utils import apply_mask_overlay_np
from core.vlm_model_registry import list_models, is_stream_only_model

from ui.mask_viz import MaskGroup, overlay_masks_by_class
from ui.components import UIComponents, LoadingStates  # ✅ 导入组件库

from ui.adapters import (
    parse_prompt_to_text_input,
    run_sam3_instance_segmentation,
    get_vlm_suggestions,
    get_dashscope_key,
    dashscope_ready,
)


# ============ 批量处理辅助函数 ============

def _safe_stem(name: str) -> str:
    """清理文件名，移除不安全字符"""
    s = (name or "image").strip().replace("\\", "_").replace("/", "_")
    for ch in [":", "*", "?", '"', "<", ">", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "_")
    s = s.strip(" .")
    return s or "image"


def _ensure_dir(path: str) -> str:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    """写入 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_vis_image(path: str, img_rgb: np.ndarray) -> None:
    """保存可视化图像"""
    Image.fromarray(img_rgb.astype(np.uint8)).save(path)


def run_paradigm_a_once(
    *,
    device: str,
    sam_proc,
    sam_model,
    sam_dtype,
    image_pil: Image.Image,
    prompt: str | list[str],
    threshold: float = 0.3,
    alpha: float = 0.5,
    multi_prompt_strategy: str = "per_prompt",
) -> tuple[dict[str, Any], np.ndarray]:
    """
    运行范式 A 单次推理并返回结果

    Args:
        device: 设备（cuda/cpu）
        sam_proc: SAM-3 处理器
        sam_model: SAM-3 模型
        sam_dtype: 数据类型
        image_pil: 输入图像
        prompt: 文本提示词（str 或 list[str]）
        threshold: SAM-3 阈值
        alpha: 掩码透明度
        multi_prompt_strategy: 多关键词策略

    Returns:
        (result_json, vis_rgb): 结果 JSON 和可视化图像
    """
    image_pil = image_pil.convert("RGB")
    raw = np.array(image_pil)
    H, W = int(image_pil.size[1]), int(image_pil.size[0])

    start_time = time.time()

    # 解析 Prompt
    text_input = parse_prompt_to_text_input(prompt)

    # 运行 SAM-3 推理
    groups: list[MaskGroup] | None = None
    if isinstance(text_input, (list, tuple)) and multi_prompt_strategy == "per_prompt":
        # 逐词推理并按类别分组
        groups = []
        words = [str(x).strip() for x in text_input if str(x).strip()]
        merged_masks = []
        merged_scores = []

        for w in words:
            r, _ = run_sam3_instance_segmentation(
                image_pil=image_pil,
                sam_proc=sam_proc,
                sam_model=sam_model,
                sam_dtype=sam_dtype,
                prompt=str(w),
                threshold=threshold,
                device=device,
            )
            m = r.get("masks")
            s = r.get("scores")

            if m is not None and getattr(m, "numel", lambda: 0)() > 0 and len(m) > 0:
                m_np = m.cpu().numpy() > 0.5
                groups.append(MaskGroup(label=str(w), masks=m_np))
                merged_masks.append(m)
            else:
                groups.append(MaskGroup(label=str(w), masks=np.zeros((0, H, W), dtype=bool)))

            if s is not None and getattr(s, "numel", lambda: 0)() > 0 and len(s) > 0:
                merged_scores.append(s.float())

        import torch
        if merged_masks and merged_scores:
            results = {"masks": torch.cat(merged_masks, dim=0), "scores": torch.cat(merged_scores, dim=0)}
        else:
            results = {"masks": torch.empty((0, 1, 1)), "scores": torch.empty((0,))}
    else:
        # 单次推理
        results, _ = run_sam3_instance_segmentation(
            image_pil=image_pil,
            sam_proc=sam_proc,
            sam_model=sam_model,
            sam_dtype=sam_dtype,
            prompt=text_input,
            threshold=threshold,
            device=device,
        )

    latency_ms = (time.time() - start_time) * 1000.0

    # 构建结果 JSON
    masks = results.get("masks")
    scores = results.get("scores")

    if masks is None or len(masks) == 0:
        # 未检测到
        result_json = {
            "mode": "paradigm_a",
            "image": {"w": W, "h": H},
            "prompt": str(prompt),
            "prompt_parsed": text_input if isinstance(text_input, list) else [text_input],
            "multi_prompt_strategy": multi_prompt_strategy,
            "threshold": float(threshold),
            "inference_results": {
                "total_instances": 0,
                "instances_by_class": {},
                "instances": []
            },
            "final": {
                "decision": "not_detected",
                "num_classes": 0,
                "total_instances": 0,
                "total_coverage": 0.0,
                "max_score": 0.0,
                "avg_score": 0.0,
                "latency_ms": float(latency_ms)
            }
        }
        return result_json, raw

    # 有检测结果
    masks_np = masks.cpu().numpy() > 0.5
    scores_np = scores.float().cpu().numpy()
    n_instances = len(masks_np)

    # 按类别统计
    instances_by_class = {}
    if groups and multi_prompt_strategy == "per_prompt":
        idx_cursor = 0
        for g in groups:
            g_len = len(g.masks)
            if g_len > 0:
                g_scores = scores_np[idx_cursor:idx_cursor + g_len]
                g_masks = masks_np[idx_cursor:idx_cursor + g_len]
                g_coverage = float(np.sum([np.sum(m) for m in g_masks]) / (H * W))

                instances_by_class[g.label] = {
                    "count": int(g_len),
                    "scores": [float(s) for s in g_scores],
                    "avg_score": float(np.mean(g_scores)),
                    "max_score": float(np.max(g_scores)),
                    "coverage_ratio": float(g_coverage)
                }
            idx_cursor += g_len

    # 全部实例列表
    instances = []
    if groups and multi_prompt_strategy == "per_prompt":
        idx_cursor = 0
        for g in groups:
            for i in range(len(g.masks)):
                if idx_cursor + i < len(scores_np):
                    instances.append({
                        "class": g.label,
                        "score": float(scores_np[idx_cursor + i]),
                        "mask_area": int(np.sum(masks_np[idx_cursor + i])),
                        "mask_area_ratio": float(np.sum(masks_np[idx_cursor + i]) / (H * W))
                    })
            idx_cursor += len(g.masks)
    else:
        for i in range(n_instances):
            instances.append({
                "class": "detected",
                "score": float(scores_np[i]),
                "mask_area": int(np.sum(masks_np[i])),
                "mask_area_ratio": float(np.sum(masks_np[i]) / (H * W))
            })

    # 计算总体统计
    total_coverage = float(np.sum([np.sum(m) for m in masks_np]) / (H * W))
    max_score = float(np.max(scores_np))
    avg_score = float(np.mean(scores_np))
    num_classes = len(instances_by_class) if instances_by_class else 1

    result_json = {
        "mode": "paradigm_a",
        "image": {"w": W, "h": H},
        "prompt": str(prompt),
        "prompt_parsed": text_input if isinstance(text_input, list) else [text_input],
        "multi_prompt_strategy": multi_prompt_strategy,
        "threshold": float(threshold),
        "inference_results": {
            "total_instances": int(n_instances),
            "instances_by_class": instances_by_class,
            "instances": instances
        },
        "final": {
            "decision": "detected",
            "num_classes": int(num_classes),
            "total_instances": int(n_instances),
            "total_coverage": float(total_coverage),
            "max_score": float(max_score),
            "avg_score": float(avg_score),
            "latency_ms": float(latency_ms)
        }
    }

    # 生成可视化
    if groups and multi_prompt_strategy == "per_prompt":
        vis_pil = overlay_masks_by_class(raw, groups, alpha=float(alpha), cmap_name="tab10")
        vis_rgb = np.array(vis_pil.convert("RGB"))
    else:
        # 单色叠加
        merged = np.any(masks_np, axis=0)
        mg = MaskGroup(label="detected", masks=merged[None, ...])
        vis_pil = overlay_masks_by_class(raw, [mg], alpha=float(alpha), cmap_name="tab10")
        vis_rgb = np.array(vis_pil.convert("RGB"))

    return result_json, vis_rgb


def _flatten_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    """将结果 JSON 扁平化为 CSV 行"""
    final = result.get("final", {})
    return {
        "name": result.get("name", ""),
        "prompt": result.get("prompt", ""),
        "strategy": result.get("multi_prompt_strategy", ""),
        "threshold": result.get("threshold", 0.0),
        "decision": final.get("decision", ""),
        "num_classes": final.get("num_classes", 0),
        "total_instances": final.get("total_instances", 0),
        "total_coverage": final.get("total_coverage", 0.0),
        "max_score": final.get("max_score", 0.0),
        "avg_score": final.get("avg_score", 0.0),
        "latency_ms": final.get("latency_ms", 0.0),
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """写入 CSV 文件"""
    import csv

    if not rows:
        return

    # 合并所有键
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============ 主渲染函数 ============


def render(*, device: str, sam_proc, sam_model, sam_dtype) -> None:
    st.markdown("### 范式 A：在线开放词汇推理")

    tab_a1, tab_a2, tab_batch = st.tabs([
        "A1 文本/推荐词推理",
        "A2 示例引导 Prompt（Exemplar→VLM）",
        "A3 批量处理",
    ])

    def _update_prompt(tag_value: str):
        tag_value = (tag_value or "").strip()
        if not tag_value:
            return

        if st.session_state.get("a_prompt_add_mode", True):
            current = str(st.session_state.get("a_prompt_input", "") or "").strip()
            current_parts = [p.strip() for p in re.split(r"[\n,;]+", current) if p.strip()]

            out_parts = []
            seen = set()
            for p in current_parts + [tag_value]:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    out_parts.append(p)

            st.session_state["a_prompt_input"] = ", ".join(out_parts)
        else:
            st.session_state["a_prompt_input"] = tag_value

    def _clear_results():
        st.session_state.update({
            "a_last_results": None,
            "a_last_latency_ms": None,
            "a_last_uploaded_name": None,
            "a_last_threshold": None,
            "a_selected_mask_idx": 0,
            "a_last_parsed_prompts": None,
            "a_prompt_mode_label": None,
        })

    with st.sidebar:
        st.markdown("## ⚙️ 参数设置")

        # === 基础参数 ===
        with st.expander(" 基础参数", expanded=True):
            threshold = st.slider(
                "置信度阈值",
                0.0, 1.0, 0.3,
                help="🎚️ 控制 SAM-3 分割的敏感度，值越高越严格"
            )

            alpha = st.slider(
                "掩码透明度",
                0.0, 1.0, 0.5,
                help="调节透明度会实时刷新叠加结果（不重新推理）"
            )

        st.markdown("---")
        st.info(f"🖥️ 当前设备: **{device.upper()}**")

    # ---------------- Tab A1 ----------------
    with tab_a1:
        left, right = st.columns([1.0, 1.0], gap="large")

        with left:
            st.markdown("### 📤 图片上传")
            uploaded_file = st.file_uploader(
                "上传测试图片",
                type=["jpg", "png", "jpeg"],
                key="a_upload",
                help="支持 JPG、PNG、JPEG 格式"
            )

            raw_image = None
            if uploaded_file:
                raw_image = Image.open(uploaded_file).convert("RGB")
                st.session_state["a_raw_rgb"] = np.array(raw_image)

                st.success(f"✅ 已加载: {uploaded_file.name}")
                st.markdown("#### 📷 图片预览")
                st.image(raw_image, use_container_width=True)
            else:
                st.info("💡 请上传一张图片开始分析")

        with right:
            st.markdown("### ⚙️ 交互控制")

            prompt = st.text_input(
                "输入文本提示",
                key="a_prompt_input",
                placeholder="例如: screw 或 screw, transistor 或 scratch\ncrack",
                help=(
                    "支持逗号/换行分隔多个关键词。\n"
                    "注意：根据 SAM-3 官方/Transformers 示例，text=list 主要用于批量（多张图片）推理，"
                    "不等价于‘单张图片多概念一次推理’。本系统对多关键词提供两种稳定策略：逐词推理合并 / 拼接成一句话。"
                ),
            )

            st.selectbox(
                "多关键词推理策略",
                options=["per_prompt", "join_string"],
                index=["per_prompt", "join_string"].index(st.session_state.get("a_multi_prompt_strategy", "per_prompt")),
                key="a_multi_prompt_strategy",
                format_func=lambda x: {"per_prompt": "逐词分别推理并合并（推荐，最稳定）", "join_string": "逗号拼接成一句话（更快，但不保证效果）"}[x],
                help=(
                    "per_prompt：对每个词分别跑一次 SAM-3，再把 masks/scores 合并（推荐，最稳定）。\n"
                    "join_string：把多个词逗号拼接成一句话一次推理（更快，但效果不稳定）。"
                ),
            )

            st.toggle(
                "VLM 词汇点击：追加到 Prompt（而不是覆盖）",
                key="a_prompt_add_mode",
                help="开启后：连续点击多个推荐词会用逗号自动追加并去重。关闭后：恢复为点击即覆盖。",
            )

            st.markdown("---")

            # === VLM 推荐 Prompt 区域 ===
            with st.expander(" VLM 推荐 Prompt", expanded=True):
                st.caption("使用 VLM 自动识别图片中的关键对象，生成推荐词汇")

                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    model_options = list_models(require="suggestions") or ["qwen-vl-max"]
                    current_model = str(st.session_state.get("a_vlm_model", "") or "").strip() or model_options[0]
                    if current_model not in model_options:
                        current_model = model_options[0]
                        st.session_state["a_vlm_model"] = current_model

                    st.session_state["a_vlm_model"] = st.selectbox(
                        "VLM 模型",
                        options=model_options,
                        index=model_options.index(current_model),
                    )
                with mcol2:
                    st.session_state["a_vlm_mode"] = st.selectbox(
                        "VLM 推理模式",
                        options=["general", "industrial_defect", "daily_damage"],
                        index=["general", "industrial_defect", "daily_damage"].index(
                            st.session_state.get("a_vlm_mode", "industrial_defect")
                            if st.session_state.get("a_vlm_mode") in ["general", "industrial_defect", "daily_damage"]
                            else "industrial_defect"
                        ),
                        format_func=lambda x: {
                            "general": "通用描述",
                            "industrial_defect": "工业缺陷专用",
                            "daily_damage": "日常物体损坏/差异",
                        }[x],
                    )

                # ✅ 检测是否选择了 QVQ 模型（与范式 C 保持一致）
                selected_model = st.session_state.get("a_vlm_model", "qwen-vl-max")
                is_qvq = is_stream_only_model(selected_model)

                # 如果是 QVQ，显示警告提示
                if is_qvq:
                    st.warning("🧠 QVQ 是【仅思考模型】，总是进行深度推理（无法关闭），因此下方思考开关对 QVQ 无效。")

                # 思考模式开关（QVQ 时禁用）
                st.toggle(
                    "开启 VLM 思考模式（更准但更慢/更易输出冗余）",
                    value=bool(st.session_state.get("a_vlm_thinking", False)),
                    key="a_vlm_thinking",
                    disabled=is_qvq,  # ✅ QVQ 模型禁用此开关
                )

                # 动态提示文字
                if is_qvq:
                    st.caption("💡 QVQ 总是开启深度思考（约800-2200字符推理过程），无需手动控制。")
                else:
                    st.caption("💡 非 QVQ 模型可以通过此开关控制是否启用思考模式。")

                effective_key = get_dashscope_key()
                if dashscope_ready(effective_key):
                    st.success("✅ VLM 已就绪")
                else:
                    st.warning("⚠️ VLM 未配置（需要 DASHSCOPE_API_KEY）")

                st.markdown("---")

                gen1, gen2 = st.columns([1, 1])
                with gen1:
                    gen_vlm_btn = st.button(
                        "🎯 生成推荐词",
                        type="primary",
                        use_container_width=True,
                        disabled=not bool(uploaded_file),
                        key="a1_gen_vlm_btn"
                    )
                with gen2:
                    st.button(
                        "🗑️ 清空结果",
                        use_container_width=True,
                        key="a1_clear_vlm_btn",
                        on_click=lambda: st.session_state.update({
                            "a_vlm_tags": None,
                            "a_vlm_desc_zh": "",
                            "a_vlm_desc_en": "",
                            "a_vlm_last_uploaded_name": None,
                        }),
                    )

                if gen_vlm_btn and uploaded_file:
                    selected_model = st.session_state.get("a_vlm_model", "qwen-vl-max")
                    selected_mode = st.session_state.get("a_vlm_mode", "industrial_defect")
                    with st.spinner("🔄 正在调用 VLM 生成推荐词..."):
                        vlm_out = get_vlm_suggestions(
                            image_pil=raw_image,
                            api_key=effective_key,
                            model_name=selected_model,
                            mode=selected_mode,
                        )
                    st.session_state["a_vlm_tags"] = getattr(vlm_out, "tags_en", [])
                    st.session_state["a_vlm_desc_zh"] = getattr(vlm_out, "desc_zh", "")
                    st.session_state["a_vlm_desc_en"] = getattr(vlm_out, "desc_en", "")
                    st.session_state["a_vlm_last_uploaded_name"] = uploaded_file.name

                desc_zh = str(st.session_state.get("a_vlm_desc_zh", "") or "").strip()
                desc_en = str(st.session_state.get("a_vlm_desc_en", "") or "").strip()
                if desc_zh or desc_en:
                    with st.expander("📝 查看整体描述（仅提示）", expanded=False):
                        if desc_zh:
                            st.markdown(f"**中文**: {desc_zh}")
                        if desc_en:
                            st.markdown(f"**English**: {desc_en}")

                tags = st.session_state.get("a_vlm_tags") or []
                if tags:
                    st.markdown("##### 🏷️ 推荐词/短语")
                    st.caption("点击词汇即可填入上方文本提示框")
                    cols = st.columns(3)
                    for i, tag in enumerate(tags[:9]):
                        with cols[i % 3]:
                            st.button(
                                f"🔖 {tag}",
                                key=f"a_tag_{i}_{tag}",
                                use_container_width=True,
                                on_click=_update_prompt,
                                args=(tag,),
                            )

            st.markdown("---")

            # === 执行分析按钮区域 ===
            st.markdown("##### 🚀 执行分析")
            run_col1, run_col2 = st.columns([1, 1])
            with run_col1:
                run_btn = st.button(
                    "▶️ 开始分析",
                    type="primary",
                    use_container_width=True,
                    disabled=not bool(uploaded_file) or not bool(st.session_state.get("models_ready")),
                    key="a1_run_analysis_btn"
                )
            with run_col2:
                st.button("🔄 清除结果", use_container_width=True, on_click=_clear_results, key="a1_clear_results_btn")

        if run_btn and uploaded_file:
            inference_progress = st.progress(0)
            inference_status = st.empty()

            t0 = time.time()

            inference_status.text("正在准备输入...")
            inference_progress.progress(20)

            text_input = parse_prompt_to_text_input(prompt)
            st.session_state["a_last_parsed_prompts"] = text_input
            inference_status.text(f"正在进行语义分割... Prompt={text_input}")
            inference_progress.progress(35)

            # 关键：为了实现“同类同色、不同类不同色”，只有在 per_prompt + 多关键词 时
            # 才能拿到每个关键词(类)对应的 masks。join_string 是混合语义，无法可靠分配到类。
            groups: list[MaskGroup] | None = None
            if isinstance(text_input, (list, tuple)) and st.session_state.get("a_multi_prompt_strategy") == "per_prompt":
                groups = []
                words = [str(x).strip() for x in text_input if str(x).strip()]
                merged_masks = []
                merged_scores = []

                for wi, w in enumerate(words):
                    inference_status.text(f"正在分割：{w} ({wi+1}/{len(words)})")
                    inference_progress.progress(35 + int(50 * (wi + 1) / max(1, len(words))))

                    r, _lat = run_sam3_instance_segmentation(
                        image_pil=raw_image,
                        sam_proc=sam_proc,
                        sam_model=sam_model,
                        sam_dtype=sam_dtype,
                        prompt=str(w),
                        threshold=threshold,
                        device=device,
                    )
                    m = r.get("masks")
                    s = r.get("scores")

                    if m is not None and getattr(m, "numel", lambda: 0)() > 0 and len(m) > 0:
                        m_np = m.cpu().numpy() > 0.5
                        groups.append(MaskGroup(label=str(w), masks=m_np))
                        merged_masks.append(m)
                    else:
                        # 空组：shape=(0,H,W)
                        H = int(raw_image.size[1])
                        W = int(raw_image.size[0])
                        groups.append(MaskGroup(label=str(w), masks=np.zeros((0, H, W), dtype=bool)))

                    if s is not None and getattr(s, "numel", lambda: 0)() > 0 and len(s) > 0:
                        merged_scores.append(s.float())

                import torch

                if merged_masks and merged_scores:
                    results = {"masks": torch.cat(merged_masks, dim=0), "scores": torch.cat(merged_scores, dim=0)}
                else:
                    results = {"masks": torch.empty((0, 1, 1)), "scores": torch.empty((0,))}

                latency = (time.time() - t0) * 1000.0
            else:
                results, latency = run_sam3_instance_segmentation(
                    image_pil=raw_image,
                    sam_proc=sam_proc,
                    sam_model=sam_model,
                    sam_dtype=sam_dtype,
                    prompt=text_input,
                    threshold=threshold,
                    device=device,
                )

            mode_label = st.session_state.get("a_prompt_mode_label")
            if isinstance(text_input, (list, tuple)):
                if mode_label == "multi:per_prompt":
                    st.success("多关键词模式：逐词分别推理并合并结果（per_prompt）。")
                elif mode_label == "multi:join_string":
                    st.info("多关键词模式：拼接成一句话一次推理（join_string）。")
            else:
                st.caption("当前为单关键词/单描述推理模式。")

            inference_progress.progress(100)
            inference_status.text(f"推理完成！")
            time.sleep(0.1)
            inference_progress.empty()
            inference_status.empty()

            st.session_state["a_last_results"] = results
            st.session_state["a_last_latency_ms"] = latency
            st.session_state["a_last_uploaded_name"] = uploaded_file.name
            st.session_state["a_last_threshold"] = threshold
            st.session_state["a_selected_mask_idx"] = 0
            st.session_state["a_last_class_groups"] = groups

        st.markdown("---")
        st.markdown("## 📊 分析结果")

        col_show1, col_show2 = st.columns(2, gap="large")

        with col_show1:
            st.markdown("### 🖼️ 原始图像")
            if raw_image is not None:
                st.image(raw_image, use_container_width=True)
            else:
                st.info("💡 等待上传图片")

        with col_show2:
            st.markdown("###  检测结果")

            parsed = st.session_state.get("a_last_parsed_prompts")
            mode_label = st.session_state.get("a_prompt_mode_label")
            if parsed is not None:
                st.caption(f"Prompt 解析结果: {parsed} | 推理策略: {mode_label}")

            results = st.session_state.get("a_last_results")
            raw_np2 = st.session_state.get("a_raw_rgb")

            if results and raw_np2 is not None and len(results.get("masks", [])) > 0:
                masks = results["masks"].cpu().numpy()
                scores = results["scores"].float().cpu().numpy()

                n = int(masks.shape[0])

                last_thr = st.session_state.get("a_last_threshold")
                if last_thr is not None and abs(float(last_thr) - float(threshold)) > 1e-9:
                    st.warning("你已调整 Threshold。结果仍基于上次推理的阈值；如需应用新阈值，请重新点击 ‘开始分析’。")

                show_k1, _ = st.columns([1, 1])
                with show_k1:
                    if n <= 1:
                        st.caption("当前只检测到 1 个实例，Top-K 固定为 1。")
                        st.session_state["a_topk"] = 1
                    else:
                        st.slider(
                            "显示 Top-K 实例",
                            1,
                            min(10, n),
                            min(int(st.session_state.get("a_topk", 5)), min(10, n)),
                            key="a_topk",
                        )

                topk = int(st.session_state.get("a_topk", 1))
                order = np.argsort(scores)[::-1]
                topk_idx = [int(i) for i in order[: min(topk, n)]]

                st.markdown("---")
                st.markdown("#### 📌 显示模式")
                st.toggle(
                    "🔀 多实例叠加模式",
                    key="a_multi_view",
                    help="同时显示多个 mask 在一张图上，无需重新推理"
                )

                if st.session_state.get("a_multi_view"):
                    st.info("💡 勾选多个实例后将自动叠加显示（支持按类别着色）")
                else:
                    st.info("💡 点击网格中的实例切换单个查看")

                if st.session_state.get("a_multi_view"):
                    st.markdown("#### ☑️ 实例勾选（勾选后自动叠加）")

                    current_selected = set(int(i) for i in (st.session_state.get("a_selected_mask_idxs") or []))
                    current_selected = {i for i in current_selected if i in topk_idx}

                    if not current_selected:
                        for i in topk_idx[: min(3, len(topk_idx))]:
                            current_selected.add(int(i))

                    grid_cols = st.columns(min(5, len(topk_idx)))
                    new_selected = set()

                    for j, mi in enumerate(topk_idx):
                        with grid_cols[j % len(grid_cols)]:
                            mi = int(mi)
                            thumb_mask = masks[mi] > 0.5

                            # 如果有按类别分组的结果，则用“类别颜色”画缩略图；否则回退旧红色
                            groups = st.session_state.get("a_last_class_groups")
                            if groups:
                                # 找到该 mask 属于哪个 group：按 concat 顺序映射
                                # groups 中每个 MaskGroup.masks 的长度决定其占用的 mask index 区间
                                idx_cursor = 0
                                color_img = None
                                for g in groups:
                                    g_len = int(np.asarray(g.masks).shape[0])
                                    if idx_cursor <= mi < idx_cursor + g_len:
                                        # 只画这一个实例，但颜色取该类
                                        g_single = MaskGroup(label=g.label, masks=(masks[mi:mi+1] > 0.5))
                                        color_img = overlay_masks_by_class(raw_np2, [g_single], alpha=0.55)
                                        break
                                    idx_cursor += g_len
                                if color_img is None:
                                    thumb = apply_mask_overlay_np(raw_np2, thumb_mask, alpha=0.55, color=(255, 0, 0))
                                else:
                                    thumb = np.array(color_img.convert("RGB"))
                            else:
                                thumb = apply_mask_overlay_np(raw_np2, thumb_mask, alpha=0.55, color=(255, 0, 0))

                            st.image(thumb, use_container_width=True)

                            checked = st.checkbox(
                                f"#{mi}  {float(scores[mi]):.3f}",
                                value=(mi in current_selected),
                                key=f"a_chk_{mi}",
                            )
                            if checked:
                                new_selected.add(mi)

                    st.session_state["a_selected_mask_idxs"] = sorted(list(new_selected))

                    if not new_selected:
                        st.info("请至少勾选 1 个实例用于叠加显示。")
                    else:
                        groups = st.session_state.get("a_last_class_groups")
                        if groups:
                            # 选中的实例按类别取并集后上色叠加
                            idx_cursor = 0
                            sel_groups: list[MaskGroup] = []
                            for g in groups:
                                g_len = int(np.asarray(g.masks).shape[0])
                                # 该类别中哪些实例被选中
                                rel = [mi - idx_cursor for mi in new_selected if idx_cursor <= mi < idx_cursor + g_len]
                                if rel:
                                    gm = np.asarray(g.masks)
                                    keep = gm[np.array(rel, dtype=int)]
                                    sel_groups.append(MaskGroup(label=g.label, masks=keep))
                                idx_cursor += g_len

                            vis_pil = overlay_masks_by_class(raw_np2, sel_groups, alpha=float(alpha))
                            st.image(
                                np.array(vis_pil.convert("RGB")),
                                use_container_width=True,
                                caption=f"多实例叠加(按类别着色): {len(new_selected)} 个 (idx={sorted(list(new_selected))})",
                            )
                        else:
                            merged = np.zeros(masks[int(next(iter(new_selected)))].shape, dtype=bool)
                            for mi in new_selected:
                                merged |= (masks[int(mi)] > 0.5)

                            vis_multi = apply_mask_overlay_np(raw_np2, merged, alpha=alpha, color=(255, 0, 0))
                            st.image(
                                vis_multi,
                                use_container_width=True,
                                caption=f"多实例叠加: {len(new_selected)} 个 (idx={sorted(list(new_selected))})",
                            )

                        cov = float(np.sum([np.sum(masks[int(mi)] > 0.5) for mi in new_selected]) / masks.shape[1] / masks.shape[2] * 100)
                        max_s = float(np.max([scores[int(mi)] for mi in new_selected]))
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("叠加覆盖率(粗略)", f"{cov:.1f}%")
                        with mc2:
                            st.metric("叠加中最高置信度", f"{max_s:.4f}")
                        with mc3:
                            if st.session_state.get("a_last_latency_ms") is not None:
                                st.metric("推理耗时", f"{st.session_state['a_last_latency_ms']:.1f}ms")
                else:
                    st.markdown("#### 🖱️ 点击选择实例")

                    def _select_mask(i: int):
                        st.session_state["a_selected_mask_idx"] = int(i)

                    grid_cols = st.columns(min(5, len(topk_idx)))
                    for j, mi in enumerate(topk_idx):
                        with grid_cols[j % len(grid_cols)]:
                            thumb_mask = masks[mi] > 0.5
                            thumb = apply_mask_overlay_np(raw_np2, thumb_mask, alpha=0.55, color=(255, 0, 0))
                            st.image(thumb, use_container_width=True)
                            st.button(
                                f"#{mi}",
                                key=f"a_pick_{mi}",
                                use_container_width=True,
                                on_click=_select_mask,
                                args=(mi,),
                            )
                            st.caption(f"📊 {float(scores[mi]):.3f}")

                    st.divider()

                    show_sel1, show_sel2 = st.columns([1, 1])
                    with show_sel1:
                        st.markdown("#####  精确切换")
                    with show_sel2:
                        options = topk_idx
                        fmt = lambda i: f"#{i}  置信度={scores[i]:.4f}"
                        selected = st.selectbox(
                            "选择实例",
                            options=options,
                            index=0 if st.session_state.get("a_selected_mask_idx") not in options else options.index(
                                int(st.session_state.get("a_selected_mask_idx"))
                            ),
                            format_func=fmt,
                            key="a_instance_select",
                            label_visibility="collapsed"
                        )
                        st.session_state["a_selected_mask_idx"] = int(selected)

                    idx = int(st.session_state.get("a_selected_mask_idx", 0))
                    idx = max(0, min(idx, n - 1))

                    best_mask = masks[idx] > 0.5
                    vis = apply_mask_overlay_np(raw_np2, best_mask, alpha=alpha, color=(255, 0, 0))
                    st.image(vis, use_container_width=True, caption=f"实例 #{idx} (score={scores[idx]:.4f})")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("当前实例置信度", f"{float(scores[idx]):.4f}")
                    with metric_col2:
                        if st.session_state.get("a_last_latency_ms") is not None:
                            st.metric("推理耗时", f"{st.session_state['a_last_latency_ms']:.1f}ms")
                    with metric_col3:
                        mask_pixels = float(np.sum(best_mask))
                        mask_area = float(mask_pixels / float(best_mask.size) * 100.0)
                        st.metric("覆盖率", f"{mask_area:.1f}%")

                    with st.expander("查看 Top-K 列表", expanded=False):
                        for rank, mi in enumerate(topk_idx, start=1):
                            st.write(f"{rank}. mask #{int(mi)}  score={float(scores[mi]):.4f}")

    # ---------------- Tab A2 ----------------
    with tab_a2:
        from streamlit_cropper import st_cropper

        st.markdown("##  示例引导 Prompt 生成")
        st.info(
            "📝 **工作流程**: 上传示例图 → 框选 ROI 区域 → VLM 自动识别 → 生成推荐词 → 写入 A1 的 Prompt\n\n"
            "💡 **说明**: 此功能仅生成 Prompt，不执行 SAM-3 推理。生成后请切回 A1 点击「开始分析」"
        )

        st.markdown("---")

        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.markdown("### 📤 上传示例图")
            ex_file = st.file_uploader(
                "选择 Exemplar 图片",
                type=["jpg", "png", "jpeg"],
                key="a2_exemplar",
                help="上传包含目标对象的示例图片"
            )
        with ex_col2:
            st.markdown("### 💡 操作提示")
            st.info("📌 在图片上拖拽红色裁剪框\n\n🎯 调整框选目标对象区域")

        if ex_file is None:
            st.warning("⚠️ 请先上传一张示例图片")
        else:
            ex_img = Image.open(ex_file).convert("RGB")

            st.markdown("---")
            st.markdown("### ✂️ 步骤 1: 框选 ROI 区域")
            st.caption("拖拽红色裁剪框调整 ROI 范围，系统将分析框内内容")

            roi = st_cropper(
                ex_img,
                realtime_update=True,
                box_color='#FF4B4B',
                aspect_ratio=None,
                return_type='image'
            )

            if roi is not None and roi.size[0] > 0 and roi.size[1] > 0:
                st.success(f"✅ ROI 已选择：{roi.size[0]} × {roi.size[1]} 像素")

            st.markdown("---")
            st.markdown("###  步骤 2: VLM 生成候选词")
            m1, m2, m3 = st.columns(3)
            with m1:
                model_options = list_models(require="suggestions") or ["qwen-vl-max"]
                current_model = str(st.session_state.get("a_vlm_model", "") or "").strip() or model_options[0]
                if current_model not in model_options:
                    current_model = model_options[0]

                vlm_model = st.selectbox(
                    "VLM 模型",
                    options=model_options,
                    index=model_options.index(current_model),
                    key="a2_vlm_model",
                )
            with m2:
                vlm_mode = st.selectbox(
                    "VLM 推理模式",
                    options=["general", "industrial_defect", "daily_damage"],
                    index=["general", "industrial_defect", "daily_damage"].index(st.session_state.get("a_vlm_mode", "industrial_defect")),
                    format_func=lambda x: {"general": "通用描述", "industrial_defect": "工业缺陷专用", "daily_damage": "日常物体损坏/差异"}[x],
                    key="a2_vlm_mode",
                )
            with m3:
                max_k = st.slider("返回数量", 3, 10, 6, key="a2_vlm_k")

            st.session_state.setdefault("a2_tags", [])
            st.session_state.setdefault("a2_desc_zh", "")
            st.session_state.setdefault("a2_desc_en", "")

            gen_btn = st.button(
                " 生成候选词（基于 ROI）",
                type="primary",
                use_container_width=True,
                key="a2_gen"
            )

            if gen_btn:
                effective_key = get_dashscope_key()
                with st.spinner("🔄 正在调用 VLM 识别 ROI 内容..."):
                    out = get_vlm_suggestions(
                        image_pil=roi,
                        api_key=effective_key,
                        model_name=str(vlm_model),
                        mode=str(vlm_mode),
                        max_keywords=int(max_k),
                    )
                st.session_state["a2_tags"] = list(getattr(out, "tags_en", []) or [])
                st.session_state["a2_desc_zh"] = str(getattr(out, "desc_zh", "") or "")
                st.session_state["a2_desc_en"] = str(getattr(out, "desc_en", "") or "")

            if st.session_state.get("a2_desc_zh") or st.session_state.get("a2_desc_en"):
                with st.expander("📝 整体描述（仅供参考）", expanded=False):
                    if st.session_state.get("a2_desc_zh"):
                        st.markdown(f"**中文**: {st.session_state['a2_desc_zh']}")
                    if st.session_state.get("a2_desc_en"):
                        st.markdown(f"**English**: {st.session_state['a2_desc_en']}")

            tags = st.session_state.get("a2_tags") or []
            if tags:
                st.markdown("---")
                st.markdown("### 🏷️ 步骤 3: 点击写入 A1 Prompt")
                st.caption("点击下方词汇会写入/追加到 A1 的 Prompt 输入框（遵循 A1 的「追加/覆盖」开关设置）")

                cols = st.columns(3)
                for i, tag in enumerate(tags[:12]):
                    with cols[i % 3]:
                        st.button(
                            f"🔖 {tag}",
                            key=f"a2_tag_{i}_{tag}",
                            use_container_width=True,
                            on_click=_update_prompt,
                            args=(tag,),
                        )

                st.success("✅ 词汇已写入后，请切回 **A1 标签页** 并点击「开始分析」")

    # ---------------- Tab A3 批量处理 ----------------
    with tab_batch:
        st.markdown("## 📦 批量处理")
        st.info(
            "🚀 **批量处理流程**: 上传多张图片 → 配置统一参数 → 自动推理所有图片 → 导出结果（JSON + CSV + 可视化图）\n\n"
            "💡 **提示**: 可以复用 A1 的 Prompt 参数，所有图片将使用相同配置"
        )

        # 提示可以复用 A1 的参数
        a1_prompt = st.session_state.get("a_prompt_input", "")
        if a1_prompt:
            st.success(f"💡 A1 当前 Prompt: `{a1_prompt}`")

        st.markdown("---")
        st.markdown("### 1️⃣ 参数配置")

        # 处理复用 A1 的逻辑
        if st.session_state.get("a3_copy_from_a1", False):
            st.session_state["a3_prompt_value"] = a1_prompt
            st.session_state["a3_copy_from_a1"] = False

        col1, col2 = st.columns([3, 1])
        with col1:
            batch_prompt = st.text_input(
                "统一 Prompt（必填）",
                value=st.session_state.get("a3_prompt_value", a1_prompt),
                key="a3_prompt",
                placeholder="例如: screw, transistor",
                help="所有图片将使用此 Prompt 进行推理"
            )
            # 同步到独立的 session_state 变量
            st.session_state["a3_prompt_value"] = batch_prompt
        with col2:
            if st.button("📋 复用 A1", use_container_width=True, help="复制 A1 当前的 Prompt"):
                st.session_state["a3_copy_from_a1"] = True
                st.rerun()

        col_strat, col_thr, col_alpha = st.columns(3)
        with col_strat:
            batch_strategy = st.selectbox(
                "多关键词策略",
                options=["per_prompt", "join_string"],
                index=["per_prompt", "join_string"].index(st.session_state.get("a_multi_prompt_strategy", "per_prompt")),
                format_func=lambda x: {"per_prompt": "逐词推理（推荐）", "join_string": "拼接推理"}[x],
                key="a3_strategy",
            )
        with col_thr:
            batch_threshold = st.slider(
                "SAM-3 阈值",
                0.0, 1.0,
                st.session_state.get("a_last_threshold", 0.3) if st.session_state.get("a_last_threshold") else 0.3,
                key="a3_threshold"
            )
        with col_alpha:
            batch_alpha = st.slider(
                "掩码透明度",
                0.0, 1.0, 0.5,
                key="a3_alpha"
            )

        st.markdown("---")
        st.markdown("### 2️⃣ 批量上传图片")

        batch_files = st.file_uploader(
            "选择多张图片（支持批量选择）",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg'],
            key="a3_batch_upload",
            help="按住 Ctrl/Cmd 可多选文件"
        )

        if batch_files:
            st.success(f"✅ 已选择 **{len(batch_files)}** 张图片")
            with st.expander("📋 查看文件列表", expanded=False):
                for i, f in enumerate(batch_files[:20], 1):
                    st.caption(f"{i}. {f.name}")
                if len(batch_files) > 20:
                    st.caption(f"... 以及其他 {len(batch_files) - 20} 个文件")

        st.markdown("---")
        st.markdown("### 3️⃣ 开始批量处理")

        # 检查按钮启用条件
        has_files = batch_files is not None and len(batch_files) > 0
        has_prompt = batch_prompt is not None and batch_prompt.strip() != ""
        models_ready = bool(st.session_state.get("models_ready", False))

        # 显示状态提示
        status_cols = st.columns(3)
        with status_cols[0]:
            if has_files:
                st.success(f"✅ 图片: {len(batch_files)} 张")
            else:
                st.error("❌ 未上传图片")
        with status_cols[1]:
            if has_prompt:
                st.success(f"✅ Prompt 已设置")
            else:
                st.error("❌ 未输入 Prompt")
        with status_cols[2]:
            if models_ready:
                st.success("✅ 模型已就绪")
            else:
                st.error("❌ 模型未初始化")

        st.markdown("---")

        col_run, col_clear = st.columns(2)
        with col_run:
            run_batch_btn = st.button(
                "🚀 开始批量处理",
                type="primary",
                use_container_width=True,
                disabled=not (has_files and has_prompt and models_ready),
                key="a3_run_batch_btn"
            )
        with col_clear:
            if st.button("🗑️ 清空结果", use_container_width=True, key="a3_clear_results_btn"):
                st.session_state["a3_batch_results"] = None
                st.session_state["a3_output_dir"] = None
                st.rerun()

        if run_batch_btn and batch_files and batch_prompt:
            # 创建输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = _ensure_dir(os.path.join("paradigm_a", f"batch_{timestamp}"))
            vis_dir = _ensure_dir(os.path.join(output_dir, "visualizations"))

            st.markdown("---")
            with st.container():
                st.markdown("### ⏳ 处理进度")

                results = []
                detected_count = 0
                not_detected_count = 0
                total_latency = 0.0

                for i, file in enumerate(batch_files):
                    # 使用组件库的进度追踪器
                    UIComponents.progress_tracker(
                        current=i+1,
                        total=len(batch_files),
                        status_text=f"📊 正在处理: {file.name}",
                        show_percentage=True
                    )

                    try:
                        img = Image.open(file).convert("RGB")

                        # 运行单次推理
                        result_json, vis_rgb = run_paradigm_a_once(
                            device=device,
                            sam_proc=sam_proc,
                            sam_model=sam_model,
                            sam_dtype=sam_dtype,
                            image_pil=img,
                            prompt=batch_prompt,
                            threshold=float(batch_threshold),
                            alpha=float(batch_alpha),
                            multi_prompt_strategy=str(batch_strategy),
                        )

                        # 添加文件名
                        result_json["name"] = file.name
                        results.append(result_json)

                        # 统计
                        if result_json["final"]["decision"] == "detected":
                            detected_count += 1
                        else:
                            not_detected_count += 1

                        total_latency += result_json["final"]["latency_ms"]

                        # 保存可视化图像
                        safe_name = _safe_stem(file.name)
                        vis_path = os.path.join(vis_dir, f"{safe_name}_result.jpg")
                        _save_vis_image(vis_path, vis_rgb)

                    except Exception as e:
                        st.error(f"处理 {file.name} 时出错：{str(e)}")
                        continue

                # 保存结果
                batch_json = {
                    "batch_info": {
                        "timestamp": timestamp,
                        "total_images": len(batch_files),
                        "processed": len(results),
                        "detected": detected_count,
                        "not_detected": not_detected_count,
                        "total_latency_ms": float(total_latency),
                        "avg_latency_ms": float(total_latency / max(1, len(results))),
                    },
                    "parameters": {
                        "prompt": batch_prompt,
                        "multi_prompt_strategy": str(batch_strategy),
                        "threshold": float(batch_threshold),
                        "alpha": float(batch_alpha),
                    },
                    "results": results
                }

                # 写入 JSON
                json_path = os.path.join(output_dir, "batch_results.json")
                _write_json(json_path, batch_json)

                # 写入 CSV
                csv_rows = [_flatten_for_csv(r) for r in results]
                csv_path = os.path.join(output_dir, "batch_summary.csv")
                _write_csv(csv_path, csv_rows)

                # 保存到 session_state
                st.session_state["a3_batch_results"] = batch_json
                st.session_state["a3_output_dir"] = output_dir

                # 显示完成信息
                LoadingStates.success_toast("批量处理完成！")

                # 使用组件库的统计面板
                UIComponents.statistics_panel({
                    "总计": f"{len(batch_files)} 张",
                    "检测到": (f"{detected_count} 张", f"{detected_count/max(1,len(results))*100:.0f}%"),
                    "未检测到": f"{not_detected_count} 张",
                    "总耗时": f"{total_latency/1000:.1f} 秒"
                }, cols=4)

        # 显示结果和下载按钮
        batch_results = st.session_state.get("a3_batch_results")
        output_dir = st.session_state.get("a3_output_dir")

        if batch_results and output_dir:
            st.markdown("---")
            st.markdown("### 📥 结果导出")

            st.info(f"📁 保存目录：`{output_dir}`")

            # 下载按钮
            col_json, col_csv = st.columns(2)

            with col_json:
                json_path = os.path.join(output_dir, "batch_results.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = f.read()
                    st.download_button(
                        label="📥 下载 JSON 结果",
                        data=json_data,
                        file_name=f"paradigm_a_batch_{batch_results['batch_info']['timestamp']}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            with col_csv:
                csv_path = os.path.join(output_dir, "batch_summary.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "r", encoding="utf-8-sig") as f:
                        csv_data = f.read()
                    st.download_button(
                        label="📥 下载 CSV 汇总",
                        data=csv_data,
                        file_name=f"paradigm_a_batch_{batch_results['batch_info']['timestamp']}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            # 结果预览
            with st.expander("📊 查看结果预览", expanded=False):
                batch_info = batch_results.get("batch_info", {})

                st.markdown("##### 汇总统计")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总图片数", batch_info.get("total_images", 0))
                with col2:
                    st.metric("处理成功", batch_info.get("processed", 0))
                with col3:
                    st.metric("检测到", batch_info.get("detected", 0))
                with col4:
                    st.metric("未检测到", batch_info.get("not_detected", 0))

                st.markdown("##### 按检测结果分类")
                results_list = batch_results.get("results", [])
                detected_list = [r for r in results_list if r.get("final", {}).get("decision") == "detected"]
                not_detected_list = [r for r in results_list if r.get("final", {}).get("decision") == "not_detected"]

                if detected_list:
                    st.write(f"**检测到目标的图片（{len(detected_list)} 张）：**")
                    for r in detected_list[:10]:  # 只显示前10个
                        final = r.get("final", {})
                        st.caption(
                            f"• {r.get('name')}: {final.get('total_instances', 0)} 个实例, "
                            f"{final.get('num_classes', 0)} 个类别, "
                            f"最高分 {final.get('max_score', 0):.3f}"
                        )
                    if len(detected_list) > 10:
                        st.caption(f"...以及其他 {len(detected_list) - 10} 张")

                if not_detected_list:
                    st.write(f"**未检测到目标的图片（{len(not_detected_list)} 张）：**")
                    for r in not_detected_list[:10]:
                        st.caption(f"• {r.get('name')}")
                    if len(not_detected_list) > 10:
                        st.caption(f"...以及其他 {len(not_detected_list) - 10} 张")

            # 处理后的图片展示
            st.markdown("---")
            st.markdown("### 🖼️ 处理结果预览")

            vis_dir = os.path.join(output_dir, "visualizations")
            if os.path.exists(vis_dir):
                vis_files = [f for f in os.listdir(vis_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

                if vis_files:
                    # 显示图片数量选择
                    num_to_show = st.slider(
                        "显示图片数量",
                        min_value=1,
                        max_value=min(20, len(vis_files)),
                        value=min(6, len(vis_files)),
                        key="a3_num_preview"
                    )

                    st.caption(f"📊 共 {len(vis_files)} 张处理后的图片，当前显示前 {num_to_show} 张")

                    # 按列显示图片
                    cols_per_row = 3
                    for i in range(0, num_to_show, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i + j
                            if idx < num_to_show and idx < len(vis_files):
                                with cols[j]:
                                    img_path = os.path.join(vis_dir, vis_files[idx])
                                    try:
                                        img = Image.open(img_path)
                                        st.image(img, use_container_width=True)

                                        # 显示对应的结果信息
                                        img_name = vis_files[idx].replace("_result.jpg", "").replace("_result.png", "")
                                        matching_result = next((r for r in results_list if _safe_stem(r.get('name', '')) == img_name), None)

                                        if matching_result:
                                            final = matching_result.get("final", {})
                                            decision = final.get("decision", "unknown")
                                            if decision == "detected":
                                                st.success(f"✅ 检测到：{final.get('total_instances', 0)} 个实例")
                                            else:
                                                st.info("ℹ️ 未检测到目标")
                                            st.caption(f"📄 {matching_result.get('name', 'unknown')}")
                                        else:
                                            st.caption(f"📄 {vis_files[idx]}")
                                    except Exception as e:
                                        st.error(f"加载图片失败: {str(e)}")

                    st.markdown("---")
                    st.info(f"💡 所有 {len(vis_files)} 张处理后的图片已保存在：`{vis_dir}`")
                else:
                    st.warning("⚠️ 可视化目录为空")
            else:
                st.warning("⚠️ 可视化目录不存在")
