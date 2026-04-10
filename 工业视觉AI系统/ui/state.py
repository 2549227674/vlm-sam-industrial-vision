"""Streamlit 应用的会话状态初始化"""

from __future__ import annotations

import streamlit as st

from core.vlm_model_registry import default_model_for_bbox, default_model_for_suggestions


def init_session_state() -> None:
    """初始化 UI 使用的所有会话状态键

    集中管理状态键，避免重构时静默破坏跨页面状态。
    """
    # 范式 A - 基础状态
    st.session_state.setdefault("a_last_uploaded_name", None)
    st.session_state.setdefault("a_raw_rgb", None)
    st.session_state.setdefault("a_last_results", None)
    st.session_state.setdefault("a_last_latency_ms", None)
    st.session_state.setdefault("a_last_threshold", None)
    st.session_state.setdefault("a_last_class_groups", None)

    # 范式 A - 提示词相关
    st.session_state.setdefault("a_prompt_input", "screw")
    st.session_state.setdefault("a_prompt_add_mode", True)  # True=追加；False=覆盖
    st.session_state.setdefault("a_prompt_fallback", None)
    st.session_state.setdefault("a_multi_prompt_strategy", "per_prompt")
    st.session_state.setdefault("a_prompt_mode_label", None)

    # 范式 A - VLM 推荐
    st.session_state.setdefault("a_vlm_last_uploaded_name", None)
    st.session_state.setdefault("a_vlm_tags", None)
    st.session_state.setdefault("a_vlm_desc_zh", "")
    st.session_state.setdefault("a_vlm_desc_en", "")
    st.session_state.setdefault("a_vlm_model", default_model_for_suggestions())
    st.session_state.setdefault("a_vlm_mode", "industrial_defect")
    st.session_state.setdefault("a_vlm_thinking", False)

    # 范式 A - 结果展示控制
    st.session_state.setdefault("a_selected_mask_idx", 0)
    st.session_state.setdefault("a_topk", 5)
    st.session_state.setdefault("a_multi_view", False)
    st.session_state.setdefault("a_selected_mask_idxs", [])
    st.session_state.setdefault("a_force_regenerate", False)

    # 范式 B - PaDiM 相关
    st.session_state.setdefault("b_last_uploaded_name", None)
    st.session_state.setdefault("b_raw_rgb", None)
    st.session_state.setdefault("b_last_feat", None)
    st.session_state.setdefault("b_last_roi", None)
    st.session_state.setdefault("b_last_sam_results", None)

    # 范式 C - VLM bbox → SAM box prompt
    st.session_state.setdefault("c_input_mode", "single")  # 'single' or 'compare'
    st.session_state.setdefault("c_vlm_model", default_model_for_bbox(fast=bool(st.session_state.get("c_fast_mode", False))))
    st.session_state.setdefault("c_vlm_thinking", False)
    st.session_state.setdefault("c_max_boxes", 3)
    st.session_state.setdefault("c_sam_thr", 0.5)
    st.session_state.setdefault("c_mask_thr", 0.5)
    st.session_state.setdefault("c_alpha", 0.5)
    st.session_state.setdefault("c_fast_mode", False)
    st.session_state.setdefault("c_bbox_pad", 0.20)
    st.session_state.setdefault("c_vlm_out", None)
    st.session_state.setdefault("c_selected_indices", [])
    st.session_state.setdefault("c_sam_runs", None)
    st.session_state.setdefault("c_final_json", None)
    st.session_state.setdefault("c_normal_uploaded_name", None)
    st.session_state.setdefault("c_test_uploaded_name", None)

    # 全局状态
    st.session_state.setdefault("models_ready", False)
    st.session_state.setdefault("models_error", None)
    st.session_state.setdefault("selected_paradigm", None)  # 选中的范式: "A", "B", "C" 或 None

