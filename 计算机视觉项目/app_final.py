import os
import time

import streamlit as st
import torch

from core.models import load_models

from ui.state import init_session_state
from ui.common import model_init_panel
from ui.constants import WEB_MODEL_PATH
from ui import paradigm_a, paradigm_b, paradigm_c
from ui.styles import get_global_styles, get_custom_components  # ✅ 导入新样式系统


# ==================== 1. 全局配置 ====================
st.set_page_config(
    page_title="基于VLM 语义引导与 SAM-3开放词汇模型的双范式工业视觉分析系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("_boot_marker", time.time())

# ✅ 应用统一样式系统
st.markdown(get_global_styles(), unsafe_allow_html=True)
st.markdown(get_custom_components(), unsafe_allow_html=True)

# 模型保存路径 (Web端专用)
os.makedirs("paradigm_b", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 2. 模型懒加载 ====================
def _ensure_models_loaded():
    """懒加载模型：避免首屏就卡在模型下载/初始化。"""
    if st.session_state.get("models_ready"):
        return st.session_state.get("models_tuple")

    try:
        sam_proc, sam_model, sam_dtype, resnet = load_models(DEVICE)
        st.session_state["models_tuple"] = (sam_proc, sam_model, sam_dtype, resnet)
        st.session_state["models_ready"] = True
        st.session_state["models_error"] = None
        return st.session_state["models_tuple"]
    except Exception as e:
        st.session_state["models_ready"] = False
        st.session_state["models_error"] = str(e)
        raise


# ==================== 3. 入口与路由 ====================
def main():
    init_session_state()

    st.title("基于VLM 语义引导与 SAM-3开放词汇模型的双范式工业视觉分析系统")
    st.caption("范式A：VLM 引导的开放词汇实例分割｜范式B：SAM-3 Purify + PaDiM 离线异常检测｜范式C：VLM 缺陷框 → SAM 精分割")
    st.info('首次使用请先点击"初始化模型"（会下载/加载 SAM-3 和 ResNet，可能需要几十秒）。')

    # ==================== 侧边栏范式选择器 ====================
    st.sidebar.title(" 系统控制台")

    with st.sidebar:
        st.markdown("### 范式选择")

        # 使用优化的单选按钮组
        paradigm_options = {
            " 范式 A：在线语义探索": "A",
            " 范式 B：离线异常检测": "B",
            " 范式 C：精准分割": "C"
        }

        selected_label = st.radio(
            label="选择分析范式",
            options=list(paradigm_options.keys()),
            index=0 if not st.session_state.get("selected_paradigm") else
                  ["A", "B", "C"].index(st.session_state.get("selected_paradigm")),
            key="paradigm_radio",
            label_visibility="collapsed"
        )

        mode = paradigm_options[selected_label]
        st.session_state["selected_paradigm"] = mode

        st.markdown("---")

        # 范式说明
        paradigm_descriptions = {
            "A": {
                "title": "📖 范式 A 说明",
                "desc": "VLM 引导的开放词汇实例分割",
                "features": ["实时语义理解", "零样本分割", "多关键词探索"]
            },
            "B": {
                "title": "📖 范式 B 说明",
                "desc": "SAM-3 Purify + PaDiM 特征学习",
                "features": ["背景净化", "特征对比", "异常定位"]
            },
            "C": {
                "title": "📖 范式 C 说明",
                "desc": "VLM 缺陷框 → SAM 精分割",
                "features": ["缺陷定位", "边界精修", "对比分析"]
            }
        }

        if mode in paradigm_descriptions:
            info = paradigm_descriptions[mode]
            with st.expander(info["title"], expanded=True):
                st.caption(info["desc"])
                for feat in info["features"]:
                    st.markdown(f"✓ {feat}")

    # 初始化模型面板（顶部与原布局一致）
    init_btn = model_init_panel(device=DEVICE)

    if init_btn and not st.session_state.get("models_ready"):
        with st.spinner("正在初始化模型(可能需要几十秒)..."):
            _ensure_models_loaded()
        st.rerun()

    # 模型占位，确保 UI 渲染不崩
    sam_proc = sam_model = resnet = None
    sam_dtype = torch.float32
    if st.session_state.get("models_ready"):
        models_tuple = st.session_state.get("models_tuple")
        if isinstance(models_tuple, (list, tuple)):
            if len(models_tuple) >= 4:
                sam_proc, sam_model, sam_dtype, resnet = models_tuple[:4]
                # 兼容旧版本缓存（曾经返回5项 exemplar_probe）
                if len(models_tuple) != 4:
                    st.session_state["models_tuple"] = (sam_proc, sam_model, sam_dtype, resnet)
            else:
                # 异常情况：缓存损坏
                st.session_state["models_ready"] = False
                st.session_state["models_error"] = "models_tuple 缓存不完整，请重新初始化模型"
        else:
            st.session_state["models_ready"] = False
            st.session_state["models_error"] = "models_tuple 缓存类型异常，请重新初始化模型"

    # 根据选中的范式渲染对应界面
    if mode == "A":
        paradigm_a.render(device=DEVICE, sam_proc=sam_proc, sam_model=sam_model, sam_dtype=sam_dtype)
    elif mode == "B":
        # 范式B依赖离线模型文件；渲染逻辑在模块内处理提示
        _ = WEB_MODEL_PATH  # keep constant referenced; also ensures folder created above
        paradigm_b.render(device=DEVICE, sam_proc=sam_proc, sam_model=sam_model, sam_dtype=sam_dtype, resnet=resnet)
    elif mode == "C":
        paradigm_c.render(device=DEVICE, sam_proc=sam_proc, sam_model=sam_model, sam_dtype=sam_dtype)



main()