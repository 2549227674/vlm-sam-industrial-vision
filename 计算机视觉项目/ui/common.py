"""共享的 UI 辅助函数"""

from __future__ import annotations

import streamlit as st


def model_init_panel(*, device: str) -> bool:

    """渲染模型初始化面板

       Args:
           device: 设备类型（cuda/cpu）

       Returns:
           bool: 用户是否点击了初始化按钮
       """
    is_ready = st.session_state.get("models_ready")

    # 使用容器包裹
    with st.container():
        st.markdown('<div class="init-status-bar">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            init_btn = st.button(
                "⚡ 初始化核心引擎" if not is_ready else "✅ 引擎已就绪",
                type="primary" if not is_ready else "secondary",
                use_container_width=True,
                disabled=is_ready
            )

        with col2:
            if is_ready:
                st.markdown(f"核心模型已加载至 `{device.upper()}`，系统响应中...")
            else:
                err = st.session_state.get("models_error")
                if err:
                    st.markdown(f"❌ <span style='color:#ff4d4f'>加载失败: {err}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("⚠️ <span style='color:#faad14'>引擎未启动：推理功能暂不可用，请点击初始化。</span>",
                                unsafe_allow_html=True)

        with col3:
            # 显示微型设备徽章
            st.markdown(f"<div style='text-align:right'><code style='color:#1890ff'>{device.upper()} MODE</code></div>",
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    return bool(init_btn)

def device_badge(device: str) -> None:
    """显示当前设备信息徽章

    Args:
        device: 设备类型（cuda/cpu）
    """
    st.info(f"当前设备: {device.upper()}")



