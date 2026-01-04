"""可复用的 UI 组件库

提供标准化的 UI 组件，确保整个应用的交互体验统一，并大幅提升代码复用度。
"""

from typing import Optional, List, Dict, Any, Callable
import streamlit as st
from PIL import Image
import numpy as np


class UIComponents:
    """UI 组件集合"""

    @staticmethod
    def paradigm_selector(current_paradigm: Optional[str] = None) -> Optional[str]:
        """卡片式范式选择器

        Args:
            current_paradigm: 当前选中的范式 ("A", "B", "C" 或 None)

        Returns:
            str: 用户选择的范式 ("A", "B", "C")，如果未选择则返回 None

        Example:
            selected = UIComponents.paradigm_selector(st.session_state.get("selected_paradigm"))
        """
        paradigm_config = [
            {
                "id": "A",
                "icon": "🔍",
                "title": "范式 A",
                "subtitle": "在线语义探索",
                "description": "VLM 引导的开放词汇实例分割",
                "color": "blue",
                "features": ["实时语义理解", "零样本分割", "多关键词探索"]
            },
            {
                "id": "B",
                "icon": "🔬",
                "title": "范式 B",
                "subtitle": "离线异常检测",
                "description": "SAM-3 Purify + PaDiM 特征学习",
                "color": "orange",
                "features": ["背景净化", "特征对比", "异常定位"]
            },
            {
                "id": "C",
                "icon": "🎯",
                "title": "范式 C",
                "subtitle": "精准分割",
                "description": "VLM 缺陷框 → SAM 精分割",
                "color": "green",
                "features": ["缺陷定位", "边界精修", "对比分析"]
            }
        ]

        st.markdown('<div class="paradigm-selector-container">', unsafe_allow_html=True)

        selected = None
        cols = st.columns(3)

        for idx, paradigm in enumerate(paradigm_config):
            with cols[idx]:
                is_active = current_paradigm == paradigm["id"]

                # 卡片容器
                card_class = f"paradigm-card paradigm-{paradigm['color']}"
                if is_active:
                    card_class += " paradigm-card-active"

                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

                # 卡片内容
                st.markdown(f"""
                    <div class="paradigm-card-header">
                        <span class="paradigm-icon">{paradigm["icon"]}</span>
                        <h3 class="paradigm-title">{paradigm["title"]}</h3>
                    </div>
                    <div class="paradigm-subtitle">{paradigm["subtitle"]}</div>
                    <p class="paradigm-description">{paradigm["description"]}</p>
                    <ul class="paradigm-features">
                        {"".join([f'<li>{feat}</li>' for feat in paradigm["features"]])}
                    </ul>
                """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # 选择按钮
                btn_label = "✓ 已选择" if is_active else "启动范式"
                if st.button(
                    btn_label,
                    key=f"paradigm_{paradigm['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    selected = paradigm["id"]

        st.markdown('</div>', unsafe_allow_html=True)

        return selected

    @staticmethod
    def section_header(
        title: str,
        icon: str = "",
        description: str = "",
        divider: bool = True
    ):
        """章节标题组件

        Args:
            title: 标题文本
            icon: 图标（emoji）
            description: 描述文本
            divider: 是否显示分隔线

        Example:
            UIComponents.section_header("参数配置", icon="⚙️", description="配置模型参数")
        """
        if icon:
            st.markdown(f"## {icon} {title}")
        else:
            st.markdown(f"## {title}")

        if description:
            st.caption(description)

        if divider:
            st.markdown("---")

    @staticmethod
    def step_indicator(steps: List[str], current: int):
        """步骤指示器

        Args:
            steps: 步骤列表
            current: 当前步骤索引（0-based）

        Example:
            UIComponents.step_indicator(["上传图片", "配置参数", "开始处理"], current=1)
        """
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if i < current:
                    # 已完成
                    st.markdown(f"### ✅ {i+1}")
                    st.caption(f"**{step}**")
                elif i == current:
                    # 进行中
                    st.markdown(f"### ⏳ {i+1}")
                    st.caption(f"**{step}**")
                else:
                    # 未开始
                    st.markdown(f"### ⭕ {i+1}")
                    st.caption(f"{step}")

    @staticmethod
    def result_card(
        title: str,
        content: Dict[str, Any],
        status: Optional[str] = None,
        expandable: bool = False,
        expanded: bool = False
    ):
        """结果卡片组件

        Args:
            title: 卡片标题
            content: 内容字典
            status: 状态（success/warning/error/info）
            expandable: 是否可展开
            expanded: 默认是否展开

        Example:
            UIComponents.result_card(
                "检测结果",
                {"实例数": 5, "置信度": 0.95},
                status="success"
            )
        """
        status_emoji = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️"
        }

        if expandable:
            with st.expander(f"📊 {title}", expanded=expanded):
                if status:
                    st.markdown(f"{status_emoji.get(status, '')} **状态**: {status}")

                for key, value in content.items():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**{key}**")
                    with col2:
                        st.write(value)
        else:
            st.markdown(f"### {title}")
            if status:
                st.markdown(f"{status_emoji.get(status, '')} **状态**: {status}")

            for key, value in content.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{key}**")
                with col2:
                    st.write(value)

    @staticmethod
    def image_grid(
        images: List[Image.Image],
        captions: Optional[List[str]] = None,
        cols: int = 3,
        show_index: bool = True,
        on_click: Optional[Callable[[int], None]] = None
    ):
        """图片网格展示组件

        Args:
            images: 图片列表
            captions: 标题列表
            cols: 列数
            show_index: 是否显示索引
            on_click: 点击回调函数

        Example:
            UIComponents.image_grid(
                images=[img1, img2, img3],
                captions=["结果1", "结果2", "结果3"],
                cols=3
            )
        """
        if not images:
            st.info("暂无图片")
            return

        for i in range(0, len(images), cols):
            cols_widgets = st.columns(cols)
            for j in range(cols):
                idx = i + j
                if idx < len(images):
                    with cols_widgets[j]:
                        st.image(images[idx], use_container_width=True)

                        caption = ""
                        if show_index:
                            caption = f"#{idx+1}"
                        if captions and idx < len(captions):
                            caption += f" - {captions[idx]}"

                        if caption:
                            st.caption(caption)

                        if on_click:
                            if st.button(f"选择", key=f"img_select_{idx}"):
                                on_click(idx)

    @staticmethod
    def statistics_panel(
        stats: Dict[str, Any],
        cols: int = 4,
        use_metric: bool = True
    ):
        """统计面板组件

        Args:
            stats: 统计数据字典 {label: value} 或 {label: (value, delta)}
            cols: 列数
            use_metric: 是否使用 metric 组件

        Example:
            UIComponents.statistics_panel({
                "总数": (10, "+2"),
                "检测到": 7,
                "未检测到": 3
            }, cols=3)
        """
        cols_widgets = st.columns(cols)
        for i, (label, data) in enumerate(stats.items()):
            with cols_widgets[i % cols]:
                if use_metric:
                    if isinstance(data, tuple):
                        value, delta = data
                        st.metric(label, value, delta=delta)
                    else:
                        st.metric(label, data)
                else:
                    st.markdown(f"**{label}**")
                    st.markdown(f"### {data}")

    @staticmethod
    def progress_tracker(
        current: int,
        total: int,
        status_text: str = "",
        show_percentage: bool = True,
        show_count: bool = True
    ):
        """进度追踪器组件

        Args:
            current: 当前进度
            total: 总数
            status_text: 状态文本
            show_percentage: 是否显示百分比
            show_count: 是否显示计数

        Example:
            UIComponents.progress_tracker(
                current=3,
                total=10,
                status_text="正在处理: test_003.jpg"
            )
        """
        progress = current / max(total, 1)

        if status_text:
            st.text(status_text)

        text = ""
        if show_percentage:
            text += f"{int(progress * 100)}%"
        if show_count:
            if text:
                text += f" ({current}/{total})"
            else:
                text += f"{current}/{total}"

        st.progress(progress, text=text if text else None)

    @staticmethod
    def parameter_group(
        title: str,
        expanded: bool = True
    ) -> bool:
        """参数组容器（使用 expander）

        Args:
            title: 组标题
            expanded: 默认是否展开

        Returns:
            bool: expander 对象（可用于 with 语句）

        Example:
            with UIComponents.parameter_group("基础参数"):
                threshold = st.slider("阈值", 0.0, 1.0, 0.3)
        """
        return st.expander(f"⚙️ {title}", expanded=expanded)

    @staticmethod
    def file_upload_area(
        label: str,
        accept_multiple: bool = False,
        file_types: Optional[List[str]] = None,
        help_text: str = "",
        key: Optional[str] = None
    ):
        """文件上传区域组件

        Args:
            label: 标签
            accept_multiple: 是否支持多选
            file_types: 文件类型列表
            help_text: 帮助文本
            key: widget key

        Returns:
            上传的文件

        Example:
            files = UIComponents.file_upload_area(
                "上传图片",
                accept_multiple=True,
                file_types=['jpg', 'png']
            )
        """
        if file_types is None:
            file_types = ['png', 'jpg', 'jpeg']

        files = st.file_uploader(
            label,
            accept_multiple_files=accept_multiple,
            type=file_types,
            help=help_text,
            key=key
        )

        if files:
            if accept_multiple:
                st.success(f"✅ 已选择 {len(files)} 个文件")
            else:
                st.success(f"✅ 已选择文件: {files.name}")

        return files

    @staticmethod
    def action_buttons(
        primary_label: str,
        primary_callback: Optional[Callable] = None,
        secondary_label: Optional[str] = None,
        secondary_callback: Optional[Callable] = None,
        disabled: bool = False,
        use_container_width: bool = True,
        key_prefix: str = "action"
    ):
        """操作按钮组

        Args:
            primary_label: 主按钮文本
            primary_callback: 主按钮回调
            secondary_label: 次按钮文本
            secondary_callback: 次按钮回调
            disabled: 是否禁用
            use_container_width: 是否使用容器宽度
            key_prefix: key 前缀

        Returns:
            tuple: (primary_clicked, secondary_clicked)

        Example:
            primary, secondary = UIComponents.action_buttons(
                "开始处理",
                secondary_label="清空结果"
            )
        """
        if secondary_label:
            col1, col2 = st.columns(2)
            with col1:
                primary_btn = st.button(
                    primary_label,
                    type="primary",
                    use_container_width=use_container_width,
                    disabled=disabled,
                    key=f"{key_prefix}_primary"
                )
            with col2:
                secondary_btn = st.button(
                    secondary_label,
                    use_container_width=use_container_width,
                    key=f"{key_prefix}_secondary"
                )

            if primary_btn and primary_callback:
                primary_callback()
            if secondary_btn and secondary_callback:
                secondary_callback()

            return primary_btn, secondary_btn
        else:
            primary_btn = st.button(
                primary_label,
                type="primary",
                use_container_width=use_container_width,
                disabled=disabled,
                key=f"{key_prefix}_primary"
            )

            if primary_btn and primary_callback:
                primary_callback()

            return primary_btn, False

    @staticmethod
    def status_badge(text: str, status: str = "info") -> str:
        """状态徽章（返回 HTML）

        Args:
            text: 文本
            status: 状态类型（success/warning/error/info）

        Returns:
            str: HTML 字符串

        Example:
            st.markdown(
                UIComponents.status_badge("检测到", "success"),
                unsafe_allow_html=True
            )
        """
        return f'<span class="status-badge status-{status}">{text}</span>'

    @staticmethod
    def info_box(
        message: str,
        type: str = "info",
        icon: str = ""
    ):
        """信息提示框

        Args:
            message: 消息文本
            type: 类型（success/info/warning/error）
            icon: 自定义图标

        Example:
            UIComponents.info_box("操作成功！", type="success")
        """
        type_config = {
            "success": ("✅", st.success),
            "info": ("ℹ️", st.info),
            "warning": ("⚠️", st.warning),
            "error": ("❌", st.error)
        }

        default_icon, func = type_config.get(type, ("ℹ️", st.info))
        display_icon = icon if icon else default_icon

        func(f"{display_icon} {message}")

    @staticmethod
    def download_buttons(
        data_dict: Dict[str, tuple],
        cols: Optional[int] = None
    ):
        """下载按钮组

        Args:
            data_dict: 数据字典 {label: (data, filename, mime)}
            cols: 列数（None 则自动）

        Example:
            UIComponents.download_buttons({
                "下载 JSON": (json_data, "result.json", "application/json"),
                "下载 CSV": (csv_data, "result.csv", "text/csv")
            })
        """
        if not data_dict:
            return

        if cols is None:
            cols = len(data_dict)

        cols_widgets = st.columns(cols)
        for i, (label, (data, filename, mime)) in enumerate(data_dict.items()):
            with cols_widgets[i % cols]:
                st.download_button(
                    label=label,
                    data=data,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True
                )

    @staticmethod
    def collapsible_code(
        code: str,
        language: str = "python",
        title: str = "查看代码",
        expanded: bool = False
    ):
        """可折叠的代码块

        Args:
            code: 代码内容
            language: 语言
            title: 标题
            expanded: 默认是否展开

        Example:
            UIComponents.collapsible_code(
                "print('Hello')",
                language="python",
                title="示例代码"
            )
        """
        with st.expander(f"💻 {title}", expanded=expanded):
            st.code(code, language=language)

    @staticmethod
    def data_table(
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        max_rows: Optional[int] = None,
        use_container_width: bool = True
    ):
        """数据表格组件

        Args:
            data: 数据列表
            title: 表格标题
            max_rows: 最大显示行数
            use_container_width: 是否使用容器宽度

        Example:
            UIComponents.data_table(
                [{"名称": "test1.jpg", "结果": "检测到"}],
                title="处理结果"
            )
        """
        if title:
            st.markdown(f"#### {title}")

        if not data:
            st.info("暂无数据")
            return

        display_data = data[:max_rows] if max_rows else data

        try:
            import pandas as pd
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=use_container_width, hide_index=True)

            if max_rows and len(data) > max_rows:
                st.caption(f"显示前 {max_rows} 行，共 {len(data)} 行")
        except ImportError:
            # 如果没有 pandas，使用 streamlit 原生
            st.dataframe(display_data, use_container_width=use_container_width)


class LoadingStates:
    """加载状态管理"""

    @staticmethod
    def spinner(message: str = "处理中..."):
        """加载旋转器（上下文管理器）

        Example:
            with LoadingStates.spinner("正在加载模型..."):
                model = load_model()
        """
        return st.spinner(f"⏳ {message}")

    @staticmethod
    def success_toast(message: str, details: Optional[str] = None):
        """成功提示

        Example:
            LoadingStates.success_toast("处理完成！", details="共处理 10 张图片")
        """
        st.success(f"🎉 {message}")
        if details:
            st.caption(details)

    @staticmethod
    def error_toast(message: str, details: Optional[str] = None, show_trace: bool = False):
        """错误提示

        Example:
            LoadingStates.error_toast("处理失败", details="文件格式不支持")
        """
        st.error(f"❌ {message}")
        if details:
            if show_trace:
                with st.expander("查看详情"):
                    st.code(details)
            else:
                st.caption(details)


class FormHelper:
    """表单辅助工具"""

    @staticmethod
    def create_form(
        form_key: str,
        submit_label: str = "提交",
        clear_on_submit: bool = False
    ):
        """创建表单（上下文管理器）

        Example:
            with FormHelper.create_form("my_form", submit_label="保存"):
                name = st.text_input("名称")
        """
        return st.form(key=form_key, clear_on_submit=clear_on_submit)

    @staticmethod
    def validation_message(
        field_name: str,
        value: Any,
        required: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> bool:
        """表单验证

        Returns:
            bool: 是否验证通过
        """
        if required and not value:
            st.error(f"❌ {field_name} 不能为空")
            return False

        if min_length and len(str(value)) < min_length:
            st.error(f"❌ {field_name} 长度不能少于 {min_length}")
            return False

        if max_length and len(str(value)) > max_length:
            st.error(f"❌ {field_name} 长度不能超过 {max_length}")
            return False

        return True

