"""统一的样式系统

提供全局 CSS 样式和自定义组件样式，确保整个应用的视觉风格统一。
"""

def get_global_styles() -> str:
    """获取全局 CSS 样式

    Returns:
        str: 完整的 CSS 样式字符串
    """
    return """
    <style>
    /* ==================== CSS 变量定义（深色主题）==================== */
    :root {
        /* 主色调 */
        --primary-color: #1890ff;
        --primary-light: #40a9ff;
        --primary-dark: #096dd9;
        
        /* 功能色 */
        --success-color: #52c41a;
        --warning-color: #faad14;
        --error-color: #ff4d4f;
        --info-color: #13c2c2;
        
        /* 深色主题背景色 */
        --bg-primary: #0a0e27;
        --bg-secondary: #141b2d;
        --bg-tertiary: #1f2940;
        --bg-hover: #2a3f5f;
        --bg-card: #1a1f3a;
        
        /* 深色主题文本色 */
        --text-primary: #e8eaf6;
        --text-secondary: #b4b7c9;
        --text-disabled: #6b7280;
        
        /* 边框和阴影 */
        --border-radius: 8px;
        --border-radius-lg: 12px;
        --border-color: #2d3748;
        --divider-color: #2d3748;
        
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
        
        /* 间距系统 */
        --spacing-xs: 8px;
        --spacing-sm: 16px;
        --spacing-md: 24px;
        --spacing-lg: 32px;
        --spacing-xl: 48px;
    }
    /* 头部 Hero 区域 */
        .header-container {
        background: linear-gradient(90deg, rgba(24, 144, 255, 0.1) 0%, rgba(31, 41, 64, 0.8) 100%);
        border: 1px solid rgba(100, 120, 180, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: left;
        position: relative;
        overflow: hidden;
    }

    .header-container::after {
        content: "INDUSTRIAL AI";
        position: absolute;
        right: -20px;
        bottom: -10px;
        font-size: 5rem;
        font-weight: 900;
        color: rgba(255, 255, 255, 0.03);
        pointer-events: none;
}

    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title-bar {
        display: flex;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
    }

    .paradigm-badge {
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* 初始化状态条 */
    .init-status-bar {
        display: flex;
        align-items: center;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 10px 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(45, 55, 72, 1);
    }
    /* ==================== 全局布局优化（深色主题）==================== */
    /* 主应用背景 - 深色渐变 */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%) !important;
    }
    
    /* 主内容区域背景 - 深色卡片 */
    .main .block-container {
        background: rgba(26, 31, 58, 0.95) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        margin: 1rem auto !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(45, 55, 72, 0.5) !important;
    }
    
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1400px !important;
    }
    
    /* 侧边栏美化 - 深色背景 */
    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        width: 320px !important;
        background: linear-gradient(180deg, #0a0e27 0%, #141b2d 100%) !important;
        border-right: 1px solid rgba(45, 55, 72, 0.5) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        padding-top: 2rem;
        background: transparent !important;
    }
    
    /* 侧边栏内容文字颜色 */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    /* 侧边栏单选按钮优化 */
    section[data-testid="stSidebar"] .stRadio > div {
        background: rgba(31, 41, 64, 0.6) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(100, 120, 180, 0.2) !important;
        border-color: rgba(100, 120, 180, 0.5) !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(24, 144, 255, 0.2) 0%, rgba(24, 144, 255, 0.1) 100%) !important;
        border: 1px solid var(--primary-color) !important;
        box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label span {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* 侧边栏展开面板优化 */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(31, 41, 64, 0.5) !important;
        border: 1px solid rgba(45, 55, 72, 0.8) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(100, 120, 180, 0.2) !important;
        border-color: var(--primary-color) !important;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(26, 31, 58, 0.3) !important;
        border: 1px solid rgba(45, 55, 72, 0.5) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 0.75rem !important;
    }
    
    /* ==================== 标题样式优化 ==================== */
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: var(--spacing-md) !important;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.3 !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: var(--spacing-md) 0 var(--spacing-sm) 0 !important;
        border-left: 4px solid var(--primary-color);
        padding-left: var(--spacing-sm);
        line-height: 1.4 !important;
    }
    
    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: var(--spacing-sm) 0 !important;
        line-height: 1.4 !important;
    }
    
    h4 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin: var(--spacing-sm) 0 var(--spacing-xs) 0 !important;
    }
    
    /* ==================== 按钮优化 ==================== */
    .stButton > button {
        border-radius: var(--border-radius) !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: var(--shadow-sm) !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, #096dd9 100%) !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-color) 100%) !important;
    }
    
    /* ==================== 输入框优化 ==================== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--border-radius) !important;
        border: 1px solid var(--border-color) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
    }
    
    /* ==================== 选择框优化 ==================== */
    .stSelectbox > div > div {
        border-radius: var(--border-radius) !important;
    }
    
    /* ==================== 滑块优化 ==================== */
    .stSlider {
        padding-top: var(--spacing-xs) !important;
        padding-bottom: var(--spacing-xs) !important;
    }
    
    /* ==================== Tab 标签页优化 ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-secondary);
        padding: 8px;
        border-radius: var(--border-radius);
        margin-bottom: var(--spacing-md);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--border-radius);
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bg-hover);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, #096dd9 100%) !important;
        color: white !important;
    }
    
    /* ==================== 状态提示优化 ==================== */
    .stSuccess {
        background-color: #f6ffed !important;
        border-left: 4px solid var(--success-color) !important;
        border-radius: var(--border-radius) !important;
        padding: var(--spacing-sm) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stInfo {
        background-color: #e6f7ff !important;
        border-left: 4px solid var(--info-color) !important;
        border-radius: var(--border-radius) !important;
        padding: var(--spacing-sm) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stWarning {
        background-color: #fffbe6 !important;
        border-left: 4px solid var(--warning-color) !important;
        border-radius: var(--border-radius) !important;
        padding: var(--spacing-sm) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stError {
        background-color: #fff2f0 !important;
        border-left: 4px solid var(--error-color) !important;
        border-radius: var(--border-radius) !important;
        padding: var(--spacing-sm) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* ==================== 图片优化 ==================== */
    .stImage > img {
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stImage > img:hover {
        transform: scale(1.02) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* ==================== 进度条优化 ==================== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--success-color) 100%) !important;
        border-radius: var(--border-radius) !important;
    }
    
    /* ==================== 指标卡片优化 ==================== */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 500 !important;
    }
    
    /* ==================== 展开面板优化 ==================== */
    .streamlit-expanderHeader {
        border-radius: var(--border-radius) !important;
        background-color: var(--bg-secondary) !important;
        font-weight: 500 !important;
        padding: var(--spacing-sm) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--bg-hover) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* ==================== 文件上传器优化 ==================== */
    [data-testid="stFileUploader"] {
        border-radius: var(--border-radius) !important;
    }
    
    [data-testid="stFileUploader"] > div {
        border-radius: var(--border-radius) !important;
        border: 2px dashed var(--border-color) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] > div:hover {
        border-color: var(--primary-color) !important;
        background-color: var(--bg-hover) !important;
    }
    
    /* ==================== 滚动条优化 ==================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: var(--border-radius);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: var(--border-radius);
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* ==================== 数据表格优化 ==================== */
    .stDataFrame {
        border-radius: var(--border-radius) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* ==================== 代码块优化 ==================== */
    .stCodeBlock {
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* ==================== 加载动画 ==================== */
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    .loading-shimmer {
        animation: shimmer 2s infinite linear;
        background: linear-gradient(
            to right,
            #f0f0f0 8%,
            #f8f8f8 18%,
            #f0f0f0 33%
        );
        background-size: 1000px 100%;
    }
    
    /* ==================== 分隔线优化 ==================== */
    hr {
        margin: var(--spacing-md) 0 !important;
        border: none !important;
        border-top: 1px solid var(--divider-color) !important;
    }
    
    /* ==================== 标注文本优化 ==================== */
    .stCaption {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
    }
    
    /* ==================== 响应式优化 ==================== */
    @media (max-width: 768px) {
        .block-container {
            padding-left: var(--spacing-sm) !important;
            padding-right: var(--spacing-sm) !important;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.25rem !important;
        }
    }
    </style>
    """


def get_custom_components() -> str:
    """获取自定义组件样式

    Returns:
        str: 自定义组件的 CSS 样式
    """
    return """
    <style>
    /* ==================== 范式选择器卡片 ==================== */
    .paradigm-selector-container {
        margin: 2rem 0;
        padding: 1rem;
    }
    
    .paradigm-card {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.95) 0%, rgba(31, 41, 64, 0.95) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 2px solid rgba(45, 55, 72, 0.5);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        min-height: 320px;
        display: flex;
        flex-direction: column;
    }
    
    .paradigm-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
        border-color: rgba(100, 120, 180, 0.8);
    }
    
    .paradigm-card-active {
        border: 2px solid var(--primary-color) !important;
        box-shadow: 0 8px 24px rgba(24, 144, 255, 0.4) !important;
        background: linear-gradient(135deg, rgba(24, 144, 255, 0.1) 0%, rgba(26, 31, 58, 0.95) 100%) !important;
    }
    
    .paradigm-card.paradigm-blue:hover {
        border-color: #667eea;
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    }
    
    .paradigm-card.paradigm-orange:hover {
        border-color: #ff6b6b;
        box-shadow: 0 12px 32px rgba(255, 107, 107, 0.4);
    }
    
    .paradigm-card.paradigm-green:hover {
        border-color: #11998e;
        box-shadow: 0 12px 32px rgba(17, 153, 142, 0.4);
    }
    
    .paradigm-card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    
    .paradigm-icon {
        font-size: 2.5rem;
        line-height: 1;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .paradigm-title {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
    }
    
    .paradigm-subtitle {
        font-size: 1rem;
        font-weight: 600;
        color: var(--primary-light);
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    
    .paradigm-card.paradigm-blue .paradigm-subtitle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .paradigm-card.paradigm-orange .paradigm-subtitle {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .paradigm-card.paradigm-green .paradigm-subtitle {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .paradigm-description {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        line-height: 1.6;
        flex-grow: 0;
    }
    
    .paradigm-features {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
        flex-grow: 1;
    }
    
    .paradigm-features li {
        padding: 0.5rem 0;
        color: var(--text-secondary);
        font-size: 0.875rem;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .paradigm-features li:before {
        content: "✓";
        position: absolute;
        left: 0;
        color: var(--success-color);
        font-weight: bold;
    }
    
    .paradigm-card.paradigm-blue .paradigm-features li:before {
        color: #667eea;
    }
    
    .paradigm-card.paradigm-orange .paradigm-features li:before {
        color: #ff6b6b;
    }
    
    .paradigm-card.paradigm-green .paradigm-features li:before {
        color: #11998e;
    }
    
    /* 响应式调整 */
    @media (max-width: 768px) {
        .paradigm-card {
            min-height: 280px;
            padding: 1rem;
        }
        
        .paradigm-icon {
            font-size: 2rem;
        }
        
        .paradigm-title {
            font-size: 1.25rem !important;
        }
    }
    
    /* ==================== 结果卡片 ==================== */
    .result-card {
        background: white;
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: var(--spacing-sm);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-color);
        transform: translateY(-2px);
    }
    
    /* ==================== 状态徽章 ==================== */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .status-success {
        background-color: #f6ffed;
        color: var(--success-color);
        border: 1px solid #b7eb8f;
    }
    
    .status-warning {
        background-color: #fffbe6;
        color: var(--warning-color);
        border: 1px solid #ffe58f;
    }
    
    .status-error {
        background-color: #fff2f0;
        color: var(--error-color);
        border: 1px solid #ffccc7;
    }
    
    .status-info {
        background-color: #e6f7ff;
        color: var(--info-color);
        border: 1px solid #91d5ff;
    }
    
    /* ==================== 参数面板 ==================== */
    .param-panel {
        background: var(--bg-tertiary);
        border-radius: var(--border-radius);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-md);
        border: 1px solid var(--border-color);
    }
    
    /* ==================== 步骤指示器 ==================== */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        margin: var(--spacing-md) 0;
    }
    
    .step {
        display: flex;
        align-items: center;
        gap: var(--spacing-xs);
    }
    
    .step-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        background: var(--bg-secondary);
        color: var(--text-secondary);
    }
    
    .step-number.active {
        background: var(--primary-color);
        color: white;
    }
    
    .step-number.completed {
        background: var(--success-color);
        color: white;
    }
    </style>
    """

