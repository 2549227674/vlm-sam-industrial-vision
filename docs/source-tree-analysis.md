# 源代码树分析

**项目名称**：SAM-3 双范式工业视觉分析系统
**生成日期**：2026-01-03
**扫描模式**：快速扫描（基于目录结构）

---

## 项目目录结构

```
计算机视觉项目/
│
├── app_final.py                    # 🚀 主入口点 - Streamlit 应用启动
│                                   # 功能：全局配置、模型懒加载、范式路由
│
├── requirements.txt                # 📦 依赖清单
│                                   # 核心：Streamlit, PyTorch, OpenCV, ModelScope
│
├── core/                           # 🧠 核心算法模块（14个文件）
│   ├── __init__.py                 # 模块初始化
│   │
│   ├── models.py                   # 🔧 模型加载器
│   │                               # 功能：统一加载 SAM-3, ResNet 模型
│   │
│   ├── sam3_infer.py               # 🎯 SAM-3 推理引擎
│   │                               # 功能：分割推理、掩码生成
│   │
│   ├── vlm.py                      # 🤖 VLM 接口
│   │                               # 功能：视觉语言模型调用
│   │
│   ├── vlm_bbox.py                 # 📦 VLM 边界框检测
│   │                               # 功能：缺陷检测、边界框提取
│   │
│   ├── vlm_model_registry.py      # 📋 VLM 模型注册表
│   │                               # 功能：模型配置管理
│   │
│   ├── dashscope_stream.py        # 🌐 DashScope 流式调用
│   │                               # 功能：阿里云 API 流式接口
│   │
│   ├── padim.py                    # 🔍 PaDiM 异常检测
│   │                               # 功能：特征分布建模、异常定位
│   │
│   ├── feature_extractor.py       # 🎨 特征提取器
│   │                               # 功能：ResNet 特征提取
│   │
│   ├── paradigm_c_metrics.py      # 📊 范式C指标计算
│   │                               # 功能：IoU、精度、召回率计算
│   │
│   ├── bbox_utils.py               # 🔲 边界框工具
│   │                               # 功能：边界框转换、NMS
│   │
│   ├── cv_utils.py                 # 🖼️ CV 工具函数
│   │                               # 功能：图像处理、格式转换
│   │
│   └── _vlm_selftest.py            # 🧪 VLM 自测试
│                                   # 功能：API 连接测试
│
└── ui/                             # 🎨 UI 模块（10个文件）
    ├── __init__.py                 # 模块初始化
    │
    ├── paradigm_a.py               # 📱 范式A界面（61KB）
    │                               # 功能：VLM 引导的开放词汇实例分割
    │                               # 特性：实时语义理解、零样本分割
    │
    ├── paradigm_b.py               # 📱 范式B界面（23KB）
    │                               # 功能：SAM-3 Purify + PaDiM 异常检测
    │                               # 特性：背景净化、特征对比、异常热图
    │
    ├── paradigm_c.py               # 📱 范式C界面（55KB）
    │                               # 功能：VLM 缺陷框 → SAM 精分割
    │                               # 特性：两阶段检测、高精度分割
    │
    ├── components.py               # 🧩 通用UI组件（22KB）
    │                               # 功能：图像上传、参数调节、结果展示
    │
    ├── styles.py                   # 🎨 样式系统（24KB）
    │                               # 功能：全局CSS、自定义组件样式
    │
    ├── state.py                    # 💾 会话状态管理
    │                               # 功能：Streamlit session_state 初始化
    │
    ├── adapters.py                 # 🔌 数据适配器
    │                               # 功能：数据格式转换
    │
    ├── mask_viz.py                 # 👁️ 掩码可视化
    │                               # 功能：分割掩码渲染、叠加显示
    │
    ├── common.py                   # 🛠️ 通用函数
    │                               # 功能：模型初始化面板
    │
    └── constants.py                # 📌 常量定义
                                    # 功能：路径、配置常量
```

---

## 关键目录说明

### 1. 根目录
- **app_final.py**：应用入口，负责：
  - Streamlit 页面配置
  - 全局样式应用
  - 模型懒加载管理
  - 范式路由（A/B/C）

### 2. core/ - 核心算法层
**职责**：封装所有 AI 模型和计算机视觉算法

**关键模块**：
- **模型管理**：`models.py` - 统一模型加载接口
- **SAM-3 引擎**：`sam3_infer.py` - 分割推理核心
- **VLM 服务**：`vlm.py`, `vlm_bbox.py`, `vlm_model_registry.py` - 视觉语言模型集成
- **异常检测**：`padim.py`, `feature_extractor.py` - PaDiM 算法实现
- **工具函数**：`bbox_utils.py`, `cv_utils.py` - 通用 CV 工具

**特点**：
- 与 UI 解耦，可独立测试
- 支持 GPU/CPU 自动切换
- 模型缓存机制

### 3. ui/ - 用户界面层
**职责**：Streamlit 界面实现和用户交互

**关键模块**：
- **范式界面**：`paradigm_a.py`, `paradigm_b.py`, `paradigm_c.py` - 三个独立的分析范式
- **UI 组件**：`components.py` - 可复用的界面组件
- **样式系统**：`styles.py` - 统一的视觉风格
- **状态管理**：`state.py` - 会话状态初始化
- **可视化**：`mask_viz.py` - 分割结果渲染

**特点**：
- 模块化设计，每个范式独立
- 统一的样式系统
- 响应式布局

---

## 数据流架构

```
用户输入（图像 + 参数）
    ↓
app_final.py（路由）
    ↓
ui/paradigm_*.py（界面逻辑）
    ↓
core/（算法调用）
    ├── models.py → 加载模型
    ├── sam3_infer.py → SAM-3 推理
    ├── vlm.py → VLM 推理
    └── padim.py → 异常检测
    ↓
ui/mask_viz.py（结果可视化）
    ↓
用户界面（结果展示）
```

---

## 模块依赖关系

```
app_final.py
    ├── 依赖 → core.models
    ├── 依赖 → ui.state
    ├── 依赖 → ui.common
    ├── 依赖 → ui.paradigm_a
    ├── 依赖 → ui.paradigm_b
    └── 依赖 → ui.paradigm_c

ui.paradigm_a
    ├── 依赖 → core.sam3_infer
    ├── 依赖 → core.vlm
    ├── 依赖 → ui.components
    └── 依赖 → ui.mask_viz

ui.paradigm_b
    ├── 依赖 → core.sam3_infer
    ├── 依赖 → core.padim
    ├── 依赖 → core.feature_extractor
    └── 依赖 → ui.components

ui.paradigm_c
    ├── 依赖 → core.sam3_infer
    ├── 依赖 → core.vlm_bbox
    ├── 依赖 → core.paradigm_c_metrics
    └── 依赖 → ui.components
```

---

## 入口点与启动流程

### 启动命令
```bash
streamlit run app_final.py
```

### 启动流程
1. **全局配置**：设置页面标题、图标、布局
2. **样式加载**：应用全局 CSS 和自定义组件
3. **会话初始化**：初始化 Streamlit session_state
4. **模型懒加载**：用户点击"初始化模型"后才加载
5. **范式选择**：侧边栏单选按钮切换范式
6. **范式渲染**：根据选择渲染对应的 paradigm_*.py 界面

---

## 配置文件

### requirements.txt
- **位置**：项目根目录
- **内容**：Python 依赖包列表
- **关键依赖**：
  - Streamlit（UI 框架）
  - PyTorch（深度学习）
  - ModelScope（模型仓库）
  - DashScope（VLM API）

### 环境变量
```bash
# 可选：启用 VLM 功能
DASHSCOPE_API_KEY=sk-your-key-here
```

---

## 扩展点（为边缘-云协同升级准备）

### 1. 视频流输入扩展
**当前**：静态图像上传
**目标**：实时视频流（来自 i.MX6ULL）

**扩展位置**：
- `ui/components.py` - 添加视频流组件
- `core/cv_utils.py` - 添加视频流解码

### 2. Socket 通信模块
**目标**：与边缘节点通信

**新增模块**：
- `core/socket_server.py` - TCP 服务器
- `core/control_protocol.py` - 控制指令协议

### 3. 云台控制接口
**目标**：发送角度指令到 i.MX6ULL

**新增模块**：
- `core/pan_tilt_control.py` - 云台控制逻辑
- `core/coordinate_transform.py` - 像素坐标 → 云台角度

---

**文档生成时间**：2026-01-03
**扫描模式**：快速扫描（目录结构）
**项目根目录**：`C:\Users\Han\PycharmProjects\PythonProject8\计算机视觉项目`
