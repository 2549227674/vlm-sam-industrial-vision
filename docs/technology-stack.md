# 技术栈分析

**项目名称**：SAM-3 双范式工业视觉分析系统
**生成日期**：2026-01-03
**项目类型**：数据科学/机器学习应用（Streamlit Web App）

---

## 1. 核心技术栈

### 1.1 应用框架

| 技术 | 版本 | 用途 | 关键特性 |
|------|------|------|----------|
| **Streamlit** | ≥1.36.0 | Web UI 框架 | 快速原型开发、实时交互、会话状态管理 |
| **streamlit-antd-components** | ≥0.2.0 | UI 组件库 | Ant Design 风格组件 |

### 1.2 深度学习与计算机视觉

| 技术 | 版本 | 用途 | 关键特性 |
|------|------|------|----------|
| **PyTorch** | ≥2.0.0 | 深度学习框架 | GPU 加速、模型推理 |
| **torchvision** | ≥0.15.0 | 视觉模型库 | 预训练模型、图像变换 |
| **ModelScope** | ≥1.9.0 | 模型仓库 | SAM-3 模型加载 |
| **Transformers** | ≥4.35.0 | NLP/VLM 模型 | VLM 模型支持 |
| **OpenCV** | ≥4.8.0 | 计算机视觉 | 图像处理、视频流 |
| **Pillow** | ≥10.0.0 | 图像处理 | 图像 I/O、格式转换 |

### 1.3 AI 服务与 API

| 技术 | 版本 | 用途 | 关键特性 |
|------|------|------|----------|
| **DashScope** | ≥1.14.0 | 阿里云 VLM API | 视觉语言模型推理服务 |

### 1.4 数值计算与可视化

| 技术 | 版本 | 用途 | 关键特性 |
|------|------|------|----------|
| **NumPy** | ≥1.24.0 | 数值计算 | 数组运算、矩阵操作 |
| **Matplotlib** | ≥3.7.0 | 数据可视化 | 图表绘制、训练分布图 |

---

## 2. 架构模式

### 2.1 应用架构
- **类型**：单页 Web 应用（SPA）
- **框架**：Streamlit
- **部署模式**：本地运行（`streamlit run app_final.py`）
- **计算模式**：GPU 加速（CUDA）/ CPU 回退

### 2.2 AI 模型架构

#### 核心模型
1. **SAM-3（Segment Anything Model 3）**
   - 用途：开放词汇实例分割
   - 来源：ModelScope
   - 推理：GPU/CPU

2. **VLM（视觉语言模型）**
   - 用途：语义理解、缺陷检测
   - 来源：DashScope API
   - 模式：在线推理

3. **ResNet（残差网络）**
   - 用途：特征提取
   - 应用：PaDiM 异常检测

4. **PaDiM（Patch Distribution Modeling）**
   - 用途：异常检测
   - 方法：特征分布建模

### 2.3 三范式架构

#### 范式 A：VLM 引导的开放词汇实例分割
- **流程**：用户输入关键词 → VLM 语义理解 → SAM-3 分割
- **特点**：零样本、实时、多关键词探索

#### 范式 B：SAM-3 Purify + PaDiM 离线异常检测
- **流程**：背景净化 → 特征提取 → 对比学习 → 异常定位
- **特点**：离线训练、特征对比、异常热图

#### 范式 C：VLM 缺陷框 → SAM 精分割
- **流程**：VLM 检测缺陷边界框 → SAM-3 精细分割
- **特点**：两阶段、高精度、缺陷聚焦

---

## 3. 代码结构

### 3.1 项目组织

```
计算机视觉项目/
├── app_final.py          # 主入口（路由、范式选择）
├── requirements.txt      # 依赖清单
├── core/                 # 核心算法模块（16个文件）
│   ├── models.py         # 模型加载器
│   ├── sam3_infer.py     # SAM-3 推理引擎
│   ├── vlm.py            # VLM 接口
│   ├── vlm_bbox.py       # VLM 边界框检测
│   ├── vlm_model_registry.py  # VLM 模型注册表
│   ├── padim.py          # PaDiM 异常检测
│   ├── feature_extractor.py   # 特征提取器
│   ├── dashscope_stream.py    # DashScope 流式调用
│   ├── bbox_utils.py     # 边界框工具
│   ├── cv_utils.py       # CV 工具函数
│   ├── paradigm_c_metrics.py  # 范式C指标计算
│   └── __init__.py
└── ui/                   # UI 模块（10个文件）
    ├── paradigm_a.py     # 范式A界面（61KB）
    ├── paradigm_b.py     # 范式B界面（23KB）
    ├── paradigm_c.py     # 范式C界面（55KB）
    ├── components.py     # 通用UI组件（22KB）
    ├── styles.py         # 样式系统（24KB）
    ├── state.py          # 会话状态管理
    ├── adapters.py       # 数据适配器
    ├── mask_viz.py       # 掩码可视化
    ├── common.py         # 通用函数
    ├── constants.py      # 常量定义
    └── __init__.py
```

### 3.2 模块职责

#### Core 模块（核心算法）
- **models.py**：统一模型加载接口
- **sam3_infer.py**：SAM-3 推理逻辑
- **vlm.py / vlm_bbox.py**：VLM 调用与边界框处理
- **padim.py / feature_extractor.py**：异常检测算法
- **dashscope_stream.py**：阿里云 API 流式调用
- **cv_utils.py / bbox_utils.py**：计算机视觉工具函数

#### UI 模块（界面与交互）
- **paradigm_*.py**：三个范式的完整界面实现
- **components.py**：可复用 UI 组件（图像上传、参数调节等）
- **styles.py**：全局样式系统（CSS、自定义组件）
- **state.py**：Streamlit 会话状态初始化
- **mask_viz.py**：分割掩码可视化

---

## 4. 运行环境

### 4.1 硬件要求
- **推荐**：NVIDIA GPU（CUDA 支持）
- **最低**：CPU（性能降低）
- **内存**：≥8GB RAM
- **存储**：≥5GB（模型缓存）

### 4.2 软件环境
- **操作系统**：Windows / Linux / macOS
- **Python**：≥3.8
- **CUDA**：≥11.0（GPU 模式）

### 4.3 环境变量
```bash
# 启用 VLM 功能（可选）
DASHSCOPE_API_KEY=sk-your-key-here
```

---

## 5. 启动方式

### 5.1 安装依赖
```bash
# 完整安装
pip install -r requirements.txt

# 最小化安装（核心依赖）
pip install streamlit torch torchvision numpy opencv-python Pillow modelscope transformers
```

### 5.2 运行应用
```bash
streamlit run app_final.py
```

### 5.3 首次使用
1. 启动应用后，点击"初始化模型"按钮
2. 等待 SAM-3 和 ResNet 模型下载/加载（约30-60秒）
3. 选择范式（A/B/C）开始使用

---

## 6. 技术特点

### 6.1 优势
- ✅ **零样本能力**：VLM + SAM-3 支持开放词汇分割
- ✅ **多范式融合**：语义分割、异常检测、精准分割三合一
- ✅ **GPU 加速**：PyTorch CUDA 支持
- ✅ **快速原型**：Streamlit 快速开发与部署
- ✅ **模块化设计**：Core 与 UI 分离，易于扩展

### 6.2 局限性
- ⚠️ **依赖外部 API**：VLM 功能需要 DashScope API Key
- ⚠️ **单机部署**：当前为本地应用，未支持分布式
- ⚠️ **GPU 依赖**：CPU 模式性能显著降低
- ⚠️ **模型体积大**：首次运行需下载大型模型

---

## 7. 扩展方向（基于用户需求）

### 7.1 边缘-云协同架构升级
**目标**：将单机应用改造为分布式边缘-云系统

#### 云端（上位机 - 当前 PC）
- 保留：VLM/SAM-3 推理计算
- 新增：Socket 服务器（接收图像、返回控制指令）
- 技术栈：Python Socket + 现有 Streamlit 应用

#### 边缘端（下位机 - i.MX6ULL）
- 新增：视频流采集（mjpg-streamer/ffmpeg）
- 新增：PCA9685 I2C 驱动（云台控制）
- 新增：Socket 客户端（C/Python）
- 技术栈：Linux I2C 驱动 + Socket 通信

#### 通信协议
- **传输层**：TCP/IP（以太网）
- **数据格式**：图像帧 + 控制指令（JSON/Protobuf）
- **延迟要求**：<100ms（实时控制）

---

**文档生成时间**：2026-01-03
**扫描模式**：深度扫描（Deep Scan）
**项目根目录**：`C:\Users\Han\PycharmProjects\PythonProject8\计算机视觉项目`
