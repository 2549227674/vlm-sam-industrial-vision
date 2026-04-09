# ====================
# 数据集（不上传）
# ====================
D070*/
*轴承*
*木材*
*芯片*
VisA*/

# ====================
# 模型权重（不上传）
# ====================
*.pt
*.pth
runs/

# ====================
# Python 运行时
# ====================
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
.env

# ====================
# IDE / 编辑器
# ====================
.idea/
.vscode/
*.iml
*.iws

# ====================
# 缓存 / 日志
# ====================
*.log
.pytest_cache/
.mypy_cache/
.cache/
.docker_cache/

# ====================
# 临时 / 备份文件
# ====================
*.zip
*.tar.gz
*.tar
*.rar
*.7z
*.jpg
*.jpeg
*.png
!docs/**/*.png

# ====================
# 系统文件
# ====================
.DS_Store
Thumbs.db

# ====================
# 本地文档/简历/分析
# ====================
简历.md
简历*.md
*分析报告*.md
*Agent框架*.md
BMAD*
项目完整结构*.md
真相大白*.md
问题已*.md
各训练参数*.md
答辩分工*.md
改进*.md
```

现在来起草 README。
<minimax:tool_call>
<invoke name="Write">
<parameter name="content"># 工业视觉 AI 分析系统 | 多模态协同检测与高并发监控平台

> 基于 Qwen-VL、SAM-3、YOLOv8 的工业缺陷检测系统，支持五大范式与多生产线并发监控。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-orange.svg)](https://streamlit.io/)

---

## 一、项目概述

本项目针对工业制造场景的缺陷检测需求，设计并实现了一套**五范式工业视觉 AI 分析平台**，涵盖从在线语义探索到离线异常检测、从精准分割导出到实时流检测、从视频跟踪的全链路能力。

### 核心技术亮点

| 维度 | 技术实现 |
|------|---------|
| **多模态大模型** | Qwen-VL/QVQ + 600+ 行 bbox 结构化解析管道，五级坐标验证 + 启发式降级重试 |
| **开放词汇分割** | SAM-3（图像 + 视频）本地推理，零 API 边际成本，支持文本/框/点三种提示模式 |
| **轻量级检测** | YOLOv8n 定制训练，mAP50 达 **87.8%**（木材）/ **72.5%**（轴承），8 类缺陷识别 |
| **无监督异常** | PaDiM（ResNet-18 特征 + 马氏距离），产线冷启动无需标注数据 |
| **高并发监控** | 1-6 条生产线共享 YOLO 单例并发推理，NVML GPU 真实监控 |
| **边缘协同** | C++ ARM Linux SDK，TCP 防粘包协议 + MJPEG 双通道推流 |

---

## 二、技术架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L1 认知层 · VLM 数据流控                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐│
│  │ 强制 JSON   │     │ 五级 bbox   │     │  启发式降级重试          ││
│  │ Schema 约束 │ ──► │ 净化管道    │ ──► │  turbo → max 模型降级   ││
│  │ 600+ 行     │     │ 类型→值→排序│     │  区分失败与真无缺陷     ││
│  └─────────────┘     │ →裁剪→面积  │     └─────────────────────────┘│
│                      └─────────────┘                                  │
│                              │                                        │
│                     VLM bbox → SAM3 精修 → YOLO 数据集导出            │
├──────────────────────────────┼──────────────────────────────────────┤
│                    L2 算法层 · 本地中小模型                            │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │ PaDiM 异常检测    │    │ YOLOv8n 定制训练  │                       │
│  │ ResNet-18 特征    │    │ mAP50: 87.8%      │                       │
│  │ 马氏距离热力图    │    │ RTX 4060 优化      │                       │
│  │ 无监督，冷启动    │    │ AMP+batch=8        │                       │
│  └──────────────────┘    └──────────────────┘                       │
│           │                        │                                 │
│           ▼                        ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            SAM-3 图像分割 & SAM-3 Video 视频跟踪              │   │
│  │  文本/框/点提示 · object_ids 跨帧跟踪 · 本地推理零成本         │   │
│  └──────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│               L3 部署层 · 高并发监控与边缘协同                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ SharedModelMgr   │    │ GPUMonitor       │    │ C++ 边缘 SDK    │ │
│  │ 单例模式          │    │ NVML 真实监控    │    │ ARM Linux      │ │
│  │ 1-6 线共享 YOLO  │    │ 显存/温度/利用率  │    │ TCP+MJPEG      │ │
│  │ threading.Lock   │    │ OOM 规避          │    │ 防粘包协议      │ │
│  └──────────────────┘    └──────────────────┘    └────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   Streamlit Web   │
                    │   五范式路由入口   │
                    └──────────────────┘
```

### 五大范式

| 范式 | 名称 | 核心技术 | 典型场景 |
|------|------|---------|---------|
| **A** | 在线语义探索 | VLM + SAM-3 零样本分割 | 输入关键词即可分割任意缺陷类型 |
| **B** | 离线异常检测 | PaDiM + ResNet-18 | 产线冷启动，无监督排障 |
| **C** | 精准分割导出 | VLM bbox → SAM3 精修 → YOLO 格式 | 自动化标注流水线 |
| **D** | 实时流检测 | YOLOv8 + Socket/MJPEG | 边缘设备实时推理 |
| **E** | 视频跟踪检测 | SAM-3 Video object_ids | 批量视频缺陷跟踪 |

---

## 三、核心模块说明

### 3.1 VLM bbox 结构化解析管道

**文件**: `core/vlm_bbox.py`（600+ 行）

```python
# 五级 bbox 净化管道
def _sanitize_bbox_xyxy(b, *, w, h):
    # L1: 类型验证
    if not isinstance(b, (list, tuple)) or len(b) != 4: return None
    # L2: 值转换验证（三重强制转换）
    x1, y1, x2, y2 = [int(round(float(x))) for x in b]
    # L3: 坐标排序（x1<x2, y1<y2）
    x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
    # L4: 边界裁剪
    x1, y1 = _clamp(x1, 0, w-1), _clamp(y1, 0, h-1)
    x2, y2 = _clamp(x2, 0, w), _clamp(y2, 0, h)
    # L5: 面积保障（宽高至少 1 像素）
    if x2 <= x1: x2 = min(w, x1+1)
    if y2 <= y1: y2 = min(h, y1+1)
    return [int(x1), int(y1), int(x2), int(y2)]
```

**启发式降级决策**:
```python
def _should_fallback(out):
    # 关键词检测区分"真无缺陷"与"解析失败"
    suspicious = ["error", "exception", "http", "failed", "invalid"]
    if any(s in raw for s in suspicious): return True
    if "{" not in raw: return True  # 无 JSON 结构
    return False
```

### 3.2 PaDiM 异常检测

**文件**: `core/padim.py` + `core/feature_extractor.py`

```
正常样本 → ResNet-18 Layer3 特征 [256, 16, 16] → PaDiM 统计量（均值 + 协方差逆）
新样本测试 → 马氏距离热力图 → 异常区域定位
```

### 3.3 多轴承并发监控系统

**文件**: `bearing_core/multi_bearing_monitor.py`（687 行）

```python
class SharedModelManager:
    """共享 YOLO 单例，1-6 条产线并发复用同一模型实例"""
    def __new__(cls, model_path, device='cuda:0'):
        # threading.Lock 保证线程安全
        ...

class GPUMonitor:
    """NVML 真实 GPU 监控：显存 / 利用率 / 温度 / 功耗"""
    def get_gpu_stats(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        ...
```

### 3.4 C++ 边缘 SDK

**文件**: `edge_device/src/`

```
TCP 防粘包帧协议:
  magic(4B) + jpeg_len(4B) + jpeg_data(NB)
  magic = 0xABCD1234，ARM 小端序匹配

MJPEG 视频流:
  HTTP Multipart/x-mixed-replace，~30 FPS 实时推流
```

---

## 四、快速开始

### 环境依赖

```bash
# Python >= 3.10
# CUDA >= 11.8（GPU 推理必需）

pip install -r requirements.txt
```

### 模型下载（ModelScope）

SAM-3 模型首次运行自动从 ModelScope 下载，无需手动操作。

### 启动五范式系统

```bash
cd 计算机视觉项目
streamlit run app_final.py
```

浏览器打开 `http://localhost:8501`，左侧选择范式（A/B/C/D/E）。

### 启动多轴承监控

```bash
python start_multi_bearing_monitor.py
```

---

## 五、文件结构

```
.
├── app_final.py                     # Streamlit 入口（五范式路由）
│
├── core/                            # 核心算法层（24+ 模块）
│   ├── vlm_bbox.py                  # VLM bbox 解析（600+ 行，五级验证）
│   ├── vlm_model_registry.py        # VLM 模型注册表（8 种模型）
│   ├── dashscope_stream.py          # QVQ 流式双通道聚合
│   ├── sam3_infer.py                # SAM-3 实例分割
│   ├── sam3_video_detector.py       # SAM-3 视频跟踪
│   ├── padim.py                     # PaDiM 异常检测
│   ├── feature_extractor.py         # ResNet-18 特征提取
│   ├── yolov8_export.py            # YOLO 数据集导出
│   ├── defect_config.py             # 缺陷类别配置（5 类行业模板）
│   ├── socket_server.py            # TCP Socket（防粘包协议）
│   └── mjpeg_server.py             # MJPEG 流媒体服务
│
├── ui/                              # 界面层（15+ 组件）
│   ├── paradigm_a.py ~ paradigm_e.py # 五大范式界面
│   ├── monitoring.py                # 工业监控看板
│   └── adapters.py                  # 适配器层
│
├── bearing_core/                    # 多轴承监控系统
│   └── multi_bearing_monitor.py    # 共享模型 + GPU 监控
│
├── edge_device/                    # C++ 边缘 SDK
│   └── src/                        # TCP Socket + 视频采集
│
├── configs/                        # YAML 配置
│   ├── train_yolo_n_rtx4060.yaml   # YOLO 训练配置
│   ├── multi_bearing/              # 1-6 条生产线配置
│   └── defect_presets/            # 5 类行业缺陷预设
│
├── train_production_lines.py        # 三线 YOLO 训练脚本
├── start_multi_bearing_monitor.py  # 多轴承监控启动器
│
└── docs/
    ├── ARCHITECTURE.md             # 系统架构详解
    └── TRAINING.md                 # 训练说明
```

---

## 六、技术栈

| 类别 | 技术 |
|------|------|
| **框架** | Streamlit, PyTorch, Ultralytics YOLOv8, Transformers |
| **多模态大模型** | Qwen-VL, QVQ（阿里云 DashScope API） |
| **分割模型** | SAM-3（Meta，开源），PaDiM |
| **特征提取** | ResNet-18（PyTorch torchvision） |
| **边缘开发** | C++（ARM Linux 交叉编译），TCP Socket，MJPEG |
| **GPU 监控** | NVML（pynvml） |
| **配置管理** | YAML（PyYAML） |

---

## 七、参考资料

- SAM-3: `facebook/sam3`（ModelScope）
- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Qwen-VL: [阿里云 DashScope](https://help.aliyun.com/zh/dashscope/)
- PaDiM: *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection* ([arxiv](https://arxiv.org/abs/2011.08785))

---

*本项目为课程设计作品，数据集来源于公开数据集。*
