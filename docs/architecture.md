# 系统架构说明

---

## 一、架构设计原则

本系统采用 **core / ui / adapters** 三层分离架构，通过 Python dataclass 实现模块间的显式数据契约，避免隐性依赖：

```
app_final.py（路由层）
    │
    ▼
ui/ paradigm_*.py（范式界面层）
    │
    ▼
ui/adapters.py（适配器层 — 统一接口封装）
    │
    ▼
core/*.py（核心算法层 — 纯逻辑，无 UI 依赖）
    │
    ▼
models.py（模型加载层 — @st.cache_resource 懒加载）
    │
    ▼
SAM-3 / ResNet-18 / YOLOv8 / VLM API
```

---

## 二、三层检测体系详解

### L1 认知层：VLM 数据流控

**目标**：解决 VLM 输出格式不可控的工程痛点。

```
用户上传图像
    │
    ▼
VLM API 调用（Qwen-VL / QVQ）
    │  强制 JSON Schema Prompt 约束
    ▼
raw_text 原始文本输出
    │
    ├─── JSON 解析成功？ ──► 五级 bbox 净化管道
    │                              │
    │                              ├── L1: 类型验证（is list of 4?）
    │                              ├── L2: 值转换（int(round(float(x))))
    │                              ├── L3: 坐标排序（x1<x2, y1<y2）
    │                              ├── L4: 边界裁剪（clamp to [0,w)×[0,h)）
    │                              └── L5: 面积保障（宽高 ≥1px）
    │
    └─── JSON 解析失败？ ──► 启发式降级决策
                               │
                               ├── 有错误关键词？→ 切换 turbo→max 重试
                               └── 无 JSON 结构？→ 判定为真无缺陷
    │
    ▼
VlmBBoxOutput（dataclass，结构化输出）
    │
    ▼
SAM-3 精分割（bbox 作为提示）
    │
    ▼
YOLOv8 数据集导出（xyxy → 归一化 xywh）
```

### L2 算法层：本地中小模型

#### SAM-3 图像分割

- 支持三种提示模式：文本提示、边界框提示、点提示
- 文本提示支持 per-prompt（逐词推理）和 join-string（拼接一次推理）两种策略
- 动态 dtype：GPU 支持 bf16（Ampere+），不支持则降级 float32

#### SAM-3 视频跟踪

- 使用 `object_ids` 实现跨帧同一实例跟踪
- 无需每帧调用 VLM，零 API 边际成本
- 适用于已知缺陷类型的固定产线批量视频分析

#### PaDiM 异常检测

- ResNet-18 Layer3 输出 [256, 16, 16] 空间特征图
- 对角协方差近似（256 维向量替代 256×256 矩阵），大幅降低计算量
- 每个 16×16 位置独立建模，保留空间信息
- 无需训练样本，仅需正常样本构建统计量

#### YOLOv8 训练

- 基础模型：YOLOv8n（nano，轻量）
- RTX 4060 优化配置：batch=8 + AMP 混合精度
- 早停机制：patience=15，防止过拟合
- 余弦学习率衰减 + Warmup

### L3 部署层：并发监控与边缘协同

#### 多轴承并发监控

```
配置文件（YAML）
    │
    ▼
MultiBearingMonitor
    │
    ├── SharedModelManager（单例，threading.Lock）
    │      └── YOLO 模型一次加载，1-6 线共享
    │
    ├── BearingProductionLine × N（每条线一个线程）
    │      ├── cv2.VideoCapture（线程独立视频读取）
    │      ├── 关键帧跳帧（keyframe_interval=8，减少 87.5% 推理）
    │      └── 检测结果复用（detection_display_frames 帧内持续显示）
    │
    └── GPUMonitor（单例，NVML 真实监控）
           ├── 显存使用量（MB）
           ├── GPU 利用率（%）
           ├── 温度（℃）
           └── 功耗（W）
```

#### C++ 边缘 SDK

```
ARM 设备端（Linux）：
    V4L2 视频采集 → JPEG 编码 → TCP Socket 发送
    帧格式: magic(0xABCD1234) + len(4B) + data

服务端（Python）：
    接收帧 → magic 校验 → 长度校验 → JPEG 头部校验
    → 异常则重同步（搜索下一个 magic）
    → 正常则回调通知推理线程
```

---

## 三、配置驱动设计

### 缺陷类别热插拔

```
configs/defect_presets/
    ├── generic.yaml     # 通用缺陷
    ├── metal.yaml       # 金属表面
    ├── pcb.yaml         # PCB 电路板
    ├── textile.yaml     # 纺织品
    └── food.yaml       # 食品
```

每类预设包含：类别 ID、中英文名称、Prompt 提示词。切换预设即可更换检测缺陷类型，无需修改代码。

### 多轴承生产线配置

```
configs/multi_bearing/
    ├── bearing_1_line.yaml
    ├── bearing_2_lines.yaml
    ├── bearing_3_lines.yaml
    └── bearing_6_lines.yaml
```

每条线独立配置：视频路径、关键帧间隔、置信度阈值、优先级。

---

## 四、数据流总览

```
                    工业图像 / 视频流
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │ 范式 A  │    │  范式 B  │    │  范式 C  │
      │ VLM+SAM │    │  PaDiM   │    │ VLM+SAM  │
      │ 零样本  │    │  无监督  │    │ → YOLO   │
      └─────────┘    └──────────┘    └────┬─────┘
           │               │                │
           └───────────────┼────────────────┘
                           ▼
                    SAM-3 实例分割掩码
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │ 范式 D  │    │  范式 E  │    │  bearing  │
      │ YOLO+   │    │ SAM3     │    │  _core   │
      │ Socket  │    │ 视频跟踪  │    │ 多线并发 │
      └────┬────┘    └──────────┘    └────┬────┘
           │                              │
           ▼                              ▼
    边缘 ARM 设备推流             实时监控面板
```

---

## 五、模块依赖关系

```
app_final.py
├── ui/paradigm_a.py
│   └── adapters.py → core/vlm.py, core/sam3_infer.py
├── ui/paradigm_b.py
│   └── adapters.py → core/padim.py, core/feature_extractor.py
├── ui/paradigm_c.py
│   └── adapters.py → core/vlm_bbox.py, core/sam3_infer.py, core/yolov8_export.py
├── ui/paradigm_d.py
│   └── adapters.py → core/socket_server.py, core/mjpeg_server.py
├── ui/paradigm_e.py
│   └── adapters.py → core/sam3_video_detector.py
└── ui/monitoring.py
    └── bearing_core.multi_bearing_monitor

models.py（统一模型加载）
├── SAM-3 图像模型（facebook/sam3）
├── SAM-3 视频模型（facebook/sam3_video）
└── ResNet-18（torchvision）
```
