# 训练说明

---

## 一、环境准备

### 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | GTX 1060 6GB | RTX 4060 8GB |
| CUDA | 11.8 | 12.1 |
| 内存 | 16GB | 32GB |
| 存储 | 50GB | 100GB NVMe |

### 软件依赖

```bash
pip install ultralytics torch torchvision
# 或使用项目根目录的 requirements.txt
```

---

## 二、YOLOv8 模型训练

### 训练脚本

```bash
python train_production_lines.py
```

### 菜单选项

```
1. 全部训练（轴承 + 木材 + 芯片）
2. 仅轴承生产线
3. 仅木材生产线
4. 仅芯片生产线
```

### RTX 4060 优化配置

```python
training_params = {
    'epochs': 100,
    'imgsz': 640,           # 图像尺寸
    'batch': 8,             # RTX 4060 8GB 适配
    'workers': 2,
    'patience': 15,         # 早停，防止过拟合
    'optimizer': 'SGD',
    'lr0': 0.01,            # 初始学习率
    'lrf': 0.01,            # 最终学习率
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,     # Warmup 稳定训练
    'cos_lr': True,         # 余弦学习率衰减
    'amp': True,            # AMP 混合精度，显存降低 50%
    'close_mosaic': 10,    # 最后10轮关闭 mosaic 增强
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    'fliplr': 0.5, 'mosaic': 1.0,
}
```

### 训练数据集

| 生产线 | 数据集规模 | 类别数 | 类别 |
|--------|-----------|--------|------|
| 轴承 | 2561 张 | 8 类 | 铸造毛刺 / 裂纹 / 划痕 / 凹坑 / 抛光铸件 / 应变 / 未抛光铸件 / 毛刺 |
| 木材 | 4000 张 | 8 类 | 矿物木斑 / 黑洞 / 缺口 / 色斑 / 老化变色 / 粗糙纹理 / 虫洞 / 端裂 |
| 芯片 | 2200 张 | 5 类 | 芯片表面缺陷 |

### 实测训练指标

#### 轴承生产线

```
Epoch 1:   mAP50=0.354  (冷启动)
Epoch 30:  mAP50=0.694  (快速收敛)
Epoch 47:  mAP50=0.725  (早停触发) ← 最终结果
训练耗时: 约 66 分钟
mAP50-95:  0.528
Precision:  0.684
Recall:    0.709
```

#### 木材生产线

```
Epoch 70:  mAP50=0.867
Epoch 75:  mAP50=0.878  ← 最优结果
Epoch 93:  mAP50=0.878  (早停触发)
训练耗时: 约 85 分钟
mAP50-95:  0.644
Precision:  0.845
Recall:    0.733
```

### 训练输出

```
runs/train_bearing/
├── bearing_line_YYYYMMDD_HHMMSS/
│   ├── weights/
│   │   ├── best.pt   ← 最优权重（用于推理）
│   │   └── last.pt   ← 最终权重
│   ├── results.csv   ← 训练曲线数据
│   └── results.png   ← 训练可视化
```

---

## 三、自定义数据集训练

### 目录结构

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       ├── img_101.jpg
│       └── img_102.jpg
├── labels/
│   ├── train/
│   │   ├── img_001.txt   # YOLO 格式：class_id x_center y_center width height
│   │   └── img_002.txt
│   └── val/
│       ├── img_101.txt
│       └── img_102.txt
└── data.yaml            # 数据集配置
```

### data.yaml 格式

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 8  # 类别数量
names:
  0: scratch
  1: crack
  2: pit
  3: burr
  4: strain
  5: casting_burr
  6: polished_casting
  7: unpolished_casting
```

### 使用范式C导出数据集

如果需要从 VLM + SAM-3 流水线生成训练数据：

1. 在范式 C 中上传缺陷图像
2. 使用 VLM bbox 检测 + SAM-3 精修
3. 点击"导出为 YOLO 格式"
4. 导出目录即满足上述目录结构

---

## 四、使用自定义配置

### 修改缺陷类别

编辑 `configs/defect_presets/your_industry.yaml`，参考 `metal.yaml` 格式：

```yaml
version: "1.0"
metadata:
  name: "你的行业名称"
  domain: "your_domain"

primary_types:
  - id: "defect_a"
    display_name: "缺陷A"
  - id: "defect_b"
    display_name: "缺陷B"
```

### 多轴承配置

参考 `configs/multi_bearing/bearing_3_lines.yaml`：

```yaml
global:
  model_path: "runs/train_bearing/your_model/weights/best.pt"
  device: "cuda:0"
  conf_threshold: 0.5
  shared_model: true   # 开启模型共享

lines:
  your_line_1:
    id: 1
    name: "生产线1"
    video: "path/to/video.mp4"
    keyframe_interval: 8
    detection_display_frames: 45
```

---

## 五、常见问题

### 显存不足（OOM）

- 减小 `batch`（建议 RTX 4060 使用 `batch=8`）
- 确保 `amp=True`（混合精度）
- 检查是否有其他程序占用 GPU：`nvidia-smi`

### 训练不收敛

- 检查标签文件格式（YOLO 格式：归一化坐标）
- 检查类别数 `nc` 是否与实际类别数一致
- 尝试增加 `warmup_epochs`

### 推理精度低

- 使用 `best.pt` 而非 `last.pt`
- 调整 `conf_threshold`（置信度阈值）
- 检查数据集是否与推理场景一致
