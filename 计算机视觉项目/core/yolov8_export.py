"""YOLOv8数据集导出模块

该模块负责将范式C的VLM + SAM检测结果导出为YOLOv8训练格式，支持：
- 单张图片标注导出
- 批量图片标注导出
- 自动生成data.yaml配置
- 支持自定义缺陷类别配置

YOLOv8标注格式：
    每个图片对应一个.txt文件，格式为：
    <class_id> <x_center> <y_center> <width> <height>
    坐标均为归一化值（0-1）

作者：System
创建时间：2026-01-08
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def export_single_to_yolov8(
    detections: List[Any],
    image_w: int,
    image_h: int,
    class_names: List[str],
    output_txt_path: str,
) -> None:
    """导出单张图片的YOLOv8标注

    Args:
        detections: VlmBBoxDetection列表
        image_w: 图像宽度
        image_h: 图像高度
        class_names: 类别名称列表（顺序对应class_id）
        output_txt_path: 输出txt文件路径
    """
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for det in detections:
            defect_type = det.defect_type

            # 获取class_id
            try:
                class_id = class_names.index(defect_type)
            except ValueError:
                # 如果类型不在列表中，尝试映射为other
                try:
                    class_id = class_names.index("other")
                except ValueError:
                    # 如果连other都没有，跳过该检测
                    print(f"⚠️ 警告：缺陷类型 '{defect_type}' 不在类别列表中，已跳过")
                    continue

            # 转换为归一化坐标
            x1, y1, x2, y2 = det.bbox_xyxy
            x_center = ((x1 + x2) / 2) / image_w
            y_center = ((y1 + y2) / 2) / image_h
            width = (x2 - x1) / image_w
            height = (y2 - y1) / image_h

            # 写入文件
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def export_batch_to_yolov8(
    results: List[Dict[str, Any]],
    image_files: List[str],
    class_names: List[str],
    output_dir: str,
    split_ratio: float = 0.8,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """批量导出YOLOv8数据集

    Args:
        results: 批量检测结果列表（每个元素包含vlm_detections、image等）
        image_files: 对应的图片文件路径列表
        class_names: 类别名称列表
        output_dir: 输出目录
        split_ratio: 训练集比例（默认0.8，即80%训练20%验证）
        copy_images: 是否复制图片到输出目录（默认True）

    Returns:
        导出统计信息字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建目录结构
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"

    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        "total": 0,
        "train": 0,
        "val": 0,
        "defects_count": {name: 0 for name in class_names},
        "skipped": 0,
    }

    # 处理每张图片
    split_idx = int(len(results) * split_ratio)

    for idx, (result, img_file) in enumerate(zip(results, image_files)):
        # 提取detections
        detections = result.get("vlm_detections", [])
        if not detections:
            stats["skipped"] += 1
            continue

        # 判断是train还是val
        is_train = idx < split_idx
        images_dir = train_images_dir if is_train else val_images_dir
        labels_dir = train_labels_dir if is_train else val_labels_dir

        # 文件名
        img_path = Path(img_file)
        stem = img_path.stem

        # 复制图片（如果需要）
        if copy_images and img_path.exists():
            shutil.copy(img_path, images_dir / img_path.name)

        # 写标注文件
        txt_path = labels_dir / f"{stem}.txt"

        # 获取图像尺寸
        image_w = result.get("image", {}).get("w", 1)
        image_h = result.get("image", {}).get("h", 1)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for det_dict in detections:
                defect_type = det_dict.get("type", "other")
                bbox_xyxy = det_dict.get("bbox_xyxy", [])

                if len(bbox_xyxy) != 4:
                    continue

                # 获取class_id
                try:
                    class_id = class_names.index(defect_type)
                except ValueError:
                    try:
                        class_id = class_names.index("other")
                    except ValueError:
                        continue

                # 转换为归一化坐标
                x1, y1, x2, y2 = bbox_xyxy
                x_center = ((x1 + x2) / 2) / image_w
                y_center = ((y1 + y2) / 2) / image_h
                width = (x2 - x1) / image_w
                height = (y2 - y1) / image_h

                # 写入文件
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # 统计
                stats["defects_count"][defect_type] = stats["defects_count"].get(defect_type, 0) + 1

        # 更新统计
        stats["total"] += 1
        if is_train:
            stats["train"] += 1
        else:
            stats["val"] += 1

    # 生成data.yaml
    data_yaml_content = {
        "path": str(output_path.absolute()),  # 数据集根目录
        "train": "images/train",  # 训练集图片相对路径
        "val": "images/val",      # 验证集图片相对路径
        "nc": len(class_names),   # 类别数量
        "names": class_names,     # 类别名称列表
    }

    with open(output_path / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, allow_unicode=True, sort_keys=False)

    # 生成classes.txt（备用）
    with open(output_path / "classes.txt", 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # 生成README.md
    readme_content = f"""# YOLOv8数据集

## 数据集信息

- **总图片数**: {stats['total']}
- **训练集**: {stats['train']} 张
- **验证集**: {stats['val']} 张
- **类别数**: {len(class_names)}

## 类别列表

"""
    for idx, name in enumerate(class_names):
        count = stats["defects_count"].get(name, 0)
        readme_content += f"{idx}. `{name}` - {count} 个标注\n"

    readme_content += f"""

## 使用方法

### 训练YOLOv8模型

```bash
# 安装ultralytics
pip install ultralytics

# 训练（以YOLOv8n为例）
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 验证模型

```bash
yolo detect val data=data.yaml model=runs/detect/train/weights/best.pt
```

### 推理

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images
```

## 目录结构

```
{output_path.name}/
├── data.yaml          # YOLOv8配置文件
├── classes.txt        # 类别列表
├── README.md          # 本文件
├── images/
│   ├── train/        # 训练集图片
│   └── val/          # 验证集图片
└── labels/
    ├── train/        # 训练集标注
    └── val/          # 验证集标注
```

---
**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**生成工具**: 范式C VLM + SAM 缺陷检测系统
"""

    with open(output_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)

    return stats


def validate_yolov8_dataset(dataset_dir: str) -> Dict[str, Any]:
    """验证YOLOv8数据集完整性

    Args:
        dataset_dir: 数据集目录

    Returns:
        验证结果字典
    """
    dataset_path = Path(dataset_dir)

    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {}
    }

    # 检查必要文件
    if not (dataset_path / "data.yaml").exists():
        result["valid"] = False
        result["errors"].append("缺少 data.yaml 文件")

    # 检查目录结构
    required_dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]

    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            result["valid"] = False
            result["errors"].append(f"缺少目录: {dir_name}")

    # 检查图片和标注匹配
    if result["valid"]:
        for split in ["train", "val"]:
            img_dir = dataset_path / "images" / split
            lbl_dir = dataset_path / "labels" / split

            img_files = set(f.stem for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
            lbl_files = set(f.stem for f in lbl_dir.glob("*.txt"))

            missing_labels = img_files - lbl_files
            missing_images = lbl_files - img_files

            if missing_labels:
                result["warnings"].append(f"{split}: {len(missing_labels)} 张图片缺少标注文件")

            if missing_images:
                result["warnings"].append(f"{split}: {len(missing_images)} 个标注文件缺少对应图片")

            result["statistics"][f"{split}_images"] = len(img_files)
            result["statistics"][f"{split}_labels"] = len(lbl_files)

    return result


def quick_export_from_paradigm_c_results(
    vlm_detections: List[Any],
    image_path: str,
    output_dir: str,
    config: Any,
) -> str:
    """快速导出单张图片的检测结果为YOLOv8格式

    Args:
        vlm_detections: VlmBBoxDetection列表
        image_path: 图片路径
        output_dir: 输出目录
        config: DefectCategoryConfig实例

    Returns:
        输出的标注文件路径
    """
    from PIL import Image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取类别列表
    class_names = config.to_yolov8_classes()

    # 读取图片尺寸
    img = Image.open(image_path)
    image_w, image_h = img.size

    # 生成输出文件名
    img_name = Path(image_path).stem
    txt_path = output_path / f"{img_name}.txt"

    # 导出标注
    export_single_to_yolov8(
        detections=vlm_detections,
        image_w=image_w,
        image_h=image_h,
        class_names=class_names,
        output_txt_path=str(txt_path)
    )

    return str(txt_path)

