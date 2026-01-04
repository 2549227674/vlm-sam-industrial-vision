"""范式 C 的度量工具（VLM bbox → SAM mask）

保持度量简洁和实现轻量。所有度量从选定的掩码（布尔 HxW）和输入 bbox 计算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CRunMetrics:
    """范式 C 单次运行的度量结果"""
    mask_area_ratio_img: float  # 掩码面积占图像的比例
    mask_area_ratio_bbox: float  # 掩码面积占边界框的比例
    iou_maskbbox_vs_vlmbbox: float  # 掩码边界框与 VLM 边界框的 IoU
    frac_mask_inside_vlmbbox: float  # 掩码在 VLM 边界框内的比例
    defect_score: float  # 缺陷得分（0-1）
    status: str  # 状态：ok / low_quality / no_mask


def _bbox_area_xyxy(b: list[int]) -> int:
    """计算边界框面积"""
    x1, y1, x2, y2 = [int(v) for v in b]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    """计算掩码的最小外接矩形"""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    """计算两个边界框的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 计算交集
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    # 计算并集
    area_a = _bbox_area_xyxy(a)
    area_b = _bbox_area_xyxy(b)
    union = max(1, area_a + area_b - inter)

    return float(inter / union)


def compute_c_metrics(
    *,
    mask_bool: np.ndarray,
    image_h: int,
    image_w: int,
    vlm_bbox_xyxy: list[int],
    sam_best_score: float,
    anomaly_subtype: str = "",
) -> CRunMetrics:
    """计算单个边界框运行的最小度量集

    Args:
        mask_bool: 布尔掩码（HxW）
        image_h: 图像高度
        image_w: 图像宽度
        vlm_bbox_xyxy: VLM 边界框 [x1, y1, x2, y2]
        sam_best_score: SAM 最佳得分
        anomaly_subtype: 异常子类型（如 "missing_like"）

    Returns:
        CRunMetrics 对象，包含各项度量和状态
    """
    mask_bool = np.asarray(mask_bool).astype(bool)
    area_mask = int(mask_bool.sum())
    area_img = max(1, int(image_h) * int(image_w))
    area_bbox = max(1, _bbox_area_xyxy(vlm_bbox_xyxy))

    mask_area_ratio_img = float(area_mask / area_img)
    mask_area_ratio_bbox = float(area_mask / area_bbox)

    # 计算掩码边界框与 VLM 边界框的 IoU
    mb = _mask_bbox_xyxy(mask_bool)
    if mb is None:
        iou = 0.0
    else:
        iou = _iou_xyxy(mb, vlm_bbox_xyxy)

    # 裁剪边界框到图像范围内
    x1, y1, x2, y2 = [int(v) for v in vlm_bbox_xyxy]
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(1, min(x2, image_w))
    y2 = max(1, min(y2, image_h))

    # 计算掩码在 VLM 边界框内的比例
    inside = int(mask_bool[y1:y2, x1:x2].sum())
    frac_inside = float(inside / max(1, area_mask))

    # 最小缺陷得分（0-1）
    s = float(max(0.0, min(1.0, sam_best_score)))

    subtype = str(anomaly_subtype or "").strip().lower()

    # 针对 "missing_like" 类型的特殊处理
    if subtype == "missing_like":
        # 缺失类异常：偏好框内一致的证据；避免微小斑点
        # 使用面积作为门槛而非线性乘数
        if area_mask == 0:
            status = "no_mask"
            defect_score = 0.0
        else:
            too_small = mask_area_ratio_bbox < 0.01
            too_large = mask_area_ratio_bbox > 0.85
            low_inside = frac_inside < 0.80
            low_iou = iou < 0.20

            if too_small or too_large or low_inside or low_iou:
                status = "low_quality"
            else:
                status = "ok"

            defect_score = float(s * frac_inside)
            if status != "ok":
                defect_score *= 0.25

        return CRunMetrics(
            mask_area_ratio_img=mask_area_ratio_img,
            mask_area_ratio_bbox=mask_area_ratio_bbox,
            iou_maskbbox_vs_vlmbbox=float(iou),
            frac_mask_inside_vlmbbox=float(frac_inside),
            defect_score=float(defect_score),
            status=status,
        )

    # 默认缺陷得分计算
    defect_score = float(s * frac_inside * max(0.0, min(1.0, mask_area_ratio_bbox)))

    # 状态启发式判断
    if area_mask == 0:
        status = "no_mask"
    elif frac_inside < 0.5 or iou < 0.2:
        status = "low_quality"
    else:
        status = "ok"

    return CRunMetrics(
        mask_area_ratio_img=mask_area_ratio_img,
        mask_area_ratio_bbox=mask_area_ratio_bbox,
        iou_maskbbox_vs_vlmbbox=float(iou),
        frac_mask_inside_vlmbbox=float(frac_inside),
        defect_score=defect_score,
        status=status,
    )


