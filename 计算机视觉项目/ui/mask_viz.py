"""Streamlit UI 的掩码可视化工具

目标：Matplotlib 风格的叠加，每个类别（提示词/关键词）使用稳定的颜色，
而非每个实例。我们不要求跨运行的实例 ID 一致性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskGroup:
    """属于同一语义类别的掩码组"""
    label: str  # 类别标签
    masks: np.ndarray  # 形状：(N, H, W)，bool 或 0/1


def _normalize_label(label: str) -> str:
    """标准化标签：转小写并去除空格"""
    return (label or "").strip().lower()


def _colors_for_labels(labels: Iterable[str], *, cmap_name: str = "tab10") -> dict[str, tuple[int, int, int]]:
    """使用 matplotlib colormap 为标签分配确定性的 RGB 颜色

    Args:
        labels: 标签列表
        cmap_name: matplotlib colormap 名称

    Returns:
        标签到 RGB 颜色的映射字典
    """
    import matplotlib

    labels_norm = [_normalize_label(x) for x in labels]
    uniq = []
    seen = set()
    for x in labels_norm:
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    if not uniq:
        return {}

    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(max(1, len(uniq)))
    out: dict[str, tuple[int, int, int]] = {}
    for i, lab in enumerate(uniq):
        r, g, b, _ = cmap(i)
        out[lab] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def _clamp_u8(x: int) -> int:
    """将值限制在 [0, 255] 范围内"""
    return int(max(0, min(255, int(x))))


def _adjust_color(color: tuple[int, int, int], *, factor: float) -> tuple[int, int, int]:
    """通过乘以因子来调整 RGB 颜色（变亮/变暗）

    Args:
        color: RGB 颜色元组
        factor: 调整因子（<1 变暗，>1 变亮）

    Returns:
        调整后的 RGB 颜色
    """
    r, g, b = color
    return (_clamp_u8(r * factor), _clamp_u8(g * factor), _clamp_u8(b * factor))


def overlay_masks_by_class(
    image,
    groups: list[MaskGroup],
    *,
    alpha: float = 0.5,
    cmap_name: str = "tab10",
    draw_contours: bool = True,
    contour_width: int = 2,
    contour_alpha: float = 0.95,
    contour_color_factor: float = 0.75,
) -> Image.Image:
    """将多个掩码组叠加到图像上

    特点：
    - 相同类别标签 → 相同颜色
    - 不同标签 → 不同颜色
    - 可选绘制轮廓以使实例边界更清晰

    Args:
        image: PIL.Image 或 ndarray (H,W,3)
        groups: MaskGroup 列表
        alpha: 掩码像素的不透明度（0-1）
        cmap_name: matplotlib colormap 名称
        draw_contours: 是否绘制轮廓
        contour_width: 轮廓线宽度（像素）
        contour_alpha: 轮廓线不透明度（0-1）
        contour_color_factor: 乘以基础类别颜色使轮廓更暗（<1）或更亮（>1）

    Returns:
        PIL.Image (RGBA)
    """
    # 规范化参数
    if not (0.0 <= float(alpha) <= 1.0):
        alpha = float(max(0.0, min(1.0, alpha)))

    if not (0.0 <= float(contour_alpha) <= 1.0):
        contour_alpha = float(max(0.0, min(1.0, contour_alpha)))

    contour_width = int(max(1, contour_width))

    # 转换为 RGBA 图像
    if isinstance(image, Image.Image):
        base = image.convert("RGBA")
    else:
        base = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGBA")

    # 为每个类别分配一种颜色
    label_to_color = _colors_for_labels([g.label for g in groups], cmap_name=cmap_name)

    for g in groups:
        lab = _normalize_label(g.label)
        if not lab:
            continue
        color = label_to_color.get(lab, (255, 0, 0))

        masks = np.asarray(g.masks)
        if masks.size == 0:
            continue
        if masks.ndim == 2:
            masks = masks[None, ...]

        # 类别内的并集：任何实例像素都是前景
        union = np.any(masks.astype(bool), axis=0)
        if union.shape[0] == 0 or union.shape[1] == 0:
            continue

        # 1) 填充叠加层
        mask_u8 = (union.astype(np.uint8) * 255)
        mask_img = Image.fromarray(mask_u8, mode="L")

        overlay = Image.new("RGBA", base.size, color + (0,))
        alpha_img = mask_img.point(lambda v: int(v * float(alpha)))
        overlay.putalpha(alpha_img)
        base = Image.alpha_composite(base, overlay)

        # 2) 轮廓叠加层（可选）
        if draw_contours:
            try:
                import cv2

                # OpenCV 需要 uint8 类型
                u8 = mask_u8
                contours, _hier = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_rgb = _adjust_color(color, factor=float(contour_color_factor))

                    # 在 RGBA 画布上绘制
                    contour_layer = np.zeros((u8.shape[0], u8.shape[1], 4), dtype=np.uint8)
                    cv2.drawContours(
                        contour_layer,
                        contours,
                        contourIdx=-1,
                        color=(contour_rgb[0], contour_rgb[1], contour_rgb[2], int(255 * contour_alpha)),
                        thickness=contour_width,
                        lineType=cv2.LINE_AA,
                    )

                    contour_pil = Image.fromarray(contour_layer, mode="RGBA")
                    base = Image.alpha_composite(base, contour_pil)
            except Exception:
                # 如果 OpenCV 不可用或轮廓失败，静默跳过
                pass

    return base



