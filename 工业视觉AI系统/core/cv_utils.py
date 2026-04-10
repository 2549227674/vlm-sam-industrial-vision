"""计算机视觉工具函数

提供图像处理和掩码叠加等常用功能。
"""

from __future__ import annotations

import cv2
import numpy as np


def pad_to_square_cv2(img: np.ndarray) -> np.ndarray:
    """保持长宽比，通过黑色边框补齐为正方形

    Args:
        img: 输入图像（numpy 数组）

    Returns:
        补齐后的正方形图像
    """
    h, w = img.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def apply_mask_overlay_np(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """将二值掩码以指定透明度叠加到 RGB 图像上

    Args:
        img_rgb: RGB 图像
        mask: 二值掩码
        alpha: 透明度（0-1）
        color: 掩码颜色 (R, G, B)

    Returns:
        叠加后的图像
    """
    overlay = img_rgb.copy()
    overlay[mask] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)


def overlay_single_mask_on_image_rgb(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """将单个布尔掩码叠加到 RGB 图像上

    Args:
        image_rgb: RGB 图像
        mask_bool: 布尔掩码
        alpha: 透明度（0-1）
        color: 掩码颜色 (R, G, B)

    Returns:
        叠加后的图像
    """
    return apply_mask_overlay_np(image_rgb, mask_bool.astype(bool), alpha=alpha, color=color)


