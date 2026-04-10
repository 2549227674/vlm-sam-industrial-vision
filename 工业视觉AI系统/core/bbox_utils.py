"""边界框工具函数

用于范式 C 在将 VLM 提出的边界框传递给 SAM 之前进行可选的填充/扩展。
放在 core 中避免 UI 代码重复。
"""

from __future__ import annotations


def pad_bbox_xyxy(bbox_xyxy: list[int], *, pad_ratio: float, image_w: int, image_h: int) -> list[int]:
    """按比例填充边界框并限制在图像范围内

    Args:
        bbox_xyxy: [x1, y1, x2, y2] 像素坐标
        pad_ratio: 填充比例，例如 0.2 表示每边扩展 20%
        image_w: 图像宽度
        image_h: 图像高度

    Returns:
        填充后的边界框 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * float(pad_ratio)))
    py = int(round(bh * float(pad_ratio)))

    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(int(image_w), x2 + px)
    ny2 = min(int(image_h), y2 + py)

    # 确保边界框有效（宽高至少为1）
    if nx2 <= nx1:
        nx2 = min(int(image_w), nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(int(image_h), ny1 + 1)

    return [int(nx1), int(ny1), int(nx2), int(ny2)]



