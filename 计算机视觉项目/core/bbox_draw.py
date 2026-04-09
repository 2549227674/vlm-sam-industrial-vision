"""Utility functions for drawing bounding boxes on images."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes_on_image(
    image_pil: Image.Image,
    bboxes: list[list[int]],
    labels: list[str] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3
) -> Image.Image:
    """Draw bounding boxes on image.

    Args:
        image_pil: PIL Image
        bboxes: List of bboxes in [x1, y1, x2, y2] format
        labels: Optional labels for each bbox
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        PIL Image with bboxes drawn
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        # Draw rectangle
        for offset in range(thickness):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
                width=1
            )

        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            # Simple text drawing (no font file needed)
            text_bbox = draw.textbbox((x1, y1 - 20), label)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255))

    return img_draw


def draw_sam_masks_on_image(
    image_pil: Image.Image,
    masks: list[np.ndarray],
    bboxes: list[list[int]],
    alpha: float = 0.5
) -> Image.Image:
    """Draw SAM segmentation masks on image.

    Args:
        image_pil: PIL Image
        masks: List of binary masks
        bboxes: List of bboxes (for coloring)
        alpha: Transparency of masks

    Returns:
        PIL Image with masks overlaid
    """
    img_array = np.array(image_pil)
    overlay = img_array.copy()

    # Color palette for different masks
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    for i, mask in enumerate(masks):
        if mask is None or mask.size == 0:
            continue

        color = colors[i % len(colors)]

        # Apply color to mask region
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    result = Image.fromarray(overlay.astype(np.uint8))

    # Draw bboxes on top
    result = draw_bboxes_on_image(result, bboxes, color=(255, 255, 0), thickness=2)

    return result
