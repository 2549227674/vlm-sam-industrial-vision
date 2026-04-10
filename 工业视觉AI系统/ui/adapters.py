"""UI-to-core adapter functions.

These wrappers keep the UI code clean and hide core call details.
"""

from __future__ import annotations

import re

import streamlit as st

from ui.constants import IMG_SIZE, FEAT_DIM

from core.sam3_infer import (
    run_sam3_instance_segmentation as _run_sam3_instance_segmentation,
    run_sam3_box_prompt_instance_segmentation as _run_sam3_box_prompt_instance_segmentation,
)
from core.feature_extractor import process_single_image as _process_single_image
from core.vlm import (
    get_vlm_suggestions as _get_vlm_suggestions,
    get_dashscope_key as _get_dashscope_key,
    dashscope_ready as _dashscope_ready,
)
from core.vlm_bbox import get_vlm_defect_bboxes as _get_vlm_defect_bboxes
from core.vlm_bbox import get_vlm_defect_bboxes_compare as _get_vlm_defect_bboxes_compare


def parse_prompt_to_text_input(prompt_raw: str):
    """Parse prompt input into str or list[str]."""
    if not prompt_raw:
        return ""
    parts = [p.strip() for p in re.split(r"[\n,;]+", str(prompt_raw))]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return parts


def run_sam3_instance_segmentation(
    *,
    image_pil,
    sam_proc,
    sam_model,
    sam_dtype,
    prompt,
    threshold: float,
    device: str,
):
    """Delegate segmentation to core.sam3_infer."""
    return _run_sam3_instance_segmentation(
        image_pil=image_pil,
        sam_proc=sam_proc,
        sam_model=sam_model,
        sam_dtype=sam_dtype,
        prompt=prompt,
        threshold=float(threshold),
        device=device,
        multi_prompt_strategy=st.session_state.get("a_multi_prompt_strategy", "per_prompt"),
        session_state=st.session_state,
    )


def process_single_image(
    *,
    image_pil,
    sam_proc,
    sam_model,
    sam_dtype,
    resnet,
    prompt,
    threshold: float = 0.25,
    context_pad: float = 0.2,
    roi_mode: str = "bbox",
    device: str,
):
    """Delegate feature extraction/ROI to core.feature_extractor."""
    return _process_single_image(
        image_pil=image_pil,
        sam_proc=sam_proc,
        sam_model=sam_model,
        sam_dtype=sam_dtype,
        resnet=resnet,
        prompt=prompt,
        threshold=float(threshold),
        context_pad=float(context_pad),
        roi_mode=str(roi_mode),
        img_size=int(IMG_SIZE),
        feat_dim=int(FEAT_DIM),
        device=device,
        multi_prompt_strategy=st.session_state.get("a_multi_prompt_strategy", "per_prompt"),
        session_state=None,
    )


def get_vlm_suggestions(*, image_pil, max_keywords: int = 6, model_name: str = "qwen-vl-max", mode: str = "industrial_defect", api_key: str | None = None):
    """Delegate VLM suggestion to core.vlm.

    Assumes dependencies are installed.
    """
    import dashscope

    return _get_vlm_suggestions(
        image_pil,
        max_keywords=max_keywords,
        model_name=model_name,
        mode=mode,
        thinking=bool(st.session_state.get("a_vlm_thinking", False)),
        api_key=api_key,
        dashscope_module=dashscope,
    )


def get_dashscope_key() -> str:
    return _get_dashscope_key()


def dashscope_ready(key: str) -> bool:
    return _dashscope_ready(key)


def run_sam3_box_prompt_instance_segmentation(
    *,
    image_pil,
    sam_proc,
    sam_model,
    sam_dtype,
    boxes_xyxy: list[list[int]],
    box_labels: list[int] | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str,
):
    return _run_sam3_box_prompt_instance_segmentation(
        image_pil=image_pil,
        sam_proc=sam_proc,
        sam_model=sam_model,
        sam_dtype=sam_dtype,
        boxes_xyxy=boxes_xyxy,
        box_labels=box_labels,
        threshold=float(threshold),
        mask_threshold=float(mask_threshold),
        device=device,
    )


def get_vlm_defect_bboxes(
    *,
    image_pil,
    model_name: str = "qwen-vl-max",
    api_key: str | None = None,
    max_boxes: int = 3,
):
    import dashscope

    return _get_vlm_defect_bboxes(
        image_pil=image_pil,
        model_name=model_name,
        thinking=bool(st.session_state.get("c_vlm_thinking", False)),
        api_key=api_key,
        dashscope_module=dashscope,
        max_boxes=int(max_boxes),
    )


def get_vlm_defect_bboxes_compare(
    *,
    normal_image_pil,
    test_image_pil,
    model_name: str = "qwen-vl-max",
    api_key: str | None = None,
    max_boxes: int = 3,
):
    import dashscope

    return _get_vlm_defect_bboxes_compare(
        normal_image_pil=normal_image_pil,
        test_image_pil=test_image_pil,
        model_name=model_name,
        thinking=bool(st.session_state.get("c_vlm_thinking", False)),
        api_key=api_key,
        dashscope_module=dashscope,
        max_boxes=int(max_boxes),
    )
