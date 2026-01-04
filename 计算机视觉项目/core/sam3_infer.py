"""SAM-3 实例分割推理模块

提供两种推理模式：
1. 文本提示推理 (run_sam3_instance_segmentation)
2. 边界框提示推理 (run_sam3_box_prompt_instance_segmentation)
"""

from __future__ import annotations

import time
from typing import Any

import torch


def merge_instance_results(result_list: list[dict]) -> dict:
    """合并多个实例分割结果

    Args:
        result_list: 分割结果列表，每个元素包含 masks 和 scores

    Returns:
        合并后的结果字典，包含所有 masks 和 scores
    """
    if not result_list:
        return {"masks": torch.empty((0, 1, 1)), "scores": torch.empty((0,))}

    masks_all = []
    scores_all = []

    for r in result_list:
        if not r:
            continue
        if "masks" in r and r["masks"] is not None and len(r["masks"]) > 0:
            masks_all.append(r["masks"])
        if "scores" in r and r["scores"] is not None and len(r["scores"]) > 0:
            scores_all.append(r["scores"].float())

    if not masks_all or not scores_all:
        return {"masks": torch.empty((0, 1, 1)), "scores": torch.empty((0,))}

    return {
        "masks": torch.cat(masks_all, dim=0),
        "scores": torch.cat(scores_all, dim=0)
    }


def run_sam3_instance_segmentation(
    *,
    image_pil,
    sam_proc,
    sam_model,
    sam_dtype,
    prompt,
    threshold: float,
    device: str,
    multi_prompt_strategy: str = "per_prompt",
    session_state: Any | None = None,
):
    """运行 SAM-3 实例分割推理

    支持单个或多个文本提示词，提供两种多词推理策略：
    - per_prompt: 逐词分别推理并合并结果（推荐，最稳定）
    - join_string: 多词用逗号拼接成一句话，一次推理（更快但不保证效果）

    Args:
        image_pil: PIL 图像对象
        sam_proc: SAM-3 处理器
        sam_model: SAM-3 模型
        sam_dtype: 数据类型
        prompt: 文本提示词，可以是 str 或 list[str]
        threshold: 置信度阈值
        device: 设备（cuda/cpu）
        multi_prompt_strategy: 多词推理策略
        session_state: 可选的会话状态，用于记录推理模式

    Returns:
        (results, latency): 分割结果和推理耗时（毫秒）
    """
    start_time = time.time()

    # 处理多个提示词的情况
    if isinstance(prompt, (list, tuple)):
        words = [str(x).strip() for x in prompt if str(x).strip()]

        if session_state is not None:
            session_state["a_prompt_mode_label"] = f"multi:{multi_prompt_strategy}"

        if not words:
            return {"masks": torch.empty((0, 1, 1)), "scores": torch.empty((0,))}, 0.0

        # 策略1: 拼接成一句话
        if multi_prompt_strategy == "join_string":
            joined = ", ".join(words)
            inputs = sam_proc(images=image_pil, text=joined, return_tensors="pt").to(device, sam_dtype)

            with torch.no_grad():
                outputs = sam_model(**inputs)
                results = sam_proc.post_process_instance_segmentation(
                    outputs,
                    target_sizes=[image_pil.size[::-1]],
                    threshold=threshold
                )[0]

            latency = (time.time() - start_time) * 1000
            return results, latency

        # 策略2: 逐词推理并合并
        results_list = []
        for w in words:
            inputs = sam_proc(images=image_pil, text=w, return_tensors="pt").to(device, sam_dtype)

            with torch.no_grad():
                outputs = sam_model(**inputs)
                r = sam_proc.post_process_instance_segmentation(
                    outputs,
                    target_sizes=[image_pil.size[::-1]],
                    threshold=threshold
                )[0]

            results_list.append(r)

        merged = merge_instance_results(results_list)
        latency = (time.time() - start_time) * 1000
        return merged, latency

    # 单个提示词的情况
    if session_state is not None:
        session_state["a_prompt_mode_label"] = "single"

    inputs = sam_proc(images=image_pil, text=str(prompt), return_tensors="pt").to(device, sam_dtype)

    with torch.no_grad():
        outputs = sam_model(**inputs)
        results = sam_proc.post_process_instance_segmentation(
            outputs,
            target_sizes=[image_pil.size[::-1]],
            threshold=threshold
        )[0]

    latency = (time.time() - start_time) * 1000
    return results, latency


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
    """使用边界框提示运行 SAM-3 实例分割

    Args:
        image_pil: PIL 图像对象
        sam_proc: SAM-3 处理器
        sam_model: SAM-3 模型
        sam_dtype: 数据类型
        boxes_xyxy: 边界框列表，格式为 [[x1,y1,x2,y2], ...]，像素坐标
        box_labels: 边界框标签列表，1 表示正样本，0 表示负样本；默认全为 1
        threshold: 置信度阈值
        mask_threshold: 掩码阈值
        device: 设备（cuda/cpu）

    Returns:
        (results, latency): 分割结果和推理耗时（毫秒）

    Note:
        Transformers Sam3Processor 期望输入格式：
        - input_boxes: [batch, num_boxes, 4]
        - input_boxes_labels: [batch, num_boxes]
    """
    start_time = time.time()

    if not boxes_xyxy:
        return {
            "masks": torch.empty((0, 1, 1)),
            "scores": torch.empty((0,)),
            "boxes": torch.empty((0, 4))
        }, 0.0

    # 默认所有框都是正样本
    if box_labels is None:
        box_labels = [1] * len(boxes_xyxy)

    # 标准化数据类型
    boxes_xyxy = [[int(v) for v in b] for b in boxes_xyxy]
    box_labels = [int(v) for v in box_labels]

    # 准备输入
    inputs = sam_proc(
        images=image_pil,
        input_boxes=[boxes_xyxy],
        input_boxes_labels=[box_labels],
        return_tensors="pt",
    ).to(device, sam_dtype)

    # 推理
    with torch.no_grad():
        outputs = sam_model(**inputs)

    # 后处理
    target_sizes = (
        inputs.get("original_sizes").tolist()
        if hasattr(inputs, "get")
        else [image_pil.size[::-1]]
    )

    results = sam_proc.post_process_instance_segmentation(
        outputs,
        threshold=float(threshold),
        mask_threshold=float(mask_threshold),
        target_sizes=target_sizes,
    )[0]

    latency = (time.time() - start_time) * 1000.0
    return results, latency
