from __future__ import annotations

from typing import Any

import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import streamlit as st
from PIL import Image

from core.paradigm_c_metrics import compute_c_metrics
from core.bbox_utils import pad_bbox_xyxy
from core.vlm_model_registry import list_models, is_stream_only_model
from core.defect_config import DefectCategoryConfig, get_available_presets, load_preset_config  # ✅ 新增
from core.yolov8_export import export_batch_to_yolov8, validate_yolov8_dataset  # ✅ 新增
from ui.mask_viz import MaskGroup, overlay_masks_by_class
from ui.components import UIComponents, LoadingStates  # ✅ 导入组件库
from ui.adapters import (
    get_dashscope_key,
    dashscope_ready,
    get_vlm_defect_bboxes,
    get_vlm_defect_bboxes_compare,
    run_sam3_box_prompt_instance_segmentation,
)


def _safe_stem(name: str) -> str:
    s = (name or "image").strip().replace("\\", "_").replace("/", "_")
    for ch in [":", "*", "?", '"', "<", ">", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "_")
    s = s.strip(" .")
    return s or "image"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_vis_image(path: str, img_rgb: np.ndarray) -> None:
    Image.fromarray(img_rgb.astype(np.uint8)).save(path)


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """将嵌套的结果 JSON 展平成适合 CSV 的小字典。"""
    final = row.get("final") or {}
    return {
        "name": row.get("name", ""),
        "vlm_input_mode": row.get("vlm_input_mode", ""),
        "sam_strategy": row.get("sam_strategy", ""),
        "decision": final.get("decision", ""),
        "evidence": final.get("evidence", ""),
        "reason": final.get("reason", ""),
        "num_valid_boxes": final.get("num_valid_boxes", 0),
        "num_vlm_dets": len(row.get("vlm_detections") or []),
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    # 合并所有键（保持顺序）
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_paradigm_c_once(
    *,
    device: str,
    sam_proc,
    sam_model,
    sam_dtype,
    mode: str,  # 'single' | 'compare'
    test_pil: Image.Image,
    normal_pil: Image.Image | None,
    vlm_model: str,
    api_key: str,
    max_boxes: int,
    sam_thr: float,
    mask_thr: float,
    bbox_pad: float,
    alpha: float,
    fast_mode: bool,
    enable_sam3: bool = True,  # 🆕 是否启用SAM-3推理（默认True保持兼容）
) -> tuple[dict[str, Any], np.ndarray]:
    """运行一次范式 C 并返回 (out_json, vis_rgb)。

    合约：
      - out_json 包含 'final.decision' 与 'final.evidence'
      - vis_rgb 是用于保存/展示的 RGB numpy 数组

    Args:
      enable_sam3: 是否启用SAM-3推理。关闭时仅返回VLM bbox结果，加快数据集制作速度。
    """
    test_pil = test_pil.convert("RGB")
    raw = np.array(test_pil)
    H, W = int(test_pil.size[1]), int(test_pil.size[0])

    # 1）请求 VLM 获取候选 bbox
    if mode == "compare" and normal_pil is not None:
        vlm_out = get_vlm_defect_bboxes_compare(
            normal_image_pil=normal_pil.convert("RGB"),
            test_image_pil=test_pil,
            model_name=str(vlm_model),
            api_key=api_key,
            max_boxes=int(max_boxes),
        )
    else:
        vlm_out = get_vlm_defect_bboxes(
            image_pil=test_pil,
            model_name=str(vlm_model),
            api_key=api_key,
            max_boxes=int(max_boxes),
        )

    dets = list(getattr(vlm_out, "detections", []) or [])

    out_json: dict[str, Any] = {
        "mode": "paradigm_c",
        "vlm_input_mode": "compare" if (mode == "compare" and normal_pil is not None) else "single",
        "sam_strategy": "disabled" if not enable_sam3 else ("S1_multi_box" if bool(fast_mode) else "S2_per_box"),
        "image": {"w": int(W), "h": int(H)},
        "vlm_model": str(vlm_model),
        "vlm_thinking": bool(st.session_state.get("c_vlm_thinking", False)),
        "vlm_detections": [
            {
                "type": d.defect_type,
                "anomaly_subtype": (getattr(d, "anomaly_subtype", "") or ""),
                "bbox_xyxy": d.bbox_xyxy,
                "conf": float(d.conf),
            }
            for d in dets
        ],
        "runs": [],
        "bbox_pad_ratio": float(bbox_pad),
        "final": {"decision": "ok", "evidence": "none", "num_valid_boxes": 0},
    }

    # 无候选 bbox -> 判定为 ok
    if not dets:
        return out_json, raw

    # 🆕 如果SAM-3关闭，直接返回VLM bbox结果
    if not enable_sam3:
        out_json["final"].update({
            "decision": "defect" if len(dets) > 0 else "ok",
            "evidence": "vlm_bbox_only",
            "num_valid_boxes": len(dets),
        })
        # 绘制VLM bbox可视化
        vis = raw.copy()
        for idx, d in enumerate(dets):
            subtype = getattr(d, "anomaly_subtype", "") or ""
            subtxt = f" [{subtype}]" if subtype else ""
            vis = _draw_bbox_on_image(
                vis,
                d.bbox_xyxy,
                label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                color=(255, 0, 0),
                thickness=2,
            )
        return out_json, vis

    # 2）为 SAM prompt 准备带 padding 的 bbox
    boxes = [
        pad_bbox_xyxy(d.bbox_xyxy, pad_ratio=float(bbox_pad), image_w=W, image_h=H) if float(bbox_pad) > 0 else d.bbox_xyxy
        for d in dets
    ]

    # 3）调用 SAM 进行分割
    if fast_mode:
        results, _lat = run_sam3_box_prompt_instance_segmentation(
            image_pil=test_pil,
            sam_proc=sam_proc,
            sam_model=sam_model,
            sam_dtype=sam_dtype,
            boxes_xyxy=boxes,
            box_labels=[1] * len(boxes),
            threshold=float(sam_thr),
            mask_threshold=float(mask_thr),
            device=device,
        )

        masks = (results or {}).get("masks")
        scores = (results or {}).get("scores")

        if masks is None or len(masks) == 0:
            out_json["final"].update({"decision": "vlm_suspect", "evidence": "vlm_bbox_only", "reason": "sam_no_mask", "num_valid_boxes": 0})
            vis = raw.copy()
            for idx, d in enumerate(dets):
                subtype = getattr(d, "anomaly_subtype", "") or ""
                subtxt = f" [{subtype}]" if subtype else ""
                vis = _draw_bbox_on_image(
                    vis,
                    d.bbox_xyxy,
                    label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                    color=(255, 165, 0),
                    thickness=3,
                )
            return out_json, vis

        m_np = masks.cpu().numpy() > 0.5
        merged = np.any(m_np, axis=0)
        mg = MaskGroup(label="defect", masks=merged[None, ...])
        vis_pil = overlay_masks_by_class(raw, [mg], alpha=float(alpha), cmap_name="tab10")
        out_json["final"].update({"decision": "defect", "evidence": "sam_mask", "num_valid_boxes": int(len(dets))})
        return out_json, np.array(vis_pil.convert("RGB"))

    # S2：逐个 bbox 处理并聚合结果
    valid = 0
    merged = np.zeros((H, W), dtype=bool)
    for det, padded_box in zip(dets, boxes):
        results, _lat = run_sam3_box_prompt_instance_segmentation(
            image_pil=test_pil,
            sam_proc=sam_proc,
            sam_model=sam_model,
            sam_dtype=sam_dtype,
            boxes_xyxy=[padded_box],
            box_labels=[1],
            threshold=float(sam_thr),
            mask_threshold=float(mask_thr),
            device=device,
        )
        masks = (results or {}).get("masks")
        scores = (results or {}).get("scores")

        if masks is None or len(masks) == 0:
            m_best = np.zeros((H, W), dtype=bool)
            best_score = 0.0
        else:
            m_np = masks.cpu().numpy() > 0.5
            s_np = scores.float().cpu().numpy() if scores is not None and len(scores) > 0 else np.zeros((m_np.shape[0],), dtype=float)
            idx = int(np.argmax(s_np)) if s_np.size > 0 else 0
            m_best = m_np[idx]
            best_score = float(s_np[idx]) if s_np.size > 0 else 0.0

        met = compute_c_metrics(
            mask_bool=m_best,
            image_h=H,
            image_w=W,
            vlm_bbox_xyxy=det.bbox_xyxy,
            sam_best_score=best_score,
            anomaly_subtype=(getattr(det, "anomaly_subtype", "") or ""),
        )

        if met.status == "ok":
            merged |= m_best
            valid += 1

        out_json["runs"].append({
            "bbox_xyxy": det.bbox_xyxy,
            "bbox_xyxy_padded": padded_box,
            "anomaly_subtype": (getattr(det, "anomaly_subtype", "") or ""),
            "sam_best_score": float(best_score),
            "mask_area_ratio_img": float(met.mask_area_ratio_img),
            "mask_area_ratio_bbox": float(met.mask_area_ratio_bbox),
            "iou_maskbbox_vs_vlmbbox": float(met.iou_maskbbox_vs_vlmbbox),
            "frac_mask_inside_vlmbbox": float(met.frac_mask_inside_vlmbbox),
            "defect_score": float(met.defect_score),
            "status": met.status,
        })

    out_json["final"]["num_valid_boxes"] = int(valid)

    if valid > 0:
        out_json["final"].update({"decision": "defect", "evidence": "sam_mask"})
        mg = MaskGroup(label="defect", masks=merged[None, ...])
        vis_pil = overlay_masks_by_class(raw, [mg], alpha=float(alpha), cmap_name="tab10")
        return out_json, np.array(vis_pil.convert("RGB"))

    out_json["final"].update({"decision": "vlm_suspect", "evidence": "vlm_bbox_only", "reason": "sam_failed_or_low_quality"})
    vis = raw.copy()
    for idx, d in enumerate(dets):
        subtype = getattr(d, "anomaly_subtype", "") or ""
        subtxt = f" [{subtype}]" if subtype else ""
        vis = _draw_bbox_on_image(
            vis,
            d.bbox_xyxy,
            label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
            color=(255, 165, 0),
            thickness=3,
        )
    return out_json, vis


def run_paradigm_c_vlm_only(
    *,
    mode: str,
    test_pil: Image.Image,
    normal_pil: Image.Image | None,
    vlm_model: str,
    api_key: str,
    max_boxes: int,
) -> dict[str, Any]:
    """仅执行VLM推理，返回bbox结果（不执行SAM-3分割）。

    用于并发批量处理时的VLM推理步骤。

    Returns:
        dict: 包含 vlm_detections 和图像信息的结果字典
    """
    test_pil = test_pil.convert("RGB")
    raw = np.array(test_pil)
    H, W = int(test_pil.size[1]), int(test_pil.size[0])

    # 请求 VLM 获取候选 bbox
    if mode == "compare" and normal_pil is not None:
        vlm_out = get_vlm_defect_bboxes_compare(
            normal_image_pil=normal_pil.convert("RGB"),
            test_image_pil=test_pil,
            model_name=str(vlm_model),
            api_key=api_key,
            max_boxes=int(max_boxes),
        )
    else:
        vlm_out = get_vlm_defect_bboxes(
            image_pil=test_pil,
            model_name=str(vlm_model),
            api_key=api_key,
            max_boxes=int(max_boxes),
        )

    dets = list(getattr(vlm_out, "detections", []) or [])

    # 构建结果
    result = {
        "mode": "paradigm_c",
        "vlm_input_mode": "compare" if (mode == "compare" and normal_pil is not None) else "single",
        "sam_strategy": "disabled",
        "image": {"w": int(W), "h": int(H)},
        "vlm_model": str(vlm_model),
        "vlm_detections": [
            {
                "type": d.defect_type,
                "anomaly_subtype": (getattr(d, "anomaly_subtype", "") or ""),
                "bbox_xyxy": d.bbox_xyxy,
                "conf": float(d.conf),
            }
            for d in dets
        ],
        "runs": [],
        "final": {
            "decision": "defect" if len(dets) > 0 else "ok",
            "evidence": "vlm_bbox_only",
            "num_valid_boxes": len(dets),
        },
        "_raw_image": raw,  # 用于后续可视化
        "_dets": dets,      # 用于后续可视化
    }

    return result


def run_paradigm_c_batch_concurrent(
    *,
    test_items: list[tuple[Any, Image.Image]],
    mode: str,
    normal_pil: Image.Image | None,
    vlm_model: str,
    api_key: str,
    max_boxes: int,
    max_workers: int = 4,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """并发批量执行VLM推理。

    Args:
        test_items: [(uploaded_file, pil_image), ...] 列表
        mode: 'single' 或 'compare'
        normal_pil: 正常参考图（compare模式时使用）
        vlm_model: VLM模型名称
        api_key: API密钥
        max_boxes: 最大bbox数量
        max_workers: 最大并发线程数（默认4）
        progress_callback: 进度回调函数 callback(completed, total)

    Returns:
        结果列表，每个元素对应一张图片的VLM推理结果
    """
    results = [None] * len(test_items)
    completed_count = [0]  # 使用列表以便在闭包中修改
    lock = threading.Lock()

    def process_single(idx: int, up, pil: Image.Image) -> tuple[int, dict[str, Any]]:
        """处理单张图片"""
        name = getattr(up, "name", f"img_{idx}.jpg")
        try:
            result = run_paradigm_c_vlm_only(
                mode=mode,
                test_pil=pil,
                normal_pil=normal_pil,
                vlm_model=vlm_model,
                api_key=api_key,
                max_boxes=max_boxes,
            )
            result["name"] = name
            result["_index"] = idx
            return idx, result
        except Exception as e:
            import traceback
            return idx, {
                "name": name,
                "_index": idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "final": {"decision": "error"},
                "vlm_detections": [],
                "image": {"w": pil.size[0], "h": pil.size[1]},
                "_raw_image": np.array(pil.convert("RGB")),
                "_dets": [],
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single, i, up, pil): i
            for i, (up, pil) in enumerate(test_items)
        }

        # 收集结果
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

            # 更新进度
            with lock:
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(completed_count[0], len(test_items))

    return results


def _draw_bbox_on_image(
    img_rgb: np.ndarray,
    bbox_xyxy: list[int],
    *,
    label: str | None = None,
    color=(255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    try:
        import cv2

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        out = img_rgb.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        if label:
            # 在框的左上角附近放置标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            txt_th = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, txt_th)

            # 确保标签在图像范围内
            lx = max(0, min(x1, out.shape[1] - 1))
            ly = max(0, y1 - th - baseline - 4)

            # 绘制填充背景以提高可读性
            bg_x2 = min(out.shape[1], lx + tw + 6)
            bg_y2 = min(out.shape[0], ly + th + baseline + 6)
            cv2.rectangle(out, (lx, ly), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.putText(out, label, (lx + 3, ly + th + 3), font, font_scale, (255, 255, 255), txt_th, cv2.LINE_AA)

        return out
    except Exception:
        return img_rgb


def render(*, device: str, sam_proc, sam_model, sam_dtype) -> None:
    st.markdown("###  范式 C：VLM 缺陷检测 → SAM-3 精准分割")
    st.caption("💡 无需离线训练 | VLM 生成 BBox → SAM-3 Box Prompt 精分割")

    tab_single, tab_batch = st.tabs(["🔍 单张检测", "📦 批量处理"])

    with tab_single:
        # --- 单张图交互式 UI（既有逻辑保留） ---
        with st.sidebar:
            st.markdown("## ⚙️ 参数设置")

            # === VLM 配置 ===
            with st.expander(" VLM 配置", expanded=True):
                c_input_mode = st.radio(
                    "输入模式",
                    options=["单图", "对比(正常+检测)"],
                    index=0 if st.session_state.get("c_input_mode", "single") == "single" else 1,
                    key="c_input_mode_radio",
                )
                st.session_state["c_input_mode"] = "compare" if c_input_mode.startswith("对比") else "single"

                if st.session_state["c_input_mode"] == "compare":
                    st.caption("💡 Image A=正常参考，Image B=待检图片")

                # 从 registry 获取模型列表（根据能力过滤）
                need_two_images = st.session_state.get("c_input_mode") == "compare"
                model_options = list_models(require="bbox", two_images=True if need_two_images else None) or ["qwen-vl-max"]
                current_model = str(st.session_state.get("c_vlm_model", "") or "").strip() or model_options[0]
                if current_model not in model_options:
                    current_model = model_options[0]
                    st.session_state["c_vlm_model"] = current_model

                vlm_model = st.selectbox(
                    "VLM 模型",
                    model_options,
                    index=model_options.index(current_model),
                    key="c_vlm_model"
                )

                # 对于 stream-only 模型（如 QVQ）显示提示
                if is_stream_only_model(vlm_model):
                    st.info("⚡ QVQ 系列：具有强推理能力")
                    st.warning("🧠 QVQ 总是开启深度思考")

                max_boxes = st.slider(
                    "返回缺陷框数量",
                    1, 5,
                    int(st.session_state.get("c_max_boxes", 3)),
                    key="c_max_boxes",
                    help="VLM 返回的 Top-K 缺陷框"
                )

                # 控制 thinking 开关 - QVQ 禁用
                is_qvq = is_stream_only_model(vlm_model)
                thinking_enabled = st.toggle(
                    "开启 VLM 思考模式",
                    value=bool(st.session_state.get("c_vlm_thinking", False)),
                    key="c_vlm_thinking",
                    disabled=is_qvq,  # ✅ QVQ 模型禁用此开关
                    help="更准确但更慢，可能输出冗余信息"
                )

                if is_qvq:
                    st.caption("💡 QVQ 无需手动控制思考模式")
                else:
                    st.caption("💡 可通过此开关控制思考深度")

                key = get_dashscope_key()
                if dashscope_ready(key):
                    st.caption("✅ VLM 已就绪")
                else:
                    st.caption("⚠️ VLM 未配置")

            # === SAM-3 参数 ===
            with st.expander(" SAM-3 参数", expanded=True):
                # 🆕 SAM-3 推理开关（默认关闭以加快数据集制作）
                enable_sam3 = st.toggle(
                    "🔬 启用 SAM-3 精细分割",
                    value=bool(st.session_state.get("c_enable_sam3", False)),
                    key="c_enable_sam3",
                    help="开启后执行SAM-3分割，关闭则仅使用VLM bbox（推荐制作数据集时关闭）"
                )

                if enable_sam3:
                    st.success("✅ SAM-3 已启用：VLM bbox → SAM-3 精细分割")
                else:
                    st.info("⚡ SAM-3 已关闭：仅VLM bbox检测（快速模式，适合制作YOLO数据集）")

                st.markdown("---")

                # SAM-3 参数（仅在启用时显示）
                if enable_sam3:
                    sam_thr = st.slider(
                        "SAM 分割阈值",
                        0.0, 1.0,
                        float(st.session_state.get("c_sam_thr", 0.5)),
                        key="c_sam_thr",
                        help="控制 SAM-3 分割的敏感度"
                    )

                    mask_thr = st.slider(
                        "掩码阈值",
                        0.0, 1.0,
                        float(st.session_state.get("c_mask_thr", 0.5)),
                        key="c_mask_thr",
                        help="二值化掩码的阈值"
                    )

                    bbox_pad = st.slider(
                        "BBox 扩张比例",
                        0.0, 0.8,
                        float(st.session_state.get("c_bbox_pad", 0.20)),
                        0.05,
                        key="c_bbox_pad",
                        help="扩大 BBox 以提高 SAM 覆盖率"
                    )
                    st.caption("💡 仅用于 SAM box prompt，UI 显示原始框")
                else:
                    # 保持默认值但不显示滑块
                    sam_thr = float(st.session_state.get("c_sam_thr", 0.5))
                    mask_thr = float(st.session_state.get("c_mask_thr", 0.5))
                    bbox_pad = float(st.session_state.get("c_bbox_pad", 0.20))

            # === 🆕 缺陷类别配置 ===
            with st.expander("🏷️ 缺陷类别配置", expanded=False):
                st.markdown("**当前缺陷类别体系**")

                # 选择预设配置
                selected_preset = st.selectbox(
                    "选择预设配置",
                    options=list(get_available_presets().keys()),
                    format_func=lambda x: f"{x} - {get_available_presets()[x]}",
                    key="c_defect_preset_sidebar",
                    help="选择适合您行业的缺陷类别预设"
                )

                # 加载并显示配置信息
                try:
                    from core.defect_config import load_preset_config
                    config = load_preset_config(selected_preset)

                    # 显示类别列表
                    st.caption(f"**包含 {len(config.primary_types)} 个主类别：**")
                    for cat in config.primary_types:
                        st.caption(f"• {cat.id}: {cat.display_name}")

                    # 显示子类型
                    if config.subtypes:
                        st.caption(f"\n**{len(config.subtypes)} 个子类型：**")
                        for sub in config.subtypes:
                            st.caption(f"• {sub.id}: {sub.display_name}")

                    # 保存到session state供后续使用
                    st.session_state["c_defect_config"] = config

                except Exception as e:
                    st.error(f"❌ 加载配置失败：{str(e)}")

                st.markdown("---")
                st.info(
                    "💡 **说明**：\n"
                    "- 此配置仅影响YOLOv8导出时的类别映射\n"
                    "- VLM推理时会自动识别所有可能的缺陷\n"
                    "- 不在配置中的类别会映射为'other'"
                )

            # === 可视化与性能 ===
            with st.expander("🎨 可视化与性能", expanded=False):
                alpha = st.slider(
                    "掩码透明度",
                    0.0, 1.0,
                    float(st.session_state.get("c_alpha", 0.5)),
                    key="c_alpha",
                    help="控制掩码叠加透明度"
                )

                fast_mode = st.toggle(
                    "加速模式",
                    value=bool(st.session_state.get("c_fast_mode", False)),
                    key="c_fast_mode",
                    help="S1: 多框一次推理（更快）\nS2: 逐框推理（更稳定，默认）"
                )

                if fast_mode:
                    st.caption("⚡ S1 模式：多框一次推理")
                else:
                    st.caption("🐢 S2 模式：逐框推理（更可解释）")

            st.markdown("---")
            st.info(f"🖥️ 当前设备: **{device.upper()}**")

        left, right = st.columns([1.0, 1.0], gap="large")

        with left:
            st.markdown("### 📤 图片上传")
            ready_single = True
            if st.session_state.get("c_input_mode") == "compare":
                st.info("📝 **对比模式**: Image A = 正常参考 | Image B = 待检测图")

                up_a = st.file_uploader(
                    "上传正常样本图（Image A）",
                    type=["jpg", "png", "jpeg"],
                    key="c_upload_normal",
                    help="上传正常参考图片"
                )
                up_b = st.file_uploader(
                    "上传检测图片（Image B）",
                    type=["jpg", "png", "jpeg"],
                    key="c_upload_test",
                    help="上传待检测图片"
                )

                if not up_a or not up_b:
                    st.warning("⚠️ 对比模式需要同时上传两张图片")
                    ready_single = False
                else:
                    normal_pil = Image.open(up_a).convert("RGB")
                    test_pil = Image.open(up_b).convert("RGB")
                    raw = np.array(test_pil)

                    st.session_state["c_normal_uploaded_name"] = getattr(up_a, "name", None)
                    st.session_state["c_test_uploaded_name"] = getattr(up_b, "name", None)

                    st.success(f"✅ 已加载: {up_a.name} & {up_b.name}")

                    st.markdown("#### 📷 图片预览")
                    st.image(normal_pil, caption="🟢 Image A (正常参考)", use_container_width=True)
                    st.image(test_pil, caption="🔴 Image B (待检测)", use_container_width=True)

                    if normal_pil.size != test_pil.size:
                        st.warning("⚠️ 两张图分辨率不同，VLM 可能将尺度差异识别为异常。建议使用同尺寸图片。")
            else:
                up = st.file_uploader(
                    "上传检测图片",
                    type=["jpg", "png", "jpeg"],
                    key="c_upload",
                    help="支持 JPG、PNG、JPEG 格式"
                )
                if not up:
                    st.info("💡 请上传一张图片开始检测")
                    ready_single = False
                    normal_pil = None
                    test_pil = None
                    raw = None
                else:
                    test_pil = Image.open(up).convert("RGB")
                    raw = np.array(test_pil)

                    st.success(f"✅ 已加载: {up.name}")
                    st.markdown("#### 📷 图片预览")
                    st.image(test_pil, caption="🔴 待检测图片", use_container_width=True)
                    normal_pil = None

            if not ready_single:
                st.info("💡 单张模式等待输入，或切换到「📦 批量处理」标签使用批量功能")
            else:
                st.markdown("---")
                st.markdown("### ⚙️ 操作控制")

                # --- 按钮区 ---
                c1, c2 = st.columns(2)
                with c1:
                    gen_btn = st.button(
                        " 生成缺陷候选框",
                        type="primary",
                        use_container_width=True,
                        key="c_gen_vlm_btn"
                    )
                with c2:
                    st.button(
                        "🗑️ 清空结果",
                        use_container_width=True,
                        key="c_clear_results_btn",
                        on_click=lambda: st.session_state.update({
                            "c_vlm_out": None,
                            "c_selected_indices": [],
                            "c_sam_runs": None,
                            "c_final_json": None,
                        }),
                    )

                if gen_btn:
                    with st.spinner("🔄 VLM 推理中：正在生成缺陷候选框..."):
                        if st.session_state.get("c_input_mode") == "compare":
                            out = get_vlm_defect_bboxes_compare(
                                normal_image_pil=normal_pil,
                                test_image_pil=test_pil,
                                model_name=str(vlm_model),
                                api_key=key,
                                max_boxes=int(max_boxes),
                            )
                        else:
                            out = get_vlm_defect_bboxes(image_pil=test_pil, model_name=str(vlm_model), api_key=key, max_boxes=int(max_boxes))

                    st.session_state["c_vlm_out"] = out
                    st.session_state["c_selected_indices"] = list(range(min(1, len(out.detections))))
                    st.session_state["c_sam_runs"] = None
                    st.session_state["c_final_json"] = None

                vlm_out = st.session_state.get("c_vlm_out")
                if vlm_out is not None and vlm_out.raw_text:
                    with st.expander("🔍 VLM 原始输出（调试）", expanded=False):
                        st.text(vlm_out.raw_text)

        with right:
            st.markdown("###  检测结果")

            vlm_out = st.session_state.get("c_vlm_out")
            if vlm_out is None:
                st.info("💡 请先在左侧上传图片并点击「 生成缺陷候选框」，或切换到「📦 批量处理」标签")
            else:
                dets = list(vlm_out.detections or [])
                if not dets:
                    st.success("✅ VLM 未发现明显缺陷")
                    st.session_state["c_final_json"] = {
                        "mode": "paradigm_c",
                        "sam_strategy": "S2_per_box" if not fast_mode else "S1_multi_box",
                        "image": {"w": int(test_pil.size[0]), "h": int(test_pil.size[1])},
                        "vlm_detections": [],
                        "runs": [],
                        "final": {"decision": "ok", "num_valid_boxes": 0},
                    }
                else:
                    st.markdown("#### 📍 VLM 候选框预览")

                    # bbox 叠加预览
                    vis = raw.copy()
                    for idx, d in enumerate(dets):
                        subtype = getattr(d, "anomaly_subtype", "") or ""
                        subtxt = f" [{subtype}]" if subtype else ""
                        vis = _draw_bbox_on_image(
                            vis,
                            d.bbox_xyxy,
                            label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                            color=(255, 0, 0),
                            thickness=2,
                        )
                    st.image(vis, caption=f"🔴 检测到 {len(dets)} 个候选缺陷框", use_container_width=True)

                    # 🆕 根据 SAM-3 开关状态决定后续流程
                    enable_sam3 = bool(st.session_state.get("c_enable_sam3", False))

                    if enable_sam3:
                        # ========== SAM-3 启用：显示候选框选择和分割按钮 ==========
                        st.markdown("---")
                        st.markdown("#### ☑️ 选择需要分割的候选框")

                        options = list(range(len(dets)))
                        fmt = lambda i: (
                            f"#{i}  {dets[i].defect_type}"
                            f"{(' [' + (getattr(dets[i], 'anomaly_subtype', '') or '') + ']') if (getattr(dets[i], 'anomaly_subtype', '') or '') else ''}"
                            f"  置信度={dets[i].conf:.2f}"
                        )
                        selected = st.multiselect(
                            "勾选一个或多个候选框进行精分割",
                            options=options,
                            default=st.session_state.get("c_selected_indices") or [0],
                            format_func=fmt,
                        )
                        st.session_state["c_selected_indices"] = [int(i) for i in selected]

                        st.markdown("---")

                        run_btn = st.button(
                            "🎯 SAM-3 精准分割",
                            type="primary",
                            use_container_width=True,
                            disabled=not bool(selected) or not bool(st.session_state.get("models_ready")),
                            key="c_run_sam_btn"
                        )

                        if run_btn:
                            sel_dets = [dets[int(i)] for i in selected]

                            # 为 SAM prompt 扩充 bbox
                            W_img, H_img = int(test_pil.size[0]), int(test_pil.size[1])
                            boxes = [
                                pad_bbox_xyxy(d.bbox_xyxy, pad_ratio=float(bbox_pad), image_w=W_img, image_h=H_img)
                                if float(bbox_pad) > 0
                                else d.bbox_xyxy
                                for d in sel_dets
                            ]

                            if fast_mode:
                                # S1：多框一次推理
                                with st.spinner("⚡ SAM-3 推理中（S1 多框一次）..."):
                                    results, latency = run_sam3_box_prompt_instance_segmentation(
                                        image_pil=test_pil,
                                        sam_proc=sam_proc,
                                        sam_model=sam_model,
                                        sam_dtype=sam_dtype,
                                        boxes_xyxy=boxes,
                                        box_labels=[1] * len(boxes),
                                        threshold=float(sam_thr),
                                        mask_threshold=float(mask_thr),
                                        device=device,
                                    )

                                st.session_state["c_sam_runs"] = {
                                    "mode": "S1_multi_box",
                                    "latency_ms": float(latency),
                                    "results": results,
                                    "selected_boxes": boxes,
                                    "bbox_pad_ratio": float(bbox_pad),
                                    "selected_types": [d.defect_type for d in sel_dets],
                                    "selected_confs": [float(d.conf) for d in sel_dets],
                                }
                            else:
                                # S2：逐框推理
                                runs = []
                                total_latency = 0.0
                                with st.spinner("🔍 SAM-3 推理中（S2 逐框推理）..."):
                                    for bi, d in enumerate(sel_dets, start=1):
                                        padded_box = pad_bbox_xyxy(d.bbox_xyxy, pad_ratio=float(bbox_pad), image_w=W_img, image_h=H_img) if float(bbox_pad) > 0 else d.bbox_xyxy
                                        st.caption(f"📊 正在处理 {bi}/{len(sel_dets)}: {d.defect_type}")
                                        r, lat = run_sam3_box_prompt_instance_segmentation(
                                            image_pil=test_pil,
                                            sam_proc=sam_proc,
                                            sam_model=sam_model,
                                            sam_dtype=sam_dtype,
                                            boxes_xyxy=[padded_box],
                                            box_labels=[1],
                                            threshold=float(sam_thr),
                                            mask_threshold=float(mask_thr),
                                            device=device,
                                        )
                                        total_latency += float(lat)
                                        runs.append({"det": d, "results": r, "latency_ms": float(lat), "padded_box": padded_box})

                                st.session_state["c_sam_runs"] = {"mode": "S2_per_box", "latency_ms": float(total_latency), "runs": runs, "bbox_pad_ratio": float(bbox_pad)}

                        # --- 渲染分割结果 ---
                        st.markdown("---")
                        st.markdown("#### 🎨 分割结果")

                        sam_runs = st.session_state.get("c_sam_runs")
                        if not sam_runs:
                            st.info("💡 点击上方「 SAM-3 精准分割」按钮执行分割")
                        else:
                            H, W = int(test_pil.size[1]), int(test_pil.size[0])

                            out_json: dict[str, Any] = {
                                "mode": "paradigm_c",
                                "sam_strategy": str(sam_runs.get("mode")),
                                "image": {"w": int(W), "h": int(H)},
                                "vlm_input_mode": str(st.session_state.get("c_input_mode", "single")),
                                "vlm_detections": [
                                    {
                                        "type": d.defect_type,
                                        "anomaly_subtype": (getattr(d, "anomaly_subtype", "") or ""),
                                        "bbox_xyxy": d.bbox_xyxy,
                                        "conf": float(d.conf),
                                    }
                                    for d in dets
                                ],
                                "runs": [],
                                "final": {"decision": "ok", "num_valid_boxes": 0},
                                "bbox_pad_ratio": float(st.session_state.get("c_bbox_pad", 0.0)),
                            }

                            if sam_runs.get("mode") == "S2_per_box":
                                merged = np.zeros((H, W), dtype=bool)
                                valid = 0
                                for item in sam_runs.get("runs", []):
                                    det = item["det"]
                                    results = item["results"]

                                    masks = results.get("masks")
                                    scores = results.get("scores")

                                    if masks is None or len(masks) == 0:
                                        m_best = np.zeros((H, W), dtype=bool)
                                        best_score = 0.0
                                    else:
                                        m_np = masks.cpu().numpy() > 0.5
                                        s_np = scores.float().cpu().numpy() if scores is not None and len(scores) > 0 else np.zeros((m_np.shape[0],), dtype=float)
                                        idx = int(np.argmax(s_np)) if s_np.size > 0 else 0
                                        m_best = m_np[idx]
                                        best_score = float(s_np[idx]) if s_np.size > 0 else 0.0

                                    met = compute_c_metrics(
                                        mask_bool=m_best,
                                        image_h=H,
                                        image_w=W,
                                        vlm_bbox_xyxy=det.bbox_xyxy,
                                        sam_best_score=best_score,
                                        anomaly_subtype=(getattr(det, "anomaly_subtype", "") or ""),
                                    )

                                    if met.status == "ok":
                                        merged |= m_best
                                        valid += 1

                                    out_json["runs"].append({
                                        "bbox_xyxy": det.bbox_xyxy,
                                        "bbox_xyxy_padded": item.get("padded_box", det.bbox_xyxy),
                                        "anomaly_subtype": (getattr(det, "anomaly_subtype", "") or ""),
                                        "sam_best_score": float(best_score),
                                        "mask_area_ratio_img": float(met.mask_area_ratio_img),
                                        "mask_area_ratio_bbox": float(met.mask_area_ratio_bbox),
                                        "iou_maskbbox_vs_vlmbbox": float(met.iou_maskbbox_vs_vlmbbox),
                                        "frac_mask_inside_vlmbbox": float(met.frac_mask_inside_vlmbbox),
                                        "defect_score": float(met.defect_score),
                                        "status": met.status,
                                    })

                                out_json["final"]["num_valid_boxes"] = int(valid)

                                if valid > 0:
                                    out_json["final"]["decision"] = "defect"
                                    out_json["final"]["evidence"] = "sam_mask"

                                    mg = MaskGroup(label="defect", masks=merged[None, ...])
                                    vis_pil = overlay_masks_by_class(raw, [mg], alpha=float(alpha), cmap_name="tab10")
                                    st.image(np.array(vis_pil.convert("RGB")), caption=f"合并缺陷 mask（valid={valid}）", use_container_width=True)
                                else:
                                    # 当 SAM 无法对候选框生成可靠掩码时，不强制判为结构缺失，避免误判
                                    out_json["final"]["decision"] = "vlm_suspect"
                                    out_json["final"]["evidence"] = "vlm_bbox_only"
                                    out_json["final"]["reason"] = "sam_failed_or_low_quality"

                                    st.warning("SAM 未能对候选异常区域生成可靠 mask（status!=ok）。将仅以 VLM bbox 作为可视化证据，不输出缺陷掩码，避免误判。")

                                    # 仅使用 bbox 的可视化（橙色）
                                    vis2 = raw.copy()
                                    for idx, d in enumerate(dets):
                                        subtype = getattr(d, "anomaly_subtype", "") or ""
                                        subtxt = f" [{subtype}]" if subtype else ""
                                        vis2 = _draw_bbox_on_image(
                                            vis2,
                                            d.bbox_xyxy,
                                            label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                                            color=(255, 165, 0),
                                            thickness=3,
                                        )
                                    st.image(vis2, caption="VLM 证据：仅 bbox 高亮（SAM 未确认 mask）", use_container_width=True)

                            else:
                                # S1：合并并展示所有 raw masks（MVP 中不做严格的按框归属）
                                results = sam_runs.get("results") or {}
                                masks = results.get("masks")
                                scores = results.get("scores")

                                if masks is None or len(masks) == 0:
                                    out_json["final"]["decision"] = "vlm_suspect"
                                    out_json["final"]["evidence"] = "vlm_bbox_only"
                                    out_json["final"]["reason"] = "sam_no_mask"
                                    out_json["final"]["num_valid_boxes"] = 0

                                    st.warning("SAM 未输出 mask。将仅以 VLM bbox 作为可视化证据，不输出缺陷掩码，避免误判。")

                                    vis2 = raw.copy()
                                    for idx, d in enumerate(dets):
                                        subtype = getattr(d, "anomaly_subtype", "") or ""
                                        subtxt = f" [{subtype}]" if subtype else ""
                                        vis2 = _draw_bbox_on_image(
                                            vis2,
                                            d.bbox_xyxy,
                                            label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                                            color=(255, 165, 0),
                                            thickness=3,
                                        )
                                    st.image(vis2, caption="VLM 证据：仅 bbox 高亮（SAM 未确认 mask）", use_container_width=True)
                                else:
                                    m_np = masks.cpu().numpy() > 0.5
                                    merged = np.any(m_np, axis=0)
                                    mg = MaskGroup(label="defect", masks=merged[None, ...])
                                    vis_pil = overlay_masks_by_class(raw, [mg], alpha=float(alpha), cmap_name="tab10")
                                    st.image(np.array(vis_pil.convert("RGB")), caption=f"S1 合并 mask（num_masks={m_np.shape[0]}）", use_container_width=True)

                                    out_json["final"]["decision"] = "defect"
                                    out_json["final"]["evidence"] = "sam_mask"
                                    out_json["final"]["num_valid_boxes"] = int(len(selected))

                            st.session_state["c_final_json"] = out_json

                            with st.expander("最小 JSON 输出", expanded=True):
                                st.json(out_json)

                            # ===== 单张检测YOLOv8导出功能（SAM-3启用时） =====
                            st.markdown("---")
                            st.markdown("#### 📤 导出为YOLOv8格式")

                            with st.expander("🎯 一键导出YOLOv8标注", expanded=False):
                                st.info(
                                    "💡 **功能说明**：将当前VLM检测结果导出为YOLOv8训练格式，支持半自动化数据集标注\n\n"
                                    "📦 **导出内容**：\n"
                                    "- 标注文件（.txt，YOLO格式）\n"
                                    "- 图片文件（原图或可视化图）\n"
                                "- 配置文件（data.yaml）\n"
                                "- 类别映射文件（classes.txt）"
                            )

                            col_exp1, col_exp2 = st.columns(2)
                            with col_exp1:
                                export_img_type = st.radio(
                                    "导出图片类型",
                                    options=["原图", "带标注可视化图"],
                                    key="c_single_export_img_type",
                                    help="选择导出原图还是带标注的可视化图"
                                )
                            with col_exp2:
                                # 加载缺陷类别配置
                                preset_name = st.selectbox(
                                    "缺陷类别配置",
                                    options=list(get_available_presets().keys()),
                                    format_func=lambda x: f"{x} - {get_available_presets()[x]}",
                                    key="c_single_defect_preset",
                                    help="选择预设的缺陷类别配置"
                                )

                            if st.button("🚀 导出当前检测结果", type="primary", use_container_width=True, key="c_single_export_btn"):
                                if not out_json.get("vlm_detections"):
                                    st.warning("⚠️ 当前没有VLM检测结果，无法导出")
                                else:
                                    try:
                                        from core.yolov8_export import export_single_to_yolov8
                                        from core.defect_config import load_preset_config

                                        # 创建导出目录
                                        export_base = _ensure_dir(os.path.join(os.path.dirname(__file__), "..", "yolov8_datasets"))
                                        export_name = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                        export_dir = _ensure_dir(os.path.join(export_base, export_name))

                                        # 创建YOLOv8目录结构
                                        images_dir = _ensure_dir(os.path.join(export_dir, "images", "train"))
                                        labels_dir = _ensure_dir(os.path.join(export_dir, "labels", "train"))

                                        # 加载配置
                                        config = load_preset_config(preset_name)
                                        class_names = config.to_yolov8_classes()

                                        # 转换检测结果为VlmBBoxDetection格式
                                        from dataclasses import dataclass
                                        @dataclass
                                        class TempDetection:
                                            defect_type: str
                                            bbox_xyxy: list
                                            conf: float

                                        temp_dets = [
                                            TempDetection(
                                                defect_type=d["type"],
                                                bbox_xyxy=d["bbox_xyxy"],
                                                conf=d["conf"]
                                            )
                                            for d in out_json["vlm_detections"]
                                        ]

                                        # 保存图片
                                        img_filename = "image_001.jpg"
                                        img_path = os.path.join(images_dir, img_filename)
                                        if export_img_type == "原图":
                                            test_pil.save(img_path)
                                        else:
                                            # 保存可视化图
                                            if vis_pil:
                                                vis_pil.save(img_path)
                                            else:
                                                test_pil.save(img_path)

                                        # 导出标注
                                        txt_path = os.path.join(labels_dir, "image_001.txt")
                                        export_single_to_yolov8(
                                            detections=temp_dets,
                                            image_w=W,
                                            image_h=H,
                                            class_names=class_names,
                                            output_txt_path=txt_path
                                        )

                                        # 生成data.yaml
                                        data_yaml_path = os.path.join(export_dir, "data.yaml")
                                        yaml_content = {
                                            "path": os.path.abspath(export_dir),
                                            "train": "images/train",
                                            "val": "images/train",  # 单张图片，train和val相同
                                            "nc": len(class_names),
                                            "names": class_names
                                        }
                                        import yaml
                                        with open(data_yaml_path, 'w', encoding='utf-8') as f:
                                            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

                                        # 生成classes.txt
                                        classes_txt_path = os.path.join(export_dir, "classes.txt")
                                        with open(classes_txt_path, 'w', encoding='utf-8') as f:
                                            f.write('\n'.join(class_names))

                                        # 生成README
                                        readme_path = os.path.join(export_dir, "README.txt")
                                        with open(readme_path, 'w', encoding='utf-8') as f:
                                            f.write(f"YOLOv8数据集导出\n")
                                            f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                            f.write(f"缺陷类别配置: {preset_name}\n")
                                            f.write(f"类别数量: {len(class_names)}\n")
                                            f.write(f"检测框数量: {len(temp_dets)}\n")
                                            f.write(f"\n类别列表:\n")
                                            for i, name in enumerate(class_names):
                                                f.write(f"  {i}: {name}\n")

                                        st.success(f"✅ 导出成功！")
                                        st.info(f"📁 导出目录：`{export_dir}`")

                                        # 显示统计信息
                                        st.markdown("##### 📊 导出统计")
                                        stat_cols = st.columns(3)
                                        with stat_cols[0]:
                                            st.metric("图片数量", 1)
                                        with stat_cols[1]:
                                            st.metric("检测框数量", len(temp_dets))
                                        with stat_cols[2]:
                                            st.metric("类别数量", len(class_names))

                                        # 显示文件列表
                                        with st.expander("📄 导出文件列表", expanded=True):
                                            st.text(f"✓ {img_filename}")
                                            st.text(f"✓ labels/train/image_001.txt")
                                            st.text(f"✓ data.yaml")
                                            st.text(f"✓ classes.txt")
                                            st.text(f"✓ README.txt")

                                    except Exception as e:
                                        import traceback
                                        st.error(f"❌ 导出失败：{str(e)}")
                                        st.code(traceback.format_exc())

                    else:
                        # ========== SAM-3 关闭：直接使用VLM bbox结果 ==========
                        st.markdown("---")
                        st.success("⚡ **快速模式**：仅使用 VLM bbox 检测结果（SAM-3 已关闭）")

                        H, W = int(test_pil.size[1]), int(test_pil.size[0])

                        # 构建输出JSON（无SAM分割）
                        out_json: dict[str, Any] = {
                            "mode": "paradigm_c",
                            "sam_strategy": "disabled",
                            "image": {"w": int(W), "h": int(H)},
                            "vlm_input_mode": str(st.session_state.get("c_input_mode", "single")),
                            "vlm_detections": [
                                {
                                    "type": d.defect_type,
                                    "anomaly_subtype": (getattr(d, "anomaly_subtype", "") or ""),
                                    "bbox_xyxy": d.bbox_xyxy,
                                    "conf": float(d.conf),
                                }
                                for d in dets
                            ],
                            "runs": [],
                            "final": {
                                "decision": "defect" if len(dets) > 0 else "ok",
                                "evidence": "vlm_bbox_only",
                                "num_valid_boxes": len(dets),
                            },
                        }

                        st.session_state["c_final_json"] = out_json

                        with st.expander("📋 检测结果 JSON", expanded=False):
                            st.json(out_json)

                        # ===== 单张检测YOLOv8导出功能（SAM-3关闭时） =====
                        st.markdown("---")
                        st.markdown("#### 📤 导出为YOLOv8格式")

                        with st.expander("🎯 一键导出YOLOv8标注", expanded=True):
                            st.info(
                                "💡 **快速导出模式**：直接使用VLM检测的bbox导出为YOLO格式\n\n"
                                "📦 **导出内容**：\n"
                                "- 标注文件（.txt，YOLO格式）\n"
                                "- 图片文件（原图）\n"
                                "- 配置文件（data.yaml）\n"
                                "- 类别映射文件（classes.txt）"
                            )

                            col_exp1, col_exp2 = st.columns(2)
                            with col_exp1:
                                export_img_type = st.radio(
                                    "导出图片类型",
                                    options=["原图", "带标注可视化图"],
                                    key="c_single_export_img_type_nosam",
                                    help="选择导出原图还是带标注的可视化图"
                                )
                            with col_exp2:
                                # 加载缺陷类别配置
                                preset_name = st.selectbox(
                                    "缺陷类别配置",
                                    options=list(get_available_presets().keys()),
                                    format_func=lambda x: f"{x} - {get_available_presets()[x]}",
                                    key="c_single_defect_preset_nosam",
                                    help="选择预设的缺陷类别配置"
                                )

                            if st.button("🚀 导出当前检测结果", type="primary", use_container_width=True, key="c_single_export_btn_nosam"):
                                if not out_json.get("vlm_detections"):
                                    st.warning("⚠️ 当前没有VLM检测结果，无法导出")
                                else:
                                    try:
                                        from core.yolov8_export import export_single_to_yolov8
                                        from core.defect_config import load_preset_config

                                        # 创建导出目录
                                        export_base = _ensure_dir(os.path.join(os.path.dirname(__file__), "..", "yolov8_datasets"))
                                        export_name = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                        export_dir = _ensure_dir(os.path.join(export_base, export_name))

                                        # 创建YOLOv8目录结构
                                        images_dir = _ensure_dir(os.path.join(export_dir, "images", "train"))
                                        labels_dir = _ensure_dir(os.path.join(export_dir, "labels", "train"))

                                        # 加载配置
                                        config = load_preset_config(preset_name)
                                        class_names = config.to_yolov8_classes()

                                        # 转换检测结果为VlmBBoxDetection格式
                                        from dataclasses import dataclass
                                        @dataclass
                                        class TempDetection:
                                            defect_type: str
                                            bbox_xyxy: list
                                            conf: float

                                        temp_dets = [
                                            TempDetection(
                                                defect_type=d["type"],
                                                bbox_xyxy=d["bbox_xyxy"],
                                                conf=d["conf"]
                                            )
                                            for d in out_json["vlm_detections"]
                                        ]

                                        # 保存图片
                                        img_filename = "image_001.jpg"
                                        img_path = os.path.join(images_dir, img_filename)
                                        if export_img_type == "原图":
                                            test_pil.save(img_path)
                                        else:
                                            # 保存带bbox的可视化图
                                            Image.fromarray(vis).save(img_path)

                                        # 导出标注
                                        txt_path = os.path.join(labels_dir, "image_001.txt")
                                        export_single_to_yolov8(
                                            detections=temp_dets,
                                            image_w=W,
                                            image_h=H,
                                            class_names=class_names,
                                            output_txt_path=txt_path
                                        )

                                        # 生成data.yaml
                                        data_yaml_path = os.path.join(export_dir, "data.yaml")
                                        yaml_content = {
                                            "path": os.path.abspath(export_dir),
                                            "train": "images/train",
                                            "val": "images/train",
                                            "nc": len(class_names),
                                            "names": class_names
                                        }
                                        import yaml
                                        with open(data_yaml_path, 'w', encoding='utf-8') as f:
                                            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

                                        # 生成classes.txt
                                        classes_txt_path = os.path.join(export_dir, "classes.txt")
                                        with open(classes_txt_path, 'w', encoding='utf-8') as f:
                                            f.write('\n'.join(class_names))

                                        # 生成README
                                        readme_path = os.path.join(export_dir, "README.txt")
                                        with open(readme_path, 'w', encoding='utf-8') as f:
                                            f.write(f"YOLOv8数据集导出（快速模式，SAM-3关闭）\n")
                                            f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                            f.write(f"缺陷类别配置: {preset_name}\n")
                                            f.write(f"类别数量: {len(class_names)}\n")
                                            f.write(f"检测框数量: {len(temp_dets)}\n")
                                            f.write(f"\n类别列表:\n")
                                            for i, name in enumerate(class_names):
                                                f.write(f"  {i}: {name}\n")

                                        st.success(f"✅ 导出成功！")
                                        st.info(f"📁 导出目录：`{export_dir}`")

                                        # 显示统计信息
                                        st.markdown("##### 📊 导出统计")
                                        stat_cols = st.columns(3)
                                        with stat_cols[0]:
                                            st.metric("图片数量", 1)
                                        with stat_cols[1]:
                                            st.metric("检测框数量", len(temp_dets))
                                        with stat_cols[2]:
                                            st.metric("类别数量", len(class_names))

                                    except Exception as e:
                                        import traceback
                                        st.error(f"❌ 导出失败：{str(e)}")
                                        st.code(traceback.format_exc())

    with tab_batch:
        st.markdown("## 📦 批量处理")
        st.info(
            "🚀 **批量处理流程**: 选择模式 → 上传图片 → 自动检测所有图片 → 导出结果（JSON + CSV + 可视化图）\n\n"
            "💡 **提示**: 支持单图批量和对比批量两种模式"
        )

        st.markdown("---")
        st.markdown("### 1️⃣ 选择批量模式")

        mode = st.radio(
            "批量模式",
            options=["单图批量", "对比批量(一张Normal + 多张Test)"],
            index=0,
            horizontal=True
        )

        # re-use sidebar params
        vlm_model = str(st.session_state.get("c_vlm_model", "qwen-vl-max"))
        max_boxes = int(st.session_state.get("c_max_boxes", 3))
        sam_thr = float(st.session_state.get("c_sam_thr", 0.5))
        mask_thr = float(st.session_state.get("c_mask_thr", 0.5))
        alpha = float(st.session_state.get("c_alpha", 0.5))
        bbox_pad = float(st.session_state.get("c_bbox_pad", 0.2))
        fast_mode = bool(st.session_state.get("c_fast_mode", False))
        api_key = get_dashscope_key()

        if not dashscope_ready(api_key):
            st.error("❌ 未检测到 DASHSCOPE_API_KEY，批量模式需要 VLM 可用")
        else:
            st.markdown("---")
            st.markdown("### 2️⃣ 批量上传图片")

            # Upload files based on mode
            files_ready = False
            if mode.startswith("对比"):
                st.info("📝 **对比模式**: 上传 1 张正常参考图 + 多张待检测图")

                normal_up = st.file_uploader(
                    "上传正常参考图（单张）",
                    type=["jpg", "png", "jpeg"],
                    key="c_batch_normal",
                    help="上传一张正常样本作为参考"
                )
                tests = st.file_uploader(
                    "上传待检测图片（多张）",
                    type=["jpg", "png", "jpeg"],
                    accept_multiple_files=True,
                    key="c_batch_tests",
                    help="上传多张待检测图片"
                )

                if not normal_up or not tests:
                    st.warning("⚠️ 请上传 1 张正常图 + 多张测试图")
                else:
                    normal_pil = Image.open(normal_up).convert("RGB")
                    test_items = [(t, Image.open(t).convert("RGB")) for t in tests]
                    batch_mode = "compare"
                    files_ready = True
                    st.success(f"✅ 已选择: 1 张参考图 + **{len(tests)}** 张测试图")
            else:
                tests = st.file_uploader(
                    "上传批量图片（多张）",
                    type=["jpg", "png", "jpeg"],
                    accept_multiple_files=True,
                    key="c_batch_single",
                    help="按住 Ctrl/Cmd 可多选文件"
                )
                if not tests:
                    st.warning("⚠️ 请上传多张图片")
                else:
                    normal_pil = None
                    test_items = [(t, Image.open(t).convert("RGB")) for t in tests]
                    batch_mode = "single"
                    files_ready = True
                    st.success(f"✅ 已选择 **{len(tests)}** 张图片")

            # Only show batch run controls if files are ready
            if files_ready:
                st.markdown("---")
                st.markdown("### 3️⃣ 开始批量处理")

                # 🆕 并发模式设置
                enable_sam3 = bool(st.session_state.get("c_enable_sam3", False))

                with st.expander("⚡ 性能设置", expanded=not enable_sam3):
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        use_concurrent = st.toggle(
                            "🚀 启用并发推理",
                            value=not enable_sam3,  # SAM-3关闭时默认启用并发
                            key="c_batch_concurrent",
                            help="多线程并发调用VLM API，显著提升批量处理速度（仅VLM模式下推荐）",
                            disabled=enable_sam3,  # SAM-3启用时禁用并发（因为SAM不支持并发）
                        )
                    with perf_col2:
                        max_workers = st.slider(
                            "并发线程数",
                            min_value=2,
                            max_value=8,
                            value=4,
                            step=1,
                            key="c_batch_max_workers",
                            help="同时处理的图片数量，建议4-6线程",
                            disabled=not use_concurrent or enable_sam3,
                        )

                    if enable_sam3:
                        st.info("ℹ️ SAM-3 启用时不支持并发模式（GPU资源限制）")
                    elif use_concurrent:
                        st.success(f"⚡ 并发模式已启用：{max_workers} 线程同时处理")
                    else:
                        st.caption("🐢 串行模式：逐张处理")

                # batch run controls
                colb1, colb2, colb3 = st.columns([1, 1, 1])
                with colb1:
                    start_btn = st.button(
                        "🚀 开始批量运行",
                        type="primary",
                        use_container_width=True,
                        key="c_batch_start_btn"
                    )
                with colb2:
                    st.toggle(
                        "遇到错误继续",
                        value=True,
                        key="c_batch_continue_on_error",
                        help="开启后，单张图片出错不会中断整个批量任务"
                    )
                with colb3:
                    st.number_input(
                        "最多处理张数",
                        min_value=1,
                        max_value=len(test_items),
                        value=min(50, len(test_items)),
                        step=1,
                        key="c_batch_max_n",
                        help="限制本次批量处理的最大图片数量"
                    )

                if start_btn:
                    # 🆕 重新读取设置
                    enable_sam3 = bool(st.session_state.get("c_enable_sam3", False))
                    use_concurrent = bool(st.session_state.get("c_batch_concurrent", False)) and not enable_sam3

                    # Check if models are ready before processing (仅当SAM-3启用时)
                    if enable_sam3 and not st.session_state.get("models_ready", False):
                        st.error("❌ 模型未加载！请先在侧边栏点击「初始化模型」按钮")
                        st.warning("💡 提示：初始化模型后，需要等待加载完成（可能需要几十秒），然后才能使用批量功能")
                        st.stop()

                    # Verify SAM model objects are not None (仅当SAM-3启用时)
                    if enable_sam3 and (sam_proc is None or sam_model is None):
                        st.error("❌ SAM 模型未正确加载！请重新初始化模型")
                        st.stop()

                    # Create output directories ONLY when button is clicked
                    out_base = _ensure_dir(os.path.join(os.path.dirname(__file__), "..", "experiments"))
                    run_name = f"paradigm_c_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    out_dir = _ensure_dir(os.path.abspath(os.path.join(out_base, run_name)))
                    out_img_dir = _ensure_dir(os.path.join(out_dir, "images"))
                    out_json_dir = _ensure_dir(os.path.join(out_dir, "json"))

                    st.markdown("---")
                    st.markdown("### ⏳ 处理进度")
                    # 🆕 显示当前模式
                    if enable_sam3:
                        st.info(f"📁 结果将保存至: `{out_dir}`\n\n🔬 模式: VLM → SAM-3 精细分割（串行处理）")
                    elif use_concurrent:
                        max_workers = int(st.session_state.get("c_batch_max_workers", 4))
                        st.success(f"📁 结果将保存至: `{out_dir}`\n\n⚡ 并发模式: {max_workers} 线程同时推理（SAM-3关闭）")
                    else:
                        st.success(f"📁 结果将保存至: `{out_dir}`\n\n⚡ 快速模式: 仅VLM bbox检测（SAM-3关闭）")

                    max_n = int(st.session_state.get("c_batch_max_n", len(test_items)))
                    cont = bool(st.session_state.get("c_batch_continue_on_error", True))

                    prog = st.progress(0)
                    status = st.empty()

                    batch_results: list[dict[str, Any]] = []
                    csv_rows: list[dict[str, Any]] = []

                    # 🆕 根据模式选择处理方式
                    if use_concurrent and not enable_sam3:
                        # ========== 并发模式：多线程VLM推理 ==========
                        import time
                        start_time = time.time()
                        max_workers = int(st.session_state.get("c_batch_max_workers", 4))

                        status.text(f"⚡ 并发推理中... (0/{max_n})")

                        # 定义进度回调
                        progress_placeholder = st.empty()
                        def update_progress(completed, total):
                            prog.progress(int(100 * completed / max(1, total)))
                            status.text(f"⚡ 并发推理中... ({completed}/{total})")

                        # 执行并发推理
                        vlm_results = run_paradigm_c_batch_concurrent(
                            test_items=test_items[:max_n],
                            mode=batch_mode,
                            normal_pil=normal_pil,
                            vlm_model=vlm_model,
                            api_key=api_key,
                            max_boxes=max_boxes,
                            max_workers=max_workers,
                            progress_callback=update_progress,
                        )

                        elapsed = time.time() - start_time
                        status.text(f"✅ VLM推理完成！用时 {elapsed:.1f}s，正在生成可视化...")

                        # 后处理：生成可视化和保存文件
                        for i, result in enumerate(vlm_results):
                            name = result.get("name", f"img_{i}.jpg")
                            stem = _safe_stem(os.path.splitext(name)[0])

                            # 生成可视化图像
                            raw = result.get("_raw_image")
                            dets = result.get("_dets", [])

                            if raw is not None:
                                vis = raw.copy()
                                for idx, d in enumerate(dets):
                                    subtype = getattr(d, "anomaly_subtype", "") or ""
                                    subtxt = f" [{subtype}]" if subtype else ""
                                    vis = _draw_bbox_on_image(
                                        vis,
                                        d.bbox_xyxy,
                                        label=f"#{idx}  {d.defect_type}{subtxt}  conf={float(d.conf):.2f}",
                                        color=(255, 0, 0),
                                        thickness=2,
                                    )
                            else:
                                vis = np.zeros((100, 100, 3), dtype=np.uint8)

                            # 清理内部字段
                            out_json = {k: v for k, v in result.items() if not k.startswith("_")}
                            out_json["output"] = {
                                "json": f"json/{stem}.json",
                                "image": f"images/{stem}.jpg",
                            }

                            # 保存文件
                            json_path = os.path.join(out_json_dir, f"{stem}.json")
                            _write_json(json_path, out_json)

                            img_path = os.path.join(out_img_dir, f"{stem}.jpg")
                            _save_vis_image(img_path, vis)

                            batch_results.append(out_json)
                            csv_rows.append(_flatten_for_csv(out_json))

                        status.empty()
                        st.success(f"⚡ 并发处理完成！总用时 {elapsed:.1f}s，平均 {elapsed/max_n:.2f}s/张")

                    else:
                        # ========== 串行模式：原有逻辑 ==========
                        for i, (up, pil) in enumerate(test_items[:max_n], start=1):
                            name = getattr(up, "name", f"img_{i}.jpg")
                            stem = _safe_stem(os.path.splitext(name)[0])
                            status.text(f"[{i}/{max_n}] 正在处理: {name}")
                            prog.progress(int(100 * i / max(1, max_n)))

                            try:
                                out_json, vis = run_paradigm_c_once(
                                    device=device,
                                    sam_proc=sam_proc,
                                    sam_model=sam_model,
                                    sam_dtype=sam_dtype,
                                    mode=batch_mode,
                                    test_pil=pil,
                                    normal_pil=normal_pil,
                                    vlm_model=vlm_model,
                                    api_key=api_key,
                                    max_boxes=max_boxes,
                                    sam_thr=sam_thr,
                                    mask_thr=mask_thr,
                                    bbox_pad=bbox_pad,
                                    alpha=alpha,
                                    fast_mode=fast_mode,
                                    enable_sam3=enable_sam3,
                                )

                                out_json["name"] = name
                                out_json["output"] = {
                                    "json": f"json/{stem}.json",
                                    "image": f"images/{stem}.jpg",
                                }

                                # Write JSON file
                                json_path = os.path.join(out_json_dir, f"{stem}.json")
                                _write_json(json_path, out_json)

                                # Write visualization image
                                img_path = os.path.join(out_img_dir, f"{stem}.jpg")
                                _save_vis_image(img_path, vis)

                                batch_results.append(out_json)
                                csv_rows.append(_flatten_for_csv(out_json))

                            except Exception as e:
                                import traceback
                                err_msg = f"{str(e)}\n{traceback.format_exc()}"
                                st.error(f"处理 {name} 时出错：{str(e)}")
                                err = {"name": name, "error": str(e), "final": {"decision": "error"}}
                                batch_results.append(err)
                                csv_rows.append({"name": name, "decision": "error", "reason": str(e)})
                                if not cont:
                                    st.error(f"批量处理中断。错误详情：\n```\n{err_msg}\n```")
                                    raise

                    # write summaries
                    batch_json_path = os.path.join(out_dir, "batch_results.json")
                    batch_csv_path = os.path.join(out_dir, "batch_summary.csv")
                    _write_json(batch_json_path, batch_results)
                    _write_csv(batch_csv_path, csv_rows)

                    # ✅ 保存到session_state，供后续导出按钮使用
                    st.session_state["c_batch_results"] = batch_results
                    st.session_state["c_batch_out_dir"] = out_dir
                    st.session_state["c_batch_test_items"] = test_items

                    status.empty()
                    prog.empty()

                    # === 增强的结果展示 ===
                    st.success(f"✅ 批量处理完成！共处理 {len(batch_results)} 张图片")
                    st.info(f"📁 结果目录：`{out_dir}`")

                    # 统计信息
                    st.markdown("### 📊 处理统计")
                    stats_cols = st.columns(4)
                    num_ok = sum(1 for r in batch_results if r.get("final", {}).get("decision") == "ok")
                    num_defect = sum(1 for r in batch_results if r.get("final", {}).get("decision") == "defect")
                    num_suspect = sum(1 for r in batch_results if r.get("final", {}).get("decision") == "vlm_suspect")
                    num_error = sum(1 for r in batch_results if r.get("final", {}).get("decision") == "error")

                    with stats_cols[0]:
                        st.metric("✅ 正常", num_ok)
                    with stats_cols[1]:
                        st.metric("🔴 缺陷", num_defect)
                    with stats_cols[2]:
                        st.metric("🟡 可疑", num_suspect)
                    with stats_cols[3]:
                        st.metric("❌ 错误", num_error)

                    # CSV下载
                    st.markdown("### 📥 下载结果")
                    dl_cols = st.columns(2)
                    with dl_cols[0]:
                        st.download_button(
                            "⬇️ 下载 batch_summary.csv",
                            data=open(batch_csv_path, "rb").read(),
                            file_name="batch_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with dl_cols[1]:
                        st.download_button(
                            "⬇️ 下载 batch_results.json",
                            data=open(batch_json_path, "rb").read(),
                            file_name="batch_results.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                    # CSV表格展示
                    st.markdown("### 📋 CSV 结果表格")
                    st.dataframe(csv_rows, use_container_width=True)

                    # 可视化结果展示
                    st.markdown("### 🖼️ 可视化结果（缩略图）")

                    # 过滤显示选项
                    filter_opt = st.radio(
                        "显示筛选",
                        options=["全部", "仅缺陷", "仅正常", "仅可疑", "仅错误"],
                        horizontal=True,
                        key="c_batch_filter"
                    )

                    # 根据筛选条件过滤结果
                    filtered_results = batch_results
                    if filter_opt == "仅缺陷":
                        filtered_results = [r for r in batch_results if r.get("final", {}).get("decision") == "defect"]
                    elif filter_opt == "仅正常":
                        filtered_results = [r for r in batch_results if r.get("final", {}).get("decision") == "ok"]
                    elif filter_opt == "仅可疑":
                        filtered_results = [r for r in batch_results if r.get("final", {}).get("decision") == "vlm_suspect"]
                    elif filter_opt == "仅错误":
                        filtered_results = [r for r in batch_results if r.get("final", {}).get("decision") == "error"]

                    if not filtered_results:
                        st.info(f"没有符合筛选条件「{filter_opt}」的结果")
                    else:
                        # 使用列布局展示缩略图
                        num_cols = 3
                        for idx in range(0, len(filtered_results), num_cols):
                            cols = st.columns(num_cols)
                            for col_idx, col in enumerate(cols):
                                result_idx = idx + col_idx
                                if result_idx >= len(filtered_results):
                                    break

                                result = filtered_results[result_idx]
                                with col:
                                    name = result.get("name", "unknown")
                                    decision = result.get("final", {}).get("decision", "unknown")
                                    evidence = result.get("final", {}).get("evidence", "")
                                    num_vlm = len(result.get("vlm_detections", []))

                                    # 决策标签颜色
                                    if decision == "ok":
                                        label = "✅ 正常"
                                        color = "green"
                                    elif decision == "defect":
                                        label = "🔴 缺陷"
                                        color = "red"
                                    elif decision == "vlm_suspect":
                                        label = "🟡 可疑"
                                        color = "orange"
                                    else:
                                        label = "❌ 错误"
                                        color = "gray"

                                    # 显示图片（如果存在）
                                    img_rel_path = result.get("output", {}).get("image", "")
                                    if img_rel_path:
                                        img_full_path = os.path.join(out_dir, img_rel_path.replace("/", os.sep))
                                        if os.path.exists(img_full_path):
                                            st.image(img_full_path, caption=name, use_container_width=True)
                                        else:
                                            st.warning(f"图片不存在：{img_rel_path}")

                                    # 显示信息
                                    st.markdown(f"**{label}**")
                                    st.caption(f"📄 {name}")
                                    if decision != "error":
                                        st.caption(f"🔍 VLM检测框: {num_vlm}")
                                        st.caption(f"📌 证据: {evidence}")
                                    else:
                                        err_msg = result.get("error", "未知错误")
                                        st.caption(f"❌ {err_msg[:50]}...")

                    st.markdown("---")
                    st.caption(f"💾 完整结果已保存至：`{out_dir}`")

            # ===== 🆕 批量处理YOLOv8数据集导出功能 (移到if start_btn外部) =====
            # 从session_state读取数据，确保在按钮点击重新渲染后仍可访问
            if st.session_state.get("c_batch_results") and st.session_state.get("c_batch_out_dir"):
                batch_results = st.session_state["c_batch_results"]
                out_dir = st.session_state["c_batch_out_dir"]
                test_items = st.session_state.get("c_batch_test_items", [])

                st.markdown("---")
                st.markdown("### 📤 导出为YOLOv8数据集")

                with st.expander("🎯 一键导出YOLOv8训练数据集", expanded=True):
                    st.info(
                        "💡 **功能说明**：将批量检测结果导出为YOLOv8训练格式，实现半自动化/全自动化数据集标注\n\n"
                        "📦 **导出内容**：\n"
                        "- images/train/ 和 images/val/（按比例分割）\n"
                        "- labels/train/ 和 labels/val/（YOLO格式标注）\n"
                        "- data.yaml（训练配置文件）\n"
                        "- classes.txt（类别映射）\n"
                        "- README.txt（数据集说明）\n\n"
                        "🎓 **使用建议**：\n"
                        "- 仅导出有检测结果的图片（decision=defect或vlm_suspect）\n"
                        "- 可设置train/val分割比例\n"
                        "- 支持自定义缺陷类别配置"
                    )

                    col_batch_exp1, col_batch_exp2, col_batch_exp3 = st.columns(3)

                    with col_batch_exp1:
                        batch_export_filter = st.selectbox(
                            "导出筛选条件",
                            options=["全部图片", "仅缺陷(defect)", "缺陷+可疑(defect+suspect)", "仅有检测框的图片"],
                            index=2,
                            key="c_batch_export_filter",
                            help="选择要导出的图片范围"
                        )

                    with col_batch_exp2:
                        batch_split_ratio = st.slider(
                            "训练集比例",
                            min_value=0.5,
                            max_value=0.95,
                            value=0.8,
                            step=0.05,
                            key="c_batch_split_ratio",
                            help="训练集占总数据的比例"
                        )

                    with col_batch_exp3:
                        batch_defect_preset = st.selectbox(
                            "缺陷类别配置",
                            options=list(get_available_presets().keys()),
                            format_func=lambda x: f"{x} - {get_available_presets()[x]}",
                            key="c_batch_defect_preset",
                            help="选择预设的缺陷类别配置"
                        )

                    batch_copy_images = st.checkbox(
                        "复制图片到导出目录",
                        value=True,
                        key="c_batch_copy_images",
                        help="如果取消勾选，仅导出标注文件"
                    )

                    if st.button("🚀 导出为YOLOv8数据集", type="primary", use_container_width=True, key="c_batch_export_btn"):
                        try:
                            from core.yolov8_export import export_batch_to_yolov8
                            from core.defect_config import load_preset_config

                            # 筛选要导出的结果
                            filtered_export = []
                            filtered_image_files = []

                            for result in batch_results:
                                decision = result.get("final", {}).get("decision", "")
                                vlm_dets = result.get("vlm_detections", [])

                                # 根据筛选条件判断
                                should_export = False
                                if batch_export_filter == "全部图片":
                                    should_export = True
                                elif batch_export_filter == "仅缺陷(defect)":
                                    should_export = (decision == "defect")
                                elif batch_export_filter == "缺陷+可疑(defect+suspect)":
                                    should_export = (decision in ["defect", "vlm_suspect"])
                                elif batch_export_filter == "仅有检测框的图片":
                                    should_export = (len(vlm_dets) > 0)

                                if should_export:
                                    filtered_export.append(result)
                                    # 获取对应的原始图片文件
                                    img_name = result.get("name", "")
                                    # 尝试从test_items中找到对应的文件
                                    for uploaded_file, _ in test_items:
                                        if uploaded_file.name == img_name:
                                            filtered_image_files.append(uploaded_file.name)
                                            break

                            if not filtered_export:
                                st.warning(f"⚠️ 根据筛选条件「{batch_export_filter}」，没有可导出的图片")
                            else:
                                # 创建YOLOv8导出目录
                                yolo_export_dir = _ensure_dir(os.path.join(out_dir, "yolov8_dataset"))

                                # 加载配置
                                config = load_preset_config(batch_defect_preset)
                                class_names = config.to_yolov8_classes()

                                # 如果需要复制图片，先保存上传的图片到临时目录
                                temp_img_paths = []
                                if batch_copy_images:
                                    temp_dir = _ensure_dir(os.path.join(out_dir, "_temp_images"))
                                    for uploaded_file, pil_img in test_items:
                                        for result in filtered_export:
                                            if result.get("name") == uploaded_file.name:
                                                temp_path = os.path.join(temp_dir, uploaded_file.name)
                                                pil_img.save(temp_path)
                                                temp_img_paths.append(temp_path)
                                                break

                                # 执行导出
                                with st.spinner(f"⏳ 正在导出 {len(filtered_export)} 张图片..."):
                                    export_stats = export_batch_to_yolov8(
                                        results=filtered_export,
                                        image_files=temp_img_paths if batch_copy_images else filtered_image_files,
                                        class_names=class_names,
                                        output_dir=yolo_export_dir,
                                        split_ratio=batch_split_ratio,
                                        copy_images=batch_copy_images,
                                    )

                                # 生成README
                                readme_path = os.path.join(yolo_export_dir, "README.txt")
                                with open(readme_path, 'w', encoding='utf-8') as f:
                                    f.write(f"YOLOv8数据集批量导出\n")
                                    f.write(f"=" * 50 + "\n\n")
                                    f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                    f.write(f"缺陷类别配置: {batch_defect_preset}\n")
                                    f.write(f"筛选条件: {batch_export_filter}\n")
                                    f.write(f"训练集比例: {batch_split_ratio:.0%}\n\n")
                                    f.write(f"统计信息:\n")
                                    f.write(f"  总图片数: {export_stats['total']}\n")
                                    f.write(f"  训练集: {export_stats['train']}\n")
                                    f.write(f"  验证集: {export_stats['val']}\n")
                                    f.write(f"  跳过(无检测框): {export_stats['skipped']}\n\n")
                                    f.write(f"缺陷类别统计:\n")
                                    for cls_name, count in export_stats.get('defects_count', {}).items():
                                        if count > 0:
                                            f.write(f"  {cls_name}: {count}\n")
                                    f.write(f"\n使用方法:\n")
                                    f.write(f"  1. 确认data.yaml中的路径正确\n")
                                    f.write(f"  2. 使用YOLOv8训练: yolo detect train data=data.yaml model=yolov8n.pt epochs=100\n")
                                    f.write(f"  3. 验证模型: yolo detect val data=data.yaml model=runs/detect/train/weights/best.pt\n")

                                # 显示成功信息
                                st.success(f"✅ YOLOv8数据集导出成功！")
                                st.info(f"📁 导出目录：`{yolo_export_dir}`")

                                # 显示详细统计
                                st.markdown("##### 📊 导出统计")
                                stat_cols = st.columns(5)
                                with stat_cols[0]:
                                    st.metric("总图片数", export_stats['total'])
                                with stat_cols[1]:
                                    st.metric("训练集", export_stats['train'])
                                with stat_cols[2]:
                                    st.metric("验证集", export_stats['val'])
                                with stat_cols[3]:
                                    st.metric("跳过", export_stats['skipped'])
                                with stat_cols[4]:
                                    st.metric("类别数", len(class_names))

                                # 显示缺陷类别分布
                                if export_stats.get('defects_count'):
                                    st.markdown("##### 🏷️ 缺陷类别分布")
                                    defect_dist = export_stats['defects_count']
                                    # 只显示非零的类别
                                    non_zero_defects = {k: v for k, v in defect_dist.items() if v > 0}
                                    if non_zero_defects:
                                        dist_cols = st.columns(min(5, len(non_zero_defects)))
                                        for idx, (cls_name, count) in enumerate(non_zero_defects.items()):
                                            with dist_cols[idx % len(dist_cols)]:
                                                st.metric(cls_name, count)

                                # 显示目录结构
                                with st.expander("📂 数据集目录结构", expanded=True):
                                    st.code(f"""
yolov8_dataset/
├── images/
│   ├── train/          ({export_stats['train']} 张)
│   └── val/            ({export_stats['val']} 张)
├── labels/
│   ├── train/          ({export_stats['train']} 个.txt)
│   └── val/            ({export_stats['val']} 个.txt)
├── data.yaml           (训练配置)
├── classes.txt         (类别列表)
└── README.txt          (说明文档)
""", language="")

                                # 训练命令提示
                                st.markdown("##### 🎓 快速开始训练")
                                train_cmd = f"yolo detect train data={os.path.join(yolo_export_dir, 'data.yaml')} model=yolov8n.pt epochs=100 imgsz=640"
                                st.code(train_cmd, language="bash")

                        except Exception as e:
                            import traceback
                            st.error(f"❌ 导出失败：{str(e)}")
                            st.code(traceback.format_exc())


