from __future__ import annotations

"""VLM bbox 提取模块，用于 Paradigm C。

该模块通过 VLM（Qwen-VL via DashScope）请求输出缺陷的边界框
符合严格的 JSON 模式。它包括稳健的解析和边界框的标准化。

我们故意将其与 core.vlm（关键字建议）分开，以便
现有的 Paradigm A 流程保持不变。

更新以支持 QVQ 系列（仅流式模型）通过 DashScopeStreamAggregator。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from PIL import Image

# 为了兼容静态分析器和在没有 core 包时的本地运行，使用 TYPE_CHECKING + try/except 回退。
if TYPE_CHECKING:
    from core.vlm_model_registry import fallback_model_for_bbox, is_stream_only_model
    from core.dashscope_stream import DashScopeStreamAggregator
else:
    try:
        from core.vlm_model_registry import fallback_model_for_bbox, is_stream_only_model
    except Exception:
        # 回退占位符：在没有 registry 时提供安全的默认行为（不会触发运行时错误）
        def fallback_model_for_bbox(primary: str):
            return None

        def is_stream_only_model(name: str) -> bool:
            return False

    try:
        from core.dashscope_stream import DashScopeStreamAggregator
    except Exception:
        # 回退的轻量占位类，仅在本地没有 dashscope 实现时避免崩溃
        class DashScopeStreamAggregator:
            def call_and_aggregate(self, *args, **kwargs):
                return None, ""


@dataclass
class VlmBBoxDetection:
    defect_type: str
    bbox_xyxy: list[int]
    conf: float
    anomaly_subtype: str = ""  # 例如 'missing_like' | 'surface_like' | 'other'


@dataclass
class VlmBBoxOutput:
    image_w: int
    image_h: int
    detections: list[VlmBBoxDetection]
    raw_text: str = ""


_DEFECT_TYPES = [
    "scratch",
    "crack",
    "stain",
    "dent",
    "burr",
    "chip",
    "discoloration",
    "contamination",
    "corrosion",
    "other",
]


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _sanitize_bbox_xyxy(b: Any, *, w: int, h: int) -> Optional[list[int]]:
    """将 bbox 规范化为在像素坐标系 [0, w) x [0, h) 内的有效 xyxy 格式。

    输入可以是数值、字符串或浮点；函数会尝试转换为整数并做排序、截断和最小面积保障。
    返回值为长度为 4 的列表 [xmin, ymin, xmax, ymax]（整数），或在解析失败时返回 None。
    """
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(x))) for x in b]
    except Exception:
        return None

    # 排序以保证坐标顺序正确
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # 截断到图片边界
    x1 = _clamp(x1, 0, max(0, w - 1))
    y1 = _clamp(y1, 0, max(0, h - 1))
    x2 = _clamp(x2, 0, w)
    y2 = _clamp(y2, 0, h)

    # 确保正面积（宽和高至少为 1 像素）
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return [int(x1), int(y1), int(x2), int(y2)]


def _extract_first_json_object(text: str) -> Optional[str]:
    """从自由文本中提取第一个 JSON 对象子串（用于从 LLM/ VLM 输出中拾取 JSON）。

    该函数会移除 Markdown 代码块围栏以提高鲁棒性，然后用简单的花括号配对
    来定位第一个完整的 JSON 对象并返回该子串；若找不到则返回 None。
    """
    if not text:
        return None

    # 移除 markdown 代码块围栏以提高解析稳定性
    text2 = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text2 = text2.replace("```", "")

    # 简单的花括号配对搜索第一个 JSON 对象
    start = text2.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text2)):
        ch = text2[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text2[start : i + 1]
    return None


def _normalize_anomaly_subtype(v: Any) -> str:
    if not v:
        return ""
    s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
    allowed = {"missing_like", "surface_like", "structural_like", "visual_like", "other"}
    return s if s in allowed else "other"


def parse_vlm_bbox_output(text: str, *, image_w: int, image_h: int, max_boxes: int = 3) -> VlmBBoxOutput:
    """将 VLM 返回的 JSON 文本解析为结构化的 bbox 列表。

    虽然我们在 prompt 中要求 VLM 仅输出 JSON，但仍需做稳健的提取与校验以防止格式或解析失败。
    """
    raw = text or ""
    obj_str = _extract_first_json_object(raw)
    data: dict[str, Any] = {}

    if obj_str:
        try:
            data = json.loads(obj_str)
        except Exception:
            data = {}

    dets_raw = data.get("detections") if isinstance(data, dict) else None
    detections: list[VlmBBoxDetection] = []

    if isinstance(dets_raw, list):
        for d in dets_raw:
            if not isinstance(d, dict):
                continue

            defect_type = str(d.get("defect_type") or d.get("type") or "other").strip().lower()
            if defect_type not in _DEFECT_TYPES:
                defect_type = "other"

            anomaly_subtype = _normalize_anomaly_subtype(d.get("anomaly_subtype") or d.get("subtype"))

            bbox = _sanitize_bbox_xyxy(d.get("bbox_xyxy"), w=image_w, h=image_h)
            if bbox is None:
                continue

            try:
                conf = float(d.get("confidence", d.get("conf", 0.0)))
            except Exception:
                conf = 0.0
            conf = float(max(0.0, min(1.0, conf)))

            detections.append(
                VlmBBoxDetection(defect_type=defect_type, bbox_xyxy=bbox, conf=conf, anomaly_subtype=anomaly_subtype)
            )

    # 按置信度降序排序并截断到 max_boxes
    detections.sort(key=lambda x: x.conf, reverse=True)
    detections = detections[: int(max_boxes)]

    return VlmBBoxOutput(image_w=int(image_w), image_h=int(image_h), detections=detections, raw_text=raw)


def build_defect_bbox_prompt(*, image_w: int, image_h: int, max_boxes: int = 3) -> str:
    """构建用于请求 VLM 输出缺陷/异常 bbox 的严格 JSON prompt。

    prompt 内容尽量简短以提高 JSON 合规性并减少幻觉（hallucination）。
    """

    object_types = (
        "Capsule, Carpet, Grid, Hazelnut, Leather, Metal Nut, Pill, Screw, Tile, Toothbrush, "
        "Transistor, Wood, Zipper, Candle, PCB(1-4), Macaroni(1-2), Fryum/Pipe Fryum, Cashew, Chewing gum, "
        "pipe_fryum, pcb4, pcb3, pcb2, pcb1, macaroni2, macaroni1, fryum, chewinggum, cashew, capsules, split_csv"
    )

    w = int(image_w)
    h = int(image_h)
    k = int(max_boxes)

    return (
        # 1) 严格的格式化规则优先
        "Return JSON ONLY. Do NOT output markdown. Do NOT output any extra text outside JSON.\n"
        "Use this JSON schema exactly:\n"
        "{\n"
        "  \"image_width\": <int>,\n"
        "  \"image_height\": <int>,\n"
        "  \"detections\": [\n"
        "    {\"defect_type\": <string>, \"anomaly_subtype\": <string>, \"bbox_xyxy\": [<int>,<int>,<int>,<int>], \"confidence\": <float>}\n"
        "  ]\n"
        "}\n"
        "\n"
        # 2) 坐标规则
        f"Image size: width={w}, height={h} pixels.\n"
        "Use pixel coordinates in xyxy format: [xmin, ymin, xmax, ymax] with origin at top-left.\n"
        f"Return at most {k} detections sorted by confidence descending.\n"
        "If no obvious abnormal/anomalous region is visible, return detections as an empty list.\n"
        "\n"
        # 3) 任务定义
        "You are a visual inspection assistant. Objects are NOT limited to metal (may be fabric/wood/food/plastic/PCB/electronics/etc.).\n"
        "Find visible defect/anomaly regions on the OBJECT (not background) and output tight bboxes.\n"
        "\n"
        # 4) 上下文：数据集的对象类别
        f"Dataset object categories may include: {object_types}.\n"
        "\n"
        # 5) 优先的缺陷线索（简洁、高价值）
        "Prefer these anomaly cues first (but you may output other obvious anomalies too):\n"
        "- Surface: scratch, crack, dent, hole, stain, contamination/foreign body, roughness\n"
        "- Structural: bent, broken, cut, squeeze, deformation, missing part/component, misplaced/misaligned\n"
        "- Visual: color defect/mismatch, faulty imprint/print defect, mold, burnt/overcooked, oxidation/corrosion\n"
        "- PCB hints: missing component, wrong component, solder bridge/ball, short/open circuit, spur, mouse bite, foreign body\n"
        "\n"
        "Mapping rule: If the anomaly is mainly caused by missing part/component/bristles/wick or absent structure, set anomaly_subtype='missing_like'.\n"
        "\n"
        # 6) 灵活性规则
        "You do NOT have to strictly match the above defect types. If an abnormal region is obvious but does not match, set defect_type to 'other' and anomaly_subtype to 'other'.\n"
    )


def build_defect_bbox_prompt_compare(*, test_image_w: int, test_image_h: int, max_boxes: int = 3) -> str:
    """构建用于比较参考图（A）与测试图（B），并只对测试图（B）输出 bbox 的严格 JSON prompt。

    要求：
      - 仅对 Image B 输出 bbox
      - 坐标使用 Image B 的像素坐标（xyxy）
      - 仅输出 JSON

    prompt 同样保持简短以提高 JSON 合规性。
    """
    object_types = (
        "Capsule, Carpet, Grid, Hazelnut, Leather, Metal Nut, Pill, Screw, Tile, Toothbrush, "
        "Transistor, Wood, Zipper, Candle, PCB(1-4), Macaroni(1-2), Fryum/Pipe Fryum, Cashew, Chewing gum, "
        "pipe_fryum, pcb4, pcb3, pcb2, pcb1, macaroni2, macaroni1, fryum, chewinggum, cashew, capsules, split_csv"
    )

    w = int(test_image_w)
    h = int(test_image_h)
    k = int(max_boxes)

    return (
        "Return JSON ONLY. Do NOT output markdown. Do NOT output any extra text outside JSON.\n"
        "Output bounding boxes ONLY for the TEST image (Image B). All bbox coordinates MUST be in Image B pixel coordinates.\n"
        "Use this JSON schema exactly:\n"
        "{\n"
        "  \"image_width\": <int>,\n"
        "  \"image_height\": <int>,\n"
        "  \"detections\": [\n"
        "    {\"defect_type\": <string>, \"anomaly_subtype\": <string>, \"bbox_xyxy\": [<int>,<int>,<int>,<int>], \"confidence\": <float>}\n"
        "  ]\n"
        "}\n"
        "\n"
        "Images: Image A = NORMAL/reference (expected OK). Image B = TEST/inspection (may contain anomaly).\n"
        f"Image B size: width={w}, height={h} pixels.\n"
        "Use pixel coordinates in xyxy format: [xmin, ymin, xmax, ymax] with origin at top-left (Image B).\n"
        f"Return at most {k} detections sorted by confidence descending.\n"
        "If no clear anomaly is visible in Image B compared to Image A, return detections as an empty list.\n"
        "\n"
        "Task: Compare Image A and Image B. Find localized, structural/material differences that likely indicate a defect/anomaly in Image B.\n"
        "Ignore minor differences from global brightness/contrast/white-balance, small camera shift, or small rotation.\n"
        "Focus on the OBJECT region (not background), and output tight bboxes covering the abnormal area in Image B.\n"
        "\n"
        f"Dataset object categories may include: {object_types}.\n"
        "Prefer these anomaly cues first (but you may output other obvious anomalies too):\n"
        "- Surface: scratch, crack, dent, hole, stain, contamination/foreign body, roughness\n"
        "- Structural: bent, broken, cut, squeeze, deformation, missing part/component, misplaced/misaligned\n"
        "- Visual: color defect/mismatch, faulty imprint/print defect, mold, burnt/overcooked, oxidation/corrosion\n"
        "- PCB hints: missing component, wrong component, solder bridge/ball, short/open circuit, spur, mouse bite, foreign body\n"
        "\n"
        "Mapping rule: If the anomaly is mainly caused by missing part/component/bristles/wick or absent structure, set anomaly_subtype='missing_like'.\n"
        "\n"
        "You do NOT have to strictly match the above defect types. If an abnormal region is obvious but does not match, set defect_type to 'other' and anomaly_subtype to 'other'.\n"
    )


def _should_fallback(out: VlmBBoxOutput) -> bool:
    """启发式判断：当前结果是否需要使用更可靠的模型重试。

    逻辑要点：
    - 如果模型返回了检测项，则无需回退
    - 如果 detections 为空，但原始文本看上去像是错误信息或非 JSON 响应（例如包含 'error','exception','traceback' 等），则建议回退
    - 如果原始文本为空且 dete detections 为空，则通常不回退（可能是真正的无缺陷）
    """
    # If model returned no detections at all, and there is some raw text (or an error message),
    # a fallback retry may salvage a parse/format issue.
    if out is None:
        return True
    if out.detections:
        return False
    # Empty detections is a valid outcome when there is truly no defect.
    # But in practice, JSON compliance failures often lead to empty parse.
    # We only fallback if raw_text looks like an error / non-JSON response.
    raw = (out.raw_text or "").strip().lower()
    if not raw:
        return False
    # common non-JSON / error signals
    suspicious = ["error", "exception", "traceback", "status", "http", "failed", "invalid", "timeout"]
    if any(s in raw for s in suspicious):
        return True
    if "{" not in raw:
        return True
    return False


def get_vlm_defect_bboxes(
    image_pil: Image.Image,
    *,
    model_name: str = "qwen-vl-max",
    thinking: bool = False,
    api_key: Optional[str] = None,
    dashscope_module=None,
    max_boxes: int = 3,
) -> VlmBBoxOutput:
    """通过 DashScope 调用 Qwen-VL（或兼容模型）以获取缺陷 bounding box。

    前提：系统中已安装并配置 dashscope（或传入兼容的 dashscope_module）。
    如果未提供 dashscope 或 API key，则返回空结果（仅用于本地调试/回退）。
    """
    import os
    import time
    from http import HTTPStatus

    if dashscope_module is None:
        # no dashscope: return empty result
        w, h = image_pil.size
        time.sleep(0.1)
        return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text="")

    key = (api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        w, h = image_pil.size
        time.sleep(0.1)
        return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text="")

    dashscope_module.api_key = key

    w, h = image_pil.size
    prompt = build_defect_bbox_prompt(image_w=w, image_h=h, max_boxes=max_boxes)

    def _call_once(name: str) -> VlmBBoxOutput:
        try:
            temp_path = "temp_vlm_input_bbox.jpg"
            image_pil.save(temp_path)
            abs_path = os.path.abspath(temp_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{abs_path}"},
                        {"text": prompt},
                    ],
                }
            ]

            # Check if this model requires stream (e.g., QVQ series)
            if is_stream_only_model(name):
                # Use stream aggregator for QVQ
                # QVQ 是"仅思考模型"，总是输出思考过程（800-2200字符）
                aggregator = DashScopeStreamAggregator()
                reasoning, answer_text = aggregator.call_and_aggregate(
                    model=name,
                    messages=messages,
                    api_key=key,
                    extract_reasoning=True,  # ✅ 启用思考过程提取（QVQ 总是思考）
                )

                # 记录思考过程统计（不展示给用户，但记录日志）
                if reasoning:
                    print(f"[QVQ 推理] 模型 {name} 思考了 {len(reasoning)} 字符")
                    # 可选：未来可以从思考过程中提取置信度信号或其他有用信息

                text = answer_text
            else:
                # Use standard non-stream call for Qwen-VL / Qwen3-VL
                response = dashscope_module.MultiModalConversation.call(
                    model=name,
                    enable_thinking=bool(thinking),
                    messages=messages,
                )

                if response.status_code != HTTPStatus.OK:
                    return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text=str(getattr(response, "message", "")))

                content = response.output.choices[0].message.content
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = content[0].get("text", "")
                else:
                    text = str(content)

            return parse_vlm_bbox_output(text, image_w=w, image_h=h, max_boxes=max_boxes)
        except Exception as e:
            return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text=str(e))

    out = _call_once(str(model_name))

    # One-time fallback retry (e.g., plus/turbo -> max)
    fb = fallback_model_for_bbox(primary=str(model_name))
    if fb and fb != str(model_name) and _should_fallback(out):
        out2 = _call_once(str(fb))
        # Keep the better one: prefer non-empty detections; otherwise prefer longer raw_text for debugging.
        if out2.detections:
            return out2
        if len((out2.raw_text or "")) > len((out.raw_text or "")):
            return out2

    return out


def get_vlm_defect_bboxes_compare(
    normal_image_pil: Image.Image,
    test_image_pil: Image.Image,
    *,
    model_name: str = "qwen-vl-max",
    thinking: bool = False,
    api_key: Optional[str] = None,
    dashscope_module=None,
    max_boxes: int = 3,
) -> VlmBBoxOutput:
    """比较正常图（A）与测试图（B），并返回以测试图坐标系（Image B）表示的 bbox 结果。

    返回的 bboxes 使用 TEST 图像的像素坐标系统。
    """
    import os
    import time
    from http import HTTPStatus

    if dashscope_module is None:
        w, h = test_image_pil.size
        time.sleep(0.1)
        return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text="")

    key = (api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        w, h = test_image_pil.size
        time.sleep(0.1)
        return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text="")

    dashscope_module.api_key = key

    w, h = test_image_pil.size
    prompt = build_defect_bbox_prompt_compare(test_image_w=w, test_image_h=h, max_boxes=max_boxes)

    def _call_once(name: str) -> VlmBBoxOutput:
        try:
            normal_path = "temp_vlm_normal.jpg"
            test_path = "temp_vlm_test.jpg"
            normal_image_pil.save(normal_path)
            test_image_pil.save(test_path)
            abs_normal = os.path.abspath(normal_path)
            abs_test = os.path.abspath(test_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{abs_normal}"},
                        {"image": f"file://{abs_test}"},
                        {"text": prompt},
                    ],
                }
            ]

            # Check if this model requires stream (e.g., QVQ series)
            if is_stream_only_model(name):
                # Use stream aggregator for QVQ (Compare mode)
                # QVQ 是"仅思考模型"，总是输出思考过程
                aggregator = DashScopeStreamAggregator()
                reasoning, answer_text = aggregator.call_and_aggregate(
                    model=name,
                    messages=messages,
                    api_key=key,
                    extract_reasoning=True,  # ✅ 启用思考过程提取（QVQ 总是思考）
                )

                # 记录思考过程统计（Compare 模式）
                if reasoning:
                    print(f"[QVQ 对比推理] 模型 {name} 思考了 {len(reasoning)} 字符")

                text = answer_text
            else:
                # Use standard non-stream call for Qwen-VL / Qwen3-VL
                response = dashscope_module.MultiModalConversation.call(
                    model=name,
                    enable_thinking=bool(thinking),
                    messages=messages,
                )

                if response.status_code != HTTPStatus.OK:
                    return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text=str(getattr(response, "message", "")))

                content = response.output.choices[0].message.content
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = content[0].get("text", "")
                else:
                    text = str(content)

            return parse_vlm_bbox_output(text, image_w=w, image_h=h, max_boxes=max_boxes)
        except Exception as e:
            return VlmBBoxOutput(image_w=w, image_h=h, detections=[], raw_text=str(e))

    out = _call_once(str(model_name))

    fb = fallback_model_for_bbox(primary=str(model_name))
    if fb and fb != str(model_name) and _should_fallback(out):
        out2 = _call_once(str(fb))
        if out2.detections:
            return out2
        if len((out2.raw_text or "")) > len((out.raw_text or "")):
            return out2

    return out

