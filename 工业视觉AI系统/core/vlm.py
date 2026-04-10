from __future__ import annotations

import os
import re
import time
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

from PIL import Image

# ✅ 添加 QVQ 流式支持
from core.vlm_model_registry import is_stream_only_model
from core.dashscope_stream import DashScopeStreamAggregator


@dataclass
class VlmOutput:
    """VLM 推理输出（供界面展示与选择）。

    fields:
      tags_en: 英文候选词/短语（用于按钮选择）
      desc_zh: 中文整体描述（仅展示）
      desc_en: 英文整体描述（仅展示）
      raw_text: 模型原始文本（可用于调试/追溯）
    """

    tags_en: list[str]
    desc_zh: str
    desc_en: str
    raw_text: str = ""


def _clean_keywords(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[\n,;]+", text)
    out: list[str] = []
    for p in parts:
        k = p.strip().lower()
        k = re.sub(r"[^a-z0-9\-\s]", "", k)
        k = re.sub(r"\s+", " ", k).strip()
        if k and k not in out:
            out.append(k)
    return out


def _extract_field(text: str, key: str) -> str:
    """从模型输出中提取类似 'KEY: ...' 的字段内容。"""
    if not text:
        return ""
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else ""


def _parse_vlm_output(text: str, *, max_tags: int) -> VlmOutput:
    """把模型返回的结构化文本解析成 VlmOutput。"""
    tags_line = _extract_field(text, "TAGS_EN")
    desc_en = _extract_field(text, "DESC_EN")
    desc_zh = _extract_field(text, "DESC_ZH")

    tags = _clean_keywords(tags_line)

    # 兜底：如果模型没有按格式返回，就尝试直接把全文当成 tags
    if not tags and not (desc_en or desc_zh):
        tags = _clean_keywords(text)

    return VlmOutput(tags_en=tags[:max_tags], desc_zh=desc_zh, desc_en=desc_en, raw_text=text)


def get_dashscope_key() -> str:
    """优先从环境变量读取 DashScope Key，并且不在界面中暴露。"""
    return os.getenv("DASHSCOPE_API_KEY", "").strip()


def dashscope_ready(key: str) -> bool:
    return bool(key)


DEFAULT_NONSTREAM_TIMEOUT = 30  # seconds
MAX_RETRIES = 2


def get_vlm_suggestions(
    image_pil: Image.Image,
    *,
    max_keywords: int = 6,
    model_name: str = "qwen-vl-max",
    mode: str = "industrial_defect",
    thinking: bool = False,
    api_key: Optional[str] = None,
    dashscope_module=None,
    timeout: int = DEFAULT_NONSTREAM_TIMEOUT,
) -> VlmOutput:
    """调用 VLM 生成候选词/短语与中英文整体描述。

    参数:
      mode:
        - general：通用描述（偏物体/场景）
        - industrial_defect：工业缺陷方向（偏检测线索/缺陷现象）
        - daily_damage：日常物体损坏/差异方向（偏损坏类型/异常）
      thinking:
        - False：默认关闭（更短、更易遵守输出格式）
        - True：开启思考模式（可能更准，但更容易输出冗余文本）

    返回:
      VlmOutput，其中 tags_en 可供用户点击选择；desc_zh/desc_en 仅用于提示说明。
    """

    mock = VlmOutput(
        tags_en=["screw", "metal", "scratch", "defect", "rust", "chip"][:max_keywords],
        desc_zh="示例描述：图中可能是金属零件，表面可能存在划痕或污渍。",
        desc_en="Sample description: The image may show a metal part; there might be scratches or stains on the surface.",
        raw_text="",
    )

    if dashscope_module is None:
        time.sleep(0.2)
        return mock

    key = (api_key or get_dashscope_key()).strip()
    if not dashscope_ready(key):
        time.sleep(0.2)
        return mock

    dashscope_module.api_key = key

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image_pil.save(temp_path)
        abs_path = os.path.abspath(temp_path)

        # 说明：这里要求模型严格按三行格式输出，便于稳定解析。
        # TAGS_EN 允许包含英文短语（例如 bent lead、missing part）。
        base_format = (
            "Output strictly in the following format with exactly 3 lines:\n"
            "TAGS_EN: <comma-separated English keywords or short phrases (3-8 items)>\n"
            "DESC_EN: <1-2 sentences overall description in English>\n"
            "DESC_ZH: <1-2 sentences overall description in Chinese>\n"
            "Do not add any extra lines."
        )

        if mode == "general":
            prompt = (
                "Analyze the image. Identify the main object(s) and scene. "
                "Prefer concrete nouns, materials, and observable attributes. "
                "TAGS_EN should be short English keywords/phrases, e.g., 'transistor', 'metal screw', 'pcb board'.\n"
                + base_format
            )
        elif mode == "daily_damage":
            prompt = (
                "You are an inspector for everyday objects. Identify the object, then infer what looks different from a typical/undamaged version. "
                "Focus on damage/differences and plausible defect types. "
                "TAGS_EN can include short phrases like 'broken corner', 'missing piece', 'surface scratch'.\n"
                + base_format
            )
        else:
            prompt = (
                "You are an industrial visual inspection assistant. Identify the industrial object and any anomaly/different situation and any visible defect clues. "
                "Focus on nomaly/different situation and defect phenomena and inspection-relevant attributes. "
                "TAGS_EN can include short phrases like 'bent lead', 'missing part', 'contamination stain', 'misalignment'.\n"
                + base_format
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{abs_path}"},
                    {"text": prompt},
                ],
            }
        ]

        # ✅ 检测是否是 QVQ 等仅流式模型
        if is_stream_only_model(model_name):
            # QVQ 系列：使用流式聚合器（与范式 C 保持一致）
            aggregator = DashScopeStreamAggregator()
            reasoning, answer_text = aggregator.call_and_aggregate(
                model=model_name,
                messages=messages,
                api_key=key,
                extract_reasoning=True,  # QVQ 总是思考
                timeout=timeout,
            )

            # 记录思考过程统计
            if reasoning:
                print(f"[范式A QVQ 推理] 模型 {model_name} 思考了 {len(reasoning)} 字符")

            text = answer_text
        else:
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = dashscope_module.MultiModalConversation.call(
                        model=model_name,
                        # Some multimodal models (e.g. Qwen3-VL) support this flag.
                        enable_thinking=bool(thinking),
                        messages=messages,
                        timeout=timeout,
                    )
                    if response.status_code != HTTPStatus.OK:
                        last_err = f"HTTP {response.status_code}: {getattr(response, 'message', '')}"
                        continue

                    content = response.output.choices[0].message.content
                    if isinstance(content, list) and content and isinstance(content[0], dict):
                        text = content[0].get("text", "")
                    else:
                        text = str(content)
                    break
                except Exception as call_err:
                    last_err = str(call_err)
                    if attempt == MAX_RETRIES:
                        raise
                    time.sleep(0.3 * attempt)
            else:
                raise RuntimeError(last_err or "dashscope call failed")

        # 解析 VLM 输出
        parsed = _parse_vlm_output(text, max_tags=max_keywords)

        # 兜底：tags 为空时回退到 mock 的 tags，但描述尽量保留
        if not parsed.tags_en:
            parsed.tags_en = mock.tags_en[:max_keywords]

        return parsed

    except Exception as e:
        print(f"[VLM] call failed: {e}")
        mock.raw_text = str(e)
        return mock
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
