"""VLM 模型注册表

功能：
- 集中管理可用的 VLM 模型列表
- 声明能力标志（单图/双图对比/关键词推荐/bbox-json）
- 为范式 A（推荐）和范式 C（bbox）提供稳定的默认值

避免在 UI 文件中硬编码模型列表。

注意：本项目使用 DashScope（阿里云）MultiModalConversation，
仅注册与该接口兼容的模型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VlmModelSpec:
    """VLM 模型规格"""
    name: str

    # 能力标志（从应用角度）
    supports_single_image: bool = True  # 支持单图
    supports_two_images: bool = True  # 支持双图对比
    supports_suggestions: bool = True  # 范式 A：TAGS_EN/DESC_EN/DESC_ZH
    supports_bbox_json: bool = True  # 范式 C：严格 JSON schema

    # UI / 降级策略的启发式参数
    json_reliability: str = "medium"  # JSON 可靠性：high/medium/low/unknown
    cost_tier: str = "medium"  # 成本层级：high/medium/low/unknown

    # 流式输出要求（针对 QVQ 系列）
    requires_stream: bool = False  # True 表示模型仅支持流式输出


# 保守注册：仅注册已知使用/兼容的模型
# 可扩展更多 DashScope 多模态模型
_REGISTRY: tuple[VlmModelSpec, ...] = (
    # Qwen-VL（现有模型）
    VlmModelSpec(name="qwen-vl-max", json_reliability="high", cost_tier="high"),
    VlmModelSpec(name="qwen-vl-plus", json_reliability="medium", cost_tier="medium"),
    VlmModelSpec(name="qwen-vl-turbo", json_reliability="low", cost_tier="low"),

    # Qwen3-VL（非流式 + 可选思考模式）
    VlmModelSpec(name="qwen3-vl-plus", json_reliability="high", cost_tier="high"),
    VlmModelSpec(name="qwen3-vl-plus-2025-12-19", json_reliability="high", cost_tier="high"),
    VlmModelSpec(name="qwen3-vl-flash", json_reliability="medium", cost_tier="low"),

    # QVQ 系列（仅流式 + 总是思考；高推理能力）
    VlmModelSpec(
        name="qvq-max",
        json_reliability="high",
        cost_tier="high",
        requires_stream=True,
    ),
    VlmModelSpec(
        name="qvq-max-latest",
        json_reliability="high",
        cost_tier="high",
        requires_stream=True,
    ),
    VlmModelSpec(
        name="qvq-plus",
        json_reliability="high",
        cost_tier="medium",
        requires_stream=True,
    ),
)


def list_models(*, require: str | None = None, two_images: bool | None = None) -> list[str]:
    """按能力过滤并列出模型名称

    Args:
        require:
            - None: 无额外要求
            - 'suggestions': 必须支持推荐（范式 A）
            - 'bbox': 必须支持 bbox-json（范式 C）
        two_images:
            - True: 必须支持双图对比
            - False: 必须支持单图（所有已注册模型都支持）
            - None: 不过滤

    Returns:
        符合条件的模型名称列表
    """
    out: list[str] = []
    for spec in _REGISTRY:
        if require == "suggestions" and not spec.supports_suggestions:
            continue
        if require == "bbox" and not spec.supports_bbox_json:
            continue
        if two_images is True and not spec.supports_two_images:
            continue
        if two_images is False and not spec.supports_single_image:
            continue
        out.append(spec.name)
    return out


def default_model_for_suggestions() -> str:
    """推荐任务的默认模型（优先选择能力更强的模型）"""
    return "qwen-vl-max"


def default_model_for_bbox(*, fast: bool = False) -> str:
    """bbox-json 任务的默认模型

    Args:
        fast: 快速模式使用 turbo，否则使用 max（更准确）

    Returns:
        默认模型名称
    """
    return "qwen-vl-turbo" if fast else "qwen-vl-max"


def fallback_model_for_bbox(*, primary: str) -> str:
    """bbox-json 任务的降级模型（主模型失败时使用）

    Args:
        primary: 主模型名称

    Returns:
        降级模型名称
    """
    if primary == "qwen-vl-max":
        return "qwen-vl-max"
    return "qwen-vl-max"


def get_model_info(model_name: str) -> VlmModelSpec | None:
    """按名称获取模型规格

    Args:
        model_name: 模型名称（如 "qvq-max", "qwen3-vl-plus"）

    Returns:
        找到则返回 VlmModelSpec，否则返回 None
    """
    for spec in _REGISTRY:
        if spec.name == model_name:
            return spec
    return None


def is_stream_only_model(model_name: str) -> bool:
    """检查模型是否仅支持流式输出（如 QVQ 系列）

    Args:
        model_name: 模型名称

    Returns:
        True 表示模型仅支持流式输出
    """
    spec = get_model_info(model_name)
    return spec.requires_stream if spec else False


