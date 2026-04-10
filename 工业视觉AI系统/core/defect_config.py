"""缺陷类别配置管理模块

该模块负责加载和管理缺陷类别的动态配置，支持：
- 从YAML文件加载预设配置
- 动态生成VLM Prompt
- 类别验证和别名映射
- 用户自定义配置

作者：System
创建时间：2026-01-08
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DefectCategory:
    """单个缺陷类别定义"""
    id: str
    display_name: str
    display_name_en: str = ""
    description: str = ""


@dataclass
class DefectSubtype:
    """缺陷子类型定义"""
    id: str
    display_name: str
    description: str = ""


@dataclass
class DefectCategoryConfig:
    """缺陷类别配置类

    负责加载YAML配置文件，提供动态Prompt生成和类别验证功能。
    """

    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    primary_types: List[DefectCategory] = field(default_factory=list)
    subtypes: List[DefectSubtype] = field(default_factory=list)
    prompt_hints: Dict[str, str] = field(default_factory=dict)
    category_aliases: Dict[str, str] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

    # 缓存的类别ID列表
    _primary_type_ids: List[str] = field(default_factory=list, init=False, repr=False)
    _subtype_ids: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """初始化后处理：构建缓存列表"""
        self._primary_type_ids = [cat.id for cat in self.primary_types]
        self._subtype_ids = [sub.id for sub in self.subtypes]

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> DefectCategoryConfig:
        """从YAML文件加载配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            DefectCategoryConfig实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: YAML格式错误
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"YAML解析失败: {e}")

        # 解析primary_types
        primary_types = []
        for cat_data in data.get('primary_types', []):
            if isinstance(cat_data, dict):
                primary_types.append(DefectCategory(
                    id=cat_data.get('id', ''),
                    display_name=cat_data.get('display_name', ''),
                    display_name_en=cat_data.get('display_name_en', ''),
                    description=cat_data.get('description', '')
                ))

        # 解析subtypes
        subtypes = []
        for sub_data in data.get('subtypes', []):
            if isinstance(sub_data, dict):
                subtypes.append(DefectSubtype(
                    id=sub_data.get('id', ''),
                    display_name=sub_data.get('display_name', ''),
                    description=sub_data.get('description', '')
                ))

        return cls(
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {}),
            primary_types=primary_types,
            subtypes=subtypes,
            prompt_hints=data.get('prompt_hints', {}),
            category_aliases=data.get('category_aliases', {}),
            validation=data.get('validation', {})
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> DefectCategoryConfig:
        """从预设配置加载

        Args:
            preset_name: 预设名称（generic/pcb/metal/food/textile）

        Returns:
            DefectCategoryConfig实例
        """
        # 尝试多个可能的配置文件路径
        possible_paths = [
            # 计算机视觉项目/core/../configs/defect_presets/
            Path(__file__).parent.parent / "configs" / "defect_presets" / f"{preset_name}.yaml",
            # 计算机视觉项目/core/../../configs/defect_presets/ (项目根目录)
            Path(__file__).parent.parent.parent / "configs" / "defect_presets" / f"{preset_name}.yaml",
            # 相对于当前工作目录
            Path("configs") / "defect_presets" / f"{preset_name}.yaml",
            Path(".") / "configs" / "defect_presets" / f"{preset_name}.yaml",
            # 相对于当前工作目录的上级目录
            Path("..") / "configs" / "defect_presets" / f"{preset_name}.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return cls.from_yaml(path)

        # 如果都不存在，使用默认配置
        print(f"⚠️ 预设配置 '{preset_name}' 未找到，使用默认通用配置")
        return cls.get_default_config()

    @classmethod
    def get_default_config(cls) -> DefectCategoryConfig:
        """获取默认配置（硬编码的通用10类）

        Returns:
            默认DefectCategoryConfig实例
        """
        default_types = [
            DefectCategory("scratch", "划痕", "Scratch"),
            DefectCategory("crack", "裂纹", "Crack"),
            DefectCategory("stain", "污渍", "Stain"),
            DefectCategory("dent", "凹陷", "Dent"),
            DefectCategory("burr", "毛刺", "Burr"),
            DefectCategory("chip", "崩缺", "Chip"),
            DefectCategory("discoloration", "变色", "Discoloration"),
            DefectCategory("contamination", "污染", "Contamination"),
            DefectCategory("corrosion", "腐蚀", "Corrosion"),
            DefectCategory("other", "其他", "Other"),
        ]

        default_subtypes = [
            DefectSubtype("missing_like", "缺失类"),
            DefectSubtype("surface_like", "表面类"),
            DefectSubtype("structural_like", "结构类"),
            DefectSubtype("visual_like", "视觉类"),
            DefectSubtype("other", "其他"),
        ]

        return cls(
            version="1.0",
            metadata={"name": "默认通用配置", "industry": "generic"},
            primary_types=default_types,
            subtypes=default_subtypes,
            prompt_hints={
                "surface": "scratch, crack, dent, hole, stain, contamination, roughness",
                "structural": "bent, broken, cut, deformation, missing part",
                "visual": "color defect, discoloration, print defect"
            },
            validation={"max_categories": 20, "min_confidence": 0.3}
        )

    def get_primary_type_ids(self) -> List[str]:
        """获取所有一级类别ID列表"""
        return self._primary_type_ids.copy()

    def get_subtype_ids(self) -> List[str]:
        """获取所有二级子类型ID列表"""
        return self._subtype_ids.copy()

    def validate_defect_type(self, defect_type: str) -> str:
        """验证并规范化缺陷类型

        Args:
            defect_type: VLM返回的缺陷类型

        Returns:
            规范化后的类型ID，若不在白名单则返回"other"
        """
        defect_type = str(defect_type).strip().lower()

        # 1. 尝试别名映射
        if defect_type in self.category_aliases:
            defect_type = self.category_aliases[defect_type]

        # 2. 检查是否在白名单
        if defect_type in self._primary_type_ids:
            return defect_type

        # 3. 不在白名单，返回"other"
        return "other" if "other" in self._primary_type_ids else self._primary_type_ids[0]

    def validate_subtype(self, subtype: str) -> str:
        """验证并规范化子类型

        Args:
            subtype: VLM返回的子类型

        Returns:
            规范化后的子类型ID，若不在白名单则返回"other"
        """
        if not subtype:
            return ""

        subtype = str(subtype).strip().lower().replace("-", "_").replace(" ", "_")

        if subtype in self._subtype_ids:
            return subtype

        return "other" if "other" in self._subtype_ids else ""

    def build_defect_bbox_prompt(self, *, image_w: int, image_h: int, max_boxes: int = 3) -> str:
        """构建VLM缺陷检测Prompt（动态生成）

        Args:
            image_w: 图像宽度
            image_h: 图像高度
            max_boxes: 最多返回框数

        Returns:
            完整的VLM Prompt字符串
        """
        w = int(image_w)
        h = int(image_h)
        k = int(max_boxes)

        # 构建类别列表字符串
        type_names = ", ".join(self._primary_type_ids)

        # 构建提示词段落
        hints_lines = []
        for category, hints in self.prompt_hints.items():
            hints_lines.append(f"- {category.title()}: {hints}")
        hints_text = "\n".join(hints_lines) if hints_lines else "- General: scratch, crack, dent, stain, deformation"

        # 行业特定提示
        industry = self.metadata.get('industry', 'generic')
        industry_hint = ""
        if industry == "electronics_pcb":
            industry_hint = "\nContext: PCB/electronics manufacturing. Focus on solder, component, and circuit defects."
        elif industry == "food":
            industry_hint = "\nContext: Food quality inspection. Focus on contamination, quality, and appearance defects."
        elif industry == "metal_machining":
            industry_hint = "\nContext: Metal machining/manufacturing. Focus on surface finish and dimensional defects."
        elif industry == "textile":
            industry_hint = "\nContext: Textile/fabric inspection. Focus on fabric integrity and stitching defects."

        return (
            # 1) 严格的格式化规则
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
            "You are a visual inspection assistant. Find visible defect/anomaly regions on the OBJECT (not background) and output tight bboxes.\n"
            f"{industry_hint}\n"
            "\n"
            # 4) 类别定义
            f"Defect types for this configuration: {type_names}\n"
            "\n"
            # 5) 提示词
            "Prefer these anomaly cues:\n"
            f"{hints_text}\n"
            "\n"
            # 6) 灵活性规则
            "You do NOT have to strictly match the above defect types. If an abnormal region is obvious but does not match, set defect_type to 'other'.\n"
            "Mapping rule: If the anomaly is mainly caused by missing part/component or absent structure, set anomaly_subtype='missing_like'.\n"
        )

    def build_compare_prompt(self, *, test_image_w: int, test_image_h: int, max_boxes: int = 3) -> str:
        """构建VLM对比检测Prompt（双图模式）

        Args:
            test_image_w: 测试图像宽度
            test_image_h: 测试图像高度
            max_boxes: 最多返回框数

        Returns:
            完整的VLM Prompt字符串
        """
        w = int(test_image_w)
        h = int(test_image_h)
        k = int(max_boxes)

        type_names = ", ".join(self._primary_type_ids)

        hints_lines = []
        for category, hints in self.prompt_hints.items():
            hints_lines.append(f"- {category.title()}: {hints}")
        hints_text = "\n".join(hints_lines) if hints_lines else "- General: scratch, crack, dent, stain, deformation"

        industry = self.metadata.get('industry', 'generic')
        industry_hint = ""
        if industry == "electronics_pcb":
            industry_hint = "\nContext: PCB inspection. Compare reference vs test board."
        elif industry == "food":
            industry_hint = "\nContext: Food quality. Compare normal vs potentially defective sample."

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
            f"{industry_hint}\n"
            "\n"
            f"Defect types for this configuration: {type_names}\n"
            f"{hints_text}\n"
            "\n"
            "Mapping rule: If the anomaly is mainly caused by missing part/component or absent structure, set anomaly_subtype='missing_like'.\n"
        )

    def get_display_name(self, defect_type: str, lang: str = "zh") -> str:
        """获取缺陷类型的显示名称

        Args:
            defect_type: 类别ID
            lang: 语言（'zh'中文, 'en'英文）

        Returns:
            显示名称
        """
        for cat in self.primary_types:
            if cat.id == defect_type:
                return cat.display_name if lang == "zh" else cat.display_name_en
        return defect_type

    def to_yolov8_classes(self) -> List[str]:
        """导出为YOLOv8 classes列表

        Returns:
            类别ID列表（用于生成classes.txt和data.yaml）
        """
        return self._primary_type_ids.copy()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "version": self.version,
            "metadata": self.metadata,
            "primary_types": [
                {"id": cat.id, "display_name": cat.display_name, "display_name_en": cat.display_name_en}
                for cat in self.primary_types
            ],
            "subtypes": [
                {"id": sub.id, "display_name": sub.display_name}
                for sub in self.subtypes
            ],
            "prompt_hints": self.prompt_hints,
            "category_aliases": self.category_aliases,
            "validation": self.validation
        }


# 预设配置快速访问
PRESET_CONFIGS = {
    "generic": "通用缺陷检测（10类）",
    "pcb": "PCB电路板（11类）",
    "metal": "金属加工（9类）",
    "food": "食品质量（10类）",
    "textile": "纺织品（9类）",
}


def get_available_presets() -> Dict[str, str]:
    """获取可用的预设配置列表

    Returns:
        字典 {preset_id: display_name}
    """
    return PRESET_CONFIGS.copy()


def load_preset_config(preset_name: str) -> DefectCategoryConfig:
    """加载预设配置（便捷函数）

    Args:
        preset_name: 预设名称

    Returns:
        DefectCategoryConfig实例
    """
    return DefectCategoryConfig.from_preset(preset_name)

