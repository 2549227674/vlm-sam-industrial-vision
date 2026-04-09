"""SAM3视频检测器 - 范式E核心模块

提供基于SAM3文本prompt的视频缺陷检测与跟踪功能
无需VLM推理，直接使用本地SAM3模型进行全实例分割
"""

from __future__ import annotations

import time
import logging
from typing import Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SAM3VideoDefectDetector:
    """基于SAM3的视频缺陷检测器

    核心特性：
    - 文本prompt驱动的视频检测
    - 自动跨帧跟踪
    - 本地推理，零边际成本
    - 支持多prompt并行检测
    """

    def __init__(self, device: str = "cuda", dtype=torch.bfloat16):
        """初始化检测器

        Args:
            device: 推理设备 (cuda/cpu)
            dtype: 模型精度 (bfloat16/float16/float32)
        """
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

        logger.info(f"[Paradigm E] 初始化SAM3视频检测器: device={device}, dtype={dtype}")

    def load_models(self, sam_video_model, sam_video_processor):
        """加载预训练模型（复用app_final中已加载的模型）

        Args:
            sam_video_model: 已加载的Sam3VideoModel实例
            sam_video_processor: 已加载的Sam3VideoProcessor实例
        """
        self.model = sam_video_model
        self.processor = sam_video_processor
        logger.info("[Paradigm E] 模型加载完成（复用已有实例）")

    def detect_defects_in_video(
        self,
        video_frames: list[Image.Image] | list[np.ndarray],
        prompts: list[str],
        threshold: float = 0.5,
        max_frames: int | None = None,
        use_streaming: bool = False,
    ) -> dict[str, Any]:
        """检测视频中的缺陷并跟踪

        Args:
            video_frames: 视频帧列表（PIL Image或numpy数组）
            prompts: 缺陷类型提示词列表，如 ["scratch", "dent"]
            threshold: 置信度阈值，低于此值的检测将被过滤
            max_frames: 最大处理帧数，None表示处理所有帧
            use_streaming: 是否使用流式推理（节省内存，但可能降低质量）

        Returns:
            {
                "prompt_results": {
                    "scratch": {
                        "total_instances": 15,
                        "total_detections": 45,  # 跨多帧的总检测数
                        "frames": {
                            0: {
                                "object_ids": [1, 2],
                                "scores": [0.85, 0.72],
                                "boxes": [[x1,y1,x2,y2], ...],
                                "num_instances": 2
                            },
                            ...
                        }
                    },
                    "dent": {...}
                },
                "statistics": {
                    "total_unique_instances": 20,  # 去重后的唯一实例数
                    "total_detections": 65,  # 所有prompt的总检测数
                    "frames_processed": 100,
                    "avg_detections_per_frame": 0.65,
                    "prompts_used": ["scratch", "dent"],
                    "inference_time_sec": 12.5,
                    "fps": 8.0
                }
            }
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未加载，请先调用 load_models()")

        start_time = time.time()

        # 限制处理帧数
        if max_frames:
            video_frames = video_frames[:max_frames]

        total_frames = len(video_frames)
        logger.info(f"[Paradigm E] 开始检测: {total_frames}帧, {len(prompts)}个prompt")

        results = {"prompt_results": {}, "statistics": {}}
        all_unique_instances = set()
        total_detections = 0

        # 对每个prompt分别检测
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"[Paradigm E] 处理prompt {prompt_idx+1}/{len(prompts)}: '{prompt}'")

            try:
                prompt_result = self._detect_single_prompt(
                    video_frames=video_frames,
                    prompt=prompt,
                    threshold=threshold,
                    use_streaming=use_streaming,
                )

                results["prompt_results"][prompt] = prompt_result

                # 收集统计信息
                unique_ids = set(prompt_result["unique_instance_ids"])
                all_unique_instances.update(unique_ids)
                total_detections += prompt_result["total_detections"]

                logger.info(
                    f"[Paradigm E] '{prompt}': {len(unique_ids)}个唯一实例, "
                    f"{prompt_result['total_detections']}次检测"
                )

            except Exception as e:
                logger.error(f"[Paradigm E] 处理prompt '{prompt}' 失败: {e}")
                results["prompt_results"][prompt] = {
                    "error": str(e),
                    "total_instances": 0,
                    "total_detections": 0,
                    "frames": {},
                    "unique_instance_ids": []
                }

        # 汇总统计信息
        inference_time = time.time() - start_time
        results["statistics"] = {
            "total_unique_instances": len(all_unique_instances),
            "total_detections": total_detections,
            "frames_processed": total_frames,
            "avg_detections_per_frame": total_detections / total_frames if total_frames > 0 else 0,
            "prompts_used": prompts,
            "inference_time_sec": round(inference_time, 2),
            "fps": round(total_frames / inference_time, 2) if inference_time > 0 else 0,
            "avg_time_per_frame_ms": round(inference_time * 1000 / total_frames, 2) if total_frames > 0 else 0,
        }

        logger.info(
            f"[Paradigm E] 检测完成: {len(all_unique_instances)}个实例, "
            f"{total_detections}次检测, 耗时{inference_time:.2f}秒 ({results['statistics']['fps']:.2f} FPS)"
        )

        return results

    def _detect_single_prompt(
        self,
        video_frames: list,
        prompt: str,
        threshold: float,
        use_streaming: bool,
    ) -> dict[str, Any]:
        """对单个prompt进行视频检测

        Returns:
            {
                "total_instances": 15,
                "total_detections": 45,
                "unique_instance_ids": [1, 2, 3, ...],
                "frames": {frame_idx: {...}, ...}
            }
        """
        try:
            # 初始化视频会话
            logger.debug(f"[Paradigm E] 初始化视频会话: {len(video_frames)}帧")
            inference_session = self.processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=self.dtype,
            )

            # 添加文本prompt
            logger.debug(f"[Paradigm E] 添加文本prompt: '{prompt}'")
            inference_session = self.processor.add_text_prompt(
                inference_session=inference_session,
                text=prompt,
            )

            # 批量推理所有帧
            prompt_detections = {}
            unique_instance_ids = set()
            total_detections_count = 0

            logger.debug("[Paradigm E] 开始批量推理...")

            for model_outputs in self.model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(video_frames)
            ):
                frame_idx = model_outputs.frame_idx

                # 后处理输出
                processed = self.processor.postprocess_outputs(
                    inference_session, model_outputs
                )

                # 过滤低置信度检测
                scores = processed.get('scores', [])
                if len(scores) == 0:
                    continue

                valid_indices = [
                    i for i, score in enumerate(scores)
                    if score > threshold
                ]

                if not valid_indices:
                    continue

                # 提取有效检测结果
                object_ids = processed.get('object_ids', [])
                boxes = processed.get('boxes', torch.tensor([]))

                valid_object_ids = [object_ids[i] for i in valid_indices]
                valid_scores = [scores[i].item() if hasattr(scores[i], 'item') else float(scores[i]) for i in valid_indices]
                valid_boxes = boxes[valid_indices].tolist() if len(boxes) > 0 else []

                # 记录帧级别的检测结果
                prompt_detections[frame_idx] = {
                    "object_ids": valid_object_ids,
                    "scores": valid_scores,
                    "boxes": valid_boxes,
                    "num_instances": len(valid_object_ids),
                }

                # 更新统计
                unique_instance_ids.update(valid_object_ids)
                total_detections_count += len(valid_object_ids)

                # 定期输出进度
                if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == len(video_frames):
                    logger.debug(
                        f"[Paradigm E] 进度: {frame_idx+1}/{len(video_frames)} "
                        f"({(frame_idx+1)/len(video_frames)*100:.1f}%)"
                    )

            return {
                "total_instances": len(unique_instance_ids),
                "total_detections": total_detections_count,
                "unique_instance_ids": list(unique_instance_ids),
                "frames": prompt_detections,
            }

        except Exception as e:
            logger.error(f"[Paradigm E] _detect_single_prompt失败: {e}", exc_info=True)
            raise

    def export_results_to_json(self, results: dict, output_path: str | Path) -> None:
        """导出检测结果为JSON格式

        Args:
            results: detect_defects_in_video的返回结果
            output_path: 输出文件路径
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 移除masks（太大）和其他不可序列化的对象
        export_data = {
            "prompt_results": {},
            "statistics": results["statistics"],
        }

        for prompt, data in results["prompt_results"].items():
            export_data["prompt_results"][prompt] = {
                "total_instances": data["total_instances"],
                "total_detections": data["total_detections"],
                "unique_instance_ids": data.get("unique_instance_ids", []),
                "frames": {
                    str(frame_idx): {
                        "object_ids": frame_data["object_ids"],
                        "scores": frame_data["scores"],
                        "boxes": frame_data["boxes"],
                        "num_instances": frame_data["num_instances"],
                    }
                    for frame_idx, frame_data in data["frames"].items()
                }
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[Paradigm E] 结果已导出至: {output_path}")

    def export_results_to_csv(self, results: dict, output_path: str | Path) -> None:
        """导出检测结果为CSV格式（产品ID-缺陷映射表）

        Args:
            results: detect_defects_in_video的返回结果
            output_path: 输出文件路径
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["帧索引", "Prompt类型", "对象ID", "置信度", "边界框(x1,y1,x2,y2)"])

            for prompt, data in results["prompt_results"].items():
                for frame_idx, frame_data in data["frames"].items():
                    for obj_id, score, box in zip(
                        frame_data["object_ids"],
                        frame_data["scores"],
                        frame_data["boxes"]
                    ):
                        writer.writerow([
                            frame_idx,
                            prompt,
                            obj_id,
                            f"{score:.4f}",
                            f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}"
                        ])

        logger.info(f"[Paradigm E] CSV已导出至: {output_path}")


def get_preset_prompts() -> dict[str, list[str]]:
    """获取预设的prompt模板

    Returns:
        预设模板字典 {模板名称: [prompt列表]}
    """
    return {
        "通用缺陷": [
            "defect",
            "scratch",
            "dent",
            "crack",
            "stain",
        ],
        "螺丝产线": [
            "scratch on screw",
            "dent on screw head",
            "thread damage",
            "incomplete thread",
            "surface defect",
        ],
        "电路板": [
            "solder bridge",
            "missing component",
            "crack on PCB",
            "short circuit",
            "cold solder joint",
        ],
        "金属零件": [
            "dent on metal",
            "scratch on surface",
            "rust spot",
            "burr on edge",
            "surface damage",
        ],
        "传送带监控": [
            "defect on product",
            "damaged item",
            "quality issue",
            "surface anomaly",
        ],
    }

