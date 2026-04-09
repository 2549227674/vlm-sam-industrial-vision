from __future__ import annotations

import time
from typing import cast
import logging

import streamlit as st
import torch
from modelscope import snapshot_download
from torchvision import models


logger = logging.getLogger(__name__)


def _bf16_supported(device: str) -> bool:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                return bool(torch.cuda.is_bf16_supported())
            except Exception:
                pass
        try:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8  # Ampere+ generally supports bfloat16
        except Exception:
            return False
    return False


@st.cache_resource(show_spinner=False)
def load_models(device: str):
    """加载 SAM-3（图像）、SAM-3（视频）与 ResNet 主干网络（带 Streamlit 缓存）。

    该函数与界面层解耦，让 app_final.py 只负责界面布局与交互。

    返回:
      sam_processor, sam_model, sam_dtype, resnet_features, sam_video_model, sam_video_processor
    """
    start = time.time()

    # Transformers 中的 SAM-3 接口在部分版本可能不存在；这里失败时给出清晰错误提示
    try:
        from transformers import Sam3Processor, Sam3Model
    except Exception as e:
        raise RuntimeError(
            "当前环境无法导入 Sam3Processor/Sam3Model。\n"
            "请检查 transformers 版本是否包含 SAM-3（部分版本可能没有 Sam3Model），或重新安装支持 SAM-3 的版本。\n"
            f"原始错误: {e}"
        )

    _Sam3Processor = cast(object, Sam3Processor)
    _Sam3Model = cast(object, Sam3Model)

    print(f"[DEBUG] 开始加载模型... (耗时: {time.time()-start:.1f}s)")

    last_err = None
    model_dir = None
    for attempt in range(1, 4):
        try:
            model_dir = snapshot_download("facebook/sam3")
            break
        except Exception as e:  # pragma: no cover - download path
            last_err = e
            logger.warning("snapshot_download 失败 (第 %d 次): %s", attempt, e)
            time.sleep(0.5 * attempt)
    if not model_dir:
        raise RuntimeError(
            "下载 SAM-3 模型失败，请检查网络/权限或手动预下载 'facebook/sam3' 到缓存后重试。\n"
            f"最后错误: {last_err}"
        )

    print(f"[DEBUG] 模型目录就绪 (耗时: {time.time()-start:.1f}s)")

    sam_processor = getattr(_Sam3Processor, "from_pretrained")(model_dir)
    print(f"[DEBUG] Processor 加载完成 (耗时: {time.time()-start:.1f}s)")

    # Dtype 策略：CPU 强制 float32；GPU 优先 bfloat16，若不支持则降级 float32 并提示。
    prefer_bf16 = _bf16_supported(device)
    target_dtype = torch.bfloat16 if prefer_bf16 else torch.float32

    try:
        sam_model = getattr(_Sam3Model, "from_pretrained")(model_dir, torch_dtype=target_dtype).to(device)
        sam_dtype = target_dtype
        if not prefer_bf16:
            logger.info("GPU 不支持 bfloat16，已降级为 float32")
    except Exception as e:
        logger.warning("SAM-3 加载 %s 失败，降级为 float32: %s", target_dtype, e)
        sam_model = getattr(_Sam3Model, "from_pretrained")(model_dir, torch_dtype=torch.float32).to(device)
        sam_dtype = torch.float32

    print(f"[DEBUG] SAM-3 模型加载完成 (耗时: {time.time()-start:.1f}s)")

    # ResNet-18（取到第三个残差阶段的特征，输出 16x16 的空间特征图）
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_features = torch.nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    ).to(device).eval()
    print(f"[DEBUG] ResNet-18 加载完成 (耗时: {time.time()-start:.1f}s)")

    # 加载 SAM-3 Video 模型（用于范式E：视频检测与跟踪）
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        print(f"[DEBUG] 开始加载 SAM-3 Video 模型... (耗时: {time.time()-start:.1f}s)")

        sam_video_processor = Sam3VideoProcessor.from_pretrained(model_dir)
        sam_video_model = Sam3VideoModel.from_pretrained(
            model_dir,
            torch_dtype=sam_dtype
        ).to(device)

        print(f"[DEBUG] SAM-3 Video 模型加载完成 (耗时: {time.time()-start:.1f}s)")

    except Exception as e:
        logger.warning("SAM-3 Video 模型加载失败（范式E不可用）: %s", e)
        sam_video_model = None
        sam_video_processor = None

    return sam_processor, sam_model, sam_dtype, resnet_features, sam_video_model, sam_video_processor
