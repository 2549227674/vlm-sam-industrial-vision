from __future__ import annotations

import time
from typing import cast

import streamlit as st
import torch
from modelscope import snapshot_download
from torchvision import models


@st.cache_resource(show_spinner=False)
def load_models(device: str):
    """加载 SAM-3 与 ResNet 主干网络（带 Streamlit 缓存）。

    该函数与界面层解耦，让 app_final.py 只负责界面布局与交互。

    返回:
      sam_processor, sam_model, sam_dtype, resnet_features
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
    model_dir = snapshot_download("facebook/sam3")
    print(f"[DEBUG] 模型目录就绪 (耗时: {time.time()-start:.1f}s)")

    sam_processor = getattr(_Sam3Processor, "from_pretrained")(model_dir)
    print(f"[DEBUG] Processor 加载完成 (耗时: {time.time()-start:.1f}s)")

    try:
        sam_model = getattr(_Sam3Model, "from_pretrained")(model_dir, torch_dtype=torch.bfloat16).to(device)
        sam_dtype = torch.bfloat16
    except Exception:
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

    return sam_processor, sam_model, sam_dtype, resnet_features
