from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .cv_utils import pad_to_square_cv2
from .sam3_infer import run_sam3_instance_segmentation


def extract_layer3_features(resnet_sequential, img_tensor: torch.Tensor) -> torch.Tensor:
    """从 ResNet 顺序模块中提取 layer3 特征。

    约定 resnet_sequential 的结构为:
    [conv1, bn1, relu, maxpool, layer1, layer2, layer3]
    """
    x = resnet_sequential[0](img_tensor)# 卷积层
    x = resnet_sequential[1](x)# 标准化层
    x = resnet_sequential[2](x)# 激活层
    x = resnet_sequential[3](x)# 最大池化层 (此时尺寸减半)
    x = resnet_sequential[4](x)# Layer 1 (基础特征)
    x = resnet_sequential[5](x)# Layer 2 (进阶特征)
    x = resnet_sequential[6](x)# Layer 3 (这就是我们要的：包含空间信息的语义特征)
    return x


def extract_multiscale_features(resnet_sequential, img_tensor: torch.Tensor) -> torch.Tensor:
    """可选的历史实现：拼接 layer1/2 下采样后的特征与 layer3 特征。

    提示：当前主流程默认使用 layer3 单层特征（更稳、特征维度更小）。
    """
    x = resnet_sequential[0](img_tensor)
    x = resnet_sequential[1](x)
    x = resnet_sequential[2](x)
    x = resnet_sequential[3](x)
    f1 = resnet_sequential[4](x)
    f2 = resnet_sequential[5](f1)
    f3 = resnet_sequential[6](f2)

    f1_d = F.adaptive_avg_pool2d(f1, (16, 16))
    f2_d = F.adaptive_avg_pool2d(f2, (16, 16))
    return torch.cat([f1_d, f2_d, f3], dim=1)


def process_single_image(
    *,
    image_pil: Image.Image,
    sam_proc,
    sam_model,
    sam_dtype,
    resnet,
    prompt,
    threshold: float,
    context_pad: float,
    roi_mode: str,
    img_size: int,
    feat_dim: int,
    device: str,
    multi_prompt_strategy: str = "per_prompt",
    session_state=None,
):
    """SAM-3 定位 -> ROI 裁剪 -> ResNet 特征提取。

    返回:
      feat_np: [feat_dim, 16, 16] 的特征（numpy）
      roi_final: RGB uint8 [img_size, img_size, 3]（送入特征网络的最终 ROI）
    """
    raw_np = np.array(image_pil)

    results, _ = run_sam3_instance_segmentation(
        image_pil=image_pil,
        sam_proc=sam_proc,
        sam_model=sam_model,
        sam_dtype=sam_dtype,
        prompt=prompt,
        threshold=threshold,
        device=device,
        multi_prompt_strategy=multi_prompt_strategy,
        session_state=session_state,
    )

    if len(results["masks"]) == 0:
        return None, None

    scores = results["scores"].float()
    best_idx = torch.argmax(scores).item()
    mask = results["masks"][best_idx].cpu().numpy() > 0.5

    if roi_mode == "mask":
        raw_np = raw_np.copy()
        raw_np[~mask] = 0

    y, x = np.where(mask)
    if len(x) == 0:
        return None, None

    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()
    w, h = x2 - x1, y2 - y1

    pad = float(max(0.0, context_pad))
    h_img, w_img = raw_np.shape[:2]
    px1 = max(0, int(x1 - pad * w))
    py1 = max(0, int(y1 - pad * h))
    px2 = min(w_img, int(x2 + pad * w))
    py2 = min(h_img, int(y2 + pad * h))

    roi = raw_np[py1:py2, px1:px2]# 核心：补齐成正方形。如果物体长，就给两边补黑边。
    roi_sq = pad_to_square_cv2(roi)# 缩放到 256x256


    # 局部导入：避免在 core 模块中引入过多全局依赖
    import cv2

    roi_final = cv2.resize(roi_sq, (img_size, img_size), interpolation=cv2.INTER_AREA)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_t = transform(Image.fromarray(roi_final)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = extract_layer3_features(resnet, input_t)  # 形状: [1, 特征维, 16, 16]

    # 维度保护：尽量保持输出维度稳定
    if feat.shape[1] != feat_dim:
        # 维度不一致时，仍返回当前得到的特征（调用方需自行处理/重新训练）
        pass

    return feat.cpu().numpy()[0], roi_final
