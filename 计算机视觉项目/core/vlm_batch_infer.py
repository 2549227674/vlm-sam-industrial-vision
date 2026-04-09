"""VLM 批量推理模块 - 用于视频生成器

批量处理图片列表，返回每张图片的缺陷检测结果
"""

import os
import re
import json
import base64
import io
from typing import List, Dict, Any, Callable, Optional
from PIL import Image


def encode_image_to_base64(image: Image.Image) -> str:
    """将 PIL 图片编码为 Base64"""
    buffered = io.BytesIO()
    # 压缩图片
    img = image.copy()
    if img.width > 1920 or img.height > 1080:
        img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)

    img.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_b64}"


def parse_vlm_response(text: str) -> Dict[str, Any]:
    """解析 VLM 返回的文本"""
    # 尝试提取 JSON 代码块
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 回退
    return {
        'frames': [],
        'raw_text': text,
        'parse_error': True
    }


def vlm_multi_image_infer(
    images: List[Image.Image],
    model: str = "qwen-vl-max",
    api_key: str = None,
    max_boxes: int = 5
) -> Dict[str, Any]:
    """VLM 多图推理"""
    try:
        import dashscope
    except ImportError:
        raise RuntimeError("请安装 dashscope: pip install dashscope")

    # 构建 Prompt
    prompt = f"""你是一个工业缺陷检测专家。现在有 {len(images)} 张产品图片（按顺序）。

请逐张分析每张图片，检测是否存在缺陷，并返回严格的 JSON 格式：

```json
{{
    "frames": [
        {{
            "frame_idx": 0,
            "has_defect": true,
            "detections": [
                {{
                    "type": "scratch",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.9,
                    "description": "产品表面划痕"
                }}
            ]
        }},
        {{
            "frame_idx": 1,
            "has_defect": false,
            "detections": []
        }}
    ]
}}
```

**要求**：
1. 每张图片单独分析（frame_idx 从 0 开始）
2. bbox 格式为 [x1, y1, x2, y2]（左上角和右下角坐标）
3. 缺陷类型：scratch（划痕）、crack（裂纹）、stain（污渍）、dent（凹痕）、chip（缺口）、other（其他）
4. 每张图片最多返回 {max_boxes} 个检测框
5. 必须严格遵守 JSON 格式

开始分析："""

    # 构建 messages
    content = []
    for img in images:
        content.append({"image": encode_image_to_base64(img)})
    content.append({"text": prompt})

    messages = [{"role": "user", "content": content}]

    # 调用 API
    try:
        response = dashscope.MultiModalConversation.call(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            model=model,
            messages=messages
        )

        if response.status_code == 200:
            text = response.output.choices[0].message.content[0]["text"]
            result = parse_vlm_response(text)
            result['raw_text'] = text
            return result
        else:
            return {
                'frames': [],
                'error': f"API 错误: {response.code} - {response.message}"
            }
    except Exception as e:
        return {
            'frames': [],
            'error': str(e)
        }


def batch_infer_images(
    images: List[Image.Image],
    model: str = "qwen-vl-max",
    batch_size: int = 10,
    max_boxes: int = 5,
    api_key: str = None,
    progress_callback: Optional[Callable] = None
) -> Dict[int, Dict[str, Any]]:
    """批量推理图片列表

    Returns:
        {
            0: {'has_defect': bool, 'detections': [...]},
            1: {...},
            ...
        }
    """
    total = len(images)
    all_detections = {}

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_images = images[batch_start:batch_end]

        if progress_callback:
            progress = batch_start / total
            msg = f"处理批次 {batch_start//batch_size + 1}/{(total+batch_size-1)//batch_size}: 图片 {batch_start+1}-{batch_end}"
            progress_callback(progress, msg)

        # 调用 VLM
        result = vlm_multi_image_infer(
            batch_images,
            model=model,
            api_key=api_key,
            max_boxes=max_boxes
        )

        # 解析结果
        if 'error' in result:
            # 推理失败，填充空结果
            for i in range(len(batch_images)):
                all_detections[batch_start + i] = {
                    'has_defect': False,
                    'detections': [],
                    'error': result['error']
                }
        else:
            frames = result.get('frames', [])

            # 映射到全局索引
            for i, frame_data in enumerate(frames):
                if i < len(batch_images):
                    all_detections[batch_start + i] = {
                        'has_defect': frame_data.get('has_defect', False),
                        'detections': frame_data.get('detections', [])
                    }

            # 填充未返回的帧
            for i in range(len(frames), len(batch_images)):
                all_detections[batch_start + i] = {
                    'has_defect': False,
                    'detections': []
                }

    if progress_callback:
        progress_callback(1.0, "推理完成！")

    return all_detections

