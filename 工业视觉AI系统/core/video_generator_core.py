"""视频生成核心模块 - 生成带检测结果的传送带视频

基于 make_vedio/2.py 的逻辑，集成 VLM 检测结果
"""

import cv2
import numpy as np
import random
from typing import List, Dict, Any, Callable, Optional
from PIL import Image
from tqdm import tqdm


def generate_conveyor_video_with_detections(
    images: List[Image.Image],
    detections: Dict[int, Dict[str, Any]],
    output_file: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    speed: int = 10,
    vibration: bool = True,
    background_image: Optional[Image.Image] = None,
    progress_callback: Optional[Callable] = None
):
    """生成带检测结果的传送带视频

    Args:
        images: PIL 图片列表
        detections: 检测结果字典 {idx: {'has_defect': bool, 'detections': [...]}}
        output_file: 输出视频路径
        width: 视频宽度
        height: 视频高度
        fps: 帧率
        speed: 传送带速度
        vibration: 是否启用振动
        background_image: 自定义背景图片
        progress_callback: 进度回调函数
    """

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 背景处理
    belt_color = (40, 40, 40)
    background = None

    if background_image:
        bg_array = np.array(background_image.convert("RGB"))
        bg_bgr = cv2.cvtColor(bg_array, cv2.COLOR_RGB2BGR)
        background = cv2.resize(bg_bgr, (width, height))

    # 动画参数
    objects = []
    frame_count = 0
    spawn_timer = 0
    img_idx = 0
    product_counter = 1

    # 视频时长（根据图片数量动态调整）
    frames_per_product = 200  # 每个产品大约显示的帧数
    total_frames = min(fps * 60, len(images) * frames_per_product)  # 最多 60 秒

    if progress_callback:
        progress_callback(0.0, "开始生成视频...")

    # 生成视频
    for frame_idx in range(total_frames):
        # 绘制背景
        if background is not None:
            frame = background.copy()
            # 添加滚动线条
            line_offset = (frame_count * speed) % 100
            overlay = frame.copy()
            for i in range(0, width + 100, 100):
                x = i - line_offset
                cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        else:
            frame = np.full((height, width, 3), belt_color, dtype=np.uint8)
            line_offset = (frame_count * speed) % 100
            for i in range(0, width + 100, 100):
                x = i - line_offset
                cv2.line(frame, (x, 0), (x, height), (60, 60, 60), 2)

        # 生成新物体
        if spawn_timer <= 0 and img_idx < len(images):
            pil_img = images[img_idx]
            img_array = np.array(pil_img.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            h, w = img_bgr.shape[:2]
            scale = (height * 0.6) / h
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))

            objects.append({
                'img': img_resized,
                'x': width,
                'y': (height - new_h) // 2,
                'id': product_counter,
                'dataset_idx': img_idx,
                'orig_size': (h, w)
            })

            product_counter += 1
            img_idx += 1
            spawn_timer = (new_w + random.randint(100, 400)) // speed

        spawn_timer -= 1

        # 更新并绘制物体
        for obj in objects[:]:
            obj['x'] -= speed

            y_offset = 0
            if vibration:
                y_offset = int(np.sin(frame_count * 0.2) * 2)

            current_x = int(obj['x'])
            current_y = int(obj['y'] + y_offset)

            h, w = obj['img'].shape[:2]

            if current_x + w < 0:
                objects.remove(obj)
                continue

            x1 = max(current_x, 0)
            y1 = max(current_y, 0)
            x2 = min(current_x + w, width)
            y2 = min(current_y + h, height)

            if x1 < x2 and y1 < y2:
                img_x1 = x1 - current_x
                img_y1 = y1 - current_y
                img_x2 = img_x1 + (x2 - x1)
                img_y2 = img_y1 + (y2 - y1)

                frame[y1:y2, x1:x2] = obj['img'][img_y1:img_y2, img_x1:img_x2]

                # 绘制检测结果
                det = detections.get(obj['dataset_idx'], {})

                if det.get('has_defect'):
                    # 绘制 VLM 检测的 bbox
                    for detection in det.get('detections', []):
                        bbox = detection.get('bbox', [])
                        if len(bbox) == 4:
                            # 坐标转换：原图 → 当前显示尺寸
                            orig_h, orig_w = obj['orig_size']
                            scale_x = w / orig_w
                            scale_y = h / orig_h

                            det_x1 = int(bbox[0] * scale_x) + current_x
                            det_y1 = int(bbox[1] * scale_y) + current_y
                            det_x2 = int(bbox[2] * scale_x) + current_x
                            det_y2 = int(bbox[3] * scale_y) + current_y

                            det_x1 = max(det_x1, 0)
                            det_y1 = max(det_y1, 0)
                            det_x2 = min(det_x2, width)
                            det_y2 = min(det_y2, height)

                            # 红色检测框
                            cv2.rectangle(frame, (det_x1, det_y1), (det_x2, det_y2),
                                        (0, 0, 255), 3)

                            # 缺陷类型标签
                            defect_type = detection.get('type', 'defect')
                            confidence = detection.get('confidence', 0)
                            label_text = f"{defect_type}"
                            if confidence > 0:
                                label_text += f" {confidence:.2f}"

                            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame,
                                        (det_x1, det_y1 - label_size[1] - 10),
                                        (det_x1 + label_size[0] + 10, det_y1),
                                        (0, 0, 255), -1)
                            cv2.putText(frame, label_text,
                                      (det_x1 + 5, det_y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # 无缺陷，绿色闪烁框
                    if frame_count % 20 < 10:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制产品编号
                label = f"#{obj['id']:03d}"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = min(1.0, w / 300)
                font_thickness = max(1, int(font_scale * 2))

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                label_x = current_x + (w - text_width) // 2
                label_y = current_y - 15

                if label_y > text_height and label_x >= 0 and label_x + text_width <= width:
                    padding = 5
                    bg_x1 = label_x - padding
                    bg_y1 = label_y - text_height - padding
                    bg_x2 = label_x + text_width + padding
                    bg_y2 = label_y + baseline + padding

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                                (50, 50, 50), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2),
                                (0, 255, 0), 1)

                    cv2.putText(frame, label, (label_x, label_y),
                              font, font_scale, (0, 0, 0), font_thickness + 2)
                    cv2.putText(frame, label, (label_x, label_y),
                              font, font_scale, (255, 255, 255), font_thickness)

        # 添加帧信息
        cv2.putText(frame, f"Frame: {frame_count}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed} px/frame", (30, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

        # 更新进度
        if progress_callback and frame_idx % 30 == 0:
            progress = frame_idx / total_frames
            msg = f"生成中... {int(progress * 100)}%"
            progress_callback(progress, msg)

    out.release()

    if progress_callback:
        progress_callback(1.0, "视频生成完成！")

