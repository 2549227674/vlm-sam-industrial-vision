"""
多轴承生产线监控系统 - 核心模块

支持多条轴承生产线的并发检测
使用共享模型机制提高资源利用率
"""

import threading
import time
import yaml
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# GPU监控支持
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml未安装，GPU监控将使用PyTorch估算。安装: pip install pynvml")


class GPUMonitor:
    """GPU性能监控器（使用NVML获取真实数据）"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._initialized = False
        self._handle = None
        self._device_name = "Unknown GPU"
        self._total_memory = 0

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._device_name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(self._device_name, bytes):
                    self._device_name = self._device_name.decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._total_memory = mem_info.total / (1024 ** 2)  # MB
                self._initialized = True
                print(f"✅ GPU监控已启用: {self._device_name}, 总显存: {self._total_memory:.0f}MB")
            except Exception as e:
                print(f"⚠️ NVML初始化失败: {e}")
                self._initialized = False
        else:
            # 使用PyTorch获取基本信息
            if torch.cuda.is_available():
                self._device_name = torch.cuda.get_device_name(0)
                self._total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                print(f"📊 GPU信息(PyTorch): {self._device_name}, 总显存: {self._total_memory:.0f}MB")

    def get_gpu_stats(self):
        """获取GPU状态"""
        stats = {
            "device_name": self._device_name,
            "total_memory_mb": self._total_memory,
            "used_memory_mb": 0,
            "memory_percent": 0,
            "gpu_util_percent": 0,
            "temperature": 0,
            "power_usage": 0,
            "is_real_data": False
        }

        if self._initialized and self._handle:
            try:
                # 获取显存使用
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                stats["used_memory_mb"] = mem_info.used / (1024 ** 2)
                stats["memory_percent"] = (mem_info.used / mem_info.total) * 100

                # 获取GPU利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                stats["gpu_util_percent"] = util.gpu

                # 获取温度
                try:
                    stats["temperature"] = pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    pass

                # 获取功耗
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    stats["power_usage"] = power / 1000  # 转换为W
                except:
                    pass

                stats["is_real_data"] = True

            except Exception as e:
                print(f"⚠️ GPU监控异常: {e}")

        elif torch.cuda.is_available():
            # PyTorch备选方案（只能获取显存）
            stats["used_memory_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
            stats["memory_percent"] = (stats["used_memory_mb"] / self._total_memory) * 100 if self._total_memory > 0 else 0
            # GPU利用率无法通过PyTorch获取，使用缓存显存估算
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            stats["gpu_util_percent"] = min((reserved / self._total_memory) * 100, 100) if self._total_memory > 0 else 0

        return stats

    def shutdown(self):
        """关闭监控"""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

def get_chinese_font(size=18):
    """获取支持中文的字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    return ImageFont.load_default()


# 缺陷类别中文名（轴承缺陷）
DEFECT_NAMES_CN = {
    0: "铸造毛刺",
    1: "裂纹",
    2: "划痕",
    3: "凹坑",
    4: "抛光铸件",
    5: "应变",
    6: "未抛光铸件",
    7: "毛刺"
}


class SharedModelManager:
    """共享模型管理器（单例模式）"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def reset(cls):
        """重置单例实例（用于重新加载模型）"""
        with cls._lock:
            cls._instance = None

    def __new__(cls, model_path, device='cuda:0'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_model(model_path, device)
        return cls._instance

    def _init_model(self, model_path, device):
        """初始化共享模型"""
        print(f"🔄 加载共享模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        if hasattr(self.model.model, 'share_memory'):
            self.model.model.share_memory()  # PyTorch共享内存模式
        print(f"✅ 模型加载完成，类别数: {len(self.model.names)}")
        print(f"   类别: {list(self.model.names.values())}")

    def predict(self, frame, conf=0.5):
        """线程安全的推理"""
        with torch.no_grad():
            results = self.model(frame, conf=conf, verbose=False)
            return results[0]  # 返回第一个结果


class BearingProductionLine:
    """单条轴承生产线"""

    def __init__(self, config, model):
        self.line_id = config['id']
        self.name = config['name']
        self.video_path = config['video']
        self.model = model
        self.keyframe_interval = config.get('keyframe_interval', 5)
        self.conf_threshold = 0.5

        # 统计数据
        self.total_frames = 0
        self.detected_defects = 0
        self.defect_types = {i: 0 for i in range(8)}  # 8类缺陷
        self.fps_history = []

        # 线程控制
        self.running = False
        self.thread = None
        self.cap = None

        # 性能监控
        self.last_frame_time = time.time()
        self.frame_times = []

        # 性能历史记录（用于图表绘制）
        self._performance_history = {
            "timestamps": [],
            "fps": [],
            "inference_times": [],
            "defect_counts": [],
            "frame_counts": []
        }
        self._history_max_len = 60  # 保留最近60个数据点
        self._last_history_update = 0

        # 实时预览帧（线程安全）
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._class_names = None  # 类别名称映射

        # 检测框持续显示控制
        self._last_detection_results = None  # 保存最后的检测结果
        self._detection_display_frames = config.get('detection_display_frames', 30)  # 检测框显示帧数（默认30帧约1秒）
        self._frames_since_detection = 0  # 自上次检测以来的帧数

    def start(self):
        """启动生产线检测"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"✅ {self.name} 已启动")

    def stop(self):
        """停止生产线"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print(f"🛑 {self.name} 已停止")

    def _run(self):
        """主检测循环"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"❌ 无法打开视频: {self.video_path}")
            return

        frame_count = 0
        loop_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # 循环播放
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_count += 1
                print(f"🔄 {self.name} 循环播放第 {loop_count} 次")
                continue

            frame_count += 1
            self.total_frames += 1

            # 用于显示的帧副本
            display_frame = frame.copy()

            # 关键帧检测
            if frame_count % self.keyframe_interval == 0:
                start_time = time.time()

                try:
                    results_list = self.model.predict(frame, conf=self.conf_threshold)

                    # model.predict() 返回列表，取第一个结果
                    if isinstance(results_list, list):
                        results = results_list[0]
                    else:
                        results = results_list

                    # 获取类别名称映射
                    if self._class_names is None:
                        if hasattr(results, 'names'):
                            self._class_names = results.names
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                            self._class_names = self.model.model.names

                    # 统计缺陷
                    if results.boxes is not None and len(results.boxes) > 0:
                        self.detected_defects += len(results.boxes)
                        for box in results.boxes:
                            cls_id = int(box.cls[0].cpu().item())
                            if cls_id in self.defect_types:
                                self.defect_types[cls_id] += 1

                        # 保存检测结果用于持续显示
                        self._last_detection_results = results
                        self._frames_since_detection = 0

                    # 绘制检测结果到显示帧
                    display_frame = self._draw_detections(display_frame, results)

                except Exception as e:
                    print(f"⚠️ {self.name} 检测异常: {e}")

                # 记录推理时间
                inference_time = time.time() - start_time
                self.frame_times.append(inference_time * 1000)  # 转换为ms
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)
            else:
                # 非关键帧：如果有之前的检测结果且未过期，继续显示检测框
                self._frames_since_detection += 1
                if (self._last_detection_results is not None and
                    self._frames_since_detection < self._detection_display_frames):
                    # 持续显示上次检测的结果
                    display_frame = self._draw_detections(display_frame, self._last_detection_results)
                else:
                    # 检测框显示过期，只显示信息覆盖层
                    display_frame = self._add_info_overlay(display_frame)

            # 更新最新帧（线程安全）
            with self._latest_frame_lock:
                self._latest_frame = display_frame

            # 控制帧率（30fps）
            time.sleep(0.033)

    def _draw_detections(self, frame, results):
        """在帧上绘制检测结果（支持中文）"""
        if results is None or results.boxes is None or len(results.boxes) == 0:
            # 即使没有检测到缺陷，也添加生产线信息
            return self._add_info_overlay(frame)

        # 定义颜色（RGB格式，用于PIL）
        colors_rgb = [
            (255, 0, 0),     # 红色
            (0, 255, 0),     # 绿色
            (0, 0, 255),     # 蓝色
            (255, 255, 0),   # 黄色
            (255, 0, 255),   # 紫色
            (0, 255, 255),   # 青色
            (255, 128, 0),   # 橙色
            (128, 0, 255),   # 粉色
        ]

        # BGR 颜色（用于 OpenCV 绘制边界框）
        colors_bgr = [
            (0, 0, 255),     # 红色
            (0, 255, 0),     # 绿色
            (255, 0, 0),     # 蓝色
            (0, 255, 255),   # 黄色
            (255, 0, 255),   # 紫色
            (255, 255, 0),   # 青色
            (0, 128, 255),   # 橙色
            (255, 0, 128),   # 粉色
        ]

        # 先用 OpenCV 绘制边界框
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().item())
            color = colors_bgr[cls_id % len(colors_bgr)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 转换为 PIL 绘制中文标签
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = get_chinese_font(16)

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            color = colors_rgb[cls_id % len(colors_rgb)]

            # 获取中文标签
            cn_name = DEFECT_NAMES_CN.get(cls_id, f"缺陷{cls_id}")
            label = f"{cn_name} {conf:.2f}"

            # 计算文本大小
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)

            # 绘制标签背景
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                          fill=color)
            # 绘制标签文字
            draw.text((x1, y1 - 25), label, font=font, fill=(255, 255, 255))

        # 添加生产线信息（中文）
        info_font = get_chinese_font(20)
        info_text = f"{self.name} | 帧数: {self.total_frames} | 缺陷: {self.detected_defects}"

        # 绘制信息背景
        info_bbox = draw.textbbox((10, 5), info_text, font=info_font)
        draw.rectangle([info_bbox[0]-5, info_bbox[1]-2, info_bbox[2]+5, info_bbox[3]+2],
                      fill=(0, 0, 0, 180))
        draw.text((10, 5), info_text, font=info_font, fill=(0, 255, 136))

        # 转回 BGR
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame

    def _add_info_overlay(self, frame):
        """添加生产线信息覆盖层（无检测时使用）"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        info_font = get_chinese_font(20)
        info_text = f"{self.name} | 帧数: {self.total_frames} | 缺陷: {self.detected_defects}"

        info_bbox = draw.textbbox((10, 5), info_text, font=info_font)
        draw.rectangle([info_bbox[0]-5, info_bbox[1]-2, info_bbox[2]+5, info_bbox[3]+2],
                      fill=(0, 0, 0, 180))
        draw.text((10, 5), info_text, font=info_font, fill=(0, 255, 136))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def get_latest_frame(self):
        """获取最新检测帧（线程安全）"""
        with self._latest_frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_stats(self, update_history=True):
        """获取统计数据"""
        defect_rate = (self.detected_defects / self.total_frames * 100) if self.total_frames > 0 else 0
        avg_inference_time = np.mean(self.frame_times) if self.frame_times else 0

        # 计算实时FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time if self.last_frame_time else 1
        current_fps = self.total_frames / elapsed if elapsed > 0 else 0

        # 更新性能历史记录（每秒最多更新一次）
        if update_history and (current_time - self._last_history_update) >= 1.0:
            timestamp = time.strftime("%H:%M:%S")
            self._performance_history["timestamps"].append(timestamp)
            self._performance_history["fps"].append(current_fps)
            self._performance_history["inference_times"].append(avg_inference_time)
            self._performance_history["defect_counts"].append(self.detected_defects)
            self._performance_history["frame_counts"].append(self.total_frames)

            # 限制历史长度
            for key in self._performance_history:
                if len(self._performance_history[key]) > self._history_max_len:
                    self._performance_history[key] = self._performance_history[key][-self._history_max_len:]

            self._last_history_update = current_time

        return {
            'line_id': self.line_id,
            'name': self.name,
            'total_frames': self.total_frames,
            'detected_defects': self.detected_defects,
            'defect_rate': defect_rate,
            'defect_types': self.defect_types.copy(),
            'avg_inference_time_ms': avg_inference_time,
            'is_running': self.running,
            'current_fps': current_fps
        }

    def get_performance_history(self):
        """获取性能历史记录（用于图表绘制）"""
        return {
            'line_id': self.line_id,
            'name': self.name,
            'history': self._performance_history.copy()
        }


class MultiBearingMonitor:
    """多轴承生产线监控系统"""

    def __init__(self, config_path):
        print("=" * 60)
        print("🔧 MultiBearingMonitor 初始化 - 新版本 v2")
        print("=" * 60)

        self.config_path = config_path
        self.config = self._load_config(config_path)

        # 获取配置文件所在目录的根目录（configs/multi_bearing -> 项目根目录）
        import os
        config_dir = os.path.dirname(os.path.abspath(config_path))
        # configs/multi_bearing -> configs -> 项目根目录
        self.project_root = os.path.dirname(os.path.dirname(config_dir))
        print(f"📁 配置文件路径: {config_path}")
        print(f"📁 配置文件目录: {config_dir}")
        print(f"📁 项目根目录: {self.project_root}")

        # 重置共享模型单例（确保使用新路径）
        SharedModelManager.reset()

        # 初始化模型
        global_config = self.config['global']
        model_path = global_config['model_path']
        print(f"📁 配置中的模型路径: {model_path}")

        # 将相对路径转换为绝对路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.project_root, model_path)
        print(f"📁 模型绝对路径: {model_path}")
        print(f"📁 模型文件存在: {os.path.exists(model_path)}")

        if global_config.get('shared_model', True):
            self.model_manager = SharedModelManager(
                model_path,
                global_config['device']
            )
            model = self.model_manager
        else:
            # 非共享模式
            print(f"🔄 加载独立模型: {model_path}")
            self.model_manager = YOLO(model_path)
            model = self.model_manager

        # 初始化生产线
        self.lines = []
        for line_key, line_config in self.config['lines'].items():
            # 转换视频路径为绝对路径
            video_path = line_config.get('video', '')
            if video_path and not os.path.isabs(video_path):
                line_config['video'] = os.path.join(self.project_root, video_path)
            line = BearingProductionLine(line_config, model)
            self.lines.append(line)

        self.start_time = None

    def start_all(self):
        """启动所有生产线"""
        print(f"\n🚀 启动 {len(self.lines)} 条轴承生产线...")
        self.start_time = time.time()
        for line in self.lines:
            line.start()
            time.sleep(0.5)  # 错开启动时间
        print("✅ 所有生产线已启动\n")

    def stop_all(self):
        """停止所有生产线"""
        print("\n🛑 停止所有生产线...")
        for line in self.lines:
            line.stop()
        print("✅ 所有生产线已停止\n")

    def get_aggregated_stats(self):
        """获取汇总统计"""
        total_frames = 0
        total_defects = 0
        aggregated_types = {i: 0 for i in range(8)}
        total_inference_time = 0

        line_stats = []
        for line in self.lines:
            stats = line.get_stats()
            line_stats.append(stats)

            total_frames += stats['total_frames']
            total_defects += stats['detected_defects']
            total_inference_time += stats['avg_inference_time_ms']

            for cls_id, count in stats['defect_types'].items():
                aggregated_types[cls_id] += count

        # 计算运行时间和吞吐量
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = total_frames / elapsed_time if elapsed_time > 0 else 0
        avg_inference_ms = total_inference_time / len(self.lines) if self.lines else 0

        # 获取真实GPU监控数据
        gpu_monitor = GPUMonitor.get_instance()
        gpu_stats = gpu_monitor.get_gpu_stats()

        return {
            'total_lines': len(self.lines),
            'total_frames': total_frames,
            'total_defects': total_defects,
            'overall_defect_rate': (total_defects / total_frames * 100) if total_frames > 0 else 0,
            'defect_types': aggregated_types,
            'lines': line_stats,
            'elapsed_time': elapsed_time,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inference_ms,
            # GPU真实监控数据
            'gpu_stats': gpu_stats
        }

    def get_all_lines_performance_history(self):
        """获取所有生产线的性能历史记录"""
        histories = []
        for line in self.lines:
            histories.append(line.get_performance_history())
        return histories

    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_all_latest_frames(self):
        """获取所有生产线的最新检测帧"""
        frames = {}
        for line in self.lines:
            frame = line.get_latest_frame()
            if frame is not None:
                frames[line.line_id] = {
                    'name': line.name,
                    'frame': frame
                }
        return frames


# 测试代码
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多轴承生产线监控系统")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--duration", type=int, default=30, help="运行时长（秒）")

    args = parser.parse_args()

    # 创建监控系统
    monitor = MultiBearingMonitor(args.config)

    # 启动所有生产线
    monitor.start_all()

    try:
        # 持续监控
        for i in range(args.duration):
            time.sleep(1)
            stats = monitor.get_aggregated_stats()

            print(f"\n⏱️  运行时间: {stats['elapsed_time']:.1f}s")
            print(f"📊 总帧数: {stats['total_frames']}, 总缺陷: {stats['total_defects']}")
            print(f"⚡ 平均FPS: {stats['avg_fps']:.1f}, 平均推理时间: {stats['avg_inference_ms']:.2f}ms")

            for line_stat in stats['lines']:
                print(f"   • {line_stat['name']}: {line_stat['total_frames']}帧, {line_stat['detected_defects']}缺陷")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")

    finally:
        # 停止所有生产线
        monitor.stop_all()

        # 显示最终统计
        final_stats = monitor.get_aggregated_stats()
        print("\n" + "="*70)
        print("📊 最终统计")
        print("="*70)
        print(f"运行时长: {final_stats['elapsed_time']:.1f}s")
        print(f"总检测帧数: {final_stats['total_frames']}")
        print(f"总检出缺陷: {final_stats['total_defects']}")
        print(f"总体缺陷率: {final_stats['overall_defect_rate']:.2f}%")
        print(f"平均吞吐量: {final_stats['avg_fps']:.1f} FPS")
        print(f"平均推理时间: {final_stats['avg_inference_ms']:.2f} ms")
        print("\n各生产线详情:")
        for line_stat in final_stats['lines']:
            print(f"  {line_stat['name']}:")
            print(f"    - 检测帧数: {line_stat['total_frames']}")
            print(f"    - 缺陷数量: {line_stat['detected_defects']}")
            print(f"    - 缺陷率: {line_stat['defect_rate']:.2f}%")
        print("="*70)

