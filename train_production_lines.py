"""
三条生产线YOLOv8专用模型训练脚本
适配RTX4060显卡 (8GB显存)
支持独立训练三个专用模型：轴承、木材、芯片
"""

import os
import sys
from datetime import datetime
from ultralytics import YOLO
import torch


class ProductionLineTrainer:
    """生产线模型训练器"""

    def __init__(self, base_model='yolov8n.pt'):
        """
        初始化训练器

        Args:
            base_model: 基础模型路径，默认YOLOv8n (适合RTX4060)
        """
        self.base_model = base_model
        self.device = '0' if torch.cuda.is_available() else 'cpu'

        # 获取项目根目录的绝对路径
        self.project_root = os.path.abspath(os.path.dirname(__file__))

        # 检查CUDA可用性
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n✅ GPU检测成功: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("\n⚠️  未检测到CUDA，将使用CPU训练（速度较慢）")

        # 数据集配置 - 使用绝对路径
        self.datasets = {
            'bearing': {
                'name': '轴承生产线',
                'data_yaml': os.path.join(self.project_root, r'D070.轴承缺陷数据集(2561)\data.yaml'),
                'classes': 8,
                'description': '金属表面缺陷检测'
            },
            'wood': {
                'name': '木材生产线',
                'data_yaml': os.path.join(self.project_root, r'木材表面缺陷数据集(总4000张_训练集3556张_验证集444张)\wood\data.yaml'),
                'classes': 8,
                'description': '木材表面缺陷检测'
            },
            'chip': {
                'name': '芯片生产线',
                'data_yaml': os.path.join(self.project_root, r'芯片缺陷检测数据集\data.yaml'),
                'classes': 5,
                'description': '电子元件结构缺陷检测'
            }
        }

        # 训练超参数 (针对RTX4060优化)
        self.training_params = {
            'epochs': 100,
            'imgsz': 640,
            'batch': 8,       # RTX4060 8GB适配
            'workers': 2,
            'patience': 15,    # 早停耐心值
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': True,       # 自动混合精度训练
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0
        }

    def check_dataset(self, line_name):
        """
        检查数据集配置是否正确

        Args:
            line_name: 生产线名称 ('bearing', 'wood', 'chip')

        Returns:
            bool: 数据集是否有效
        """
        dataset_info = self.datasets[line_name]
        data_yaml = dataset_info['data_yaml']

        print(f"\n🔍 检查{dataset_info['name']}数据集...")

        if not os.path.exists(data_yaml):
            print(f"   ❌ 数据集配置文件不存在: {data_yaml}")
            return False

        print(f"   ✅ 配置文件: {data_yaml}")
        print(f"   ✅ 类别数量: {dataset_info['classes']}")
        print(f"   ✅ 检测类型: {dataset_info['description']}")

        return True

    def train_single_line(self, line_name, custom_params=None):
        """
        训练单条生产线模型

        Args:
            line_name: 生产线名称 ('bearing', 'wood', 'chip')
            custom_params: 自定义训练参数(可选)

        Returns:
            训练结果
        """
        if line_name not in self.datasets:
            print(f"❌ 错误: 未知的生产线名称 '{line_name}'")
            print(f"   可用选项: {list(self.datasets.keys())}")
            return None

        # 检查数据集
        if not self.check_dataset(line_name):
            return None

        dataset_info = self.datasets[line_name]

        # 合并训练参数
        params = self.training_params.copy()
        if custom_params:
            params.update(custom_params)

        # 设置项目名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"runs/train_{line_name}"
        run_name = f"{line_name}_line_{timestamp}"

        print(f"\n{'='*70}")
        print(f"🚀 开始训练: {dataset_info['name']}")
        print(f"{'='*70}")
        print(f"📦 基础模型: {self.base_model}")
        print(f"📊 数据集: {dataset_info['data_yaml']}")
        print(f"🎯 类别数量: {dataset_info['classes']}")
        print(f"💾 输出目录: {project_name}/{run_name}")
        print(f"🖥️  设备: {self.device}")
        print(f"⚙️  训练参数:")
        print(f"   - Epochs: {params['epochs']}")
        print(f"   - Batch Size: {params['batch']}")
        print(f"   - Image Size: {params['imgsz']}")
        print(f"   - Patience: {params['patience']}")
        print(f"{'='*70}\n")

        try:
            # 加载模型
            model = YOLO(self.base_model)

            # 开始训练
            results = model.train(
                data=dataset_info['data_yaml'],
                project=project_name,
                name=run_name,
                device=self.device,
                **params
            )

            print(f"\n{'='*70}")
            print(f"✅ {dataset_info['name']}训练完成!")
            print(f"{'='*70}")
            print(f"📁 模型保存位置: {project_name}/{run_name}/weights/")
            print(f"   - best.pt  : 最佳模型")
            print(f"   - last.pt  : 最后一轮模型")
            print(f"📊 训练日志: {project_name}/{run_name}/")
            print(f"{'='*70}\n")

            return results

        except Exception as e:
            print(f"\n❌ 训练失败: {dataset_info['name']}")
            print(f"   错误信息: {str(e)}")
            return None

    def train_all_lines(self, sequential=True):
        """
        训练所有三条生产线的模型

        Args:
            sequential: 是否顺序训练(True)还是并行训练(False)
                       建议RTX4060使用顺序训练避免显存不足

        Returns:
            dict: 所有训练结果
        """
        results = {}

        print("\n" + "="*70)
        print("🏭 三条生产线专用模型训练计划")
        print("="*70)
        print("📋 训练顺序:")
        for i, (line_name, info) in enumerate(self.datasets.items(), 1):
            print(f"   {i}. {info['name']} ({info['classes']}类) - {info['description']}")
        print("="*70 + "\n")

        input("按Enter键开始训练...")

        if sequential:
            # 顺序训练 (推荐)
            for line_name in ['bearing', 'wood', 'chip']:
                results[line_name] = self.train_single_line(line_name)

                if results[line_name] is None:
                    print(f"\n⚠️  {self.datasets[line_name]['name']}训练失败，继续下一个...")
                    continue

                print(f"\n{'⏳ 等待5秒后开始下一个训练...'}\n")
                import time
                time.sleep(5)
        else:
            # 并行训练 (需要更大显存)
            print("⚠️  警告: 并行训练需要大量显存，RTX4060可能不足！")
            print("   建议使用sequential=True进行顺序训练\n")
            # 这里可以实现多进程并行训练，但不推荐在RTX4060上使用

        # 打印总结
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """打印训练总结"""
        print("\n" + "="*70)
        print("📊 训练总结")
        print("="*70)

        success_count = sum(1 for r in results.values() if r is not None)
        total_count = len(results)

        print(f"✅ 成功训练: {success_count}/{total_count}")
        print("\n详细结果:")

        for line_name, result in results.items():
            info = self.datasets[line_name]
            status = "✅ 成功" if result is not None else "❌ 失败"
            print(f"   {status} - {info['name']}")

        print("="*70)
        print("\n🎉 所有训练任务执行完毕!")
        print("\n📌 下一步:")
        print("   1. 查看训练日志和指标曲线")
        print("   2. 使用best.pt模型进行验证")
        print("   3. 将模型集成到工业看板系统")
        print("="*70 + "\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='三条生产线YOLOv8模型训练')
    parser.add_argument('--line', type=str, choices=['bearing', 'wood', 'chip', 'all'],
                       default='all', help='选择训练的生产线')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='基础模型 (yolov8n.pt/yolov8s.pt/yolov8m.pt)')

    args = parser.parse_args()

    # 创建训练器
    trainer = ProductionLineTrainer(base_model=args.model)

    # 自定义参数
    custom_params = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz
    }

    # 执行训练
    if args.line == 'all':
        trainer.train_all_lines()
    else:
        trainer.train_single_line(args.line, custom_params)


if __name__ == '__main__':
    # 如果没有命令行参数，提供交互式菜单
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("🏭 三条生产线YOLOv8训练系统")
        print("="*70)
        print("\n请选择训练模式:")
        print("  1. 训练所有三条生产线 (推荐)")
        print("  2. 仅训练轴承生产线")
        print("  3. 仅训练木材生产线")
        print("  4. 仅训练芯片生产线")
        print("  5. 退出")

        choice = input("\n请输入选项 (1-5): ").strip()

        trainer = ProductionLineTrainer()

        if choice == '1':
            trainer.train_all_lines()
        elif choice == '2':
            trainer.train_single_line('bearing')
        elif choice == '3':
            trainer.train_single_line('wood')
        elif choice == '4':
            trainer.train_single_line('chip')
        elif choice == '5':
            print("退出训练系统")
        else:
            print("无效选项，退出")
    else:
        main()

