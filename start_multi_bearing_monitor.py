"""
多轴承生产线监控系统 - 快速启动脚本

使用方法:
    python start_multi_bearing_monitor.py
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bearing_core.multi_bearing_monitor import MultiBearingMonitor


def select_config():
    """选择配置文件"""
    print("\n" + "="*70)
    print("🏭 多轴承生产线监控系统")
    print("="*70)
    print("\n请选择测试配置:")
    print("  1️⃣  单条生产线（基准测试）")
    print("  2️⃣  两条生产线（并发测试）")
    print("  3️⃣  三条生产线（极限测试）")
    print()

    choice = input("请输入选项 (1/2/3): ").strip()

    config_map = {
        '1': 'configs/multi_bearing/bearing_1_line.yaml',
        '2': 'configs/multi_bearing/bearing_2_lines.yaml',
        '3': 'configs/multi_bearing/bearing_3_lines.yaml'
    }

    if choice not in config_map:
        print("❌ 无效选项")
        sys.exit(1)

    return config_map[choice]


def main():
    """主函数"""
    # 选择配置
    config_path = select_config()

    print(f"\n📄 配置文件: {config_path}")

    # 询问运行时长
    duration_str = input("运行时长（秒，默认30）: ").strip()
    duration = int(duration_str) if duration_str else 30

    print(f"\n⏱️  将运行 {duration} 秒...\n")

    # 创建监控系统
    print("🔄 正在初始化监控系统...")
    monitor = MultiBearingMonitor(config_path)

    # 启动所有生产线
    monitor.start_all()

    try:
        # 持续监控
        print(f"\n{'='*70}")
        print("📊 实时监控数据（每秒更新）")
        print(f"{'='*70}\n")

        for i in range(duration):
            time.sleep(1)
            stats = monitor.get_aggregated_stats()

            # 清屏（可选）
            # print("\033[2J\033[H", end="")

            print(f"\r⏱️  {stats['elapsed_time']:.1f}s | "
                  f"总帧: {stats['total_frames']} | "
                  f"缺陷: {stats['total_defects']} | "
                  f"FPS: {stats['avg_fps']:.1f} | "
                  f"推理: {stats['avg_inference_ms']:.2f}ms", end="")

            # 每10秒显示详细信息
            if (i + 1) % 10 == 0:
                print()  # 换行
                for line_stat in stats['lines']:
                    print(f"  • {line_stat['name']}: "
                          f"{line_stat['total_frames']}帧, "
                          f"{line_stat['detected_defects']}缺陷, "
                          f"缺陷率 {line_stat['defect_rate']:.2f}%")

        print("\n")  # 最后换行

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")

    finally:
        # 停止所有生产线
        monitor.stop_all()

        # 显示最终统计
        final_stats = monitor.get_aggregated_stats()
        print("\n" + "="*70)
        print("📊 最终统计报告")
        print("="*70)
        print(f"运行时长: {final_stats['elapsed_time']:.1f}s")
        print(f"总检测帧数: {final_stats['total_frames']:,}")
        print(f"总检出缺陷: {final_stats['total_defects']}")
        print(f"总体缺陷率: {final_stats['overall_defect_rate']:.2f}%")
        print(f"平均吞吐量: {final_stats['avg_fps']:.1f} FPS")
        print(f"平均推理时间: {final_stats['avg_inference_ms']:.2f} ms")

        print("\n各生产线详情:")
        for line_stat in final_stats['lines']:
            print(f"\n  📍 {line_stat['name']}")
            print(f"     检测帧数: {line_stat['total_frames']:,}")
            print(f"     缺陷数量: {line_stat['detected_defects']}")
            print(f"     缺陷率: {line_stat['defect_rate']:.2f}%")
            print(f"     推理时间: {line_stat['avg_inference_time_ms']:.2f} ms")

            # 显示缺陷类型分布
            defect_types = line_stat['defect_types']
            if any(count > 0 for count in defect_types.values()):
                print("     缺陷类型分布:")
                for cls_id, count in defect_types.items():
                    if count > 0:
                        print(f"       - 类型 {cls_id}: {count} 次")

        print("="*70)
        print("✅ 测试完成！")


if __name__ == "__main__":
    main()

