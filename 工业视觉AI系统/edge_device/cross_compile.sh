#!/bin/bash
#
# 交叉编译脚本 - 在 Ubuntu 虚拟机上编译 i.MX6ULL 程序
#
# 前提条件：
#   1. 安装交叉编译器: sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
#   2. 安装 ARM 版本的依赖库（见下方说明）
#

set -e

echo "=========================================="
echo "  Edge Device 交叉编译 (i.MX6ULL)"
echo "=========================================="

# 检查交叉编译器
if ! command -v arm-linux-gnueabihf-g++ &> /dev/null; then
    echo "错误: 未找到交叉编译器"
    echo "请安装: sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf"
    exit 1
fi

echo "交叉编译器: $(which arm-linux-gnueabihf-g++)"

# 创建构建目录
BUILD_DIR="build_arm"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 运行 CMake
echo ""
echo "运行 CMake..."
cmake -DCMAKE_TOOLCHAIN_FILE=../arm-linux-gnueabihf.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# 编译
echo ""
echo "编译中..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "  编译完成!"
echo "=========================================="
echo ""
echo "生成的可执行文件:"
ls -la edge_device* 2>/dev/null || echo "  (无)"
echo ""
echo "复制到开发板:"
echo "  scp edge_device root@<板子IP>:/root/"
echo ""
echo "在板子上运行:"
echo "  ./edge_device /dev/video1 192.168.1.10 8888"
echo ""

