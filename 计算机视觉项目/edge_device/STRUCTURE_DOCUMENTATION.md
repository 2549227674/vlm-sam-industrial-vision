# 📁 Edge Device 结构功能文档

## 📋 概述

`edge_device` 是一个基于 **i.MX6ULL** 嵌入式平台的边缘端视频采集与传输模块，使用 **C++** 开发。该模块作为"五范式工业视觉分析系统"中**范式D（边缘-云协同实时检测）** 的边缘端组件，负责实时视频采集和云端通信。

**核心特点：**
- 🚀 **零依赖 JPEG 编码** - 摄像头硬件直接输出 MJPEG，无需 libjpeg
- ⚡ **高性能** - 640×480 @ 30 FPS
- 🔧 **交叉编译友好** - 仅依赖 pthread

---

## 📂 目录结构

```
edge_device/
├── CMakeLists.txt              # CMake 构建配置
├── arm-linux-gnueabihf.cmake   # 交叉编译工具链配置
├── cross_compile.sh            # 交叉编译脚本
├── README.md                   # 项目说明
├── STRUCTURE_DOCUMENTATION.md  # 本文档
└── src/                        # 源代码目录
    ├── main.cpp                # 程序入口
    ├── protocol.h              # 通信协议定义
    ├── video_capture.h         # V4L2 视频采集头文件
    ├── video_capture.cpp       # V4L2 MJPEG 视频采集实现
    ├── socket_client.h         # Socket 客户端头文件
    └── socket_client.cpp       # TCP Socket 通信实现
```

---

## 🎯 工作原理

### MJPEG 硬件编码（零 CPU 开销）

```
┌─────────────────────────────────────────────────────┐
│                  USB 摄像头                          │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│  │  CMOS     │ →  │  硬件     │ →  │  MJPEG    │   │
│  │  传感器   │    │  JPEG编码 │    │  输出     │   │
│  └───────────┘    └───────────┘    └───────────┘   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ USB 传输
┌─────────────────────────────────────────────────────┐
│               i.MX6ULL 边缘设备                      │
│                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│  │   V4L2    │ →  │  直接     │ →  │  TCP      │   │
│  │   采集    │    │  转发     │    │  发送     │   │
│  └───────────┘    └───────────┘    └───────────┘   │
│                                                     │
│  ✅ 无 JPEG 编码开销（摄像头已编码）                  │
│  ✅ 无 libjpeg 依赖                                  │
│  ✅ CPU 占用极低                                     │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ TCP:8888
┌─────────────────────────────────────────────────────┐
│              云端服务器 (RTX 4060)                   │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐   │
│  │  Socket   │ →  │  JPEG     │ →  │  YOLOv8   │   │
│  │  接收     │    │  解码     │    │  推理     │   │
│  └───────────┘    └───────────┘    └───────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 🧩 模块功能详解

### 1️⃣ `main.cpp` - 程序入口

**功能**：采集 MJPEG 视频帧 → 发送到云端服务器

```cpp
// 工作流程
初始化摄像头(MJPEG 模式) → 连接云端服务器 → 循环 {
    从 V4L2 读取 MJPEG 帧（已是 JPEG 格式）
    通过 TCP 发送到云端
}
```

**命令行参数**：
```bash
./edge_device [设备] [服务器IP] [端口]
./edge_device /dev/video1 192.168.1.100 8888
```

### 2️⃣ `video_capture` - V4L2 MJPEG 采集

基于原生 V4L2 API，直接采集摄像头的 MJPEG 输出。

| 特性 | 值 |
|------|-----|
| 分辨率 | 640 × 480 |
| 帧率 | 30 FPS |
| 格式 | **MJPEG（摄像头硬件编码）** |
| 缓冲区 | 4 个 mmap 缓冲区 |

**关键代码**：
```cpp
// 设置 MJPEG 格式
fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;

// captureFrame() 直接返回 JPEG 数据，无需转换
```

### 3️⃣ `protocol.h` - 通信协议定义

**上行帧结构**（边缘 → 云端）：
```cpp
struct UplinkFrame {
    uint32_t magic;          // 帧头魔数 0xABCD1234
    uint32_t jpeg_length;    // JPEG 数据长度
    // JPEG 数据紧随其后
};
```

### 4️⃣ `socket_client` - TCP Socket 客户端

与云端服务器通信。

| 特性 | 值 |
|------|-----|
| 协议 | TCP/IP |
| 发送超时 | 5 秒 |
| 默认端口 | 8888 |
| 自动重连 | 是（5秒间隔） |

---

## 📈 性能指标

| 指标 | 值 |
|------|------------|
| 分辨率 | 640×480 |
| 帧率 | **30 FPS** |
| JPEG 大小 | ~20-50 KB/帧 |
| 带宽需求 | ~1-1.5 MB/s |
| CPU 占用 | **极低**（无编码开销） |
| 内存占用 | < 5 MB |
| 依赖库 | **仅 pthread** |

---

## 🔧 编译

### 本地编译（开发板上）
```bash
cd edge_device
mkdir build && cd build
cmake ..
make
```

### 交叉编译（Ubuntu 虚拟机）
```bash
# 1. 安装交叉编译器
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# 2. 运行编译脚本
chmod +x cross_compile.sh
./cross_compile.sh

# 3. 复制到开发板
scp build_arm/edge_device root@<板子IP>:/root/
```

---

## 🚀 运行

```bash
# 在开发板上运行
./edge_device /dev/video1 192.168.1.100 8888

# 输出示例：
# Edge Device Starting...
# Video device: /dev/video1
# Server: 192.168.1.100:8888
# Actual resolution: 640x480
# Video capture initialized (MJPEG mode @ 30 FPS)
# Connected to server
# Starting video streaming...
# FPS: 29.8
```

---

## 🔗 与项目整体的关系

```
计算机视觉项目/
├── app_final.py              # Streamlit 云端界面
├── core/
│   ├── socket_server.py      # 接收 edge_device 的视频流
│   └── ...
├── ui/
│   └── paradigm_d.py         # 范式D 界面
└── edge_device/              # 本模块
    └── edge_device           # → socket_server.py
```

---

## 📝 开发状态

### ✅ 已完成
- [x] V4L2 MJPEG 视频采集
- [x] TCP Socket 通信
- [x] 交叉编译支持
- [x] 自动重连机制

### ❌ 已移除（简化架构）
- ~~云台控制~~
- ~~温湿度传感器~~
- ~~LED 控制~~
- ~~HTTP MJPEG 流媒体~~
- ~~OpenCV 依赖~~
- ~~libjpeg 依赖~~

---

## 📚 参考资料

- **V4L2 API**: https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/v4l2.html
- **MJPEG 格式**: https://en.wikipedia.org/wiki/Motion_JPEG
