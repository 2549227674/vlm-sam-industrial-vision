# Edge Device - 边缘端视频采集与传输

## 项目概述

这是基于 i.MX6ULL 的边缘端视频采集与传输模块，提供两种运行模式：

| 模式 | 可执行文件 | 用途 |
|------|-----------|------|
| Socket 模式 | `edge_device` | 发送到云端服务器进行 AI 推理 |
| MJPEG 模式 | `edge_device_mjpeg` | 浏览器直接查看视频流 |

## 功能特性

- V4L2 视频采集（640x480 @ 20 FPS）
- JPEG 压缩（质量 85）
- TCP Socket 通信（自定义协议）
- HTTP MJPEG 流媒体（浏览器兼容）

## 系统要求

### 硬件
- i.MX6ULL 开发板
- USB 摄像头或 CSI 摄像头

### 软件依赖
- Linux 内核（支持 V4L2）
- CMake >= 3.10
- GCC/G++ 编译器
- libjpeg-dev（Socket 模式必需）
- OpenCV（MJPEG 模式可选）

## 编译安装

### 方式1: 本机编译（在开发板上直接编译）

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install build-essential cmake libjpeg-dev

# 可选：安装 OpenCV（用于 MJPEG 模式）
sudo apt-get install libopencv-dev

# 编译
cd edge_device
mkdir build && cd build
cmake ..
make
```

### 方式2: 交叉编译（在 Ubuntu 虚拟机上编译）

**适用于在 PC/虚拟机上编译，然后拷贝到 i.MX6ULL 开发板运行。**

#### 步骤 1: 安装交叉编译器

```bash
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
```

#### 步骤 2: 安装 ARM 版本的 libjpeg

```bash
# 方法 A: 使用 multiarch（推荐）
sudo dpkg --add-architecture armhf
sudo apt-get update
sudo apt-get install libjpeg-dev:armhf

# 方法 B: 从开发板拷贝库文件
# 在开发板上: scp /usr/lib/arm-linux-gnueabihf/libjpeg* user@虚拟机IP:/tmp/
# 在虚拟机上: sudo cp /tmp/libjpeg* /usr/arm-linux-gnueabihf/lib/
```

#### 步骤 3: 交叉编译

```bash
cd edge_device

# 使用脚本（推荐）
chmod +x cross_compile.sh
./cross_compile.sh

# 或手动执行
mkdir build_arm && cd build_arm
cmake -DCMAKE_TOOLCHAIN_FILE=../arm-linux-gnueabihf.cmake ..
make
```

#### 步骤 4: 拷贝到开发板

```bash
scp build_arm/edge_device root@<板子IP>:/root/
```

#### 步骤 5: 在开发板上运行

```bash
chmod +x edge_device
./edge_device /dev/video0 <云端IP> 8888
```

编译完成后会生成：
- `edge_device` - Socket 模式（始终生成）
- `edge_device_mjpeg` - MJPEG 模式（需要 OpenCV）

## 运行

### Socket 模式（发送到云端服务器）

```bash
./edge_device [video_device] [server_host] [server_port]

# 示例
./edge_device /dev/video0 192.168.1.100 8888
```

### MJPEG 模式（浏览器直接查看）

```bash
./edge_device_mjpeg [video_device] [port]

# 示例
./edge_device_mjpeg /dev/video0 8080

# 然后在浏览器打开：http://<板子IP>:8080
```

## 通信协议

### Socket 模式 - 上行消息（边缘 -> 云端）

```c
struct UplinkFrame {
    uint32_t magic;          // 帧头魔数 0xABCD1234
    uint32_t jpeg_length;    // JPEG 数据长度
    // JPEG 数据紧随其后
};
```

### MJPEG 模式 - HTTP 响应

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg
Content-Length: <size>

<JPEG data>
```

## 性能指标

| 指标 | Socket 模式 | MJPEG 模式 |
|------|------------|------------|
| 分辨率 | 640x480 | 640x480 |
| 帧率 | 20 FPS | 30 FPS |
| 延迟 | 低 | 中等 |
| 多客户端 | 否 | 是 |

## 故障排查

### 1. 无法打开视频设备

```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
v4l2-ctl --device=/dev/video0 --all
```

### 2. 连接服务器失败（Socket 模式）

- 检查服务器地址和端口
- 确认云端服务器已启动
- 检查防火墙设置

### 3. 浏览器无法访问（MJPEG 模式）

- 确认端口未被占用
- 检查防火墙是否开放端口
- 尝试使用不同浏览器

## 开发状态

### 已完成
- [x] V4L2 视频采集
- [x] JPEG 压缩
- [x] Socket 通信
- [x] HTTP MJPEG 流媒体

### 待实现
- [ ] 自动重连优化
- [ ] 错误日志记录
- [ ] 多摄像头支持

## 许可证

本项目为教育和研究目的开发。

## 联系方式

如有问题，请联系项目维护者。
