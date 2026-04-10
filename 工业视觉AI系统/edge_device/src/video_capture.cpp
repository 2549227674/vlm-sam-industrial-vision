#include "video_capture.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cstring>
#include <iostream>

struct Buffer {
    void* start;
    size_t length;
};

VideoCapture::VideoCapture(const std::string& device, int width, int height, int fps)
    : device_(device), width_(width), height_(height), fps_(fps), fd_(-1), buffers_(nullptr), buffer_count_(0) {}

VideoCapture::~VideoCapture() {
    close();
}

bool VideoCapture::open() {
    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "Failed to open device: " << device_ << std::endl;
        return false;
    }

    if (!initDevice() || !initMmap() || !startCapture()) {
        close();
        return false;
    }

    return true;
}

void VideoCapture::close() {
    if (fd_ >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);

        if (buffers_) {
            Buffer* bufs = static_cast<Buffer*>(buffers_);
            for (size_t i = 0; i < buffer_count_; ++i) {
                munmap(bufs[i].start, bufs[i].length);
            }
            delete[] bufs;
            buffers_ = nullptr;
        }

        ::close(fd_);
        fd_ = -1;
    }
}

bool VideoCapture::initDevice() {
    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "Failed to query device capabilities" << std::endl;
        return false;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width_;
    fmt.fmt.pix.height = height_;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;  // 直接使用摄像头硬件 MJPEG 编码
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        std::cerr << "Failed to set format" << std::endl;
        return false;
    }

    // 更新实际设置的分辨率
    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    std::cout << "Actual resolution: " << width_ << "x" << height_ << std::endl;

    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps_;

    if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        std::cerr << "Warning: Failed to set frame rate" << std::endl;
        // 不返回 false，帧率设置失败不是致命错误
    }

    return true;
}

bool VideoCapture::initMmap() {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        std::cerr << "Failed to request buffers" << std::endl;
        return false;
    }

    buffer_count_ = req.count;
    Buffer* bufs = new Buffer[buffer_count_];
    buffers_ = bufs;

    for (size_t i = 0; i < buffer_count_; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "Failed to query buffer" << std::endl;
            return false;
        }

        bufs[i].length = buf.length;
        bufs[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

        if (bufs[i].start == MAP_FAILED) {
            std::cerr << "Failed to mmap buffer" << std::endl;
            return false;
        }
    }

    return true;
}

bool VideoCapture::startCapture() {
    for (size_t i = 0; i < buffer_count_; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "Failed to queue buffer" << std::endl;
            return false;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        std::cerr << "Failed to start streaming" << std::endl;
        return false;
    }

    return true;
}

// 直接读取 MJPEG 帧（摄像头已经编码好）
bool VideoCapture::captureFrame(std::vector<uint8_t>& jpeg_data) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        return false;
    }

    Buffer* bufs = static_cast<Buffer*>(buffers_);
    // 摄像头输出的数据已经是 JPEG 格式，直接使用
    jpeg_data.assign(static_cast<uint8_t*>(bufs[buf.index].start),
                     static_cast<uint8_t*>(bufs[buf.index].start) + buf.bytesused);

    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        std::cerr << "Failed to requeue buffer" << std::endl;
        return false;
    }

    return true;
}

