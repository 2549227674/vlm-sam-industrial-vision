#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <string>
#include <vector>

class VideoCapture {
public:
    VideoCapture(const std::string& device, int width, int height, int fps);
    ~VideoCapture();

    bool open();
    void close();
    bool captureFrame(std::vector<uint8_t>& jpeg_data);  // 直接返回 MJPEG 数据

private:
    std::string device_;
    int width_;
    int height_;
    int fps_;
    int fd_;
    void* buffers_;
    size_t buffer_count_;

    bool initDevice();
    bool initMmap();
    bool startCapture();
};

#endif // VIDEO_CAPTURE_H
