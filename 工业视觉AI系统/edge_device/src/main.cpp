#include "video_capture.h"
#include "socket_client.h"
#include "protocol.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>

volatile bool running = true;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal received. Shutting down..." << std::endl;
    running = false;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);

    std::string device = "/dev/video0";
    std::string host = "127.0.0.1";
    int port = 8888;

    if (argc >= 2) device = argv[1];
    if (argc >= 3) host = argv[2];
    if (argc >= 4) port = std::atoi(argv[3]);

    std::cout << "Edge Device Starting..." << std::endl;
    std::cout << "Video device: " << device << std::endl;
    std::cout << "Server: " << host << ":" << port << std::endl;

    VideoCapture capture(device, 1280, 720, 30);  // 720p HD MJPEG @ 30 FPS
    if (!capture.open()) {
        std::cerr << "Failed to open video device" << std::endl;
        return 1;
    }
    std::cout << "Video capture initialized (MJPEG mode @ 30 FPS)" << std::endl;

    SocketClient client(host, port);
    while (running && !client.connect()) {
        std::cout << "Retrying connection in 5 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    if (!running) {
        return 0;
    }

    std::cout << "Starting video streaming..." << std::endl;

    auto last_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    auto fps_start = std::chrono::steady_clock::now();

    while (running) {
        std::vector<uint8_t> jpeg_data;
        if (!capture.captureFrame(jpeg_data)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (!client.sendFrame(jpeg_data)) {
            std::cerr << "Failed to send frame, reconnecting..." << std::endl;
            while (running && !client.connect()) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
            continue;
        }

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fps_start).count();
        if (elapsed >= 5) {
            float fps = frame_count / (float)elapsed;
            std::cout << "FPS: " << fps << std::endl;
            frame_count = 0;
            fps_start = now;
        }

        auto frame_duration = std::chrono::milliseconds(33); // 30 FPS
        auto next_frame_time = last_frame_time + frame_duration;
        std::this_thread::sleep_until(next_frame_time);
        last_frame_time = next_frame_time;
    }

    std::cout << "Shutting down..." << std::endl;
    capture.close();
    client.disconnect();

    return 0;
}
