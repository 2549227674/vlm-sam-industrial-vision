#include "socket_client.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>

SocketClient::SocketClient(const std::string& host, int port)
    : host_(host), port_(port), sockfd_(-1) {}

SocketClient::~SocketClient() {
    disconnect();
}

bool SocketClient::connect() {
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }

    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sockfd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    tv.tv_sec = 10;
    setsockopt(sockfd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << host_ << std::endl;
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    if (::connect(sockfd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to " << host_ << ":" << port_ << std::endl;
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    std::cout << "Connected to " << host_ << ":" << port_ << std::endl;
    return true;
}

void SocketClient::disconnect() {
    if (sockfd_ >= 0) {
        ::close(sockfd_);
        sockfd_ = -1;
    }
}

bool SocketClient::reconnect() {
    disconnect();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return connect();
}

bool SocketClient::sendFrame(const std::vector<uint8_t>& jpeg_data) {
    if (!isConnected()) {
        return false;
    }

    UplinkFrame frame;
    frame.magic = FRAME_MAGIC;
    frame.jpeg_length = jpeg_data.size();

    ssize_t sent = send(sockfd_, &frame, sizeof(frame), 0);
    if (sent != sizeof(frame)) {
        std::cerr << "Failed to send frame header" << std::endl;
        return false;
    }

    size_t total_sent = 0;
    while (total_sent < jpeg_data.size()) {
        sent = send(sockfd_, jpeg_data.data() + total_sent, jpeg_data.size() - total_sent, 0);
        if (sent <= 0) {
            std::cerr << "Failed to send JPEG data" << std::endl;
            return false;
        }
        total_sent += sent;
    }

    return true;
}

