#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include <string>
#include <vector>
#include "protocol.h"

class SocketClient {
public:
    SocketClient(const std::string& host, int port);
    ~SocketClient();

    bool connect();
    void disconnect();
    bool sendFrame(const std::vector<uint8_t>& jpeg_data);
    bool isConnected() const { return sockfd_ >= 0; }

private:
    std::string host_;
    int port_;
    int sockfd_;

    bool reconnect();
};

#endif // SOCKET_CLIENT_H
