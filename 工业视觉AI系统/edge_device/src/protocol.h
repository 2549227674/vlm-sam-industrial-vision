#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <cstdint>

// Frame header magic number
const uint32_t FRAME_MAGIC = 0xABCD1234;

// Uplink message: Edge -> Cloud
struct __attribute__((packed)) UplinkFrame {
    uint32_t magic;           // Frame header magic
    uint32_t jpeg_length;     // JPEG data length
    // JPEG data follows
};


#endif // PROTOCOL_H