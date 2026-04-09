"""Socket server for receiving video stream from edge device."""

import asyncio
import struct
import logging
from typing import Optional, Callable
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

# Protocol constants
FRAME_MAGIC = 0xABCD1234
UPLINK_HEADER_SIZE = 8  # magic(4) + jpeg_len(4)
MAX_JPEG_LEN = 512 * 1024  # 512 KB 上限，防止 DoS
READ_TIMEOUT = 10  # seconds


@dataclass
class UplinkFrame:
    """Uplink frame from edge device."""
    jpeg_data: bytes



class SocketServer:
    """Async socket server for edge device communication."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self.running = False
        self.latest_frame: Optional[UplinkFrame] = None
        self.frame_callback: Optional[Callable[[UplinkFrame], None]] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._lock = threading.Lock()

    def set_frame_callback(self, callback: Callable[[UplinkFrame], None]):
        """Set callback for new frames."""
        self.frame_callback = callback

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection."""
        if self.writer:
            logger.warning("Existing client connected; closing previous writer")
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass

        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected: {addr}")
        print(f"[SocketServer] Client connected: {addr}")
        self.writer = writer

        # Magic bytes in little-endian: 0xABCD1234 -> bytes: 34 12 CD AB
        MAGIC_BYTES = struct.pack('<I', FRAME_MAGIC)

        try:
            while self.running:
                # Read header with timeout
                header_data = await asyncio.wait_for(reader.readexactly(UPLINK_HEADER_SIZE), timeout=READ_TIMEOUT)
                # 使用小端字节序 '<' 匹配ARM边缘设备（ARM默认小端）
                magic, jpeg_len = struct.unpack('<I I', header_data)

                if magic != FRAME_MAGIC:
                    logger.warning(f"Invalid magic: {magic:08x}, attempting to resync...")
                    print(f"[SocketServer] Invalid magic: {magic:08x}, resyncing...")

                    # 尝试在缓冲区中找到正确的magic标记
                    # 将已读取的数据和新读取的数据一起搜索
                    buffer = header_data
                    found = False
                    max_search = 1024 * 1024  # 最多搜索1MB
                    searched = 0

                    while searched < max_search and self.running:
                        # 在buffer中查找magic
                        pos = buffer.find(MAGIC_BYTES)
                        if pos != -1:
                            # 找到了magic，丢弃前面的数据
                            buffer = buffer[pos:]
                            # 确保有完整的header
                            while len(buffer) < UPLINK_HEADER_SIZE:
                                more = await asyncio.wait_for(reader.read(UPLINK_HEADER_SIZE - len(buffer)), timeout=READ_TIMEOUT)
                                if not more:
                                    raise asyncio.IncompleteReadError(buffer, UPLINK_HEADER_SIZE)
                                buffer += more
                            # 重新解析
                            magic, jpeg_len = struct.unpack('<I I', buffer[:UPLINK_HEADER_SIZE])
                            if magic == FRAME_MAGIC:
                                print(f"[SocketServer] Resync successful! jpeg_len={jpeg_len}")
                                found = True
                                break

                        # 继续读取更多数据
                        try:
                            more = await asyncio.wait_for(reader.read(4096), timeout=READ_TIMEOUT)
                            if not more:
                                break
                            buffer += more
                            searched += len(more)
                            # 只保留最后8字节之前的搜索窗口
                            if len(buffer) > 8192:
                                buffer = buffer[-8192:]
                        except asyncio.TimeoutError:
                            break

                    if not found:
                        logger.error("Failed to resync, closing connection")
                        print("[SocketServer] Failed to resync, closing connection")
                        break
                    continue

                if jpeg_len <= 0 or jpeg_len > MAX_JPEG_LEN:
                    logger.warning(f"jpeg_len out of bound: {jpeg_len}")
                    print(f"[SocketServer] jpeg_len out of bound: {jpeg_len}")
                    break

                # Read JPEG data with timeout
                jpeg_data = await asyncio.wait_for(reader.readexactly(jpeg_len), timeout=READ_TIMEOUT)

                # Validate JPEG data (should start with FFD8)
                if len(jpeg_data) >= 2 and jpeg_data[0] == 0xFF and jpeg_data[1] == 0xD8:
                    # Create frame
                    frame = UplinkFrame(
                        jpeg_data=jpeg_data
                    )

                    with self._lock:
                        self.latest_frame = frame

                    # Call callback
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    logger.warning(f"Invalid JPEG data (not starting with FFD8)")
                    print(f"[SocketServer] Invalid JPEG data")

        except asyncio.TimeoutError:
            logger.warning(f"Client read timeout: {addr}")
            print(f"[SocketServer] Client read timeout: {addr}")
        except asyncio.IncompleteReadError:
            logger.info(f"Client disconnected: {addr}")
            print(f"[SocketServer] Client disconnected: {addr}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            print(f"[SocketServer] Error: {e}")
        finally:
            self.writer = None
            writer.close()
            await writer.wait_closed()


    async def start(self):
        """Start server."""
        import socket
        self.running = True
        try:
            self.server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port,
                reuse_address=True,  # 允许端口复用
            )
            logger.info(f"Socket server started on {self.host}:{self.port}")
            print(f"[SocketServer] Listening on {self.host}:{self.port}")
        except OSError as e:
            self.running = False
            logger.error(f"Failed to start server: {e}")
            print(f"[SocketServer] Failed to bind {self.host}:{self.port} - {e}")
            raise

    async def stop(self):
        """Stop server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("Socket server stopped")

    def get_latest_frame(self) -> Optional[UplinkFrame]:
        """Get latest frame (thread-safe, returns a copy)."""
        with self._lock:
            if not self.latest_frame:
                return None
            frame = self.latest_frame
        # Return a shallow copy with jpeg_data bytes copied to avoid caller mutation
        return UplinkFrame(
            jpeg_data=bytes(frame.jpeg_data) if frame.jpeg_data else b"",
        )


# Global server instance
_server_instance: Optional[SocketServer] = None
_server_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _run_server_loop(server: SocketServer):
    """Run server in background thread."""
    global _event_loop
    _event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_event_loop)
    try:
        _event_loop.run_until_complete(server.start())
        _event_loop.run_forever()
    except Exception as e:
        print(f"[SocketServer] Error in server loop: {e}")
        server.running = False


def start_server(host: str = "0.0.0.0", port: int = 8888) -> SocketServer:
    """Start socket server in background thread."""
    global _server_instance, _server_thread

    # 如果已有实例且正在运行，直接返回
    if _server_instance and _server_instance.running:
        print(f"[SocketServer] Already running on port {port}")
        return _server_instance

    # 清理旧实例
    if _server_instance:
        try:
            stop_server()
        except:
            pass

    _server_instance = SocketServer(host, port)
    _server_thread = threading.Thread(target=_run_server_loop, args=(_server_instance,), daemon=True)
    _server_thread.start()

    # 等待服务器启动
    import time
    time.sleep(0.5)

    if _server_instance.running:
        print(f"[SocketServer] Started successfully on {host}:{port}")
    else:
        print(f"[SocketServer] Failed to start on {host}:{port}")

    return _server_instance


def stop_server():
    """Stop socket server."""
    global _server_instance, _event_loop

    if _server_instance and _event_loop:
        asyncio.run_coroutine_threadsafe(_server_instance.stop(), _event_loop)
        _event_loop.call_soon_threadsafe(_event_loop.stop)
        if _server_thread:
            _server_thread.join(timeout=5)
        _server_instance = None


def get_server() -> Optional[SocketServer]:
    """Expose current server instance (may be None if not started)."""
    return _server_instance

