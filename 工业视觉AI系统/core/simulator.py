"""Production line simulator with real image upload."""

import threading
import time
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
from copy import deepcopy


@dataclass
class ProductionLine:
    """Production line configuration."""
    line_id: str
    name: str
    images: List[Image.Image] = None
    current_index: int = 0
    fps: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if self.images is None:
            self.images = []


class ProductionLineSimulator:
    """Simulates multiple production lines with uploaded images."""

    def __init__(self):
        self.lines = [
            ProductionLine("line_1", "生产线 1", fps=1.0),
            ProductionLine("line_2", "生产线 2", fps=0.8),
            ProductionLine("line_3", "生产线 3", fps=1.2),
        ]
        self.running = False
        self.threads = []
        self.frame_callback = None

        # Statistics
        self.stats = {line.line_id: {"frames": 0, "defects": 0} for line in self.lines}
        self.lock = threading.Lock()

    def set_frame_callback(self, callback):
        """Set callback for new frames."""
        self.frame_callback = callback

    def upload_images(self, line_id: str, images: List[Image.Image]):
        """Upload images to a production line."""
        for line in self.lines:
            if line.line_id == line_id:
                with self.lock:
                    line.images = images
                    line.current_index = 0
                break

    def start(self):
        """Start all production lines."""
        if self.running:
            return

        self.running = True
        for line in self.lines:
            if line.enabled and len(line.images) > 0:
                thread = threading.Thread(
                    target=self._simulate_line,
                    args=(line,),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)

    def stop(self):
        """Stop all production lines."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)
        self.threads = []

    def _simulate_line(self, line: ProductionLine):
        """Simulate a single production line."""
        while self.running:
            with self.lock:
                if line.current_index >= len(line.images):
                    line.current_index = 0  # Loop back

                if len(line.images) == 0:
                    time.sleep(1)
                    continue

                image = line.images[line.current_index]
                line.current_index += 1

                # Update statistics
                self.stats[line.line_id]["frames"] += 1

            # Call callback with a shallow copy to avoid external mutation of stored images
            if self.frame_callback:
                self.frame_callback(line.line_id, image.copy())

            # Sleep according to FPS
            time.sleep(1.0 / line.fps)

    def get_line_stats(self, line_id: str) -> dict:
        """Get statistics for a specific line."""
        with self.lock:
            stats = self.stats.get(line_id, {"frames": 0, "defects": 0})
            frames = stats["frames"]
            defects = stats["defects"]
            defect_rate = (defects / frames * 100) if frames > 0 else 0.0

            # Get image count
            line = next((l for l in self.lines if l.line_id == line_id), None)
            image_count = len(line.images) if line else 0

            return {
                "frames": frames,
                "defects": defects,
                "defect_rate": defect_rate,
                "image_count": image_count
            }

    def update_defect_count(self, line_id: str, has_defect: bool):
        """Update defect count for a line."""
        if has_defect:
            with self.lock:
                self.stats[line_id]["defects"] += 1

    def get_all_stats(self) -> dict:
        """Get statistics for all lines."""
        return {line.line_id: self.get_line_stats(line.line_id) for line in self.lines}

    def get_lines(self) -> list:
        """Get list of production lines."""
        return self.lines

    def set_line_enabled(self, line_id: str, enabled: bool):
        """Enable or disable a production line."""
        for line in self.lines:
            if line.line_id == line_id:
                line.enabled = enabled
                break


# Global simulator instance
_simulator: Optional[ProductionLineSimulator] = None


def get_simulator() -> ProductionLineSimulator:
    """Get or create global simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = ProductionLineSimulator()
    return _simulator


def stop_simulator():
    """Stop global simulator."""
    global _simulator
    if _simulator:
        _simulator.stop()
        _simulator = None
