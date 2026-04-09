"""Thread pool manager for concurrent inference tasks."""

import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Inference task."""
    task_id: str
    line_id: str
    image_data: Any
    prompt: dict  # Changed from str to dict to support configuration
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    # 添加可选上下文，用于传递模型句柄与设备信息，避免后台线程访问 Streamlit 状态
    context: Optional[dict] = None


class ThreadPoolManager:
    """Thread pool manager for concurrent inference."""

    def __init__(self, max_workers: int = 4, max_queue_size: int = 64, rate_limit_per_sec: float = 10.0):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue(maxsize=max_queue_size)
        self.active_tasks = {}
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread = None

        # Metrics
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_timeouts = 0
        self.last_error: Optional[str] = None

        # Rate limiting (token bucket)
        self.rate_limit_per_sec = rate_limit_per_sec
        self._tokens = rate_limit_per_sec
        self._last_refill = time.time()

    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        max_tokens = self.rate_limit_per_sec * 2  # allow small burst
        self._tokens = min(max_tokens, self._tokens + elapsed * self.rate_limit_per_sec)
        self._last_refill = now

    def start(self):
        """Start the thread pool manager."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the thread pool manager."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        # 取消活跃任务
        with self.lock:
            for _, fut in list(self.active_tasks.values()):
                fut.cancel()
            self.active_tasks.clear()
        # 清空队列
        try:
            while True:
                self.task_queue.get_nowait()
        except Empty:
            pass
        self.executor.shutdown(wait=True)

    def submit_task(self, task: Task) -> str:
        """Submit a task to the queue."""
        with self.lock:
            self._refill_tokens()
            if self._tokens < 1:
                raise RuntimeError("rate limit exceeded; please slow down and retry")
            self._tokens -= 1

        if self.task_queue.full():
            raise RuntimeError("task queue is full; please retry later")

        task.submit_time = time.time()
        task.status = TaskStatus.PENDING
        self.task_queue.put(task)

        with self.lock:
            self.total_submitted += 1

        return task.task_id

    def _process_queue(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                future = self.executor.submit(self._execute_task, task)

                with self.lock:
                    self.active_tasks[task.task_id] = (task, future)

            except Empty:
                continue
            except Exception as e:
                print(f"Error processing queue: {e}")

    def _execute_task(self, task: Task):
        """Execute a single task."""
        task.start_time = time.time()
        task.status = TaskStatus.RUNNING

        try:
            # Paradigm C: VLM bbox inference
            from ui.adapters import get_vlm_defect_bboxes
            from core.bbox_draw import draw_bboxes_on_image

            context = task.context or {}
            device = context.get("device", "cpu")
            sam_proc = context.get("sam_proc")
            sam_model = context.get("sam_model")
            sam_dtype = context.get("sam_dtype")

            # Get VLM bbox detections
            vlm_result = get_vlm_defect_bboxes(
                image_pil=task.image_data,
                model_name=task.prompt.get("vlm_model", "qwen-vl-max"),
                api_key=context.get("dashscope_api_key"),
                max_boxes=task.prompt.get("max_boxes", 3)
            )

            # Normalize detection list for both dict and dataclass output
            detections = []
            if vlm_result:
                detections = getattr(vlm_result, "detections", None) or vlm_result.get("detections", [])

            # Draw bboxes on image
            processed_image = task.image_data
            if detections:
                bboxes = []
                labels = []
                for det in detections:
                    if isinstance(det, dict):
                        bboxes.append(det.get("bbox_xyxy") or det.get("bbox"))
                        labels.append(det.get("label", det.get("defect_type", "defect")))
                    else:
                        bboxes.append(getattr(det, "bbox_xyxy", None))
                        labels.append(getattr(det, "defect_type", "defect"))
                bboxes = [b for b in bboxes if b]
                if bboxes:
                    processed_image = draw_bboxes_on_image(task.image_data, bboxes, labels)

            # Optional SAM inference
            if task.prompt.get("enable_sam", False) and detections:
                try:
                    from ui.adapters import run_sam3_box_prompt_instance_segmentation
                    from core.bbox_draw import draw_sam_masks_on_image

                    if sam_proc and sam_model:
                        bboxes = []
                        for det in detections:
                            if isinstance(det, dict):
                                bboxes.append(det.get("bbox_xyxy") or det.get("bbox"))
                            else:
                                bboxes.append(getattr(det, "bbox_xyxy", None))
                        bboxes = [b for b in bboxes if b]
                        if bboxes:
                            sam_result, _ = run_sam3_box_prompt_instance_segmentation(
                                image_pil=task.image_data,
                                sam_proc=sam_proc,
                                sam_model=sam_model,
                                sam_dtype=sam_dtype,
                                boxes_xyxy=bboxes,
                                threshold=task.prompt.get("sam_threshold", 0.5),
                                mask_threshold=task.prompt.get("mask_threshold", 0.5),
                                device=device
                            )

                            if sam_result and sam_result.get("masks"):
                                processed_image = draw_sam_masks_on_image(
                                    task.image_data,
                                    sam_result["masks"],
                                    bboxes,
                                    alpha=0.4
                                )
                except Exception as sam_error:
                    print(f"SAM inference failed: {sam_error}")
                    # Continue with VLM-only result

            # Determine if defects detected
            has_defect = bool(detections)

            task.result = {
                "detections": detections or [],
                "decision": getattr(vlm_result, "decision", None) or (vlm_result.get("decision") if vlm_result else "unknown") or "unknown",
                "evidence": getattr(vlm_result, "evidence", None) or (vlm_result.get("evidence") if vlm_result else "") or "",
                "has_defect": has_defect,
                "processed_image": processed_image
            }
            task.status = TaskStatus.COMPLETED

            with self.lock:
                self.total_completed += 1

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

            with self.lock:
                self.total_failed += 1
                self.last_error = str(e)

        finally:
            task.end_time = time.time()

            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)

                # Keep only last 100 completed tasks
                if len(self.completed_tasks) > 100:
                    self.completed_tasks = self.completed_tasks[-100:]

    def get_metrics(self) -> dict:
        """Get current metrics."""
        with self.lock:
            queue_size = self.task_queue.qsize()
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)

            # Calculate utilization
            utilization = (active_count / self.max_workers) * 100 if self.max_workers > 0 else 0

            # Calculate average processing time
            if self.completed_tasks:
                avg_time = sum(t.end_time - t.start_time for t in self.completed_tasks[-20:]) / min(20, len(self.completed_tasks))
            else:
                avg_time = 0.0

            return {
                "queue_size": queue_size,
                "active_tasks": active_count,
                "max_workers": self.max_workers,
                "utilization": utilization,
                "total_submitted": self.total_submitted,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "total_timeouts": self.total_timeouts,
                "avg_processing_time": avg_time,
                "completed_tasks": completed_count,
                "rate_limit_per_sec": self.rate_limit_per_sec,
                "last_error": self.last_error,
            }

    def get_active_tasks(self) -> list:
        """Get list of active tasks."""
        with self.lock:
            return [task for task, _ in self.active_tasks.values()]

    def get_completed_tasks(self, limit: int = 20) -> list:
        """Get recent completed tasks."""
        with self.lock:
            return self.completed_tasks[-limit:]


# Global thread pool instance
_thread_pool: Optional[ThreadPoolManager] = None


def get_thread_pool(max_workers: int = 4, max_queue_size: int = 64) -> ThreadPoolManager:
    """Get or create global thread pool instance."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolManager(max_workers=max_workers, max_queue_size=max_queue_size)
        _thread_pool.start()
    return _thread_pool


def stop_thread_pool():
    """Stop global thread pool."""
    global _thread_pool
    if _thread_pool:
        _thread_pool.stop()
        _thread_pool = None
