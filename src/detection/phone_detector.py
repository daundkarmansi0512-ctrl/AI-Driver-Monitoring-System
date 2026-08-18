"""
Phone detection using YOLOv8 (ultralytics) with asynchronous background processing.

Uses a pretrained YOLO nano model to detect "cell phone" (COCO class 67)
in the video frame.

Performance Features:
- Asynchronous worker thread: YOLO inference runs entirely off the main thread.
- Frame resizing: inference runs on downscaled frames (e.g. 320px) to minimize CPU usage.
- Throttled inference: limits inference rate (e.g. max 3-4 calls/sec) while main loop runs at 30 FPS.
- Persistence duration: phone must be continuously visible for a configurable duration (e.g. 1.5s)
  before an alert fires.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class PhoneDetector:
    """Detect cell phone usage via YOLOv8 object detection with async support."""

    # "cell phone" is class index 67 in the standard COCO dataset
    CELL_PHONE_CLASS = 67

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.5,
        phone_duration: float = 1.5,
        async_interval: float = 0.25,
    ) -> None:
        """
        Args:
            model_name:
                YOLO model file (e.g. "yolov8n.pt" nano model).
            confidence:
                Minimum detection confidence to accept.
            phone_duration:
                Seconds the phone must be continuously visible before the alert fires.
            async_interval:
                Minimum seconds between background YOLO inferences to prevent CPU overload.
        """

        self.confidence = confidence
        self.phone_duration = phone_duration
        self.async_interval = async_interval

        # Temporal alert state
        self._phone_visible_start: float = 0.0
        self._phone_currently_visible: bool = False
        self._is_phone_detected: bool = False
        self._alert_fired: bool = False

        # Model state
        self._model = None
        self._available = False

        # Asynchronous worker state
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()
        self._is_worker_running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._latest_detected = False
        self._last_request_time = 0.0

        # Profiling stats
        self._last_inference_ms = 0.0
        self._max_inference_ms = 0.0
        self._total_inference_ms = 0.0
        self._inference_count = 0

        # Try to load YOLO
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_name)

            names = self._model.names
            if self.CELL_PHONE_CLASS in names:
                phone_label = names[self.CELL_PHONE_CLASS]
                print(
                    f"✅ Phone detector ready "
                    f"(class {self.CELL_PHONE_CLASS} = '{phone_label}')."
                )
                self._available = True
                self._start_worker()
            else:
                print(
                    f"⚠️ Model does not contain class {self.CELL_PHONE_CLASS}. "
                    f"Phone detection disabled."
                )

        except ImportError:
            print("⚠️ ultralytics not installed. Phone detection disabled.")
        except Exception as exc:
            print(f"⚠️ Could not load YOLO model: {exc}. Phone detection disabled.")

    @property
    def available(self) -> bool:
        """True if the YOLO model loaded successfully."""
        return self._available

    # --------------------------------------------------
    # Asynchronous Worker
    # --------------------------------------------------

    def _start_worker(self) -> None:
        """Start the background inference worker thread."""
        if not self._available:
            return

        self._is_worker_running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PhoneDetectionWorker",
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Continuous background loop processing incoming frames."""
        while self._is_worker_running:
            self._new_frame_event.wait(timeout=0.5)
            if not self._is_worker_running:
                break

            with self._lock:
                frame_to_process = self._pending_frame
                self._pending_frame = None
                self._new_frame_event.clear()

            if frame_to_process is None:
                continue

            t0 = time.perf_counter()
            detected = self._run_model_inference(frame_to_process)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            with self._lock:
                self._latest_detected = detected
                self._last_inference_ms = elapsed_ms
                if elapsed_ms > self._max_inference_ms:
                    self._max_inference_ms = elapsed_ms
                self._total_inference_ms += elapsed_ms
                self._inference_count += 1

    def request_async_detection(
        self,
        frame: np.ndarray,
        resize_width: int = 320,
    ) -> None:
        """
        Pass a frame to the background worker if it is ready.
        Non-blocking and takes ~0ms on the main loop.
        """
        if not self._available or not self._is_worker_running:
            return

        now = time.time()
        if now - self._last_request_time < self.async_interval:
            return

        # Prepare downscaled frame for the worker
        infer_frame = frame
        if resize_width > 0 and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            infer_frame = cv2.resize(frame, (resize_width, new_h))
        else:
            infer_frame = frame.copy()

        with self._lock:
            # If the worker hasn't picked up the previous frame yet, overwrite it
            self._pending_frame = infer_frame
            self._last_request_time = now
            self._new_frame_event.set()

    @property
    def latest_phone_visible(self) -> bool:
        """Return the latest detection state reported by the background worker."""
        with self._lock:
            return self._latest_detected

    def get_profiling_stats(self) -> Tuple[float, float, float]:
        """Return (last_ms, avg_ms, max_ms) for background YOLO inference."""
        with self._lock:
            avg_ms = (
                (self._total_inference_ms / self._inference_count)
                if self._inference_count > 0
                else 0.0
            )
            return self._last_inference_ms, avg_ms, self._max_inference_ms

    # --------------------------------------------------
    # Core Inference (Worker internal)
    # --------------------------------------------------

    def _run_model_inference(self, frame: np.ndarray) -> bool:
        """Run YOLO model on a single frame."""
        if not self._available or self._model is None:
            return False

        try:
            results = self._model.predict(
                frame,
                classes=[self.CELL_PHONE_CLASS],
                conf=self.confidence,
                verbose=False,
            )
            if results and len(results[0].boxes) > 0:
                return True
        except Exception:
            pass

        return False

    def detect(self, frame: np.ndarray, resize_width: int = 0) -> bool:
        """Synchronous detection fallback."""
        if not self._available:
            return False

        infer_frame = frame
        if resize_width > 0 and frame.shape[1] > resize_width:
            scale = resize_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            infer_frame = cv2.resize(frame, (resize_width, new_h))

        return self._run_model_inference(infer_frame)

    # --------------------------------------------------
    # Temporal Alert State Management
    # --------------------------------------------------

    def update(self, phone_visible: bool) -> bool:
        """
        Apply duration filter to latest detection state.

        Args:
            phone_visible:
                True if phone is currently visible.

        Returns:
            True ONCE when phone usage is first confirmed
            (after duration threshold).  The visual state
            ``is_phone_detected`` stays active until the
            phone disappears.
        """
        if phone_visible:
            if not self._phone_currently_visible:
                self._phone_currently_visible = True
                self._phone_visible_start = time.time()
                self._alert_fired = False
                return False

            elapsed = time.time() - self._phone_visible_start

            if elapsed >= self.phone_duration:
                self._is_phone_detected = True

                if not self._alert_fired:
                    self._alert_fired = True
                    return True
        else:
            self._phone_currently_visible = False
            self._phone_visible_start = 0.0
            self._is_phone_detected = False
            self._alert_fired = False

        return False

    @property
    def is_phone_detected(self) -> bool:
        """True while phone has been visible past the duration limit."""
        return self._is_phone_detected

    def close(self) -> None:
        """Stop background worker thread."""
        self._is_worker_running = False
        self._new_frame_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
