"""Camera management utilities for the driver monitoring system.

This module provides a simple OpenCV-based wrapper for opening a webcam,
reading frames, and releasing the camera safely.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CameraManager:
    """Manage webcam access in a simple and beginner-friendly way.

    The class opens a camera device, reads frames when requested,
    and releases system resources when the camera is no longer needed.

    Args:
        camera_index: The index of the webcam to open (usually 0).
        width: Optional desired frame width.
        height: Optional desired frame height.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.camera_index: int = camera_index
        self.width: Optional[int] = width
        self.height: Optional[int] = height
        self.capture: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Open the camera and prepare it for capturing frames.

        Returns:
            True if the camera starts successfully.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """

        # Camera is already open
        if self.capture is not None and self.capture.isOpened():
            return True

        # Open the camera using DirectShow backend (better on Windows)
        self.capture = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        # Check if camera opened successfully
        if not self.capture.isOpened():
            self.capture = None
            raise RuntimeError(
                f"Failed to open camera (Index: {self.camera_index})."
            )

        # Reduce frame buffering (helps prevent frozen frames)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set camera resolution if specified
        if self.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        if self.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera.

        Returns:
            A NumPy array containing the captured frame,
            or None if no valid frame could be read.

        Raises:
            RuntimeError: If an unexpected camera error occurs.
        """

        try:
            if self.capture is None:
                self.start()

            if self.capture is None or not self.capture.isOpened():
                return None

            success, frame = self.capture.read()

            if not success or frame is None:
                return None

            return frame

        except Exception as exc:
            raise RuntimeError(
                f"Error reading camera frame: {exc}"
            ) from exc

    def release(self) -> None:
        """Release the camera and free system resources."""

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            finally:
                self.capture = None