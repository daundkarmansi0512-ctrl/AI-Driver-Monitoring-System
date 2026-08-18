"""Face detection module using MediaPipe."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Detect faces in an image using MediaPipe."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ) -> None:

        self.mp_face_detection = mp.solutions.face_detection

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_faces(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces.

        Args:
            frame: BGR frame.
            rgb_frame: Optional pre-converted RGB frame.
                       Avoids redundant BGR->RGB conversion.

        Returns:
            frame:
                Frame with bounding boxes.

            detections:
                List of (x, y, w, h) for every detected face.
        """

        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_frame)

        detections: List[Tuple[int, int, int, int]] = []

        if not results.detections:
            return frame, detections

        image_height, image_width, _ = frame.shape

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * image_width)
            y = int(bbox.ymin * image_height)
            w = int(bbox.width * image_width)
            h = int(bbox.height * image_height)

            detections.append((x, y, w, h))

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        return frame, detections

    @staticmethod
    def face_count(
        detections: List[Tuple[int, int, int, int]],
    ) -> int:
        """Return number of detected faces."""

        return len(detections)

    @staticmethod
    def has_single_face(
        detections: List[Tuple[int, int, int, int]],
    ) -> bool:
        """True only when exactly one face is detected."""

        return len(detections) == 1