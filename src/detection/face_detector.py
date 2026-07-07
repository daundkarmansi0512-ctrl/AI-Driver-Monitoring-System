"""Face detection module using MediaPipe."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Detect faces in an image using MediaPipe."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_faces(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, List]:
        """
        Detect faces and draw bounding boxes.

        Args:
            frame: Input webcam frame.

        Returns:
            Tuple containing:
            - Frame with face bounding boxes.
            - List of detected face coordinates.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_frame)

        detections = []

        if results.detections:

            image_height, image_width, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * image_width)
                y = int(bbox.ymin * image_height)
                w = int(bbox.width * image_width)
                h = int(bbox.height * image_height)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                detections.append((x, y, w, h))

        return frame, detections