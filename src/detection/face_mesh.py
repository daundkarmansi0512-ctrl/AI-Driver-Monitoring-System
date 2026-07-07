"""Face mesh module using MediaPipe."""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    """Detect facial landmarks using MediaPipe Face Mesh."""

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=1,
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_landmarks(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, Optional[list]]:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec,
                )

            return frame, results.multi_face_landmarks

        return frame, None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()