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
        draw_landmarks: bool = False,
    ) -> None:

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_landmarks = draw_landmarks

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
        rgb_frame: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list | None]:
        """Detect facial landmarks.

        Args:
            frame: BGR frame (used for drawing).
            rgb_frame: Optional pre-converted RGB frame.
                       If provided, skips the BGR->RGB conversion.
        """

        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            if self.draw_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec,
                )

            # Return only the landmark list
            return frame, face_landmarks.landmark

        return frame, None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()