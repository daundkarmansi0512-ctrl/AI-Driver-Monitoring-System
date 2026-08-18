"""
Eye Aspect Ratio (EAR) calculation utilities.
"""

from __future__ import annotations

import math
from typing import List


class EyeAspectRatio:
    """Calculate the Eye Aspect Ratio (EAR)."""

    @staticmethod
    def distance(point1, point2) -> float:
        """
        Calculate the Euclidean distance between two points.
        """

        return math.sqrt(
            (point2.x - point1.x) ** 2 +
            (point2.y - point1.y) ** 2
        )

    @classmethod
    def calculate(cls, landmarks: List, eye_indices: List[int]) -> float:
        """
        Calculate the Eye Aspect Ratio.

        Eye landmark order:

            p2      p3
             •------•
           /          \
        p1              p4
           \          /
             •------•
            p6      p5
        """

        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_1 = cls.distance(p2, p6)
        vertical_2 = cls.distance(p3, p5)

        horizontal = cls.distance(p1, p4)

        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return ear