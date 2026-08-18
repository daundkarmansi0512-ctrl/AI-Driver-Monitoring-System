"""
Yawn detection using Mouth Aspect Ratio (MAR).

Uses existing MediaPipe Face Mesh landmarks to measure
how wide the mouth is open.  A sustained wide-open mouth
is classified as a yawn, while brief openings (talking)
are ignored by the duration filter.
"""

from __future__ import annotations

import math
import time


class YawnDetector:
    """Detect yawns from mouth landmark aspect ratio."""

    def __init__(
        self,
        mar_threshold: float = 0.65,
        yawn_duration: float = 1.5,
    ) -> None:
        """
        Args:
            mar_threshold:
                MAR above this value means the mouth is
                open wide enough to be a potential yawn.
            yawn_duration:
                Seconds the mouth must stay open above
                the threshold before counting as a yawn.
                This filters out normal talking.
        """

        self.mar_threshold = mar_threshold
        self.yawn_duration = yawn_duration

        # When did the mouth first open wide?
        self._open_start: float = 0.0
        self._is_open: bool = False

        # Has the one-shot alert fired for this episode?
        self._alert_fired: bool = False

        # Is a yawn currently in progress? (for HUD)
        self._is_yawning: bool = False

        # Total yawn count
        self.yawn_count: int = 0

    @staticmethod
    def calculate_mar(landmarks: list, mouth_indices: list) -> float:
        """
        Calculate the Mouth Aspect Ratio.

        Mouth landmark order (from MOUTH_MAR):
            [0] left corner     (61)
            [1] right corner    (291)
            [2] top outer lip   (13)
            [3] bottom outer lip (14)
            [4] top inner lip   (82)
            [5] bottom inner lip (18)

        MAR = (outer_vertical + inner_vertical) / (2 × horizontal)

        High MAR = mouth wide open (yawn)
        Low MAR  = mouth closed or slightly open (normal)
        """

        left = landmarks[mouth_indices[0]]
        right = landmarks[mouth_indices[1]]
        top_outer = landmarks[mouth_indices[2]]
        bot_outer = landmarks[mouth_indices[3]]
        top_inner = landmarks[mouth_indices[4]]
        bot_inner = landmarks[mouth_indices[5]]

        # Vertical distances (outer and inner lip pairs)
        outer_vertical = math.sqrt(
            (top_outer.x - bot_outer.x) ** 2
            + (top_outer.y - bot_outer.y) ** 2
        )
        inner_vertical = math.sqrt(
            (top_inner.x - bot_inner.x) ** 2
            + (top_inner.y - bot_inner.y) ** 2
        )

        # Horizontal distance (mouth width)
        horizontal = math.sqrt(
            (left.x - right.x) ** 2
            + (left.y - right.y) ** 2
        )

        if horizontal == 0:
            return 0.0

        mar = (outer_vertical + inner_vertical) / (2.0 * horizontal)

        return mar

    def update(self, mar: float) -> bool:
        """
        Feed the latest MAR value each frame.

        Returns:
            True ONCE when a yawn is first detected for
            this episode.  The visual state (is_yawning)
            stays active until the mouth closes.
        """

        if mar >= self.mar_threshold:
            # Mouth is wide open

            if not self._is_open:
                # Just opened — start timer
                self._is_open = True
                self._open_start = time.time()
                self._alert_fired = False
                return False

            # Still open — check duration
            elapsed = time.time() - self._open_start

            if elapsed >= self.yawn_duration:
                self._is_yawning = True

                if not self._alert_fired:
                    self._alert_fired = True
                    self.yawn_count += 1
                    return True

        else:
            # Mouth closed — reset
            self._is_open = False
            self._open_start = 0.0
            self._is_yawning = False
            self._alert_fired = False

        return False

    @property
    def is_yawning(self) -> bool:
        """True while a yawn is in progress."""
        return self._is_yawning
