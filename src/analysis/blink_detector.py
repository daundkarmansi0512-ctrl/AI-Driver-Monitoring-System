"""
Blink detection using Eye Aspect Ratio (EAR).
"""

from __future__ import annotations


class BlinkDetector:
    """Detect blinks using Eye Aspect Ratio."""

    def __init__(
        self,
        threshold: float = 0.25,
        min_closed_frames: int = 2,
        max_closed_frames: int = 8,
    ) -> None:

        self.threshold = threshold
        self.min_closed_frames = min_closed_frames
        self.max_closed_frames = max_closed_frames

        self.closed_frames = 0
        self.blink_count = 0

    def update(self, ear: float) -> bool:
        """
        Update blink detector.

        Returns:
            True if a blink is detected.
        """

        # Eye is closed
        if ear < self.threshold:
            self.closed_frames += 1
            return False

        # Eye opened again
        if self.closed_frames > 0:

            if (
                self.min_closed_frames
                <= self.closed_frames
                <= self.max_closed_frames
            ):
                self.blink_count += 1

                self.closed_frames = 0
                return True

            self.closed_frames = 0

        return False