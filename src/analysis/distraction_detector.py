"""
Distraction detection using head pose direction.

Tracks how long the driver's head has been turned away
from the forward (CENTER) position.  Brief glances are
ignored — only sustained off-center head positions
trigger an alert.
"""

from __future__ import annotations

import time


class DistractionDetector:
    """Detect driver distraction from sustained head turning."""

    def __init__(
        self,
        distraction_duration: float = 2.0,
        cooldown: float = 5.0,
    ) -> None:
        """
        Args:
            distraction_duration:
                Seconds the head must remain off-center
                before distraction is flagged.
            cooldown:
                Seconds to suppress repeated alerts
                after one fires.
        """

        self.distraction_duration = distraction_duration

        # When did the head first leave CENTER?
        self._off_center_start: float = 0.0

        # Is the head currently off-center?
        self._is_off_center: bool = False

        # Current direction while distracted (e.g. "LEFT")
        self._current_direction: str = "CENTER"

        # Active distraction state (for HUD / visual)
        self._is_distracted: bool = False

        # True after the one-shot alert has fired for
        # this off-center episode.  Prevents repeated
        # console/log alerts while the head stays turned.
        # Resets when the head returns to CENTER.
        self._alert_fired: bool = False

    def update(self, head_direction: str) -> bool:
        """
        Feed the latest head direction each frame.

        Args:
            head_direction:
                One of "CENTER", "LEFT", "RIGHT",
                "UP", "DOWN" from HeadPoseEstimator.

        Returns:
            True ONCE when distraction is first detected
            for this episode.  The visual state stays
            active (is_distracted = True) until the head
            returns to CENTER, but update() will not
            return True again for the same episode.
        """

        if head_direction == "CENTER":
            # Head returned to center — reset everything
            self._is_off_center = False
            self._off_center_start = 0.0
            self._is_distracted = False
            self._alert_fired = False
            self._current_direction = "CENTER"
            return False

        # Head is NOT centered
        self._current_direction = head_direction

        if not self._is_off_center:
            # Just started looking away — begin timer
            self._is_off_center = True
            self._off_center_start = time.time()
            return False

        # Still looking away — check duration
        elapsed = time.time() - self._off_center_start

        if elapsed >= self.distraction_duration:
            # Mark as distracted (for HUD display)
            self._is_distracted = True

            # Fire ONE alert per episode — never repeat
            # until the head returns to center first.
            if not self._alert_fired:
                self._alert_fired = True
                return True

        return False

    @property
    def is_distracted(self) -> bool:
        """True while head has been off-center past the duration limit."""
        return self._is_distracted

    @property
    def current_direction(self) -> str:
        """The current head direction (CENTER/LEFT/RIGHT/UP/DOWN)."""
        return self._current_direction

    @property
    def off_center_seconds(self) -> float:
        """How many seconds the head has been off-center."""
        if not self._is_off_center:
            return 0.0
        return time.time() - self._off_center_start
