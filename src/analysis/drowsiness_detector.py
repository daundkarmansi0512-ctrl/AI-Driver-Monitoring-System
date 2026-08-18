"""
Drowsiness detection using Eye Aspect Ratio (EAR).
"""

from __future__ import annotations

import time


class DrowsinessDetector:
    """Detect prolonged eye closure with alert cooldown."""

    def __init__(
        self,
        threshold: float,
        closed_time: float = 1.0,
        cooldown: float = 5.0,
    ) -> None:
        """
        Args:
            threshold:
                EAR below this value means eyes closed.
            closed_time:
                Seconds eyes must be closed before
                drowsiness is flagged.
            cooldown:
                Seconds to suppress repeated alerts
                after one fires.  Prevents the same
                drowsiness episode from flooding the
                terminal and UI every frame.
        """

        self.threshold = threshold
        self.closed_time = closed_time
        self.cooldown = cooldown

        self.eye_closed = False
        self.closed_start = 0.0
        self.last_alert_time = 0.0

    def update(
        self,
        ear: float,
        ear_reliable: bool = True,
    ) -> bool:
        """
        Returns True when drowsiness is detected.

        Args:
            ear:
                Current Eye Aspect Ratio value.
            ear_reliable:
                False when the head is rotated far enough
                that the 2D EAR measurement is unreliable
                (e.g. high yaw).  In that case the
                closed-eye timer is PAUSED — not ignored
                forever.  When the head returns to center
                and ear_reliable becomes True again,
                drowsiness detection resumes immediately
                from a fresh state.

        After returning True once, further alerts are
        suppressed for ``cooldown`` seconds even if the
        eyes remain closed.
        """

        # ---- Pose makes EAR unreliable ----
        # Reset the closed-eye timer so a sideways head
        # never counts toward drowsiness, but as soon as
        # the head returns to center detection restarts.
        if not ear_reliable:
            self.eye_closed = False
            self.closed_start = 0.0
            return False

        # ---- Normal EAR-based detection ----
        if ear < self.threshold:

            if not self.eye_closed:
                self.eye_closed = True
                self.closed_start = time.time()

            elif (
                time.time() - self.closed_start
                >= self.closed_time
            ):
                now = time.time()

                if now - self.last_alert_time >= self.cooldown:
                    self.last_alert_time = now
                    return True

        else:
            self.eye_closed = False
            self.closed_start = 0.0

        return False

    @property
    def is_drowsy(self) -> bool:
        """True while eyes are closed beyond the time limit."""

        if not self.eye_closed:
            return False

        return (
            time.time() - self.closed_start
            >= self.closed_time
        )