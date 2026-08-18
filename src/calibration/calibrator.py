"""
Driver calibration utilities.

Collects valid EAR samples and computes
a personalized blink threshold.
"""

from __future__ import annotations

from statistics import mean


class DriverCalibrator:
    """Handle driver EAR calibration."""

    def __init__(self, sample_target: int = 90) -> None:
        """
        Args:
            sample_target:
                Number of valid EAR samples required.
                ~90 samples ≈ 3 seconds at 30 FPS.
        """

        self.sample_target = sample_target

        self.is_calibrating = False
        self.is_complete = False

        self.open_ear_values: list[float] = []

    def start(self) -> None:
        """Start calibration."""

        self.is_calibrating = True
        self.is_complete = False

        self.open_ear_values.clear()

    def update(self, ear: float) -> None:
        """
        Add one valid EAR sample.
        """

        if not self.is_calibrating:
            return

        self.open_ear_values.append(ear)

        if len(self.open_ear_values) >= self.sample_target:

            self.is_calibrating = False
            self.is_complete = True

    def get_average_open_ear(self) -> float:
        """Average EAR during calibration."""

        if not self.open_ear_values:
            return 0.0

        return mean(self.open_ear_values)

    def get_suggested_threshold(self) -> float:
        """Personalized EAR threshold."""

        return self.get_average_open_ear() * 0.85

    @property
    def samples_collected(self) -> int:
        """Number of collected samples."""

        return len(self.open_ear_values)

    @property
    def samples_remaining(self) -> int:
        """Remaining samples."""

        remaining = self.sample_target - len(self.open_ear_values)

        if remaining < 0:
            remaining = 0

        return remaining

    @property
    def progress(self) -> float:
        """Calibration progress (0–100%)."""

        return (
            len(self.open_ear_values)
            / self.sample_target
        ) * 100