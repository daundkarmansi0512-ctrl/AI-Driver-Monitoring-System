"""
Application Flow Manager.

Controls the application's UI flow:

WELCOME
    ↓
INSTRUCTIONS
    ↓
COUNTDOWN
    ↓
IDENTIFICATION
    ↓  ↘
    ↓   CALIBRATION → RESULTS → PROFILE_SAVED
    ↓                               ↓
    ↓←←←←←←←←←←←←←←←←←←←←←←←←←←←←↓
    ↓
MONITORING
"""

from __future__ import annotations

import time

from src.ui.app_states import AppState


class FlowManager:
    """Controls application states and timing."""

    def __init__(self) -> None:

        self.state = AppState.WELCOME
        self.state_start_time = time.time()

    # ---------------------------------------------------
    # State Utilities
    # ---------------------------------------------------

    def change_state(self, new_state: AppState) -> None:
        """Switch to another application state."""

        self.state = new_state
        self.state_start_time = time.time()

    def start_identification(self) -> None:
        """Move to driver identification."""

        self.change_state(AppState.IDENTIFICATION)

    def start_calibration(self) -> None:
        """Move to driver calibration."""

        self.change_state(AppState.CALIBRATION)

    def start_results(self) -> None:
        """
        Move immediately to the Results screen.

        Used after calibration finishes.
        """

        self.change_state(AppState.RESULTS)

    def start_monitoring(self) -> None:
        """
        Move directly to monitoring.

        Used when a previously registered driver
        has been recognized.
        """

        self.change_state(AppState.MONITORING)

    def elapsed(self) -> float:
        """Return seconds spent in current state."""

        return time.time() - self.state_start_time

    # ---------------------------------------------------
    # State Update
    # ---------------------------------------------------

    def update(self) -> None:
        """
        Update timed state transitions.

        States that require logic-based transitions
        (IDENTIFICATION, CALIBRATION) are handled
        explicitly by main.py instead of here.
        """

        # -------------------------------
        # Welcome Screen (2 seconds)
        # -------------------------------
        if self.state == AppState.WELCOME:

            if self.elapsed() >= 2:
                self.change_state(AppState.INSTRUCTIONS)

        # -------------------------------
        # Instruction Screen (5 seconds)
        # -------------------------------
        elif self.state == AppState.INSTRUCTIONS:

            if self.elapsed() >= 5:
                self.change_state(AppState.COUNTDOWN)

        # -------------------------------
        # Countdown (3 seconds)
        # -------------------------------
        elif self.state == AppState.COUNTDOWN:

            if self.elapsed() >= 3:
                self.change_state(AppState.IDENTIFICATION)

        # -------------------------------
        # Driver Identification
        # (transition handled by main.py)
        # -------------------------------
        elif self.state == AppState.IDENTIFICATION:
            pass

        # -------------------------------
        # Calibration
        # (transition handled by main.py)
        # -------------------------------
        elif self.state == AppState.CALIBRATION:
            pass

        # -------------------------------
        # Calibration Result Screen (3 seconds)
        # -------------------------------
        elif self.state == AppState.RESULTS:

            if self.elapsed() >= 3:
                self.change_state(AppState.PROFILE_SAVED)

        # -------------------------------
        # Profile Saved Screen (2 seconds)
        # -------------------------------
        elif self.state == AppState.PROFILE_SAVED:

            if self.elapsed() >= 2:
                self.change_state(AppState.MONITORING)

        # -------------------------------
        # Monitoring
        # -------------------------------
        elif self.state == AppState.MONITORING:
            pass

    # ---------------------------------------------------
    # Screen Information
    # ---------------------------------------------------

    @property
    def countdown_value(self) -> int:
        """
        Return countdown number.

        3
        2
        1
        """

        remaining = 3 - int(self.elapsed())

        if remaining < 1:
            remaining = 1

        return remaining

    @property
    def state_name(self) -> str:
        """Readable state name."""

        names = {
            AppState.WELCOME: "WELCOME",
            AppState.INSTRUCTIONS: "INSTRUCTIONS",
            AppState.COUNTDOWN: "COUNTDOWN",
            AppState.IDENTIFICATION: "IDENTIFICATION",
            AppState.CALIBRATION: "CALIBRATION",
            AppState.RESULTS: "RESULTS",
            AppState.PROFILE_SAVED: "PROFILE SAVED",
            AppState.MONITORING: "MONITORING",
        }

        return names[self.state]

    # ---------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------

    def is_welcome(self) -> bool:
        return self.state == AppState.WELCOME

    def is_instruction(self) -> bool:
        return self.state == AppState.INSTRUCTIONS

    def is_countdown(self) -> bool:
        return self.state == AppState.COUNTDOWN

    def is_identification(self) -> bool:
        return self.state == AppState.IDENTIFICATION

    def is_calibration(self) -> bool:
        return self.state == AppState.CALIBRATION

    def is_results(self) -> bool:
        return self.state == AppState.RESULTS

    def is_profile_saved(self) -> bool:
        return self.state == AppState.PROFILE_SAVED

    def is_monitoring(self) -> bool:
        return self.state == AppState.MONITORING