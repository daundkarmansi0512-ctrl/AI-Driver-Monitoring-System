"""
Screen drawing utilities for the Driver Monitoring System.
"""

from __future__ import annotations

import cv2
import numpy as np


class ScreenManager:
    """Draw UI elements on the video frame."""

    # --------------------------------------------------
    # Core Text Drawing
    # --------------------------------------------------

    @staticmethod
    def draw_center_text(
        frame: np.ndarray,
        text: str,
        scale: float = 1.0,
        color=(255, 255, 255),
        thickness: int = 2,
        y_offset: int = 0,
    ) -> None:
        """Draw text centred horizontally and vertically."""

        h, w = frame.shape[:2]

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            thickness,
        )

        x = (w - tw) // 2
        y = (h + th) // 2 + y_offset

        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness,
        )

    @staticmethod
    def draw_multiline_text(
        frame: np.ndarray,
        lines: list[str],
        start_y: int = 80,
        scale: float = 0.8,
        color=(255, 255, 255),
        thickness: int = 2,
        line_spacing: int = 40,
    ) -> None:
        """Draw multiple lines of text."""

        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (40, start_y + i * line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness,
            )

    @staticmethod
    def draw_status(
        frame: np.ndarray,
        status: str,
        color=(0, 255, 0),
    ) -> None:
        """Draw current system status at the top."""

        cv2.putText(
            frame,
            f"STATUS : {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2,
        )

    # --------------------------------------------------
    # Calibration Screens
    # --------------------------------------------------

    @staticmethod
    def draw_calibration_timer(
        frame: np.ndarray,
        seconds_left: float,
    ) -> None:
        """Draw calibration timer."""

        cv2.putText(
            frame,
            f"Time Left : {seconds_left:.1f}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2,
        )

    @staticmethod
    def draw_progress_bar(
        frame: np.ndarray,
        progress: float,
        color=(0, 255, 0),
    ) -> None:
        """Draw a progress bar centred on the frame."""

        h, w = frame.shape[:2]

        bar_x = 50
        bar_y = h // 2 - 15
        bar_w = w - 100
        bar_h = 30

        # Outline
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (100, 100, 100), 2,
        )

        # Fill
        fill_w = int(bar_w * min(progress, 100) / 100)

        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x + 2, bar_y + 2),
                (bar_x + fill_w, bar_y + bar_h - 2),
                color, -1,
            )

        # Percentage label
        text = f"{int(progress)}%"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        tx = bar_x + (bar_w - tw) // 2
        ty = bar_y + (bar_h + th) // 2

        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2,
        )

    @staticmethod
    def draw_results(
        frame: np.ndarray,
        average_ear: float,
        threshold: float,
    ) -> None:
        """Draw calibration results screen."""

        lines = [
            "CALIBRATION COMPLETE",
            "",
            f"Average EAR : {average_ear:.3f}",
            f"Threshold : {threshold:.3f}",
        ]

        ScreenManager.draw_multiline_text(
            frame, lines,
            start_y=120, scale=0.9,
            color=(0, 255, 0),
        )

    @staticmethod
    def draw_profile_saved(
        frame: np.ndarray,
    ) -> None:
        """Display centred PROFILE SAVED message."""

        h, w = frame.shape[:2]
        box_w, box_h = 340, 80
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2

        cv2.rectangle(
            frame, (x1, y1),
            (x1 + box_w, y1 + box_h),
            (0, 180, 0), -1,
        )

        text = "PROFILE SAVED"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3,
        )
        tx = x1 + (box_w - tw) // 2
        ty = y1 + (box_h + th) // 2

        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 3,
        )

    # --------------------------------------------------
    # Monitoring HUD
    # --------------------------------------------------

    @staticmethod
    def draw_monitoring_hud(
        frame: np.ndarray,
        driver_name: str,
        ear: float,
        blink_count: int,
        head_direction: str = "CENTER",
        attention_state: str = "OK",
        yawn_count: int = 0,
    ) -> None:
        """
        Draw the monitoring heads-up display with a
        semi-transparent background panel for readability.

        Args:
            frame:           Video frame to draw on.
            driver_name:     Display name of the driver.
            ear:             Current EAR value.
            blink_count:     Total blinks detected.
            head_direction:  Head direction (CENTER/LEFT/RIGHT/UP/DOWN).
            attention_state: Overall attention state.
            yawn_count:      Total yawns detected.
        """

        h, w = frame.shape[:2]

        # Semi-transparent dark panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 5), (300, 225), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(
            frame, f"Driver: {driver_name}",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2,
        )

        cv2.putText(
            frame, f"EAR: {ear:.2f}",
            (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2,
        )

        cv2.putText(
            frame, f"Blinks: {blink_count}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2,
        )

        cv2.putText(
            frame, f"Yawns: {yawn_count}",
            (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2,
        )

        cv2.putText(
            frame, f"Head: {head_direction}",
            (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2,
        )

        # Attention state — colour changes with severity
        att_colors = {
            "DROWSY": (0, 0, 255),         # Red
            "DISTRACTED": (0, 165, 255),    # Orange
            "YAWNING": (0, 255, 255),       # Yellow
            "PHONE": (255, 0, 128),         # Purple
        }
        att_color = att_colors.get(
            attention_state, (0, 255, 0),   # Green default
        )

        cv2.putText(
            frame, f"Attention: {attention_state}",
            (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, att_color, 2,
        )

        # Status line
        cv2.putText(
            frame, "Status: Monitoring",
            (20, 190), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1,
        )

    # --------------------------------------------------
    # Alerts
    # --------------------------------------------------

    @staticmethod
    def draw_drowsiness_alert(
        frame: np.ndarray,
    ) -> None:
        """Draw drowsiness warning banner at the top."""

        h, w = frame.shape[:2]

        # Red banner
        cv2.rectangle(
            frame, (10, 10), (w - 10, 100),
            (0, 0, 255), -1,
        )

        text = "DROWSINESS DETECTED!"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3,
        )

        cv2.putText(
            frame, text,
            ((w - tw) // 2, 55 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 3,
        )

    @staticmethod
    def draw_distraction_alert(
        frame: np.ndarray,
        direction: str = "",
    ) -> None:
        """Draw distraction warning banner (orange) at the top."""

        h, w = frame.shape[:2]

        # Orange banner
        cv2.rectangle(
            frame, (10, 10), (w - 10, 100),
            (0, 140, 255), -1,
        )

        text = "DISTRACTION WARNING"
        if direction:
            text = f"DISTRACTION: LOOKING {direction}"

        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3,
        )

        cv2.putText(
            frame, text,
            ((w - tw) // 2, 55 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 3,
        )

    @staticmethod
    def draw_yawn_alert(
        frame: np.ndarray,
    ) -> None:
        """Draw yawn warning banner (yellow) at the top."""

        h, w = frame.shape[:2]

        # Yellow banner
        cv2.rectangle(
            frame, (10, 10), (w - 10, 100),
            (0, 220, 255), -1,
        )

        text = "YAWN DETECTED!"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3,
        )

        cv2.putText(
            frame, text,
            ((w - tw) // 2, 55 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 3,
        )

    @staticmethod
    def draw_phone_alert(
        frame: np.ndarray,
    ) -> None:
        """Draw phone usage warning banner (purple) at the top."""

        h, w = frame.shape[:2]

        # Purple banner
        cv2.rectangle(
            frame, (10, 10), (w - 10, 100),
            (200, 0, 128), -1,
        )

        text = "PHONE USAGE DETECTED!"
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3,
        )

        cv2.putText(
            frame, text,
            ((w - tw) // 2, 55 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 3,
        )

    @staticmethod
    def draw_driver_change_alert(
        frame: np.ndarray,
        expected_driver: str,
        current_driver: str,
    ) -> None:
        """Draw a driver-change alert overlay."""

        h, w = frame.shape[:2]

        # Semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (w - 20, 200), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, "DRIVER CHANGE DETECTED",
            (40, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2,
        )

        cv2.putText(
            frame, f"Expected: {expected_driver}",
            (40, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2,
        )

        cv2.putText(
            frame, f"Current: {current_driver}",
            (40, 145), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2,
        )

        cv2.putText(
            frame, "MONITORING PAUSED",
            (40, 185), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 200, 255), 2,
        )

    @staticmethod
    def draw_monitoring_started(
        frame: np.ndarray,
    ) -> None:
        """Monitoring started screen."""

        ScreenManager.draw_center_text(
            frame, "Monitoring Started",
            scale=1.2, color=(0, 255, 0),
        )