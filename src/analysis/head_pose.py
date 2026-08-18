"""Simple head-pose estimation using facial landmarks."""

from __future__ import annotations

import cv2
import numpy as np


class HeadPoseEstimator:
    """Estimate approximate head orientation from MediaPipe landmarks.

    Features:
      - solvePnP-based yaw/pitch/roll from 6 facial landmarks
      - Neutral pose calibration: the driver's natural resting
        position becomes the zero-point for all angles
      - Hysteresis on direction classification to prevent
        flickering between CENTER and LEFT/RIGHT/UP/DOWN
    """

    # MediaPipe landmark indices used for pose estimation
    NOSE = 1
    CHIN = 152
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        calibration_frames: int = 30,
        debug: bool = False,
    ) -> None:
        """
        Args:
            frame_width:  Width of the video frame in pixels.
            frame_height: Height of the video frame in pixels.
            calibration_frames:
                Number of initial frames to average for
                neutral-pose baseline.  ~30 frames ≈ 1s at 30 FPS.
            debug:
                If True, print raw and calibrated angles
                to the console every frame (for development).
        """

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.debug = debug

        focal_length = frame_width

        self.camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2],
                [0, focal_length, frame_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Generic 3D face model points (mm-scale, nose at origin)
        self.model_points = np.array(
            [
                [0.0, 0.0, 0.0],          # Nose tip
                [0.0, -63.6, -12.5],      # Chin
                [-43.3, 32.7, -26.0],     # Left eye corner
                [43.3, 32.7, -26.0],      # Right eye corner
                [-28.9, -28.9, -24.1],    # Left mouth corner
                [28.9, -28.9, -24.1],     # Right mouth corner
            ],
            dtype=np.float64,
        )

        # ---- Neutral pose calibration ----
        # The driver's natural resting angles become the
        # zero point.  Without this, "looking at camera"
        # often reads as pitch ≈ +10° → falsely classified
        # as DOWN.
        self._calibration_target = calibration_frames
        self._calibration_samples: list[tuple[float, float, float]] = []
        self._is_calibrated = False
        self._neutral_yaw = 0.0
        self._neutral_pitch = 0.0
        self._neutral_roll = 0.0

        # ---- Hysteresis state ----
        # Prevents the direction from flickering when the
        # angle hovers near a threshold boundary.
        self._current_direction = "CENTER"

    @property
    def is_calibrated(self) -> bool:
        """True once the neutral baseline has been established."""
        return self._is_calibrated

    # --------------------------------------------------
    # Raw angle estimation (solvePnP)
    # --------------------------------------------------

    def _estimate_raw(
        self, landmarks: list,
    ) -> tuple[float, float, float]:
        """Return raw yaw, pitch, roll from solvePnP."""

        image_points = np.array(
            [
                [
                    landmarks[self.NOSE].x * self.frame_width,
                    landmarks[self.NOSE].y * self.frame_height,
                ],
                [
                    landmarks[self.CHIN].x * self.frame_width,
                    landmarks[self.CHIN].y * self.frame_height,
                ],
                [
                    landmarks[self.LEFT_EYE].x * self.frame_width,
                    landmarks[self.LEFT_EYE].y * self.frame_height,
                ],
                [
                    landmarks[self.RIGHT_EYE].x * self.frame_width,
                    landmarks[self.RIGHT_EYE].y * self.frame_height,
                ],
                [
                    landmarks[self.LEFT_MOUTH].x * self.frame_width,
                    landmarks[self.LEFT_MOUTH].y * self.frame_height,
                ],
                [
                    landmarks[self.RIGHT_MOUTH].x * self.frame_width,
                    landmarks[self.RIGHT_MOUTH].y * self.frame_height,
                ],
            ],
            dtype=np.float64,
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        projection_matrix = np.hstack(
            (rotation_matrix, translation_vector)
        )

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            projection_matrix
        )

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return yaw, pitch, roll

    # --------------------------------------------------
    # Calibrated estimation (public API)
    # --------------------------------------------------

    def estimate(
        self, landmarks: list,
    ) -> tuple[float, float, float]:
        """Return calibrated yaw, pitch, roll angles.

        During the first ``calibration_frames`` calls the
        raw angles are collected to compute the neutral
        baseline.  After calibration is complete all
        returned angles are relative to that baseline so
        the driver's natural resting position reads as
        (0, 0, 0).
        """

        raw_yaw, raw_pitch, raw_roll = self._estimate_raw(landmarks)

        # ---- Collect calibration samples ----
        if not self._is_calibrated:
            self._calibration_samples.append(
                (raw_yaw, raw_pitch, raw_roll)
            )

            if len(self._calibration_samples) >= self._calibration_target:
                # Average all samples to get the baseline
                yaws = [s[0] for s in self._calibration_samples]
                pitches = [s[1] for s in self._calibration_samples]
                rolls = [s[2] for s in self._calibration_samples]

                self._neutral_yaw = sum(yaws) / len(yaws)
                self._neutral_pitch = sum(pitches) / len(pitches)
                self._neutral_roll = sum(rolls) / len(rolls)

                self._is_calibrated = True

                if self.debug:
                    print(
                        f"[HeadPose] Neutral baseline: "
                        f"yaw={self._neutral_yaw:.1f} "
                        f"pitch={self._neutral_pitch:.1f} "
                        f"roll={self._neutral_roll:.1f}"
                    )

            # During calibration, return raw values
            # (direction defaults to CENTER)
            return raw_yaw, raw_pitch, raw_roll

        # ---- Subtract neutral baseline ----
        cal_yaw = raw_yaw - self._neutral_yaw
        cal_pitch = raw_pitch - self._neutral_pitch
        cal_roll = raw_roll - self._neutral_roll

        if self.debug:
            print(
                f"[HeadPose] raw=({raw_yaw:.1f}, {raw_pitch:.1f}) "
                f"cal=({cal_yaw:.1f}, {cal_pitch:.1f})"
            )

        return cal_yaw, cal_pitch, cal_roll

    # --------------------------------------------------
    # Direction classification with hysteresis
    # --------------------------------------------------

    def get_head_direction(
        self,
        yaw: float,
        pitch: float,
        yaw_threshold: float = 18.0,
        pitch_threshold: float = 15.0,
        hysteresis: float = 6.0,
    ) -> str:
        """Convert angles into a human-readable direction.

        Uses hysteresis to prevent flickering: a direction
        is entered when the angle exceeds ``threshold`` and
        is only exited when the angle drops below
        ``threshold - hysteresis``.

        Args:
            yaw:             Calibrated yaw angle.
            pitch:           Calibrated pitch angle.
            yaw_threshold:   Degrees to enter LEFT/RIGHT.
            pitch_threshold: Degrees to enter UP/DOWN.
            hysteresis:      Degrees of dead-zone to prevent
                             flickering on state exit.
        """

        # During calibration, always report CENTER
        if not self._is_calibrated:
            return "CENTER"

        direction = self._current_direction

        # ---- Check if we should LEAVE current direction ----
        yaw_exit = yaw_threshold - hysteresis
        pitch_exit = pitch_threshold - hysteresis

        if direction == "RIGHT" and yaw < yaw_exit:
            direction = "CENTER"
        elif direction == "LEFT" and yaw > -yaw_exit:
            direction = "CENTER"
        elif direction == "DOWN" and pitch < pitch_exit:
            direction = "CENTER"
        elif direction == "UP" and pitch > -pitch_exit:
            direction = "CENTER"

        # ---- Check if we should ENTER a new direction ----
        # Yaw takes priority over pitch (same as before)
        if direction == "CENTER":
            if yaw > yaw_threshold:
                direction = "RIGHT"
            elif yaw < -yaw_threshold:
                direction = "LEFT"
            elif pitch > pitch_threshold:
                direction = "DOWN"
            elif pitch < -pitch_threshold:
                direction = "UP"

        self._current_direction = direction
        return direction