"""Entry point for the AI Driver Monitoring System."""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.analysis.blink_detector import BlinkDetector
from src.analysis.drowsiness_detector import DrowsinessDetector
from src.analysis.distraction_detector import DistractionDetector
from src.analysis.eye_aspect_ratio import EyeAspectRatio
from src.analysis.landmark_indices import LEFT_EYE, RIGHT_EYE, MOUTH_MAR
from src.analysis.head_pose import HeadPoseEstimator
from src.analysis.yawn_detector import YawnDetector
from src.analysis.event_logger import EventLogger
from src.analysis import config

from src.calibration.calibrator import DriverCalibrator

from src.camera.camera_manager import CameraManager

from src.detection.face_detector import FaceDetector
from src.detection.face_mesh import FaceMeshDetector
from src.detection.phone_detector import PhoneDetector

from src.ui.flow_manager import FlowManager
from src.ui.screen_manager import ScreenManager

from src.recognition.profile_manager import ProfileManager
from src.recognition.face_recognizer import FaceRecognizer


# ==========================================================
# Performance Profiler
# ==========================================================


class PipelineProfiler:
    """Tracks rolling averages and performance metrics across pipeline stages."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.frame_deltas = collections.deque(maxlen=window_size)
        self.timings = {
            "camera": collections.deque(maxlen=window_size),
            "face_det": collections.deque(maxlen=window_size),
            "face_mesh": collections.deque(maxlen=window_size),
            "ear": collections.deque(maxlen=window_size),
            "mar": collections.deque(maxlen=window_size),
            "head_pose": collections.deque(maxlen=window_size),
            "phone_main": collections.deque(maxlen=window_size),
            "identity_main": collections.deque(maxlen=window_size),
            "hud": collections.deque(maxlen=window_size),
            "display": collections.deque(maxlen=window_size),
            "loop_total": collections.deque(maxlen=window_size),
        }

    def record(self, category: str, duration_ms: float) -> None:
        if category in self.timings:
            self.timings[category].append(duration_ms)

    def record_frame_delta(self, delta_s: float) -> None:
        if delta_s > 0:
            self.frame_deltas.append(delta_s)

    def get_avg(self, category: str) -> float:
        vals = self.timings.get(category)
        return (sum(vals) / len(vals)) if vals else 0.0

    @property
    def fps(self) -> float:
        if not self.frame_deltas:
            return 0.0
        avg_dt = sum(self.frame_deltas) / len(self.frame_deltas)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0


# ==========================================================
# Background Identity Checker (Threading)
# ==========================================================


class _IdentityCheckState:
    """Thread-safe container for background identity results."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: tuple | None = None
        self._checking: bool = False
        self._last_duration_ms: float = 0.0

    @property
    def is_checking(self) -> bool:
        with self._lock:
            return self._checking

    @property
    def last_duration_ms(self) -> float:
        with self._lock:
            return self._last_duration_ms

    def start_check(self) -> None:
        with self._lock:
            self._checking = True

    def finish_check(
        self,
        result: tuple[bool, str | None, float],
        duration_ms: float = 0.0,
    ) -> None:
        with self._lock:
            self._result = result
            self._checking = False
            self._last_duration_ms = duration_ms

    def consume_result(
        self,
    ) -> tuple[bool, str | None, float] | None:
        """Return and clear the latest result."""
        with self._lock:
            r = self._result
            self._result = None
            return r


def _identity_worker(
    recognizer: FaceRecognizer,
    frame_copy: np.ndarray,
    expected_embedding: any,
    state: _IdentityCheckState,
) -> None:
    """
    Background worker for identity checking.
    Runs InsightFace on a copy of the camera frame off the main thread.
    """
    try:
        t0 = time.perf_counter()
        result = recognizer.check_identity(
            frame_copy,
            expected_embedding,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        state.finish_check(result, elapsed_ms)

    except Exception as exc:
        print(f"⚠️ Identity check error: {exc}")
        state.finish_check((False, None, 0.0), 0.0)


# ==========================================================
# Main Application
# ==========================================================


def main() -> None:
    """Start the AI Driver Monitoring System."""

    # --------------------------------------------------
    # INITIALIZE COMPONENTS
    # --------------------------------------------------

    camera = CameraManager()
    face_detector = FaceDetector()
    face_mesh = FaceMeshDetector(draw_landmarks=config.DRAW_FACE_MESH)

    calibrator = DriverCalibrator(sample_target=90)

    blink_detector = BlinkDetector(
        threshold=0.25,
        min_closed_frames=1,
        max_closed_frames=8,
    )

    drowsiness_detector = DrowsinessDetector(
        threshold=0.25,
        closed_time=config.DROWSINESS_CLOSED_TIME,
        cooldown=config.DROWSINESS_COOLDOWN,
    )

    distraction_detector = DistractionDetector(
        distraction_duration=config.DISTRACTION_DURATION,
    )

    yawn_detector = YawnDetector(
        mar_threshold=config.YAWN_MAR_THRESHOLD,
        yawn_duration=config.YAWN_DURATION,
    )

    phone_detector = PhoneDetector(
        model_name=config.PHONE_MODEL_NAME,
        confidence=config.PHONE_CONFIDENCE,
        phone_duration=config.PHONE_DURATION,
        async_interval=config.PHONE_ASYNC_INTERVAL,
    )

    event_logger = EventLogger()
    profiler = PipelineProfiler(window_size=30)

    flow = FlowManager()
    face_recognizer = FaceRecognizer()

    # --------------------------------------------------
    # IDENTIFICATION STATE
    # --------------------------------------------------

    recognized_driver_id: str | None = None
    recognized_profile: dict | None = None
    identification_attempted = False
    identification_result_time: float | None = None

    expected_driver_embedding = None

    # --------------------------------------------------
    # DRIVER CHANGE DETECTION
    # --------------------------------------------------

    last_identity_check = 0.0
    identity_check_interval = 3.0
    driver_change_count = 0
    driver_change_limit = 3
    driver_change_alert = False
    current_detected_driver: str | None = None

    identity_state = _IdentityCheckState()

    # --------------------------------------------------
    # MONITORING CACHE & STATE
    # --------------------------------------------------

    monitoring_message_shown = False
    frame_counter = 0

    cached_detections: list = []
    cached_face_count: int = 0
    cached_landmarks: list | None = None

    cached_yaw: float = 0.0
    cached_pitch: float = 0.0
    cached_roll: float = 0.0
    cached_head_direction: str = "CENTER"

    last_loop_timestamp = time.perf_counter()

    try:
        camera.start()

        print("=" * 40)
        print("AI DRIVER MONITORING SYSTEM")
        print("=" * 40)

        first_frame = camera.read_frame()

        if first_frame is None:
            print("❌ Failed to read frame from camera.")
            return

        head_pose = HeadPoseEstimator(
            first_frame.shape[1],
            first_frame.shape[0],
            calibration_frames=config.HEAD_POSE_CALIBRATION_FRAMES,
            debug=config.HEAD_POSE_DEBUG,
        )

        frame = first_frame

        while True:
            t_loop_start = time.perf_counter()
            loop_delta = t_loop_start - last_loop_timestamp
            last_loop_timestamp = t_loop_start
            profiler.record_frame_delta(loop_delta)

            # --------------------------------------------------
            # 1. Camera Capture
            # --------------------------------------------------
            t0 = time.perf_counter()
            frame = camera.read_frame()
            t_camera = (time.perf_counter() - t0) * 1000.0
            profiler.record("camera", t_camera)

            if frame is None:
                break

            frame_counter += 1
            is_monitoring = flow.is_monitoring()

            # --------------------------------------------------
            # 2. Shared Color Conversion & Detection Scheduling
            # --------------------------------------------------
            # Run face detection:
            # - Always during calibration / welcome / identification
            # - Every N frames during monitoring
            run_face_det = (
                not is_monitoring
                or (frame_counter % config.FACE_DETECT_EVERY_N_FRAMES == 0)
                or (cached_face_count == 0)
            )

            # Run face mesh:
            # - Always during calibration
            # - Every N frames during monitoring
            run_face_mesh = (
                not is_monitoring
                or (frame_counter % config.FACE_MESH_EVERY_N_FRAMES == 0)
                or (cached_landmarks is None)
            )

            rgb_frame: np.ndarray | None = None
            if run_face_det or run_face_mesh:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face Detection
            t0 = time.perf_counter()
            if run_face_det:
                frame, detections = face_detector.detect_faces(frame, rgb_frame=rgb_frame)
                cached_detections = detections
                cached_face_count = face_detector.face_count(detections)
            else:
                detections = cached_detections
            face_count = cached_face_count
            t_face_det = (time.perf_counter() - t0) * 1000.0
            profiler.record("face_det", t_face_det)

            # Face Mesh
            t0 = time.perf_counter()
            if run_face_mesh:
                frame, landmarks = face_mesh.detect_landmarks(frame, rgb_frame=rgb_frame)
                cached_landmarks = landmarks
            else:
                landmarks = cached_landmarks
            t_face_mesh = (time.perf_counter() - t0) * 1000.0
            profiler.record("face_mesh", t_face_mesh)

            flow.update()

            # ==========================================
            # WELCOME
            # ==========================================
            if flow.is_welcome():
                ScreenManager.draw_center_text(
                    frame,
                    "AI DRIVER MONITORING SYSTEM",
                    scale=1.2,
                )

            # ==========================================
            # INSTRUCTIONS
            # ==========================================
            elif flow.is_instruction():
                ScreenManager.draw_status(frame, "INSTRUCTIONS")
                ScreenManager.draw_multiline_text(
                    frame,
                    [
                        "Driver Identification",
                        "",
                        "1. Sit comfortably in front of the camera.",
                        "2. Face the camera directly.",
                        "3. Make sure only one person is visible.",
                        "4. Keep your face clearly visible.",
                        "",
                        "The system will identify you automatically.",
                        "New drivers will be calibrated.",
                        "",
                        "Please remain still during calibration.",
                    ],
                    line_spacing=35,
                )

            # ==========================================
            # COUNTDOWN
            # ==========================================
            elif flow.is_countdown():
                ScreenManager.draw_status(frame, "GET READY")
                ScreenManager.draw_center_text(
                    frame,
                    str(flow.countdown_value),
                    scale=3,
                )

            # ==========================================
            # IDENTIFICATION
            # ==========================================
            elif flow.is_identification():
                if identification_attempted:
                    elapsed = time.time() - identification_result_time

                    if recognized_driver_id is not None:
                        driver_label = recognized_driver_id.replace("_", " ").title()
                        ScreenManager.draw_status(frame, "DRIVER IDENTIFIED", color=(0, 255, 0))
                        ScreenManager.draw_center_text(
                            frame,
                            f"Driver: {driver_label}",
                            scale=0.9,
                            color=(0, 255, 0),
                        )
                        ScreenManager.draw_center_text(
                            frame,
                            "Loading profile...",
                            scale=0.7,
                            color=(200, 200, 200),
                            y_offset=50,
                        )
                        if elapsed >= 3.0:
                            flow.start_monitoring()

                    else:
                        ScreenManager.draw_status(frame, "NEW DRIVER DETECTED", color=(0, 255, 255))
                        ScreenManager.draw_center_text(
                            frame,
                            "Calibration Required",
                            scale=0.9,
                            color=(0, 255, 255),
                        )
                        if elapsed >= 3.0:
                            flow.start_calibration()

                elif face_count == 0:
                    ScreenManager.draw_status(frame, "IDENTIFYING DRIVER...")
                    ScreenManager.draw_center_text(frame, "No Driver Detected", scale=1.0, color=(0, 0, 255))

                elif face_count > 1:
                    ScreenManager.draw_status(frame, "IDENTIFYING DRIVER...")
                    ScreenManager.draw_center_text(frame, "Multiple Faces Detected", scale=1.0, color=(0, 0, 255))

                else:
                    ScreenManager.draw_status(frame, "IDENTIFYING DRIVER...")
                    ScreenManager.draw_center_text(frame, "Please look at the camera", scale=0.8)

                    print("\n🔍 Checking driver identity...")
                    recognized_driver_id = face_recognizer.find_matching_driver(frame)
                    identification_attempted = True
                    identification_result_time = time.time()

                    if recognized_driver_id is not None:
                        recognized_profile = ProfileManager.load_profile(recognized_driver_id)
                        if recognized_profile is not None:
                            saved_threshold = recognized_profile["blink_threshold"]
                            blink_detector.threshold = saved_threshold
                            drowsiness_detector.threshold = saved_threshold
                            expected_driver_embedding = face_recognizer.load_embedding(recognized_driver_id)
                            print(f"✅ Driver identified: {recognized_driver_id}")
                            print(f"   EAR threshold: {saved_threshold:.3f}")
                        else:
                            print("⚠️ Profile not found, treating as new driver.")
                            recognized_driver_id = None
                    else:
                        print("❌ No matching driver. New driver detected.")

            # ==========================================
            # CALIBRATION
            # ==========================================
            elif flow.is_calibration():
                if face_count == 0:
                    ScreenManager.draw_center_text(frame, "No Driver Detected", scale=1.0, color=(0, 0, 255))
                    ScreenManager.draw_status(frame, "Calibration Paused")

                elif face_count > 1:
                    ScreenManager.draw_center_text(frame, "Multiple Faces Detected", scale=1.0, color=(0, 0, 255))
                    ScreenManager.draw_status(frame, "Calibration Paused")

                else:
                    if not calibrator.is_calibrating and not calibrator.is_complete:
                        calibrator.start()

                    ScreenManager.draw_status(frame, "CALIBRATING DRIVER")
                    ScreenManager.draw_progress_bar(frame, calibrator.progress)

                    h = frame.shape[0]
                    cv2.putText(
                        frame,
                        "Keep your eyes naturally open",
                        (40, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        1,
                    )
                    cv2.putText(
                        frame,
                        "Please remain still",
                        (40, h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        1,
                    )

                    if landmarks is not None:
                        left_ear = EyeAspectRatio.calculate(landmarks, LEFT_EYE)
                        right_ear = EyeAspectRatio.calculate(landmarks, RIGHT_EYE)
                        ear = (left_ear + right_ear) / 2
                        calibrator.update(ear)

                    if calibrator.is_complete:
                        threshold = calibrator.get_suggested_threshold()
                        blink_detector.threshold = threshold
                        drowsiness_detector.threshold = threshold

                        driver_id = ProfileManager.get_next_driver_id()
                        ProfileManager.save_profile(
                            calibrator.get_average_open_ear(),
                            threshold,
                            driver_id,
                        )

                        if detections:
                            face_recognizer.save_face(
                                frame,
                                detections[0],
                                driver_id,
                            )

                        recognized_driver_id = driver_id
                        expected_driver_embedding = face_recognizer.load_embedding(driver_id)

                        print("\n===== CALIBRATION COMPLETE =====")
                        print(f"Average Open EAR : {calibrator.get_average_open_ear():.3f}")
                        print(f"Personalized Threshold : {threshold:.3f}")
                        print(f"Driver ID: {driver_id}")

                        flow.start_results()

            # ==========================================
            # RESULTS → PROFILE_SAVED
            # ==========================================
            elif flow.is_results():
                ScreenManager.draw_results(
                    frame,
                    calibrator.get_average_open_ear(),
                    calibrator.get_suggested_threshold(),
                )

            elif flow.is_profile_saved():
                ScreenManager.draw_profile_saved(frame)
                if recognized_driver_id:
                    driver_label = recognized_driver_id.replace("_", " ").title()
                    h, w = frame.shape[:2]
                    text = f"Driver: {driver_label}"
                    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(
                        frame,
                        text,
                        ((w - tw) // 2, h // 2 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # ==========================================
            # MONITORING
            # ==========================================
            elif flow.is_monitoring():
                if not monitoring_message_shown:
                    print("\n===== MONITORING STARTED =====")
                    if recognized_driver_id:
                        print(f"Driver: {recognized_driver_id}")
                    print(f"EAR threshold: {blink_detector.threshold:.3f}\n")
                    monitoring_message_shown = True

                if face_count == 0:
                    ScreenManager.draw_center_text(
                        frame, "No Driver Detected", scale=1.0, color=(0, 0, 255)
                    )
                    ScreenManager.draw_status(frame, "Waiting for driver...")

                elif face_count > 1:
                    ScreenManager.draw_center_text(
                        frame, "Multiple Faces Detected", scale=1.0, color=(0, 0, 255)
                    )
                    ScreenManager.draw_status(frame, "Only one driver allowed")

                else:
                    # ---- Background Identity Check ----
                    t0 = time.perf_counter()
                    check_result = identity_state.consume_result()
                    if check_result is not None:
                        is_same, other_id, _sim = check_result
                        if is_same:
                            if driver_change_alert:
                                driver_change_alert = False
                                driver_change_count = 0
                                print("✅ Original driver returned. Resuming.")
                            else:
                                driver_change_count = 0
                        else:
                            if not driver_change_alert:
                                driver_change_count += 1
                                current_detected_driver = other_id
                                print(f"⚠️ Driver mismatch ({driver_change_count}/{driver_change_limit})")
                                if driver_change_count >= driver_change_limit:
                                    driver_change_alert = True
                                    print("🚨 DRIVER CHANGE DETECTED!")
                                    event_logger.log(
                                        driver=(recognized_driver_id or "unknown"),
                                        event="driver_change",
                                    )

                    # Trigger periodic background identity check
                    if (
                        expected_driver_embedding is not None
                        and not identity_state.is_checking
                        and (time.time() - last_identity_check >= identity_check_interval)
                    ):
                        last_identity_check = time.time()
                        frame_copy = frame.copy()
                        identity_state.start_check()
                        threading.Thread(
                            target=_identity_worker,
                            args=(
                                face_recognizer,
                                frame_copy,
                                expected_driver_embedding,
                                identity_state,
                            ),
                            daemon=True,
                        ).start()
                    t_identity = (time.perf_counter() - t0) * 1000.0
                    profiler.record("identity_main", t_identity)

                    if driver_change_alert:
                        expected_label = (
                            recognized_driver_id.replace("_", " ").title()
                            if recognized_driver_id
                            else "Unknown"
                        )
                        current_label = (
                            current_detected_driver.replace("_", " ").title()
                            if current_detected_driver
                            else "Unknown"
                        )
                        ScreenManager.draw_driver_change_alert(
                            frame, expected_label, current_label
                        )

                    else:
                        if landmarks is not None:
                            # ----------------------------------
                            # 3. EAR Calculation (Every frame)
                            # ----------------------------------
                            t0 = time.perf_counter()
                            left_ear = EyeAspectRatio.calculate(landmarks, LEFT_EYE)
                            right_ear = EyeAspectRatio.calculate(landmarks, RIGHT_EYE)
                            ear = (left_ear + right_ear) / 2.0
                            t_ear = (time.perf_counter() - t0) * 1000.0
                            profiler.record("ear", t_ear)

                            # ----------------------------------
                            # 4. Head Pose Estimation (Scheduled)
                            # ----------------------------------
                            t0 = time.perf_counter()
                            run_head_pose = (
                                frame_counter % config.HEAD_POSE_EVERY_N_FRAMES == 0
                                or not head_pose.is_calibrated
                            )
                            if run_head_pose:
                                cached_yaw, cached_pitch, cached_roll = head_pose.estimate(landmarks)
                                cached_head_direction = head_pose.get_head_direction(
                                    cached_yaw,
                                    cached_pitch,
                                    yaw_threshold=config.HEAD_YAW_THRESHOLD,
                                    pitch_threshold=config.HEAD_PITCH_THRESHOLD,
                                    hysteresis=config.HEAD_DIRECTION_HYSTERESIS,
                                )
                            yaw = cached_yaw
                            pitch = cached_pitch
                            roll = cached_roll
                            head_direction = cached_head_direction
                            t_head_pose = (time.perf_counter() - t0) * 1000.0
                            profiler.record("head_pose", t_head_pose)

                            ear_reliable = abs(yaw) < config.EAR_POSE_SUPPRESS_YAW

                            # ----------------------------------
                            # 5. Blink & Drowsiness Detection
                            # ----------------------------------
                            if ear_reliable:
                                blink_detected = blink_detector.update(ear)
                                if blink_detected:
                                    event_logger.log(
                                        driver=(recognized_driver_id or "unknown"),
                                        event="blink",
                                        head_direction=head_direction,
                                        ear=ear,
                                    )

                            drowsy_alert = drowsiness_detector.update(
                                ear, ear_reliable=ear_reliable
                            )
                            if drowsy_alert:
                                print("⚠️ DROWSINESS DETECTED!")
                                event_logger.log(
                                    driver=(recognized_driver_id or "unknown"),
                                    event="drowsiness",
                                    head_direction=head_direction,
                                    ear=ear,
                                )

                            # ----------------------------------
                            # 6. Distraction Detection
                            # ----------------------------------
                            distraction_alert = distraction_detector.update(head_direction)
                            if distraction_alert:
                                direction = distraction_detector.current_direction
                                print(f"⚠️ DISTRACTION: LOOKING {direction}!")
                                event_logger.log(
                                    driver=(recognized_driver_id or "unknown"),
                                    event=f"distraction_{direction.lower()}",
                                    head_direction=direction,
                                    ear=ear,
                                    duration=distraction_detector.off_center_seconds,
                                )

                            # ----------------------------------
                            # 7. Yawn Detection (MAR)
                            # ----------------------------------
                            t0 = time.perf_counter()
                            mar = YawnDetector.calculate_mar(landmarks, MOUTH_MAR)
                            yawn_alert = yawn_detector.update(mar)
                            if yawn_alert:
                                print("⚠️ YAWN DETECTED!")
                                event_logger.log(
                                    driver=(recognized_driver_id or "unknown"),
                                    event="yawn",
                                    head_direction=head_direction,
                                    ear=ear,
                                )
                            t_mar = (time.perf_counter() - t0) * 1000.0
                            profiler.record("mar", t_mar)

                            # ----------------------------------
                            # 8. Async Phone Detection
                            # ----------------------------------
                            t0 = time.perf_counter()
                            phone_detector.request_async_detection(
                                frame, resize_width=config.PHONE_RESIZE_WIDTH
                            )
                            phone_alert = phone_detector.update(
                                phone_detector.latest_phone_visible
                            )
                            if phone_alert:
                                print("⚠️ PHONE USAGE DETECTED!")
                                event_logger.log(
                                    driver=(recognized_driver_id or "unknown"),
                                    event="phone_usage",
                                    head_direction=head_direction,
                                    ear=ear,
                                )
                            t_phone_main = (time.perf_counter() - t0) * 1000.0
                            profiler.record("phone_main", t_phone_main)

                            # ----------------------------------
                            # 9. UI Alerts & HUD Rendering
                            # ----------------------------------
                            t0 = time.perf_counter()
                            if drowsiness_detector.is_drowsy:
                                ScreenManager.draw_drowsiness_alert(frame)
                            elif phone_detector.is_phone_detected:
                                ScreenManager.draw_phone_alert(frame)
                            elif distraction_detector.is_distracted:
                                ScreenManager.draw_distraction_alert(
                                    frame, direction=distraction_detector.current_direction
                                )
                            elif yawn_detector.is_yawning:
                                ScreenManager.draw_yawn_alert(frame)

                            if drowsiness_detector.is_drowsy:
                                attention_state = "DROWSY"
                            elif phone_detector.is_phone_detected:
                                attention_state = "PHONE"
                            elif distraction_detector.is_distracted:
                                attention_state = "DISTRACTED"
                            elif yawn_detector.is_yawning:
                                attention_state = "YAWNING"
                            else:
                                attention_state = "OK"

                            driver_label = (
                                recognized_driver_id.replace("_", " ").title()
                                if recognized_driver_id
                                else "Unknown"
                            )

                            ScreenManager.draw_monitoring_hud(
                                frame,
                                driver_label,
                                ear,
                                blink_detector.blink_count,
                                head_direction=head_direction,
                                attention_state=attention_state,
                                yawn_count=yawn_detector.yawn_count,
                            )
                            t_hud = (time.perf_counter() - t0) * 1000.0
                            profiler.record("hud", t_hud)

            # --------------------------------------------------
            # 10. FPS Display & Frame Presentation
            # --------------------------------------------------
            current_fps = profiler.fps
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}",
                (w - 140, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            t0 = time.perf_counter()
            cv2.imshow("AI Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            t_display = (time.perf_counter() - t0) * 1000.0
            profiler.record("display", t_display)

            t_loop_total = (time.perf_counter() - t_loop_start) * 1000.0
            profiler.record("loop_total", t_loop_total)

            # --------------------------------------------------
            # Periodic Performance Logging
            # --------------------------------------------------
            if (
                config.SHOW_PROFILING
                and is_monitoring
                and frame_counter % config.PROFILING_LOG_INTERVAL == 0
            ):
                _last_yolo, avg_yolo, max_yolo = phone_detector.get_profiling_stats()
                print(
                    f"\n[Performance Profile] FPS: {current_fps:.1f} | Loop: {profiler.get_avg('loop_total'):.1f}ms\n"
                    f"  ├─ Camera Read:    {profiler.get_avg('camera'):.1f}ms\n"
                    f"  ├─ Face Detection: {profiler.get_avg('face_det'):.1f}ms (every {config.FACE_DETECT_EVERY_N_FRAMES} frames)\n"
                    f"  ├─ Face Mesh:      {profiler.get_avg('face_mesh'):.1f}ms (every {config.FACE_MESH_EVERY_N_FRAMES} frames)\n"
                    f"  ├─ EAR / Blinks:   {profiler.get_avg('ear'):.2f}ms\n"
                    f"  ├─ MAR / Yawn:     {profiler.get_avg('mar'):.2f}ms\n"
                    f"  ├─ Head Pose:      {profiler.get_avg('head_pose'):.2f}ms (every {config.HEAD_POSE_EVERY_N_FRAMES} frames)\n"
                    f"  ├─ Phone (Main):   {profiler.get_avg('phone_main'):.2f}ms [Worker: avg {avg_yolo:.1f}ms, max {max_yolo:.1f}ms]\n"
                    f"  ├─ Identity (Main):{profiler.get_avg('identity_main'):.2f}ms [Worker: last {identity_state.last_duration_ms:.0f}ms]\n"
                    f"  └─ HUD & Display:  {(profiler.get_avg('hud') + profiler.get_avg('display')):.1f}ms"
                )

    finally:
        phone_detector.close()
        face_mesh.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()