"""Entry point for the AI Driver Monitoring System."""

import cv2

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector


def main() -> None:
    """Start the AI Driver Monitoring System."""

    camera = CameraManager()
    detector = FaceDetector()

    try:
        camera.start()

        print("Camera started successfully.")
        print("Press 'Q' to quit.")

        while True:

            frame = camera.read_frame()

            if frame is None:
                print("Failed to read frame.")
                break

            frame, detections = detector.detect_faces(frame)

            cv2.imshow("AI Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()