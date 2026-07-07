"""Entry point for the AI Driver Monitoring System."""

import cv2

from src.camera.camera_manager import CameraManager
from src.detection.face_detector import FaceDetector
from src.detection.face_mesh import FaceMeshDetector


def main() -> None:
    """Start the AI Driver Monitoring System."""

    camera = CameraManager()
    face_detector = FaceDetector()
    face_mesh = FaceMeshDetector()

    try:
        camera.start()

        print("Camera started successfully.")
        print("Press 'Q' to quit.")

        while True:

            frame = camera.read_frame()

            if frame is None:
                print("Failed to read frame.")
                break

            # Step 1: Detect face
            frame, detections = face_detector.detect_faces(frame)

            # Step 2: Detect facial landmarks
            frame, landmarks = face_mesh.detect_landmarks(frame)

            cv2.imshow("AI Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        face_mesh.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()