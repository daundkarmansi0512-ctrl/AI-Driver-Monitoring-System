import cv2

from src.recognition.face_recognizer import FaceRecognizer


def main() -> None:

    recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW,
   )

    if not camera.isOpened():
        print("❌ Could not open camera.")
        return

    print()
    print("========================================")
    print("DRIVER EMBEDDING CAPTURE")
    print("========================================")
    print("Look directly at the camera.")
    print("Press SPACE to capture.")
    print("Press Q to quit.")
    print()

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Could not read camera frame.")
            break

        display = frame.copy()

        faces = recognizer.face_app.get(frame)

        cv2.putText(
            display,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "SPACE = capture | Q = quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Driver Embedding Capture",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):

            if len(faces) != 1:

                print(
                    "⚠️ Please make sure exactly "
                    "one face is visible."
                )

                continue

            face = faces[0]

            embedding = face.embedding

            if embedding is None:

                print(
                    "❌ Could not generate embedding."
                )

                continue

            embedding = embedding.astype(
                "float32"
            )

            # Normalize embedding
            norm = cv2.norm(
                embedding
            )

            if norm == 0:

                print(
                    "❌ Invalid embedding."
                )

                continue

            embedding = embedding / norm

            recognizer.save_embedding(
                embedding,
                "driver_001",
            )

            print()
            print(
                "========================================"
            )
            print(
                "✅ DRIVER 001 EMBEDDING SAVED"
            )
            print(
                "========================================"
            )
            print(
                f"Shape: {embedding.shape}"
            )
            print()

            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()