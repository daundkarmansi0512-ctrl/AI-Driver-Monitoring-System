from src.recognition.face_recognizer import FaceRecognizer


def main() -> None:

    recognizer = FaceRecognizer()

    frame = cv2.imread(
        "test_face.jpg"
    )

    if frame is None:
        print("❌ Could not load test_face.jpg.")
        return

    print("✅ Test image loaded.")

    faces = recognizer.face_app.get(frame)

    print(
        f"Faces detected: {len(faces)}"
    )

    if not faces:
        print("❌ No face detected.")
        return

    face = faces[0]

    embedding = face.embedding

    if embedding is None:
        print("❌ No embedding generated.")
        return

    print(
        f"✅ Embedding generated: "
        f"{embedding.shape}"
    )


if __name__ == "__main__":
    import cv2

    main()