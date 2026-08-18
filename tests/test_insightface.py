import cv2

from insightface.app import FaceAnalysis


def main() -> None:

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    image = cv2.imread(
        "test_face.jpg"
    )

    if image is None:
        print("❌ Could not load test_face.jpg.")
        return

    faces = app.get(image)

    print(
        f"Faces detected: {len(faces)}"
    )

    for index, face in enumerate(faces):

        print(
            f"Face {index + 1}:"
        )

        print(
            f"  Bounding box: {face.bbox}"
        )

        print(
            f"  Embedding shape: "
            f"{face.embedding.shape}"
        )


if __name__ == "__main__":
    main()