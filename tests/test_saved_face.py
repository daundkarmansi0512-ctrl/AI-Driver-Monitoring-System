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

    face = cv2.imread(
        "data/drivers/driver_001/face.jpg"
    )

    if face is None:
        print("❌ Could not load driver_001 face.")
        return

    # Add some surrounding space
    padded = cv2.copyMakeBorder(
        face,
        80,
        80,
        80,
        80,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    faces = app.get(padded)

    print(
        f"Faces detected: {len(faces)}"
    )

    for index, detected in enumerate(faces):

        print(
            f"Face {index + 1} embedding shape: "
            f"{detected.embedding.shape}"
        )


if __name__ == "__main__":
    main()