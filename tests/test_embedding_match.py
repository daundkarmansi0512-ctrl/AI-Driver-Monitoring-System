import cv2

from src.recognition.face_recognizer import FaceRecognizer


def main() -> None:

    recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        print("❌ Camera could not be opened.")
        return

    print()
    print("========================================")
    print("DRIVER RECOGNITION TEST")
    print("========================================")
    print("Look at the camera.")
    print("Press SPACE to test recognition.")
    print("Press Q to quit.")
    print()

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Could not read frame.")
            break

        cv2.imshow(
            "Recognition Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):

            print()
            print("Testing recognition...")

            driver_id = (
                recognizer.find_matching_driver(
                    frame
                )
            )

            if driver_id is not None:

                print(
                    f"✅ Recognized as: "
                    f"{driver_id}"
                )

            else:

                print(
                    "❌ Driver not recognized."
                )

            print()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()