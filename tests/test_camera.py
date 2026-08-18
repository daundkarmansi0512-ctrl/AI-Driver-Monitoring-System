import cv2


def main() -> None:

    camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        print("❌ Camera could not be opened.")
        return

    print("✅ Camera opened.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Frame read failed.")
            break

        cv2.imshow(
            "Camera Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()