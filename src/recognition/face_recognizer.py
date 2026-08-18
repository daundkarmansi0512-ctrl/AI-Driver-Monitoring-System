"""
Face recognition helper.

Uses InsightFace to generate facial embeddings and
recognize previously registered drivers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Handles driver face storage and ML recognition."""

    EMBEDDING_FILE = "face_embedding.npy"

    # --------------------------------------------------
    # Recognition threshold (cosine similarity).
    #
    # The 512-D embeddings are L2-normalized, so
    # their dot product equals cosine similarity:
    #
    #   1.0  = identical face
    #   0.0  = completely unrelated
    #
    # Observed values during initial testing:
    #   Same person  ≈ 0.576
    #   Wrong person ≈ 0.279
    #
    # 0.45 is an INITIAL starting threshold that sits
    # between those two observations.  It should be
    # tuned further with more same-person and
    # different-person samples.
    #
    # Higher = stricter (fewer false accepts,
    #                     more false rejects)
    # Lower  = more lenient (more false accepts,
    #                        fewer false rejects)
    # --------------------------------------------------

    RECOGNITION_THRESHOLD = 0.45

    def __init__(self) -> None:

        self.driver_directory = Path("data/drivers")

        self.driver_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Loading InsightFace...")

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=(640, 640),
        )

        print("✅ InsightFace ready.")

    # --------------------------------------------------
    # DRIVER FOLDER
    # --------------------------------------------------

    def get_driver_folder(
        self,
        driver_id: str,
    ) -> Path:

        folder = self.driver_directory / driver_id

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        return folder

    # --------------------------------------------------
    # FACE PATH
    # --------------------------------------------------

    def get_face_path(
        self,
        driver_id: str,
    ) -> Path:

        return (
            self.get_driver_folder(driver_id)
            / "face.jpg"
        )

    # --------------------------------------------------
    # EMBEDDING PATH
    # --------------------------------------------------

    def get_embedding_path(
        self,
        driver_id: str,
    ) -> Path:

        return (
            self.get_driver_folder(driver_id)
            / self.EMBEDDING_FILE
        )

    # --------------------------------------------------
    # SAVE FACE + EMBEDDING
    # --------------------------------------------------

    def save_face(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        driver_id: str,
    ) -> None:
        """
        Save the driver's face image and generate
        the ML embedding from the original frame.
        """

        x, y, w, h = bbox

        height, width = frame.shape[:2]

        x = max(0, x)
        y = max(0, y)

        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            return

        # ------------------------------------------
        # Save cropped face image
        # ------------------------------------------

        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            return

        folder = self.get_driver_folder(
            driver_id
        )

        face_path = folder / "face.jpg"

        cv2.imwrite(
            str(face_path),
            face,
        )

        print(
            f"✅ Face saved for {driver_id}."
        )

        # ------------------------------------------
        # Generate ML embedding
        # FROM ORIGINAL FRAME
        # ------------------------------------------

        embedding = self.extract_embedding_from_frame(
            frame,
            bbox,
        )

        if embedding is None:

            print(
                "⚠️ Could not generate "
                "face embedding."
            )

            return

        self.save_embedding(
            embedding,
            driver_id,
        )

    # --------------------------------------------------
    # EXTRACT EMBEDDING FROM ORIGINAL FRAME
    # --------------------------------------------------

    def extract_embedding_from_frame(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """
        Generate a face embedding from the original
        camera frame.

        InsightFace performs face detection on the
        original frame instead of the tiny saved crop.
        """

        if frame is None or frame.size == 0:
            return None

        faces = self.face_app.get(
            frame
        )

        if not faces:
            print(
                "⚠️ InsightFace found no face."
            )

            return None

        x, y, w, h = bbox

        target_center_x = x + (w / 2)
        target_center_y = y + (h / 2)

        best_face = None
        best_distance = float("inf")

        # ------------------------------------------
        # Match InsightFace detection with our
        # FaceDetector bounding box
        # ------------------------------------------

        for detected_face in faces:

            box = detected_face.bbox

            center_x = (
                box[0] + box[2]
            ) / 2

            center_y = (
                box[1] + box[3]
            ) / 2

            distance = (
                (center_x - target_center_x) ** 2
                +
                (center_y - target_center_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_face = detected_face

        if best_face is None:
            return None

        embedding = best_face.embedding

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # Normalize embedding
        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding

    # --------------------------------------------------
    # SAVE EMBEDDING
    # --------------------------------------------------

    def save_embedding(
        self,
        embedding: np.ndarray,
        driver_id: str,
    ) -> None:

        embedding_path = (
            self.get_embedding_path(
                driver_id
            )
        )

        np.save(
            str(embedding_path),
            embedding,
        )

        print(
            f"✅ Face embedding saved "
            f"for {driver_id}."
        )

    # --------------------------------------------------
    # LOAD EMBEDDING
    # --------------------------------------------------

    def load_embedding(
        self,
        driver_id: str,
    ) -> np.ndarray | None:

        embedding_path = (
            self.get_embedding_path(
                driver_id
            )
        )

        if not embedding_path.exists():
            return None

        embedding = np.load(
            str(embedding_path)
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    # --------------------------------------------------
    # FIND MATCHING DRIVER
    # --------------------------------------------------

    def find_matching_driver(
        self,
        current_frame: np.ndarray,
        threshold: float | None = None,
    ) -> str | None:
        """
        Find a matching driver using an InsightFace
        embedding generated from the current frame.

        Compares the live face against every saved
        driver embedding and returns the best match
        above the threshold.

        Args:
            current_frame:
                Clean camera frame (no annotations).
            threshold:
                Cosine similarity threshold.
                If None, uses RECOGNITION_THRESHOLD.

        Returns:
            The driver_id of the best match,
            or None if no match exceeds the threshold.
        """

        if threshold is None:
            threshold = self.RECOGNITION_THRESHOLD

        if (
            current_frame is None
            or current_frame.size == 0
        ):
            return None

        faces = self.face_app.get(
            current_frame
        )

        if not faces:
            return None

        # For now, use the largest detected face.
        current_face = max(
            faces,
            key=lambda item: (
                item.bbox[2] - item.bbox[0]
            ) * (
                item.bbox[3] - item.bbox[1]
            ),
        )

        current_embedding = (
            current_face.embedding
        )

        if current_embedding is None:
            return None

        current_embedding = np.asarray(
            current_embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            current_embedding
        )

        if norm == 0:
            return None

        current_embedding = (
            current_embedding / norm
        )

        best_driver_id = None
        best_similarity = -1.0

        for driver_folder in (
            self.driver_directory.iterdir()
        ):

            if not driver_folder.is_dir():
                continue

            driver_id = driver_folder.name

            saved_embedding = (
                self.load_embedding(
                    driver_id
                )
            )

            if saved_embedding is None:
                continue

            similarity = float(
                np.dot(
                    current_embedding,
                    saved_embedding,
                )
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_driver_id = driver_id

        if (
            best_driver_id is not None
            and best_similarity >= threshold
        ):

            print(
                f"✅ Driver recognized: "
                f"{best_driver_id} "
                f"(similarity: "
                f"{best_similarity:.3f})"
            )

            return best_driver_id

        print(
            f"❌ No matching driver found. "
            f"Best similarity: "
            f"{best_similarity:.3f}"
        )

        return None

    # --------------------------------------------------
    # CHECK IDENTITY (for driver-change detection)
    # --------------------------------------------------

    def check_identity(
        self,
        current_frame: np.ndarray,
        expected_embedding: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[bool, str | None, float]:
        """
        Check whether the face in the current frame
        matches the expected driver.

        This is used for periodic identity checks
        during monitoring.  It compares the live face
        against the cached expected_embedding first
        (fast in-memory dot product), and only falls
        back to scanning all saved profiles if there
        is a mismatch.

        Args:
            current_frame:
                Clean camera frame (no annotations).
            expected_embedding:
                The 512-D embedding of the expected
                driver (cached in memory).
            threshold:
                Cosine similarity threshold.
                If None, uses RECOGNITION_THRESHOLD.

        Returns:
            A tuple of:
                is_same_driver:
                    True if the face matches expected.
                other_driver_id:
                    If a different registered driver,
                    their ID.  None if unknown.
                similarity:
                    Cosine similarity with expected.
        """

        if threshold is None:
            threshold = self.RECOGNITION_THRESHOLD

        if (
            current_frame is None
            or current_frame.size == 0
        ):
            return False, None, 0.0

        faces = self.face_app.get(current_frame)

        if not faces:
            return False, None, 0.0

        # Use the largest face in the frame.
        current_face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0])
                * (f.bbox[3] - f.bbox[1])
            ),
        )

        current_embedding = current_face.embedding

        if current_embedding is None:
            return False, None, 0.0

        current_embedding = np.asarray(
            current_embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(current_embedding)

        if norm == 0:
            return False, None, 0.0

        current_embedding = current_embedding / norm

        # ----- Fast path: compare against cached -----

        similarity = float(
            np.dot(
                current_embedding,
                expected_embedding,
            )
        )

        if similarity >= threshold:
            return True, None, similarity

        # ----- Mismatch: check all saved profiles -----

        best_driver_id = None
        best_similarity = -1.0

        for driver_folder in (
            self.driver_directory.iterdir()
        ):
            if not driver_folder.is_dir():
                continue

            driver_id = driver_folder.name

            saved_embedding = (
                self.load_embedding(driver_id)
            )

            if saved_embedding is None:
                continue

            sim = float(
                np.dot(
                    current_embedding,
                    saved_embedding,
                )
            )

            if sim > best_similarity:
                best_similarity = sim
                best_driver_id = driver_id

        if (
            best_driver_id is not None
            and best_similarity >= threshold
        ):
            return False, best_driver_id, similarity

        return False, None, similarity

    # --------------------------------------------------
    # HAS SAVED FACE
    # --------------------------------------------------

    def has_saved_face(
        self,
        driver_id: str,
    ) -> bool:

        return self.get_face_path(
            driver_id
        ).exists()

    # --------------------------------------------------
    # LOAD FACE
    # --------------------------------------------------

    def load_face(
        self,
        driver_id: str,
    ) -> np.ndarray | None:

        face_path = self.get_face_path(
            driver_id
        )

        if not face_path.exists():
            return None

        return cv2.imread(
            str(face_path)
        )

    # --------------------------------------------------
    # DELETE FACE + EMBEDDING
    # --------------------------------------------------

    def delete_face(
        self,
        driver_id: str,
    ) -> None:

        face_path = self.get_face_path(
            driver_id
        )

        embedding_path = (
            self.get_embedding_path(
                driver_id
            )
        )

        if face_path.exists():
            face_path.unlink()

        if embedding_path.exists():
            embedding_path.unlink()

        print(
            f"🗑 Face data deleted "
            f"for {driver_id}."
        )