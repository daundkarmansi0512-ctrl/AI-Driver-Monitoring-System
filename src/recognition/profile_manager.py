"""
Profile Manager

Handles saving and loading driver calibration profiles.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ProfileManager:
    """Save and load driver profiles."""

    DRIVER_DIRECTORY = Path("data/drivers")

    @classmethod
    def get_driver_folder(
        cls,
        driver_id: str,
    ) -> Path:
        """Return the folder for a driver."""

        folder = cls.DRIVER_DIRECTORY / driver_id

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        return folder

    @classmethod
    def get_next_driver_id(cls) -> str:
        """Return the next available driver ID."""

        cls.DRIVER_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        driver_numbers = []

        for folder in cls.DRIVER_DIRECTORY.iterdir():

            if not folder.is_dir():
                continue

            if not folder.name.startswith("driver_"):
                continue

            number = folder.name.replace(
                "driver_",
                "",
            )

            if number.isdigit():
                driver_numbers.append(
                    int(number)
                )

        if not driver_numbers:
            next_number = 1
        else:
            next_number = max(driver_numbers) + 1

        return f"driver_{next_number:03d}"

    @classmethod
    def save_profile(
        cls,
        average_ear: float,
        threshold: float,
        driver_id: str,
    ) -> None:
        """Save driver profile."""

        folder = cls.get_driver_folder(driver_id)

        profile_file = folder / "profile.json"

        profile = {
            "driver_id": driver_id,
            "driver_name": driver_id.replace("_", " ").title(),
            "average_open_ear": average_ear,
            "blink_threshold": threshold,
            "calibration_date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        with open(
            profile_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                profile,
                file,
                indent=4,
            )

        print(f"✅ {driver_id} profile saved.")

    @classmethod
    def load_profile(
        cls,
        driver_id: str,
    ) -> dict | None:
        """Load saved driver profile."""

        profile_file = (
            cls.DRIVER_DIRECTORY
            / driver_id
            / "profile.json"
        )

        if not profile_file.exists():
            return None

        with open(
            profile_file,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    @classmethod
    def profile_exists(
        cls,
        driver_id: str,
    ) -> bool:
        """Check whether a driver's profile exists."""

        profile_file = (
            cls.DRIVER_DIRECTORY
            / driver_id
            / "profile.json"
        )

        return profile_file.exists()

    @classmethod
    def delete_profile(
        cls,
        driver_id: str,
    ) -> None:
        """Delete a driver's profile."""

        profile_file = (
            cls.DRIVER_DIRECTORY
            / driver_id
            / "profile.json"
        )

        if profile_file.exists():

            profile_file.unlink()

            print(
                f"🗑 {driver_id} profile deleted."
            )