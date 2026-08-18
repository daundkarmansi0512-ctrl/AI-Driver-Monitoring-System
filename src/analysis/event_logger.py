"""
Simple CSV-based event logger for the Driver Monitoring System.

Appends one row per significant event so the session can
be reviewed later.  Does NOT log every frame — only
state transitions (e.g. drowsiness detected, distraction
started, driver changed).
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path


class EventLogger:
    """Log monitoring events to a CSV file."""

    HEADERS = [
        "timestamp",
        "driver",
        "event",
        "head_direction",
        "ear",
        "duration",
    ]

    def __init__(
        self,
        log_dir: str = "logs",
    ) -> None:

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "events.csv"

        # Write header if the file is new or empty
        write_header = (
            not self.log_file.exists()
            or self.log_file.stat().st_size == 0
        )

        if write_header:
            with open(
                self.log_file, "w",
                newline="", encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def log(
        self,
        driver: str,
        event: str,
        head_direction: str = "",
        ear: float = 0.0,
        duration: float = 0.0,
    ) -> None:
        """
        Append one event row to the CSV log.

        Args:
            driver:         Driver ID or name.
            event:          Event type, e.g. "blink",
                            "drowsiness", "distraction_left".
            head_direction: Current head direction.
            ear:            Current EAR value.
            duration:       How long the condition lasted
                            (seconds), if applicable.
        """

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            driver,
            event,
            head_direction,
            f"{ear:.3f}",
            f"{duration:.1f}",
        ]

        try:
            with open(
                self.log_file, "a",
                newline="", encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as exc:
            # Never crash the main loop because of logging
            print(f"⚠️ Logging error: {exc}")
