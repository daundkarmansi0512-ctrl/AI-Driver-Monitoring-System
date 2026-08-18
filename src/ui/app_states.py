"""
Application states.
"""

from enum import Enum


class AppState(Enum):

    WELCOME = 1

    INSTRUCTIONS = 2

    COUNTDOWN = 3

    IDENTIFICATION = 4

    CALIBRATION = 5

    RESULTS = 6

    PROFILE_SAVED = 7

    MONITORING = 8