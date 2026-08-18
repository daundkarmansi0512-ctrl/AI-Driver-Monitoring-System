"""
Facial landmark indices used throughout the Driver Monitoring System.

These indices are provided by MediaPipe Face Mesh.
"""

# Left Eye
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,
]

# Right Eye
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380,
]

# Mouth (outer corners — used for basic reference)
MOUTH = [
    61,
    291,
    13,
    14,
]

# ==========================================================
# Mouth landmarks for Mouth Aspect Ratio (MAR) calculation
#
# MediaPipe Face Mesh mouth landmark layout:
#
#          13 (top lip center)
#           •
#          / \
#    61 •     • 291
#  (left)     (right)
#          \ /
#           •
#          14 (bottom lip center)
#
# Additional inner vertical points for better MAR:
#   82  = upper inner lip (between 13 and mouth interior)
#   18  = lower inner lip (between 14 and mouth interior)
# ==========================================================

# Mouth landmarks for MAR calculation:
# [left_corner, right_corner, top_outer, bottom_outer, top_inner, bottom_inner]
MOUTH_MAR = [
    61,    # Left mouth corner
    291,   # Right mouth corner
    13,    # Top lip center (outer)
    14,    # Bottom lip center (outer)
    82,    # Upper inner lip
    18,    # Lower inner lip
]