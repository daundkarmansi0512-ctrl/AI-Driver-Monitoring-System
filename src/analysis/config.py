"""
Central configuration for the Driver Monitoring System.

All detection thresholds and performance settings live here so they are easy to
find and tune without hunting through multiple files.
"""

# ==========================================================
# HEAD POSE THRESHOLDS
# ==========================================================

# Yaw angle (degrees) beyond which the driver is
# considered to be looking left or right.
HEAD_YAW_THRESHOLD = 18.0

# Pitch angle (degrees) beyond which the driver is
# considered to be looking up or down.
HEAD_PITCH_THRESHOLD = 15.0

# Hysteresis (degrees) to prevent direction flickering.
# Direction is entered at threshold, exited at
# threshold - hysteresis.
HEAD_DIRECTION_HYSTERESIS = 6.0

# Number of frames to capture for neutral pose baseline.
# ~30 frames ≈ 1 second at 30 FPS.
HEAD_POSE_CALIBRATION_FRAMES = 30

# Run head pose estimation every N frames to save CPU.
# The cached result is reused between estimations.
HEAD_POSE_EVERY_N_FRAMES = 3

# Print raw/calibrated yaw/pitch each frame.
# Set to True during development, False for normal use.
HEAD_POSE_DEBUG = False

# ==========================================================
# INFERENCE & SCHEDULING OPTIMIZATIONS
# ==========================================================

# Run MediaPipe face detection every N frames during monitoring.
# Only needed for face count (0, 1, multiple).
# Cached result is reused between updates.
FACE_DETECT_EVERY_N_FRAMES = 3

# Run MediaPipe face mesh every N frames during monitoring.
# Landmarks are cached and reused for EAR/MAR between updates.
# Set 1 for every frame, 2 for every other frame.
FACE_MESH_EVERY_N_FRAMES = 2

# Draw the full 468-point mesh tesselation on the frame.
# Disable for ~5-10ms saving per frame.
DRAW_FACE_MESH = False

# ==========================================================
# EAR — POSE RELIABILITY
# ==========================================================

# When absolute yaw exceeds this value the 2D eye
# landmarks are foreshortened and EAR becomes unreliable.
# Drowsiness detection is PAUSED (not permanently ignored)
# while yaw is beyond this limit.  It resumes immediately
# once the head returns inside the range.
EAR_POSE_SUPPRESS_YAW = 15.0

# ==========================================================
# DROWSINESS DETECTION
# ==========================================================

# How long eyes must stay closed before drowsiness fires.
DROWSINESS_CLOSED_TIME = 1.0  # seconds

# Cooldown between repeated drowsiness alerts.
DROWSINESS_COOLDOWN = 5.0  # seconds

# ==========================================================
# DISTRACTION DETECTION
# ==========================================================

# Seconds the head must remain off-center before a
# distraction alert fires.
DISTRACTION_DURATION = 2.0  # seconds

# ==========================================================
# YAWN DETECTION
# ==========================================================

# MAR (Mouth Aspect Ratio) above this value means the
# mouth is open wide enough to potentially be a yawn.
YAWN_MAR_THRESHOLD = 0.65

# Seconds the mouth must stay open above the MAR
# threshold before a yawn is counted.  This rejects
# normal talking which is brief and lower-MAR.
YAWN_DURATION = 1.5  # seconds

# ==========================================================
# PHONE DETECTION (YOLOv8)
# ==========================================================

# YOLO model name or path.  "yolov8n.pt" is the nano
# model (~6 MB) which auto-downloads on first run.
PHONE_MODEL_NAME = "yolov8n.pt"

# Minimum confidence to accept a "cell phone" detection.
PHONE_CONFIDENCE = 0.5

# Minimum time interval (seconds) between background YOLO inferences.
# 0.25s = at most 4 background inferences per second.
# Because YOLO runs in a background thread, the main video loop is never blocked!
PHONE_ASYNC_INTERVAL = 0.25

# Resize frame to this width before sending to YOLO background worker.
# Smaller = faster inference and lower CPU load.
PHONE_RESIZE_WIDTH = 320

# Seconds the phone must be continuously detected
# before the alert fires.  Prevents false alerts from
# a single noisy frame.
PHONE_DURATION = 1.5  # seconds

# ==========================================================
# PROFILING
# ==========================================================

# Show per-component timing breakdown and FPS in the console.
# Set True to diagnose performance, False for clean logs.
SHOW_PROFILING = True

# How often to print detailed profiling stats (in number of frames).
PROFILING_LOG_INTERVAL = 60
