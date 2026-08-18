# AI Driver Monitoring System

A real-time **AI-powered driver monitoring system** that uses computer vision and machine learning to detect drowsiness, yawning, distraction, phone usage, identify drivers, and alert on driver changes — all through a standard webcam.

Built as a portfolio / research project demonstrating practical ML and computer-vision techniques.

---

## Motivation

Drowsy driving is a leading cause of road accidents worldwide. This project explores how commodity hardware (a laptop webcam) combined with modern ML models can provide real-time driver safety monitoring — including **personalized drowsiness thresholds**, **head-pose-based distraction detection**, **yawn detection**, **phone usage detection**, **face-recognition-based driver identification**, and **driver-change detection**.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Driver Identification** | InsightFace generates 512-D face embeddings; cosine similarity matches against saved profiles |
| **New-Driver Registration** | Unknown drivers are automatically calibrated and registered with a unique ID |
| **Personalized Calibration** | Collects ~90 EAR samples to compute a per-driver blink threshold |
| **Blink Detection** | Eye Aspect Ratio (EAR) tracks blinks using MediaPipe Face Mesh landmarks |
| **Drowsiness Detection** | Prolonged eye closure triggers an on-screen alert with debounce/cooldown |
| **Head Pose Estimation** | Yaw/Pitch/Roll estimated from facial landmarks via OpenCV solvePnP with neutral-pose calibration |
| **Distraction Detection** | Sustained off-center head position triggers a distraction alert with timer-based logic |
| **Pose-Aware EAR** | EAR-based drowsiness is paused when head rotation makes eye measurements unreliable |
| **Yawn Detection** | Mouth Aspect Ratio (MAR) from MediaPipe landmarks detects sustained mouth opening |
| **Phone Detection** | YOLOv8 nano model detects cell phone usage with duration-based filtering |
| **Driver-Change Detection** | Background thread periodically verifies the seated driver matches the expected one |
| **Event Logging** | CSV-based logging of blinks, drowsiness, yawns, distraction, phone usage, and driver changes |
| **On-Screen UI** | Real-time HUD with driver name, EAR, blinks, yawns, head direction, and attention state |
| **Smooth Camera** | Identity checks run on a background thread — the camera feed never freezes |

---

## Architecture

```
Camera
  │
  ▼
Face Detection (MediaPipe)
  │
  ▼
Face Landmarks (MediaPipe Face Mesh — 468 points)
  ├── EAR Calculation
  │     ├── Blink Detection
  │     └── Drowsiness Detection ◄── (paused when pose unreliable)
  │
  ├── MAR Calculation
  │     └── Yawn Detection (timer-based)
  │
  ├── Head Pose Estimation (OpenCV solvePnP + neutral calibration)
  │     ├── Head Direction (CENTER/LEFT/RIGHT/UP/DOWN)
  │     ├── EAR Reliability Check
  │     └── Distraction Detection (timer-based)
  │
  └── Face Recognition (InsightFace)
        └── Driver Identity / Change Detection (threaded)

Phone Detection (YOLOv8n — every Nth frame)
  └── Phone Usage Detection (timer-based)

              ▼
Decision / Alert Logic
  ├── Attention State (OK / DROWSY / YAWNING / DISTRACTED / PHONE)
  ├── Monitoring HUD
  └── Event Logging (CSV)
```

### Application Flow

```
WELCOME (2s) → INSTRUCTIONS (5s) → COUNTDOWN (3s) → IDENTIFICATION
                                                          │
                                              ┌───────────┴───────────┐
                                              │                       │
                                        KNOWN DRIVER            UNKNOWN DRIVER
                                              │                       │
                                        Load profile            CALIBRATION
                                        Skip calibration             │
                                              │               RESULTS (3s)
                                              │                       │
                                              │               PROFILE SAVED (2s)
                                              │                       │
                                              └───────────────────────┘
                                                          │
                                                     MONITORING
                                          ┌──────────┬────┼────────┬──────────┐
                                          │          │    │        │          │
                                     EAR/Blink/  Head Pose/ MAR/   Phone   Background
                                     Drowsiness  Distraction Yawn  (YOLO)  Identity
                                     (per frame) (per frame)       (N frames)(3s)
```

---

## Driver States

The system distinguishes six states:

| State | Trigger | Alert Color |
|-------|---------|-------------|
| **NORMAL (OK)** | Eyes open, head centered, no phone | Green |
| **DROWSY** | Eyes closed (low EAR) for ≥1 second | Red |
| **YAWNING** | Mouth wide open (high MAR) for ≥1.5 seconds | Yellow |
| **DISTRACTED** | Head turned away from center for ≥2 seconds | Orange |
| **PHONE** | Cell phone visible for ≥1.5 seconds | Purple |
| **DRIVER CHANGE** | Different person detected (3 consecutive mismatches) | Red overlay |

### Why head rotation doesn't cause false drowsiness

When the driver turns their head sideways, the 2D eye landmarks become foreshortened — the eye appears narrower even though it's wide open. This causes EAR to drop artificially.

**Solution:** When head yaw exceeds ±15°, the system marks EAR as **unreliable** and **pauses** the drowsiness timer. The instant the head returns to center, drowsiness detection resumes. Meanwhile, the distraction detector tracks the off-center position.

### Head pose calibration

The system captures the driver's natural resting head position during the first ~1 second of monitoring. This becomes the "neutral baseline" — all subsequent angles are relative to it. This prevents the common problem where a natural forward-looking position reads as "looking down" due to camera angle.

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **OpenCV** | Camera capture, frame display, drawing, solvePnP |
| **MediaPipe** | Face detection, Face Mesh (468 landmarks) |
| **InsightFace** | Face recognition — `buffalo_l` model, 512-D embeddings |
| **ONNX Runtime** | InsightFace model inference (CPU) |
| **Ultralytics YOLOv8** | Phone detection — `yolov8n.pt` nano model (~6 MB) |
| **NumPy** | Embedding storage, numerical operations |
| **Threading** | Non-blocking background identity checks |

---

## ML / Computer-Vision Components

### 1. Face Recognition (InsightFace)

- **Model:** `buffalo_l` (ArcFace-based)
- **Output:** 512-dimensional L2-normalized face embedding
- **Similarity Metric:** Cosine similarity (dot product of normalized vectors)
- **Threshold:** `0.45` (configurable)
- **Usage:** Initial identification + periodic driver-change verification

### 2. Face Mesh (MediaPipe)

- 468 facial landmarks with iris refinement
- Eye landmarks (6 per eye) for EAR
- Mouth landmarks (6 points) for MAR
- 6 key landmarks for head pose estimation
- Runs every frame (~5ms)

### 3. Eye Aspect Ratio (EAR)

```
        p2      p3
         •------•
       /          \
    p1              p4
       \          /
         •------•
        p6      p5

EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
```

### 4. Mouth Aspect Ratio (MAR)

```
          13 (top outer)
           •
          / \
    61 •     • 291
          \ /
           •
          14 (bottom outer)

MAR = (outer_vertical + inner_vertical) / (2 × horizontal)
```

- Wide open mouth (yawn) → MAR ≈ 0.7–1.0
- Closed/talking → MAR ≈ 0.1–0.4
- Duration filter: must stay open ≥1.5s to count as yawn

### 5. Head Pose Estimation

- Uses 6 facial landmarks (nose, chin, eyes, mouth corners)
- 3D model points mapped to 2D image via OpenCV `solvePnP`
- Neutral pose calibration from first ~30 frames
- Hysteresis prevents direction flickering
- Outputs: CENTER, LEFT, RIGHT, UP, DOWN

### 6. Phone Detection (YOLOv8)

- **Model:** `yolov8n.pt` (nano, ~6 MB, auto-downloads on first run)
- **Class:** "cell phone" (COCO class 67)
- **CPU inference** — runs every 3rd frame to save resources
- **Duration filter:** phone must be visible ≥1.5s before alert
- **Modular:** if ultralytics is not installed, phone detection is silently disabled

### 7. Personalized Calibration

- Collects 90 EAR samples (~3 seconds at 30 FPS)
- Computes average open-eye EAR per driver
- Threshold = `average_open_EAR × 0.85`

### 8. Temporal Logic

- **Blink:** EAR drops below threshold for 1–8 frames
- **Drowsiness:** EAR below threshold for ≥1 second (head centered only)
- **Yawn:** MAR above threshold for ≥1.5 seconds
- **Distraction:** Head off-center for ≥2 seconds
- **Phone usage:** Phone visible for ≥1.5 seconds
- **Driver change:** 3 consecutive identity mismatches

---

## Event Logging

Events are logged to `logs/events.csv` (one row per state transition, not per frame):

| Column | Description |
|--------|-------------|
| `timestamp` | When the event occurred |
| `driver` | Driver ID |
| `event` | Event type (blink, drowsiness, yawn, distraction_left, phone_usage, driver_change, etc.) |
| `head_direction` | Head direction at the time |
| `ear` | EAR value |
| `duration` | Duration of the condition (seconds) |

---

## Configuration

All detection thresholds are in [`src/analysis/config.py`](src/analysis/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HEAD_YAW_THRESHOLD` | 18.0° | Yaw beyond which head is off-center |
| `HEAD_PITCH_THRESHOLD` | 15.0° | Pitch beyond which head is off-center |
| `HEAD_DIRECTION_HYSTERESIS` | 6.0° | Dead-zone to prevent flickering |
| `HEAD_POSE_CALIBRATION_FRAMES` | 30 | Frames for neutral baseline |
| `HEAD_POSE_DEBUG` | False | Print raw angles (development) |
| `EAR_POSE_SUPPRESS_YAW` | 15.0° | Yaw beyond which EAR is unreliable |
| `DROWSINESS_CLOSED_TIME` | 1.0s | Eye closure for drowsiness |
| `DROWSINESS_COOLDOWN` | 5.0s | Cooldown between drowsiness alerts |
| `DISTRACTION_DURATION` | 2.0s | Off-center for distraction |
| `YAWN_MAR_THRESHOLD` | 0.65 | MAR for yawn detection |
| `YAWN_DURATION` | 1.5s | Mouth open for yawn |
| `PHONE_MODEL_NAME` | yolov8n.pt | YOLO model file |
| `PHONE_CONFIDENCE` | 0.5 | Detection confidence |
| `PHONE_DETECT_EVERY_N_FRAMES` | 3 | Frame skip for performance |
| `PHONE_DURATION` | 1.5s | Phone visible for alert |

---

## Project Structure

```
AI-Driver-Monitoring-System/
│
├── main.py                          # Application entry point
│
├── src/
│   ├── analysis/
│   │   ├── blink_detector.py        # Blink counting via EAR
│   │   ├── config.py                # Central thresholds configuration
│   │   ├── distraction_detector.py  # Timer-based distraction detection
│   │   ├── drowsiness_detector.py   # Prolonged eye closure detection
│   │   ├── event_logger.py          # CSV event logger
│   │   ├── eye_aspect_ratio.py      # EAR calculation
│   │   ├── head_pose.py             # Head pose estimation (solvePnP + calibration)
│   │   ├── landmark_indices.py      # MediaPipe landmark IDs
│   │   └── yawn_detector.py         # MAR-based yawn detection
│   │
│   ├── calibration/
│   │   └── calibrator.py            # Per-driver EAR calibration
│   │
│   ├── camera/
│   │   └── camera_manager.py        # Webcam wrapper (DirectShow)
│   │
│   ├── detection/
│   │   ├── face_detector.py         # MediaPipe face detection
│   │   ├── face_mesh.py             # MediaPipe Face Mesh
│   │   └── phone_detector.py        # YOLOv8 phone detection
│   │
│   ├── recognition/
│   │   ├── face_recognizer.py       # InsightFace embeddings + matching
│   │   └── profile_manager.py       # JSON profile storage
│   │
│   └── ui/
│       ├── app_states.py            # AppState enum (8 states)
│       ├── flow_manager.py          # State machine + transitions
│       └── screen_manager.py        # OpenCV drawing utilities
│
├── data/
│   └── drivers/                     # Per-driver folders
│       └── driver_001/
│           ├── face.jpg             # Saved face image
│           ├── face_embedding.npy   # 512-D embedding
│           └── profile.json         # Calibration data
│
├── logs/
│   └── events.csv                   # Session event log
│
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A working webcam
- Windows (tested), Linux/macOS should also work
- Internet connection (first run only — to download YOLOv8n weights ~6 MB)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Driver-Monitoring-System.git
cd AI-Driver-Monitoring-System

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

```bash
python main.py
```

On the first run:
- **InsightFace** downloads the `buffalo_l` model (~250 MB) for face recognition
- **YOLOv8n** downloads the nano weights (~6 MB) for phone detection
- Both are cached locally and reused on subsequent runs

Press **q** to quit.

---

## How It Works

### Driver Registration (First Use)

1. System shows welcome screen, instructions, countdown
2. InsightFace scans your face — no match found
3. **"NEW DRIVER DETECTED"** appears
4. Calibration begins: keep eyes naturally open for ~3 seconds
5. System computes your personalized EAR threshold
6. Your face embedding + profile are saved as `driver_001`
7. Monitoring begins

### Returning Driver

1. Same welcome/countdown sequence
2. InsightFace recognizes you — **"DRIVER IDENTIFIED: Driver 001"**
3. Your saved EAR threshold is loaded
4. Monitoring begins immediately — **no calibration needed**

### During Monitoring

- **Head pose calibrates** during the first ~1 second (establishes neutral baseline)
- **Blinks** are counted and displayed on the HUD
- **Yawns** are detected via MAR and counted
- **Head direction** is shown (CENTER/LEFT/RIGHT/UP/DOWN)
- **Drowsiness** is detected when eyes stay closed ≥1 second (head centered)
- **Distraction** is detected when head stays off-center ≥2 seconds
- **Phone usage** is detected when a cell phone is visible ≥1.5 seconds
- A **background thread** checks your identity every 3 seconds
- If a **different person** sits down, the system alerts after 3 consecutive mismatches
- All events are **logged to CSV** for later review

---

## Testing Checklist

| # | Scenario | Expected Result |
|---|----------|-----------------|
| 1 | Face straight, eyes open | Attention: OK, Head: CENTER |
| 2 | Eyes closed ≥1 second | Red: DROWSINESS DETECTED |
| 3 | Head left, eyes open | Head: LEFT, no drowsiness |
| 4 | Head right, eyes open | Head: RIGHT, no drowsiness |
| 5 | Head up | Head: UP |
| 6 | Head down | Head: DOWN |
| 7 | Brief head glance | No distraction alert |
| 8 | Sustained head turn (>2s) | Orange: DISTRACTION |
| 9 | Mouth wide open (>1.5s) | Yellow: YAWN DETECTED |
| 10 | Normal talking | No yawn alert |
| 11 | Phone visible briefly | No phone alert |
| 12 | Phone visible >1.5s | Purple: PHONE USAGE DETECTED |
| 13 | No face visible | "No Driver Detected" |
| 14 | Registered driver sits down | Identified → monitoring |
| 15 | Different person sits down | Driver change alert |
| 16 | Head turned + eyes open | Distraction only, NOT drowsiness |

---

## Known Limitations

- **Head pose estimation** uses a simplified 3D face model — accuracy varies with face shape and camera angle
- **Neutral calibration** assumes the driver looks straight at the camera during the first ~1 second
- **Single-camera system** — no depth sensing or IR
- **CPU-only inference** — InsightFace and YOLO run on CPU (GPU would be faster)
- **Recognition threshold (0.45)** is based on limited testing
- **Lighting sensitivity** — extreme darkness or backlighting reduces accuracy
- **MAR for yawn detection** can be affected by facial hair or masks
- **Phone detection** depends on YOLOv8n accuracy — may miss small or partially hidden phones
- **Not a production safety system** — this is a research/portfolio project

---

## License

This project is for educational and portfolio purposes.
