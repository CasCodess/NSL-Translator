"""
Shared helper for hand landmark detection.

Newer mediapipe releases removed the old `mp.solutions.hands` API.
Detection now goes through the Tasks API (`HandLandmarker`), which needs
a small model file downloaded once. This module wraps that so the other
scripts don't have to deal with it directly.
"""

import os
import urllib.request
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Standard 21-point hand skeleton connections (thumb, fingers, palm base).
# Hardcoded because mp.solutions.hands.HAND_CONNECTIONS no longer exists.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def ensure_model():
    """Download the hand landmark model once if it isn't already present."""
    if not os.path.isfile(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)
    return MODEL_PATH


def create_landmarker(num_hands=2):
    """Create a HandLandmarker set up for synchronous per-frame video use."""
    model_path = ensure_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return HandLandmarker.create_from_options(options)


def landmarks_to_row(result):
    """Flatten up to 2 hands x 21 landmarks x (x, y, z) into a fixed-length row."""
    row = [0.0] * (2 * 21 * 3)
    if result.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
            for i, lm in enumerate(hand_landmarks):
                base = (hand_idx * 21 + i) * 3
                row[base] = lm.x
                row[base + 1] = lm.y
                row[base + 2] = lm.z
    return row


def draw_landmarks(frame, result):
    """Draw hand points and connecting lines directly with OpenCV."""
    import cv2

    if not result.hand_landmarks:
        return
    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 1)
        for (x, y) in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
