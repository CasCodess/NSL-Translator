"""
Step 3: Real-Time Translation
-------------------------------
Runs the trained model live on your webcam feed, showing the
recognised sign and building up a sentence as you sign.

Run:
    python translate_live.py

Controls:
    'c' -> clear the current sentence
    'q' -> quit
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from collections import deque, Counter

from hand_utils import create_landmarker, landmarks_to_row, draw_landmarks

MODEL_DIR = "model"


def load_model():
    clf_path = os.path.join(MODEL_DIR, "nsl_model.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if not (os.path.isfile(clf_path) and os.path.isfile(le_path)):
        raise FileNotFoundError("No trained model found. Run train_model.py first.")
    clf = joblib.load(clf_path)
    le = joblib.load(le_path)
    return clf, le


def run(camera_index=0, confidence_threshold=0.6, smoothing_window=8):
    clf, le = load_model()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open webcam. Check camera_index or permissions.")
        return

    landmarker = create_landmarker(num_hands=2)
    last_ts = -1

    recent_preds = deque(maxlen=smoothing_window)
    sentence = []
    last_added = None

    print("Press 'c' to clear sentence, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        result = landmarker.detect_for_video(mp_image, ts_ms)
        draw_landmarks(frame, result)

        display_text = ""
        if result.hand_landmarks:
            features = np.array(landmarks_to_row(result)).reshape(1, -1)
            probs = clf.predict_proba(features)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]

            if confidence >= confidence_threshold:
                label = le.inverse_transform([best_idx])[0]
                recent_preds.append(label)
                display_text = f"{label} ({confidence:.2f})"

                if len(recent_preds) == smoothing_window:
                    most_common, count = Counter(recent_preds).most_common(1)[0]
                    if count >= smoothing_window * 0.7 and most_common != last_added:
                        sentence.append(most_common)
                        last_added = most_common
                        recent_preds.clear()

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {display_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Sentence: " + " ".join(sentence), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("NSL Translator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
            last_added = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run()
