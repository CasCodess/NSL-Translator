"""
Step 1: Data Collection
------------------------
Records hand-landmark samples from your webcam for a sign you name
(e.g. a Namibian Sign Language letter, word, or phrase you're teaching
the system) and appends them to data/landmarks.csv.

Run:
    python collect_data.py

Controls while the webcam window is open:
    's' -> save the current frame's landmarks as a sample
    'q' -> quit early

Note: the first run downloads a small hand-detection model file
(~7 MB), which needs an internet connection just that once.
"""

import cv2
import mediapipe as mp
import csv
import os
import time

from hand_utils import create_landmarker, landmarks_to_row, draw_landmarks

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def collect(label, num_samples=200, camera_index=0):
    csv_path = os.path.join(DATA_DIR, "landmarks.csv")
    file_exists = os.path.isfile(csv_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open webcam. Check camera_index or permissions.")
        return

    landmarker = create_landmarker(num_hands=2)
    last_ts = -1

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"]
            for hand_idx in range(2):  # up to 2 hands
                for i in range(21):     # 21 landmarks per hand
                    header += [f"h{hand_idx}_x{i}", f"h{hand_idx}_y{i}", f"h{hand_idx}_z{i}"]
            writer.writerow(header)

        count = 0
        print(f"Collecting samples for '{label}'.")
        print("Show the sign clearly, press 's' to save a sample, 'q' to stop early.")

        while count < num_samples:
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
            row_data = landmarks_to_row(result)

            cv2.putText(
                frame,
                f"Label: {label} | Samples: {count}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Collect NSL Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and result.hand_landmarks:
                writer.writerow([label] + row_data)
                count += 1
                time.sleep(0.05)
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"Done. Saved samples for '{label}' to {csv_path}")


if __name__ == "__main__":
    label = input("Enter the sign label (e.g. 'HELLO', 'THANK_YOU', 'A'): ").strip().upper()
    n = input("How many samples to collect? [default 200]: ").strip()
    n = int(n) if n else 200
    collect(label, n)
