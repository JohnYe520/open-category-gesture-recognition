import cv2
import mediapipe as mp
import numpy as np
import os
import time
from utils import landmarks_to_array, normalize_landmarks

# Keys for each gesture.
KNOWN_GESTURES = {
    'u': 'up',
    'd': 'down',
    'l': 'left',
    'r': 'right',
    's': 'stop',
    'z': 'zero',
}

UNKNOWN_GESTURES = {
    'x': 'unknown_misc'
}

# Where the samples go.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data")

# MediaPipe setup.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam.
cap = cv2.VideoCapture(0)

print("[INFO] Press key to save sample. Press q to quit.")

while True:
    # Get frame from camera.
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip so it looks like a mirror.
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_vec = None

    # Get hand landmarks if hand is found.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = landmarks_to_array(hand_landmarks)
            landmark_vec = normalize_landmarks(landmarks)

    cv2.imshow("Collect Landmarks", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Convert key code to character.
    key_char = chr(key) if key != 255 else None

    # Save known gesture sample.
    if key_char in KNOWN_GESTURES and landmark_vec is not None:
        label = KNOWN_GESTURES[key_char]
        folder = os.path.join(SAVE_DIR, "known", label)
        os.makedirs(folder, exist_ok=True)

        filename = os.path.join(folder, f"{label}_{int(time.time()*1000)}.npy")
        np.save(filename, landmark_vec)
        print(f"[SAVED] {filename}")

    # Save unknown gesture sample.
    elif key_char in UNKNOWN_GESTURES and landmark_vec is not None:
        label = UNKNOWN_GESTURES[key_char]
        folder = os.path.join(SAVE_DIR, "unknown", label)
        os.makedirs(folder, exist_ok=True)

        filename = os.path.join(folder, f"{label}_{int(time.time()*1000)}.npy")
        np.save(filename, landmark_vec)
        print(f"[SAVED] {filename}")

# Clean up.
cap.release()
cv2.destroyAllWindows()
