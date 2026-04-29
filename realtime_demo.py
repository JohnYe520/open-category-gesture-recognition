import cv2
import mediapipe as mp
import numpy as np
import joblib
from utils import landmarks_to_array, normalize_landmarks
from unknown_detection import predict_with_unknown

# Load trained model.
model, class_names = joblib.load("models/gesture_model.pkl")

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

print("[INFO] Press q to quit.")

while True:
    # Get camera frame.
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror preview.
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label_text = "No Hand"

    # Predict gesture if a hand is visible.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = landmarks_to_array(hand_landmarks)
            vec = normalize_landmarks(landmarks)

            pred, conf = predict_with_unknown(model, vec)

            if pred == "unknown":
                label_text = f"UNKNOWN ({conf:.2f})"
            else:
                label_text = f"{class_names[pred]} ({conf:.2f})"

    # Draw prediction on screen.
    cv2.putText(frame, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Realtime Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up.
cap.release()
cv2.destroyAllWindows()
