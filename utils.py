import numpy as np


def landmarks_to_array(hand_landmarks):
    # Keep only x/y points.
    points = []
    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y])
    return np.array(points)


def normalize_landmarks(landmarks):
    # Move wrist to the origin.
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale hand size.
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 1e-6:
        landmarks = landmarks / max_dist

    return landmarks.flatten()
