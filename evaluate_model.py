import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from unknown_detection import predict_with_unknown


KNOWN_DIR = "data/known"
UNKNOWN_DIR = "data/unknown"
MODEL_PATH = "models/gesture_model.pkl"
THRESHOLD = 0.70
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_labeled_samples(root_dir):
    samples = []
    labels = []

    if not os.path.isdir(root_dir):
        return np.array(samples), labels

    class_names = sorted(
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )

    for class_name in class_names:
        folder = os.path.join(root_dir, class_name)
        for file_name in os.listdir(folder):
            if file_name.endswith(".npy"):
                path = os.path.join(folder, file_name)
                samples.append(np.load(path))
                labels.append(class_name)

    return np.array(samples), labels


def predict_labels(model, class_names, samples, threshold):
    predictions = []

    for sample in samples:
        pred, _ = predict_with_unknown(model, sample, threshold=threshold)
        if pred == "unknown":
            predictions.append("unknown")
        else:
            predictions.append(class_names[pred])

    return predictions


def evaluate_known_set(model, class_names, known_x, known_y):
    predictions = predict_labels(model, class_names, known_x, THRESHOLD)

    accuracy = accuracy_score(known_y, predictions)
    macro_f1 = f1_score(known_y, predictions, average="macro", labels=class_names)

    print("=== Known Gesture Evaluation ===")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Samples: {len(known_y)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(
        classification_report(
            known_y,
            predictions,
            labels=class_names + ["unknown"],
            zero_division=0,
        )
    )


def evaluate_unknown_set(model, class_names, unknown_x):
    predictions = predict_labels(model, class_names, unknown_x, THRESHOLD)
    detected_unknown = sum(pred == "unknown" for pred in predictions)
    misclassified_known = len(predictions) - detected_unknown

    total_unknown = len(predictions)
    unknown_detection_rate = detected_unknown / total_unknown if total_unknown else 0.0
    false_positive_rate = misclassified_known / total_unknown if total_unknown else 0.0

    print("=== Unknown Gesture Evaluation ===")
    print(f"Samples: {total_unknown}")
    print(f"Detected as unknown: {detected_unknown}")
    print(f"Misclassified as known: {misclassified_known}")
    print(f"Unknown detection rate: {unknown_detection_rate:.4f}")
    print(f"False positive rate: {false_positive_rate:.4f}")


def collect_threshold_results(model, class_names, known_x, known_y, unknown_x):
    results = []

    for threshold in THRESHOLDS:
        known_predictions = predict_labels(model, class_names, known_x, threshold)
        known_accuracy = accuracy_score(known_y, known_predictions)
        known_f1 = f1_score(known_y, known_predictions, average="macro", labels=class_names)

        unknown_predictions = predict_labels(model, class_names, unknown_x, threshold)
        detected_unknown = sum(pred == "unknown" for pred in unknown_predictions)
        total_unknown = len(unknown_predictions)
        misclassified_known = total_unknown - detected_unknown

        unknown_detection_rate = detected_unknown / total_unknown if total_unknown else 0.0
        false_positive_rate = misclassified_known / total_unknown if total_unknown else 0.0

        results.append({
            "threshold": threshold,
            "known_accuracy": known_accuracy,
            "known_f1": known_f1,
            "unknown_detection_rate": unknown_detection_rate,
            "false_positive_rate": false_positive_rate,
        })

    return results


def plot_threshold_results(results):
    thresholds = [row["threshold"] for row in results]
    known_accuracy = [row["known_accuracy"] for row in results]
    known_f1 = [row["known_f1"] for row in results]
    unknown_detection = [row["unknown_detection_rate"] for row in results]
    false_positive = [row["false_positive_rate"] for row in results]

    plt.figure(figsize=(9, 5.5))
    plt.plot(thresholds, known_accuracy, marker="o", linewidth=2, label="Known Accuracy")
    plt.plot(thresholds, known_f1, marker="s", linewidth=2, label="Known Macro F1")
    plt.plot(thresholds, unknown_detection, marker="^", linewidth=2, label="Unknown Detection Rate")
    plt.plot(thresholds, false_positive, marker="d", linewidth=2, label="False Positive Rate")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Comparison")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    model, class_names = joblib.load(MODEL_PATH)
    known_x, known_y = load_labeled_samples(KNOWN_DIR)
    unknown_x, _ = load_labeled_samples(UNKNOWN_DIR)

    print(f"[INFO] Loaded model from {MODEL_PATH}")
    print(f"[INFO] Known classes: {class_names}")
    print(f"[INFO] Threshold: {THRESHOLD:.2f}")
    print()

    if len(known_y) == 0:
        print("[WARN] No known samples found.")
        return

    evaluate_known_set(model, class_names, known_x, known_y)
    print()

    if len(unknown_x) == 0:
        print("[WARN] No unknown samples found.")
        return

    evaluate_unknown_set(model, class_names, unknown_x)
    print()

    results = collect_threshold_results(model, class_names, known_x, known_y, unknown_x)
    plot_threshold_results(results)


if __name__ == "__main__":
    main()
