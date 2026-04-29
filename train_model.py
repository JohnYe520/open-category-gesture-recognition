import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = "data/known"

# Training data.
X = []
y = []

# Folder names become class names.
class_names = sorted(os.listdir(DATA_DIR))

# Load every saved landmark file.
for idx, cls in enumerate(class_names):
    folder = os.path.join(DATA_DIR, cls)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            arr = np.load(os.path.join(folder, file))
            X.append(arr)
            y.append(idx)

X = np.array(X)
y = np.array(y)

print("[INFO] Classes:", class_names)

# Split data for a quick test score.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM model for gesture classification.
model = SVC(probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=class_names))

# Save model and class list together.
os.makedirs("models", exist_ok=True)
joblib.dump((model, class_names), "models/gesture_model.pkl")

print("[INFO] Model saved.")
