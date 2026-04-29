# Open-Category Hand Gesture Recognition + Battle Game

This project is a MediaPipe landmark-based hand gesture recognition system with open-category detection. Instead of training on raw camera images, it extracts hand landmarks, normalizes them, trains an SVM classifier, and uses a confidence threshold to reject gestures that do not look like one of the known classes.

The project supports six known gestures:

- `up`
- `down`
- `left`
- `right`
- `stop`
- `zero`

It also supports collecting unknown gesture samples for threshold testing and open-category evaluation.

---

## Project Structure

| Path | Description |
|------|-------------|
| `collect_landmarks.py` | Webcam tool for collecting MediaPipe hand landmark samples. |
| `train_model.py` | Trains an SVM classifier on `data/known/` and saves `models/gesture_model.pkl`. |
| `evaluate_model.py` | Evaluates known-class accuracy and unknown detection across confidence thresholds. |
| `realtime_demo.py` | Live webcam demo that predicts known gestures or labels low-confidence input as unknown. |
| `game.py` | Main gesture battle game using live directional gestures. |
| `unknown_detection.py` | Helper for confidence-threshold-based unknown detection. |
| `utils.py` | Landmark extraction and normalization helpers. |
| `data/known/` | Saved landmark samples for known gestures. |
| `data/unknown/` | Saved landmark samples for unknown gestures. |
| `models/gesture_model.pkl` | Trained gesture classifier and class-name list. |
| `assets/` | Sprite assets used by the game. |

---

## Setup

Install the required Python packages:

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib matplotlib pygame
```

Most scripts use paths relative to this folder, so run commands from inside `open_category`:

```bash
cd open_category
```

---

## 1. Collect Landmark Data

Run:

```bash
python collect_landmarks.py
```

The webcam window opens and draws MediaPipe hand landmarks when a hand is detected. Press a key to save the current landmark vector.

| Key | Saved Label | Folder |
|-----|-------------|--------|
| `u` | `up` | `data/known/up/` |
| `d` | `down` | `data/known/down/` |
| `l` | `left` | `data/known/left/` |
| `r` | `right` | `data/known/right/` |
| `s` | `stop` | `data/known/stop/` |
| `z` | `zero` | `data/known/zero/` |
| `x` | `unknown_misc` | `data/unknown/unknown_misc/` |
| `q` | quit | none |

Each saved file is a normalized `.npy` landmark vector.

---

## 2. Train the Gesture Model

Run:

```bash
python train_model.py
```

The script:

- Loads samples from `data/known/`
- Uses folder names as class labels
- Splits the data into train/test sets
- Trains an SVM classifier with probability output
- Prints a classification report
- Saves the trained model to `models/gesture_model.pkl`

---

## 3. Evaluate Known and Unknown Detection

Run:

```bash
python evaluate_model.py
```

This script loads `models/gesture_model.pkl`, evaluates known gesture samples, evaluates unknown samples, and plots results for several confidence thresholds.

The default unknown threshold in `evaluate_model.py` is:

```python
THRESHOLD = 0.70
```

Higher thresholds usually reject more inputs as unknown. Lower thresholds usually accept more inputs as known classes.

---

## 4. Realtime Gesture Demo

Run:

```bash
python realtime_demo.py
```

The demo opens the webcam, detects one hand, predicts the gesture, and displays the label with confidence. If the model confidence is below the unknown threshold, the prediction is shown as `UNKNOWN`.

Press `q` to quit.

---

## 5. Gesture Battle Game

Run:

```bash
python game.py
```

Gameplay:

- A random sequence of four direction gestures is generated.
- Match the highlighted direction with your hand gesture.
- Hold the gesture briefly to confirm it.
- Completing the full sequence damages the enemy.
- The enemy attacks on a timer if you take too long.
- The match ends when either HP bar reaches zero or the round timer expires.

Controls:

| Key | Action |
|-----|--------|
| `ESC` | Quit the game |

The game uses `up`, `down`, `left`, and `right` as battle inputs. The `stop`, `zero`, and unknown outputs are ignored by the battle logic.

---

## Notes

- A webcam is required for data collection, realtime demo, and gameplay.
- The GitHub version keeps `data/` as an empty placeholder folder because collected `.npy` samples can be large. The trained model can still be uploaded as `models/gesture_model.pkl`.
- `game.py` expects `assets/pixel_characters.png` to exist.
- If the model file is missing, run `python train_model.py` before starting the demo or game.
