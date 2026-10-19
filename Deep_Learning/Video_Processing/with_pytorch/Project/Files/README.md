# Cricket Umpire Gesture Recognition (CNN + LSTM)

Predict which signal a cricket umpire is making — **Out, Six, Four, Wide,
No-Ball, Leg-Bye, Bye, TV Umpire, No-Signal** (or any classes you define)
— from a short video clip.

## How it works

```
video clip
   │
   ▼
[1] Sample 20 evenly-spaced frames  (OpenCV)
   │
   ▼
[2] Each frame → MobileNetV2 (frozen, ImageNet-pretrained)  → 1280-d feature vector
   │
   ▼
[3] Sequence of 20 feature vectors → 2-layer LSTM  (learns motion over time)
   │
   ▼
[4] Dense + Softmax → predicted gesture + confidence
```

CNN = per-frame *what does this look like*. LSTM = *how does it change
over time* (arm rising steadily = Out, side-to-side wave = Wide, etc).

## Files

| File | Purpose |
|---|---|
| `umpire_gesture_recognition.py` | Model definition, training, and prediction |
| `prepare_dataset.py` | Organizes **your own real videos** into the folder structure the model expects |
| `generate_sample_dataset.py` | Generates **synthetic placeholder clips** so you can test the pipeline works before you have real footage |

## 1. Install dependencies

```bash
pip install tensorflow opencv-python numpy scikit-learn --break-system-packages
```

## 2. Get videos into the right folder structure

The trainer expects:

```
dataset/
  train/
    out/       clip1.mp4  clip2.mp4 ...
    six/       ...
    four/      ...
    wide/      ...
    no_ball/   ...
  val/
    out/       ...
    six/       ...
    ...
```

Folder names under `train/` become your class labels automatically.

### Option A — You already have labeled real clips
Put them all in one folder, e.g. `raw_videos/`, then either:

**A CSV mapping** (`labels.csv` with columns `filename,label`):
```bash
python prepare_dataset.py --raw_dir raw_videos --csv labels.csv --out_dir dataset
```

**Or filename prefixes** (`out_001.mp4`, `six_002.mp4`, `wide_001.mp4`, ...):
```bash
python prepare_dataset.py --raw_dir raw_videos --infer_from_filename --out_dir dataset
```

This copies (never moves) your files, splitting each class 80/20 into
train/val by default (`--val_split 0.2` to change it).

### Option B — You don't have real footage yet
Generate a synthetic dataset just to confirm the pipeline runs end-to-end:

```bash
python generate_sample_dataset.py --out_dir dataset --clips_per_class 20
```

⚠️ Synthetic clips are simple animated shapes, not real umpires — they
only prove the code works. Swap in real footage via Option A before you
care about actual accuracy.

## 3. Train

```bash
python umpire_gesture_recognition.py train --data_dir dataset --epochs 30
```

Produces:
- `model.h5` — trained weights
- `labels.json` — maps class index → gesture name

Training tips:
- Start with **15–30 clips per class minimum**; more is better.
- Keep clips short (2–4 seconds) and centered on the gesture.
- If validation accuracy stalls, try more clips per class before tuning
  the model — data usually matters more than architecture here.
- `EarlyStopping` and `ModelCheckpoint` are already wired in, so it
  stops automatically and keeps the best-performing weights.

## 4. Predict on a new video

```bash
python umpire_gesture_recognition.py predict --video new_clip.mp4 --model model.h5 --labels labels.json
```

Output:
```
Predicted gesture: out  (confidence: 92.4%)

All class probabilities:
  out          92.4%
  six           4.1%
  wide          2.0%
  four          1.0%
  no_ball       0.5%
```

## Improving accuracy later

- **Pose estimation front-end**: run frames through MediaPipe/OpenPose
  first and feed joint keypoints (instead of raw MobileNetV2 features)
  into the LSTM — often much more sample-efficient for gesture tasks.
- **Fine-tune the CNN**: unfreeze the last few MobileNetV2 layers once
  you have a few hundred clips per class.
- **Data augmentation**: random crop/flip/brightness on frames during
  training to reduce overfitting on small datasets.
- **More frames**: increase `SEQUENCE_LENGTH` in
  `umpire_gesture_recognition.py` if gestures are fast/short.

## Notes

- All three scripts run locally — no internet access is required except
  for the one-time download of MobileNetV2's ImageNet weights the first
  time you train.
- Real umpire footage must come from you (recorded or properly licensed)
  — this project doesn't include or fetch any copyrighted match footage.
