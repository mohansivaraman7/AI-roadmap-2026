"""
Cricket Umpire Gesture Recognition — CNN + LSTM
=================================================

Goal
----
Given a short video clip of a cricket umpire making a signal (Out, Six,
Four, Wide, No-Ball, Leg-Bye, Bye, TV-Umpire/Review, No-Signal), predict
which gesture is being shown.

Approach
--------
1. FRAME EXTRACTION   : Sample N frames evenly across each video clip.
2. SPATIAL FEATURES    : Run every frame through a pretrained CNN
                          (MobileNetV2, frozen) to get a compact feature
                          vector per frame. This is much cheaper than
                          training a 3D-CNN from scratch and works well
                          with small cricket-gesture datasets.
3. TEMPORAL MODELING   : Feed the sequence of per-frame feature vectors
                          into an LSTM, which learns how the pose changes
                          over time (e.g. arm raising = Out, waving arm
                          side to side = Wide).
4. CLASSIFICATION HEAD : Dense + Softmax over gesture classes.
5. INFERENCE            : Given a brand-new video, extract frames, run
                          them through the same CNN+LSTM, and output the
                          predicted gesture with confidence.

Expected dataset layout (you supply this)
------------------------------------------
dataset/
  train/
    out/          clip1.mp4  clip2.mp4 ...
    six/          clip1.mp4  clip2.mp4 ...
    four/         ...
    wide/         ...
    no_ball/      ...
    leg_bye/      ...
    bye/          ...
    tv_umpire/    ...
    no_signal/    ...
  val/
    out/ ...
    six/ ...
    ...

Each sub-folder name under train/ and val/ becomes a class label
automatically (sorted alphabetically -> index).

Usage
-----
# 1) Train
python umpire_gesture_recognition.py train --data_dir dataset --epochs 30

# 2) Predict on a new video
python umpire_gesture_recognition.py predict --video new_clip.mp4 --model model.h5 --labels labels.json

Dependencies
------------
pip install tensorflow opencv-python numpy scikit-learn --break-system-packages
"""

import os
import json
import argparse
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
IMG_SIZE = 128          # frame resize (H=W=IMG_SIZE)
SEQUENCE_LENGTH = 20     # number of frames sampled per video clip
FEATURE_DIM = 1280       # MobileNetV2 (alpha=1.0, include_top=False) output dim
BATCH_SIZE = 8
DEFAULT_EPOCHS = 30


# --------------------------------------------------------------------------
# 1. Frame extraction
# --------------------------------------------------------------------------
def extract_frames(video_path, seq_len=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """Uniformly sample `seq_len` frames from a video and resize them."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not read frames from {video_path}")

    # Evenly spaced frame indices across the whole clip
    indices = np.linspace(0, max(total_frames - 1, 0), seq_len, dtype=int)

    frames = []
    idx_set = set(indices.tolist())
    current = 0
    grabbed = {}

    while current <= max(indices) and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            grabbed[current] = frame
        current += 1
    cap.release()

    # Preserve requested order; pad by repeating last frame if video was short
    ordered = []
    last = None
    for i in indices:
        if i in grabbed:
            last = grabbed[i]
        if last is None:
            # video shorter than expected / unreadable start — use a black frame
            last = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        ordered.append(last)

    return np.array(ordered, dtype=np.float32)  # (seq_len, H, W, 3)


# --------------------------------------------------------------------------
# 2. Dataset builder — turns a folder of videos into (X, y) arrays
# --------------------------------------------------------------------------
def build_label_map(data_dir_split):
    classes = sorted(
        d for d in os.listdir(data_dir_split)
        if os.path.isdir(os.path.join(data_dir_split, d))
    )
    return {c: i for i, c in enumerate(classes)}


def load_split(data_dir_split, label_map, seq_len=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    X, y = [], []
    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    for class_name, class_idx in label_map.items():
        class_dir = os.path.join(data_dir_split, class_name)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(video_exts):
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                frames = extract_frames(fpath, seq_len, img_size)
            except ValueError as e:
                print(f"  [skip] {e}")
                continue
            X.append(frames)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# --------------------------------------------------------------------------
# 3. Model — TimeDistributed CNN (frozen MobileNetV2) + LSTM
# --------------------------------------------------------------------------
def build_cnn_lstm_model(num_classes, seq_len=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    # Frozen, pretrained per-frame feature extractor
    cnn_base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    cnn_base.trainable = False  # freeze — fine-tune later if you have lots of data

    video_input = layers.Input(shape=(seq_len, img_size, img_size, 3), name="video_input")

    # Preprocess + extract features for every frame in the sequence
    x = layers.TimeDistributed(
        layers.Lambda(preprocess_input), name="preprocess"
    )(video_input)
    x = layers.TimeDistributed(cnn_base, name="cnn_features")(x)  # (batch, seq_len, FEATURE_DIM)

    # Temporal modeling over the sequence of frame features
    x = layers.Masking()(x)
    x = layers.LSTM(256, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(128, dropout=0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="gesture")(x)

    model = models.Model(video_input, outputs, name="umpire_gesture_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# 4. Training
# --------------------------------------------------------------------------
def train(data_dir, epochs=DEFAULT_EPOCHS, out_model="model.h5", out_labels="labels.json"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    label_map = build_label_map(train_dir)
    print("Classes found:", label_map)

    print("Loading training videos (this extracts + resizes frames)...")
    X_train, y_train = load_split(train_dir, label_map)
    print(f"  train set: {X_train.shape}, labels: {y_train.shape}")

    print("Loading validation videos...")
    X_val, y_val = load_split(val_dir, label_map)
    print(f"  val set: {X_val.shape}, labels: {y_val.shape}")

    model = build_cnn_lstm_model(num_classes=len(label_map))
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        callbacks.ModelCheckpoint(out_model, save_best_only=True, monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=cb,
    )

    model.save(out_model)
    with open(out_labels, "w") as f:
        json.dump({v: k for k, v in label_map.items()}, f, indent=2)

    print(f"\nSaved model -> {out_model}")
    print(f"Saved label map -> {out_labels}")


# --------------------------------------------------------------------------
# 5. Inference on a brand-new video
# --------------------------------------------------------------------------
def predict(video_path, model_path="model.h5", labels_path="labels.json"):
    model = tf.keras.models.load_model(model_path, compile=False)

    with open(labels_path) as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)  # (1, seq_len, H, W, 3)

    probs = model.predict(frames)[0]
    top_idx = int(np.argmax(probs))

    print(f"\nPredicted gesture: {idx_to_label[top_idx]}  (confidence: {probs[top_idx]:.2%})")
    print("\nAll class probabilities:")
    for i, p in sorted(enumerate(probs), key=lambda x: -x[1]):
        print(f"  {idx_to_label[i]:<12} {p:.2%}")

    return idx_to_label[top_idx], float(probs[top_idx])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket umpire gesture recognition (CNN+LSTM)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the CNN+LSTM model")
    p_train.add_argument("--data_dir", required=True, help="Path to dataset/ (with train/ and val/)")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--out_model", default="model.h5")
    p_train.add_argument("--out_labels", default="labels.json")

    p_pred = sub.add_parser("predict", help="Predict the gesture in a new video")
    p_pred.add_argument("--video", required=True)
    p_pred.add_argument("--model", default="model.h5")
    p_pred.add_argument("--labels", default="labels.json")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data_dir, epochs=args.epochs, out_model=args.out_model, out_labels=args.out_labels)
    elif args.command == "predict":
        predict(args.video, model_path=args.model, labels_path=args.labels)
