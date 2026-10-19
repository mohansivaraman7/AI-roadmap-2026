"""
generate_sample_dataset.py
===========================
Creates a small SYNTHETIC dataset of short .mp4 clips so you can test
that the full train -> predict pipeline in umpire_gesture_recognition.py
actually runs end-to-end, before you have real umpire footage.

Each "gesture" class gets its own distinct animated motion pattern
(different shape / direction / speed) drawn on a plain background, so
the CNN+LSTM has something genuinely learnable to distinguish — this is
only for testing the pipeline's plumbing, NOT for real accuracy. Swap
these out for real umpire clips (see prepare_dataset.py) once you're
ready to train a model you'll actually use.

Usage:
    python generate_sample_dataset.py --out_dir dataset --clips_per_class 10
"""

import os
import argparse
import numpy as np
import cv2

CLASSES = ["out", "six", "four", "wide", "no_ball"]

IMG_SIZE = 128
NUM_FRAMES = 30
FPS = 10


def make_clip(path, motion_type, seed):
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (IMG_SIZE, IMG_SIZE))

    bg_color = (30, 30, 30)
    fg_color = tuple(int(c) for c in rng.integers(120, 255, size=3))

    for t in range(NUM_FRAMES):
        frame = np.full((IMG_SIZE, IMG_SIZE, 3), bg_color, dtype=np.uint8)
        progress = t / (NUM_FRAMES - 1)  # 0 -> 1

        cx, cy = IMG_SIZE // 2, IMG_SIZE // 2

        if motion_type == "out":
            # arm raising straight up: a vertical bar growing upward
            height = int(progress * 80)
            cv2.rectangle(frame, (cx - 6, cy + 40 - height), (cx + 6, cy + 40), fg_color, -1)
            cv2.circle(frame, (cx, cy + 40 - height - 10), 10, fg_color, -1)

        elif motion_type == "six":
            # both arms raised up (two bars)
            height = int(progress * 70)
            cv2.rectangle(frame, (cx - 30, cy + 30 - height), (cx - 18, cy + 30), fg_color, -1)
            cv2.rectangle(frame, (cx + 18, cy + 30 - height), (cx + 30, cy + 30), fg_color, -1)

        elif motion_type == "four":
            # one arm waving side to side at chest height
            x_offset = int(np.sin(progress * np.pi * 2) * 30)
            cv2.rectangle(frame, (cx + x_offset - 25, cy - 5), (cx + x_offset + 25, cy + 5), fg_color, -1)

        elif motion_type == "wide":
            # both arms extended horizontally, staying static-ish
            cv2.rectangle(frame, (cx - 50, cy - 5), (cx + 50, cy + 5), fg_color, -1)
            wobble = int(np.sin(progress * np.pi * 4) * 4)
            cv2.rectangle(frame, (cx - 50, cy - 5 + wobble), (cx - 40, cy + 5 + wobble), fg_color, -1)

        elif motion_type == "no_ball":
            # one arm extended sideways, held horizontal
            length = int(progress * 45)
            cv2.rectangle(frame, (cx, cy), (cx + length, cy + 4), fg_color, -1)

        writer.write(frame)

    writer.release()


def generate(out_dir, clips_per_class=10, val_split=0.2):
    seed_counter = 0
    for split, n_clips in (("train", int(clips_per_class * (1 - val_split))),
                            ("val", max(1, int(clips_per_class * val_split)))):
        for cls in CLASSES:
            cls_dir = os.path.join(out_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n_clips):
                path = os.path.join(cls_dir, f"{cls}_{i:03d}.mp4")
                make_clip(path, cls, seed=seed_counter)
                seed_counter += 1
            print(f"  {split}/{cls}: {n_clips} clips")

    print(f"\nSynthetic dataset created at: {out_dir}")
    print("Test the pipeline with:")
    print(f"  python umpire_gesture_recognition.py train --data_dir {out_dir} --epochs 5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic umpire-gesture dataset for pipeline testing")
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--clips_per_class", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    generate(args.out_dir, clips_per_class=args.clips_per_class, val_split=args.val_split)
