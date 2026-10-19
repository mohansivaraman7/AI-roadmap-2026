"""
prepare_dataset.py
===================
Organizes YOUR OWN raw umpire-gesture video clips into the folder
structure that umpire_gesture_recognition.py expects:

    dataset/train/<class>/*.mp4
    dataset/val/<class>/*.mp4

You don't need to move files by hand — put all your recorded/downloaded
clips in one folder, tell this script which label each one belongs to
(either via a CSV, or by naming files with a class prefix), and it will
split + copy them into the right place.

--------------------------------------------------------------------
OPTION A — CSV mapping (recommended, most flexible)
--------------------------------------------------------------------
Create a CSV like:

    filename,label
    clip001.mp4,out
    clip002.mp4,six
    clip003.mp4,wide
    ...

Then run:
    python prepare_dataset.py --raw_dir raw_videos --csv labels.csv --out_dir dataset --val_split 0.2

--------------------------------------------------------------------
OPTION B — filename prefix (quick and dirty)
--------------------------------------------------------------------
Name your files like: out_001.mp4, out_002.mp4, six_001.mp4, wide_001.mp4
(i.e. "<label>_<anything>.<ext>"), put them all in one folder, then run:

    python prepare_dataset.py --raw_dir raw_videos --infer_from_filename --out_dir dataset --val_split 0.2

--------------------------------------------------------------------
Both options randomly split each class into train/val (default 80/20)
and COPY (not move) the files, so your originals are untouched.
"""

import os
import csv
import shutil
import random
import argparse

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_csv_mapping(csv_path):
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = row["label"].strip().lower()
    return mapping


def infer_label_from_filename(fname):
    # "out_001.mp4" -> "out"   |   "no_ball_014.mp4" -> "no_ball"
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    # drop trailing numeric part(s), e.g. "no_ball_014" -> "no_ball"
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts).lower() if parts else "unknown"


def organize(raw_dir, out_dir, csv_path=None, infer_from_filename=False, val_split=0.2, seed=42):
    if not csv_path and not infer_from_filename:
        raise ValueError("Provide either --csv or --infer_from_filename")

    random.seed(seed)

    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(VIDEO_EXTS)]
    if not files:
        raise ValueError(f"No video files found in {raw_dir}")

    if csv_path:
        mapping = load_csv_mapping(csv_path)
        missing = [f for f in files if f not in mapping]
        if missing:
            print(f"[warning] {len(missing)} files have no CSV label and will be skipped: {missing[:5]}...")
        labeled = {f: mapping[f] for f in files if f in mapping}
    else:
        labeled = {f: infer_label_from_filename(f) for f in files}

    # group by class
    by_class = {}
    for fname, label in labeled.items():
        by_class.setdefault(label, []).append(fname)

    print(f"Found {len(labeled)} labeled videos across {len(by_class)} classes:")
    for label, flist in by_class.items():
        print(f"  {label:<12} {len(flist)} clips")

    for label, flist in by_class.items():
        random.shuffle(flist)
        n_val = max(1, int(len(flist) * val_split)) if len(flist) > 1 else 0
        val_files = flist[:n_val]
        train_files = flist[n_val:]

        for split_name, split_files in (("train", train_files), ("val", val_files)):
            dest_dir = os.path.join(out_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(raw_dir, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)

        print(f"  -> {label}: {len(train_files)} train / {len(val_files)} val")

    print(f"\nDataset ready at: {out_dir}")
    print("You can now run:")
    print(f"  python umpire_gesture_recognition.py train --data_dir {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize raw videos into train/val class folders")
    parser.add_argument("--raw_dir", required=True, help="Folder containing all your raw video clips")
    parser.add_argument("--out_dir", default="dataset", help="Output dataset folder (default: dataset)")
    parser.add_argument("--csv", default=None, help="CSV file with columns: filename,label")
    parser.add_argument("--infer_from_filename", action="store_true",
                         help="Infer label from filename prefix instead of using a CSV")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of each class held out for validation")
    args = parser.parse_args()

    organize(args.raw_dir, args.out_dir, csv_path=args.csv,
              infer_from_filename=args.infer_from_filename, val_split=args.val_split)
