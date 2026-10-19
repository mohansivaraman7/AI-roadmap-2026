"""
Video Classifier - Real Video Processing
==========================================

Requirements:
    pip install torch opencv-python numpy

Usage:
    python video_classifier_program.py --video sample.mp4
"""

import argparse
import cv2
"""pip install opencv-python"""
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Model (same architecture, prints removed for real use)
# ============================================================

class VideoClassifier(nn.Module):

    def __init__(self, num_classes=5, debug=False):
        super().__init__()
        self.debug = debug

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        # 128x128 -> pool -> 64x64 -> pool -> 32x32
        # Feature size: 32 * 32 * 32 = 32768
        self.lstm = nn.LSTM(
            input_size=32768,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, frames, channels, H, W)
        batch, frames, C, H, W = x.shape

        x = x.view(batch * frames, C, H, W)
        x = self.cnn(x)
        x = x.view(batch, frames, -1)

        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)

        if self.debug:
            print("Final Output :", x.shape)

        return x


# ============================================================
# Real video loading
# ============================================================

def load_video_frames(video_path, num_frames=30, resize=(128, 128)):
    """
    Reads a real video file and returns a tensor of shape
    (1, num_frames, 3, H, W), ready to feed into VideoClassifier.

    Frames are sampled uniformly across the whole video so that
    videos of any length/fps are reduced to a fixed frame count.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Uniformly spaced frame indices to sample
    sample_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())

    frames = []
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in sample_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)

        current_idx += 1
        if len(frames) >= num_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not extract any frames from: {video_path}")

    # Pad by repeating the last frame if the video was shorter than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.array(frames, dtype=np.float32) / 255.0     # (F, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))              # (F, C, H, W)

    video_tensor = torch.from_numpy(frames).unsqueeze(0)      # (1, F, C, H, W)
    return video_tensor


# ============================================================
# Main
# ============================================================
"""python video_classifier_program.py --video sample.mp4"""
def main():
    parser = argparse.ArgumentParser(description="Run VideoClassifier on a real video file")
    parser.add_argument("--video", type=str, required=True, help="Path to video file (mp4, avi, etc.)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to sample")
    parser.add_argument("--size", type=int, default=128, help="Frame resize (size x size)")
    args = parser.parse_args()

    print(f"Loading video: {args.video}")
    video_tensor = load_video_frames(
        args.video,
        num_frames=args.frames,
        resize=(args.size, args.size)
    )
    print("Video Tensor Shape:", video_tensor.shape)

    model = VideoClassifier(num_classes=5, debug=True)
    model.eval()  # inference mode (no dropout/batchnorm updates - good practice even without those layers)

    with torch.no_grad():
        prediction = model(video_tensor)

    probs = torch.softmax(prediction, dim=1)
    predicted_class = torch.argmax(prediction, dim=1)

    print("\nClass probabilities:", probs.squeeze().tolist())
    print("Predicted Class:", predicted_class.item())


if __name__ == "__main__":
    main()
