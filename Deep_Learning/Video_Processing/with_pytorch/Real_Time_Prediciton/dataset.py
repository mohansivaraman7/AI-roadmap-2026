import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BowlingDataset(Dataset):

    def __init__(self,
                 root_dir,
                 num_frames=30,
                 image_size=128):

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size

        # ---------------------------------------
        # Automatically detect classes
        # ---------------------------------------

        self.classes = sorted([
            folder
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ])

        # Example
        #
        # Fast_Bowling -> 0
        # Spin_Bowling -> 1

        self.class_to_index = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.samples = []

        # ---------------------------------------
        # Collect every video path
        # ---------------------------------------

        for class_name in self.classes:

            class_folder = os.path.join(
                root_dir,
                class_name
            )

            for file in os.listdir(class_folder):

                if file.endswith(".mp4"):

                    video_path = os.path.join(
                        class_folder,
                        file
                    )

                    label = self.class_to_index[class_name]

                    self.samples.append(
                        (video_path, label)
                    )

        print("--------------------------------")
        print("Classes :", self.classes)
        print("Videos  :", len(self.samples))
        print("--------------------------------")

    # ---------------------------------------

    def __len__(self):

        return len(self.samples)

    # ---------------------------------------

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_numbers = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        frames = []

        current_frame = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if current_frame in frame_numbers:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame = cv2.resize(
                    frame,
                    (self.image_size,
                     self.image_size)
                )

                frame = frame.astype(np.float32)

                frame = frame / 255.0

                frames.append(frame)

            current_frame += 1

        cap.release()

        # ------------------------------
        # If video shorter than 30 frames
        # ------------------------------

        while len(frames) < self.num_frames:

            frames.append(frames[-1])

        frames = np.array(frames)

        # (Frames,H,W,C)

        frames = np.transpose(
            frames,
            (0,3,1,2)
        )

        return torch.tensor(
            frames,
            dtype=torch.float32
        )

    # ---------------------------------------

    def __getitem__(self, index):

        video_path, label = self.samples[index]

        video = self.load_video(video_path)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return video, label