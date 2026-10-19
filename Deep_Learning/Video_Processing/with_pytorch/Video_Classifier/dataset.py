import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):

    def __init__(self, root_dir, num_frames=30):

        self.root_dir = root_dir
        self.num_frames = num_frames

        self.samples = []

        self.classes = sorted(os.listdir(root_dir))

        self.class_to_idx = {
            cls:i for i,cls in enumerate(self.classes)
        }

        for cls in self.classes:

            folder = os.path.join(root_dir, cls)

            for file in os.listdir(folder):

                if file.endswith(".mp4"):

                    self.samples.append(
                        (
                            os.path.join(folder,file),
                            self.class_to_idx[cls]
                        )
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self,index):

        path,label = self.samples[index]

        cap = cv2.VideoCapture(path)

        frames=[]

        while True:

            ret,frame = cap.read()

            if not ret:
                break

            frame=cv2.resize(frame,(128,128))
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame=frame/255.0

            frame=np.transpose(frame,(2,0,1))

            frames.append(frame)

        cap.release()

        if len(frames)>=self.num_frames:

            frames=frames[:self.num_frames]

        else:

            while len(frames)<self.num_frames:

                frames.append(frames[-1])

        frames=np.array(frames,dtype=np.float32)

        return torch.tensor(frames),torch.tensor(label)