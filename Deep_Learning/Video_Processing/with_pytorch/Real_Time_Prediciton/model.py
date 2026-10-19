import torch
import torch.nn as nn


class VideoClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()

        # CNN extracts features from each frame
        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten()
        )

        # 128x128
        #
        # Pool1 -> 64x64
        #
        # Pool2 -> 32x32
        #
        # 32 channels
        #
        # 32 x 32 x 32 = 32768

        self.lstm = nn.LSTM(

            input_size=32768,

            hidden_size=256,

            num_layers=1,

            batch_first=True

        )

        self.fc = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        # x
        #
        # (batch,frames,3,128,128)

        batch_size, frames, C, H, W = x.shape

        # Merge batch and frames

        x = x.view(
            batch_size * frames,
            C,
            H,
            W
        )

        # CNN

        x = self.cnn(x)

        # Separate frames again

        x = x.view(
            batch_size,
            frames,
            -1
        )

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc(x)

        return x