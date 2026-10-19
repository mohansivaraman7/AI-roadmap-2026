# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import torch
import torch.nn as nn

# ==========================================
# 2. CREATE MODEL
# ==========================================

class VideoClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN Feature Extractor
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

        # Image size:
        # 128x128
        #
        # After Pool1
        # 64x64
        #
        # After Pool2
        # 32x32
        #
        # Feature size:
        # 32 x 32 x 32 = 32768

        self.lstm = nn.LSTM(
            input_size=32768,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(
            256,
            5          # Suppose 5 classes
        )

    def forward(self, x):

        print("\nInput Shape :", x.shape)

        # x shape
        # (batch, frames, channels, H, W)

        batch, frames, C, H, W = x.shape

        # ----------------------------------
        # Merge batch and frames
        # ----------------------------------

        x = x.view(batch * frames, C, H, W)

        print("After View :", x.shape)

        # ----------------------------------
        # CNN
        # ----------------------------------

        x = self.cnn(x)

        print("After CNN :", x.shape)

        # ----------------------------------
        # Restore frame dimension
        # ----------------------------------

        x = x.view(batch, frames, -1)

        print("After Reshape :", x.shape)

        # ----------------------------------
        # LSTM
        # ----------------------------------

        output, (hidden, cell) = self.lstm(x)

        print("LSTM Output :", output.shape)
        print("Hidden Shape :", hidden.shape)

        # Last hidden state

        x = hidden[-1]

        print("Last Hidden :", x.shape)

        # ----------------------------------
        # Fully Connected
        # ----------------------------------

        x = self.fc(x)

        print("Final Output :", x.shape)

        return x


# ==========================================
# 3. CREATE MODEL OBJECT
# ==========================================

model = VideoClassifier()

# ==========================================
# 4. CREATE DUMMY VIDEO INPUT
# ==========================================

# Batch = 2 videos
# Frames = 30
# RGB = 3 channels
# Height = 128
# Width = 128

video = torch.randn(
    2,
    30,
    3,
    128,
    128
)

print("Video Shape :", video.shape)

# ==========================================
# 5. PASS INPUT TO MODEL
# ==========================================

prediction = model(video)

# ==========================================
# 6. PREDICTED CLASS
# ==========================================

predicted_class = torch.argmax(prediction, dim=1)

print("\nPredicted Class :", predicted_class)