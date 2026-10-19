# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ==========================================
# 2. LOAD MNIST DATA
# ==========================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = nn.Sequential(

            # Input: (1, 28, 28)

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Output: (32, 14, 14)

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            # Output: (64, 7, 7)

        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )

        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x


model = CNN()

print(model)


# ==========================================
# 4. LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 5. OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. TRAINING LOOP
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(
            outputs,
            labels
        )

        # Clear Old Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss = {running_loss/len(train_loader):.4f}"
    )


# ==========================================
# 7. TESTING
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = 100 * correct / total

print(f"\nTest Accuracy : {accuracy:.2f}%")