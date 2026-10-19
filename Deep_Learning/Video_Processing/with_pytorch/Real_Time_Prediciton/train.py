import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import VideoClassifier
from dataset import BowlingDataset

# -----------------------------------------
# Device
# -----------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device :", device)

# -----------------------------------------
# Dataset
# -----------------------------------------

train_dataset = BowlingDataset(
    root_dir="BowlingDataset/train"
)

test_dataset = BowlingDataset(
    root_dir="BowlingDataset/test"
)

# -----------------------------------------
# DataLoader
# -----------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

# -----------------------------------------
# Model
# -----------------------------------------

model = VideoClassifier(
    num_classes=2
)

model.to(device)

# -----------------------------------------
# Loss Function
# -----------------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------------
# Optimizer
# -----------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# -----------------------------------------
# Training
# -----------------------------------------

epochs = 20

best_accuracy = 0

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for videos, labels in train_loader:

        videos = videos.to(device)

        labels = labels.to(device)

        # --------------------------

        predictions = model(videos)

        # --------------------------

        loss = criterion(
            predictions,
            labels
        )

        # --------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # --------------------------

        running_loss += loss.item()

        _, predicted = torch.max(
            predictions,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        correct / total
    ) * 100

    # ============================================
    # Validation
    # ============================================

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for videos, labels in test_loader:

            videos = videos.to(device)

            labels = labels.to(device)

            predictions = model(videos)

            _, predicted = torch.max(
                predictions,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    test_accuracy = (
        correct / total
    ) * 100

    print(
        f"Epoch {epoch+1}/{epochs}"
    )

    print(
        f"Loss : {running_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy : {test_accuracy:.2f}%"
    )

    print("-----------------------------------")

    # -----------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        torch.save(

            model.state_dict(),

            "bowling_classifier.pth"

        )

        print("Model Saved")

print()

print("Training Finished")

print("Best Accuracy :", best_accuracy)