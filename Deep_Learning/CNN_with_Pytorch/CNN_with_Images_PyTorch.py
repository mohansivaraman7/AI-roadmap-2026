import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================
# 1. TRANSFORM
# ============================================================

transform_to_tensor = transforms.ToTensor()


# ============================================================
# 2. MNIST DATASET
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform_to_tensor
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform_to_tensor
)


# ============================================================
# 3. DATA LOADERS
# ============================================================

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


# ============================================================
# 4. CNN MODEL
# ============================================================

model = nn.Sequential(

    # --------------------------------------------------------
    # CNN LAYER 1
    # --------------------------------------------------------

    nn.Conv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=3,
        padding=1
    ),

    nn.ReLU(),

    nn.MaxPool2d(
        kernel_size=2
    ),


    # --------------------------------------------------------
    # CNN LAYER 2
    # --------------------------------------------------------

    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    ),

    nn.ReLU(),

    nn.MaxPool2d(
        kernel_size=2
    ),


    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    nn.Flatten(),


    # --------------------------------------------------------
    # FULLY CONNECTED LAYERS
    # --------------------------------------------------------

    nn.Linear(
        32 * 7 * 7,
        128
    ),

    nn.ReLU(),

    nn.Linear(
        128,
        10
    )
)


print(model)


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 7. TRAINING
# ============================================================

epochs = 5

for epoch in range(epochs):

    for images, labels in train_loader:

        # ----------------------------------------------------
        # Forward Propagation
        # ----------------------------------------------------

        outputs = model(images)


        # ----------------------------------------------------
        # Calculate Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Clear Gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update Weights
        # ----------------------------------------------------

        optimizer.step()


    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss = {loss.item():.4f}"
    )


# ============================================================
# 8. TESTING
# ============================================================

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


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = 100 * correct / total

print(
    f"\nAccuracy: {accuracy:.2f}%"
)