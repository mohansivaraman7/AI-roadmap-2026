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
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
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


# ==========================================
# 3. DEFINE ANN
# ==========================================

model = nn.Sequential(

    nn.Flatten(),      # 28x28 -> 784

    nn.Linear(784,128),
    nn.ReLU(),

    nn.Linear(128,64),
    nn.ReLU(),

    nn.Linear(64,10)
)

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

    for images, labels in train_loader:

        # Forward Propagation
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Clear Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss = {loss.item():.4f}"
    )


# ==========================================
# 7. TESTING
# ==========================================

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

print(
    f"\nAccuracy: {accuracy:.2f}%"
)