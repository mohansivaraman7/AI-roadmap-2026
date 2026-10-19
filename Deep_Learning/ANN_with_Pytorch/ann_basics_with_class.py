# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 2. LOAD MNIST DATASET
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


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ==========================================
# 3. DEFINE MODEL (ANN)
# ==========================================

class MNIST_ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(28*28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)   # 10 classes (0-9)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)   # flatten image
        return self.model(x)


model = MNIST_ANN()


# ==========================================
# 4. LOSS & OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==========================================
# 5. TRAINING LOOP
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")


# ==========================================
# 6. TESTING
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total

print("\nTest Accuracy:", accuracy)