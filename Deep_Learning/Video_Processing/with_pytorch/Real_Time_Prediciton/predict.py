import torch
from model import VideoClassifier
from dataset import BowlingDataset

# Create model
model = VideoClassifier(num_classes=2)

# Load trained weights
model.load_state_dict(torch.load("bowling_classifier.pth"))

model.eval()

# Class names
classes = [
    "Fast Bowling",
    "Spin Bowling"
]

# Load one sample from the test dataset
dataset = BowlingDataset("BowlingDataset/test")

video, label = dataset[0]

video = video.unsqueeze(0)

with torch.no_grad():

    prediction = model(video)

    probabilities = torch.softmax(prediction, dim=1)

    predicted = torch.argmax(probabilities, dim=1).item()

print("--------------------------------")

print("Prediction :", classes[predicted])

print("Confidence :", probabilities[0][predicted].item())

print("--------------------------------")