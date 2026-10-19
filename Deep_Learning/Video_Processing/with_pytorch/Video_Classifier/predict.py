import torch

from dataset import VideoDataset
from models import VideoClassifier

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset=VideoDataset("dataset")

model=VideoClassifier(
    len(dataset.classes)
)

model.load_state_dict(
    torch.load("video_classifier.pth")
)

model.to(device)

model.eval()

video,label=dataset[0]

video=video.unsqueeze(0)

video=video.to(device)

with torch.no_grad():

    prediction=model(video)

pred=torch.argmax(
    prediction,
    dim=1
)

print(
    dataset.classes[pred.item()]
)