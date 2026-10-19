import torch

from dataset import VideoDataset
from models import VideoClassifier

from torch.utils.data import DataLoader

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset=VideoDataset("dataset")

loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model=VideoClassifier(
    len(dataset.classes)
).to(device)

loss_fn=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs=10

for epoch in range(epochs):

    model.train()

    total_loss=0

    for videos,labels in loader:

        videos=videos.to(device)

        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(videos)

        loss=loss_fn(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    print(
        f"Epoch {epoch+1} Loss={total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "video_classifier.pth"
)