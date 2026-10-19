from dataset import VideoDataset
from torch.utils.data import DataLoader

dataset = VideoDataset("dataset")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for videos,labels in loader:

    print(videos.shape)

    print(labels)

    break