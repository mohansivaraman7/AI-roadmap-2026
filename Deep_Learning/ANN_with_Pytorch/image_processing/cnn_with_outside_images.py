from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)