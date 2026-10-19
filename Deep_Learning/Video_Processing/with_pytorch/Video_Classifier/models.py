import torch
import torch.nn as nn
import torchvision.models as models

class VideoClassifier(nn.Module):

    def __init__(self,num_classes):

        super().__init__()

        #cnn=models.resnet18(weights="DEFAULT")
        cnn = models.resnet18(weights=None)

        self.cnn=nn.Sequential(
            *list(cnn.children())[:-1]
        )

        self.lstm=nn.LSTM(

            input_size=512,
            hidden_size=256,
            batch_first=True
        )

        self.fc=nn.Linear(
            256,
            num_classes
        )

    def forward(self,x):

        batch,time,C,H,W=x.shape

        x=x.view(batch*time,C,H,W)

        features=self.cnn(x)

        features=features.view(batch,time,512)

        output,(hidden,cell)=self.lstm(features)

        prediction=self.fc(hidden[-1])

        return prediction