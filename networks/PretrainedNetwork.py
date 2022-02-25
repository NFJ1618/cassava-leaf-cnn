import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PretrainedNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()

        self.prebase = models.resnet18(pretrained=True)
        self.prebase = torch.nn.Sequential(*(list(self.prebase.children())[:-1]))

        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100,5)
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        with torch.no_grad():
            x = self.prebase(x)
        x = self.bnl1(self.fc1(x))
        x = F.relu(x)
        x = self.drop2(x)
        x = self.bnl2(self.fc2(x))
        x = F.relu(x)
        x = self.drop3(x)
        x = self.fc3(x)
        x = self.drop4(x)
        x = self.fc4(x)
        # (n, 5)
        return x


