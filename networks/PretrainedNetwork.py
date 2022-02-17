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

        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100,5)
        # self.sigmoid = nn.Sigmoid()

        #test
        
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(3,2)
        self.pool3 = nn.MaxPool2d(3,2)

        #batch normalization
        # self.bnl1 = nn.BatchNorm1d(4000)
        # self.bnl2 = nn.BatchNorm1d(4000)

        self.bnl1 = nn.BatchNorm1d(1000)
        self.bnl2 = nn.BatchNorm1d(1000)

        self.bnc1 = nn.BatchNorm2d(96)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(384)
        self.bnc4 = nn.BatchNorm2d(384)
        self.bnc5 = nn.BatchNorm2d(384)

        #dropout
        #self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.9)


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


