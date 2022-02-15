import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256*4*4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100,5)
        # self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3,96, kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(96,256,kernel_size=4, stride=1,padding=1)
        self.conv3 = nn.Conv2d(256,256,kernel_size=4,padding=2)

        #test
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(3,3)

        #batch normalization
        self.bnl1 = nn.BatchNorm1d(500)
        self.bnl2 = nn.BatchNorm1d(100)
        self.bnc1 = nn.BatchNorm2d(96)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)


    def forward(self, x):
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # return x

        #(n, 3, 224, 224)
        x = self.bnc1(self.conv1(x))
        x = F.relu(x)
        # (n, 96, 52, 52)
        x = self.pool1(x)
        # (n, 96, 26, 26)
        x = self.bnc2(self.conv2(x))
        x = F.relu(x)
        # (n, 256, 24, 24)
        x = self.pool1(x)
        # (n, 256, 12, 12)
        x = self.conv3(x)
        # (n, 256, 12, 12)
        x = self.conv3(x)
        # (n, 256, 12, 12)
        x = self.pool2(x)
        # (n, 256, 4, 4)
        x = torch.reshape(x, (-1, 256*4*4))
        # (n, 256*4*4)
        x = self.bnl1(self.fc1(x))
        x = F.relu(x)
        # (n, 500)
        x = self.bnl2(self.fc2(x))
        x = F.relu(x)
        # (n, 100)
        x = self.fc3(x)
        # (n, 5)
        return x


