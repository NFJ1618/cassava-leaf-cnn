import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()

        self.prebase = models.resnet18(pretrained=True)

        self.fc1 = nn.Linear(384*6*6, 4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4000, 100)
        self.fc4 = nn.Linear(100,5)
        # self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3,96, kernel_size=16,stride=4)
        self.conv2 = nn.Conv2d(96,256,kernel_size=8, stride=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
        #self.conv5 = nn.Conv2d(384,384,kernel_size=5,padding=2)

        #test
        
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(3,2)
        self.pool3 = nn.MaxPool2d(3,2)

        #batch normalization
        self.bnl1 = nn.BatchNorm1d(4000)
        self.bnl2 = nn.BatchNorm1d(4000)
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

        # #print(x.shape)
        # #(n, 3, 512, 512)
        # x = self.bnc1(self.conv1(x))
        # x = F.relu(x)
        # #print(x.shape)
        # # (n, 96, 125, 125)
        # x = self.pool1(x)
        # #print(x.shape)
        # # (n, 96, 62, 62)
        # x = self.bnc2(self.conv2(x))
        # x = F.relu(x)
        # #print(x.shape)
        # # (n, 256, 28, 28)
        # x = self.pool2(x)
        # #print(x.shape)
        # # (n, 384, 13, 13)
        # x = self.bnc3(self.conv3(x))
        # x = F.relu(x)
        # #print(x.shape)
        # # (n, 384, 13, 13)
        # x = self.bnc4(self.conv4(x))
        # x = F.relu(x)
        # #print(x.shape)
        # # (n, 384, 13, 13)
        # #x = self.bnc5(self.conv5(x))
        # #x = F.relu(x)
        # #print(x.shape)
        # # (n, 384, 13, 13)
        # x = self.pool3(x)
        # #print(x.shape)
        # # (n, 384, 6, 6)

        with torch.no_grad():
            x = self.prebase(x)

        x = torch.reshape(x, (-1, 384*6*6))
        print(x.shape)
        # (n, 384*6*6)
        #x = self.drop1(x)
        x = self.bnl1(self.fc1(x))
        x = F.relu(x)
        #print(x.shape)
        # (n, 4000)
        x = self.drop2(x)
        x = self.bnl2(self.fc2(x))
        x = F.relu(x)
        #print(x.shape)
        # (n, 4000)
        x = self.drop3(x)
        x = self.fc3(x)
        #print(x.shape)
        # (n, 100)
        x = self.drop4(x)
        x = self.fc4(x)
        # (n, 5)
        return x


