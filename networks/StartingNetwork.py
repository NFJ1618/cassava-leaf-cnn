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

        self.fc1 = nn.Linear(12*150*200, 6*50*50)
        self.fc2 = nn.Linear(6*50*50, 128)
        self.fc3 = nn.Linear(128,5)
        # self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3,6, kernel_size=6,padding=2)
        self.conv2 = nn.Conv2d(6,12,kernel_size=2, padding=1)

        self.pool = nn.MaxPool2d(2,2)



    def forward(self, x):
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = self.sigmoid(x)
        # return x

        #(n, 3, 600, 800)
        x = self.conv1(x)
        x = F.relu(x)
        # (n, 6, 600, 800)
        x = self.pool(x)
        # (n, 6, 300, 400)
        x = self.conv2(x)
        x = F.relu(x)
        # (n, 12, 300, 400)
        x = self.pool(x)
        # (n, 12, 150, 200)
        x = torch.reshape(x, (-1, 12 * 150 * 200))
        # (n, 12 * 150 * 200)
        x = self.fc1(x)
        x = F.relu(x)
        # (n, 6*50*50)
        x = self.fc2(x)
        x = F.relu(x)
        # (n, 128)
        x = self.fc3(x)
        # (n, 5)
        return x


