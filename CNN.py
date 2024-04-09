import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embedding_length=100):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(self.kernel_size, 300))
        self.conv2 = nn.Conv1d(8, 16, kernel_size=self.kernel_size)
        self.fc1 = nn.Linear(16 * (embedding_length-(self.kernel_size-1)*2), 64)
        self.fc2 = nn.Linear(64, 1)
        self.GELU = nn.GELU()
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.GELU(self.bn1(self.conv1(x)))
        x = x.squeeze_(-1)
        x = self.GELU(self.bn2(self.conv2(x)))
        x = x.view(-1, 16 * 192)
        x = self.GELU(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze_(-1)

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))


class LargeCNN(nn.Module):
    def __init__(self, embedding_length=100):
        super(LargeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(51,151))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=26)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=10)

        self.fc1 = nn.Linear(8 * (8**2), 1028)
        self.fc2 = nn.Linear(1028, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm1d(1028)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 8 * 8**2)
        x = self.gelu(self.bn4(self.fc1(x)))
        x = self.gelu(self.bn5(self.fc2(x)))
        x = self.gelu(self.bn6(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze_(-1)

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))


class FlatCNN(nn.Module):
    def __init__(self, embedding_length=200):
        super(FlatCNN, self).__init__()
        self.conv1 = nn.Conv1d(300, 200, kernel_size=6)
        self.conv2 = nn.Conv1d(200,100, kernel_size=6)
        self.conv3 = nn.Conv1d(100, 50, kernel_size=6)

        self.fc1 = nn.Linear(50 * (embedding_length-15), 64)
        self.fc2 = nn.Linear(64, 1)

        self.GELU = nn.GELU()

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(50)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.GELU(self.bn1(self.conv1(x)))
        x = self.GELU(self.bn2(self.conv2(x)))
        x = self.GELU(self.bn3(self.conv3(x)))
        x = x.view(-1, 50 * (200-15))
        x = self.GELU(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze_(-1)

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))


