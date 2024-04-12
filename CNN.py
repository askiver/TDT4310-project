import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embedding_length=100):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(self.kernel_size, 300))
        self.conv2 = nn.Conv1d(8, 16, kernel_size=self.kernel_size)
        self.fc1 = nn.Linear(16 * (embedding_length - (self.kernel_size - 1) * 2), 64)
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
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(51, 151))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=26)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=10)

        self.fc1 = nn.Linear(8 * (8 ** 2), 1028)
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
        x = x.view(-1, 8 * 8 ** 2)
        x = self.gelu(self.bn4(self.fc1(x)))
        x = self.gelu(self.bn5(self.fc2(x)))
        x = self.gelu(self.bn6(self.fc3(x)))
        x = self.fc4(x)
        return x.squeeze_(-1)

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))


class FlatCNN(nn.Module):
    def __init__(self, embedding_length=200, vector_dimension=300, num_networks=5, network_depth=3, kernel_size=2, output_channels=64):
        super(FlatCNN, self).__init__()
        self.conv_layers = [[] for _ in range(num_networks)]

        for i in range(num_networks):
            input_channels = vector_dimension
            for j in range(network_depth):
                self.conv_layers[i].append(
                    nn.Conv1d(input_channels, output_channels*2**j, kernel_size=kernel_size * (1 + i)))
                input_channels = output_channels*2**j

        conv_output_size = 0
        for i in range(num_networks):
            conv_output_size += output_channels*2**(network_depth-1) * (embedding_length - (kernel_size * (1 + i) - 1) * network_depth)

        self.fc1 = nn.Linear(conv_output_size, 1000)
        self.fc2 = nn.Linear(1000, 1)

        self.GELU = nn.GELU()

        self.bn_layers = [[nn.BatchNorm1d(vector_dimension) for _ in range(network_depth)] for _ in range(num_networks)]
        self.bn1 = nn.BatchNorm1d(1000)

    def forward(self, x):
        conv_outputs = []
        for i, layers in enumerate(self.conv_layers):
            out = x
            for layer in layers:
                out = layer(out)
            out = out.view(out.size(0), -1)
            conv_outputs.append(out)
        # flatten x
        x = torch.cat(conv_outputs, dim=1)
        x = self.GELU(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze_(-1)

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))

    def to(self, device):
        for net in self.conv_layers:
            for layer in net:
                layer.to(device)
        for net in self.bn_layers:
            for layer in net:
                layer.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.bn1.to(device)
        return self
