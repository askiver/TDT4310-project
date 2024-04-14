import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embedding_length=100, vector_dimension=768):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(self.kernel_size, vector_dimension))
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
    def __init__(self, embedding_length=200, vector_dimension=768, num_networks=3, network_depth=3, kernel_size=3,
                 output_channels=64):
        super(FlatCNN, self).__init__()
        self.conv_layers = [[] for _ in range(num_networks)]
        self.bn_layers = [[] for _ in range(num_networks)]

        for i in range(num_networks):
            input_channels = vector_dimension
            for j in range(network_depth):
                self.conv_layers[i].append(
                    nn.Conv1d(input_channels, output_channels * 2 ** j, kernel_size=kernel_size + 2 * i,
                              padding=(kernel_size + 2 * i - 1) // 2))
                self.bn_layers[i].append(nn.BatchNorm1d(output_channels * 2 ** j))
                input_channels = output_channels * 2 ** j

        conv_output_size = 0
        for i in range(num_networks):
            conv_output_size += output_channels * 2 ** (network_depth - 1) * (embedding_length // 2 ** network_depth)

        self.fc1 = nn.Linear(conv_output_size, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, 1)

        self.GELU = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(200)

    def forward(self, x):
        conv_outputs = []
        for i, layers in enumerate(self.conv_layers):
            out = x
            for j, layer in enumerate(layers):
                out = self.pool(self.GELU(self.bn_layers[i][j](layer(out))))
            out = out.view(out.size(0), -1)
            conv_outputs.append(out)
        # flatten x
        x = torch.cat(conv_outputs, dim=1)
        x = self.GELU(self.bn1(self.fc1(x)))
        x = self.GELU(self.bn2(self.fc2(x)))
        x = self.fc3(x)
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
        self.fc3.to(device)
        self.bn1.to(device)
        self.bn2.to(device)
        return self


class SimpleFlatCNN(nn.Module):
    def __init__(self, binary_classification=True, embedding_length=200, vector_dimension=300, kernel_size=5, output_channels=64):
        super(SimpleFlatCNN, self).__init__()
        self.conv1 = nn.Conv1d(vector_dimension, output_channels, kernel_size=kernel_size, padding=(kernel_size - 1)//2)
        self.conv2 = nn.Conv1d(output_channels, output_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1)//2)
        self.conv3 = nn.Conv1d(output_channels * 2, output_channels * 4, kernel_size=kernel_size,
                               padding=(kernel_size - 1)//2)
        self.fc1 = nn.Linear((output_channels * 4 * embedding_length) // 2 ** 3, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 1)

        self.GELU = nn.GELU()
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels * 2)
        self.bn3 = nn.BatchNorm1d(output_channels * 4)
        self.bn4 = nn.BatchNorm1d(400)
        self.bn5 = nn.BatchNorm1d(200)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.average_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool = self.max_pool

        self.final_act = nn.Identity() if binary_classification else nn.ReLU()

    def forward(self, x):
        x = self.GELU(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.GELU(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.GELU(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.GELU(self.bn4(self.fc1(x)))
        x = self.GELU(self.bn5(self.fc2(x)))
        x = self.final_act(self.fc3(x))
        if x.size(-1) == 1:
            x = x.squeeze(-1)
        return x

    def forward_accuracy(self, x):
        x = self.forward(x)
        return torch.round(torch.sigmoid(x))

    def forward_score(self, x):
        x = self.forward(x)
        x = torch.clamp(x, min=1, max=8)
        # Return the score rounded to the nearest integer
        return torch.round(x)
