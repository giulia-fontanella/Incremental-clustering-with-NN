from torch import nn
import torch.nn.functional as F
import torch

# Define Neural Network architecture


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Define the layers we will use
        self.fc1 = nn.Linear(1030, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.activation = nn.LeakyReLU(inplace=False)



    def forward(self, x):

        # Apply layers with ReLu activation function (or LeakyReLU,...)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = torch.sigmoid(x)

        # Apply sigmoid activation function to get output in [0,1]
        return x


