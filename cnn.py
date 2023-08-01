import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 1st Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1),
            nn.BatchNorm1d(32), #881 size
            nn.ReLU()
        )
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1),
            nn.BatchNorm1d(64), #862 size
            nn.ReLU()
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(862 * 64, 20)  # 900 is the flattened size from the previous layers
        self.fc2 = nn.Linear(20, 5)         # 5 output neurons for the 5 possible classes

        # Dropout layer with inactivation probability of 0.1
        self.dropout = nn.Dropout(0.1)

        # Softmax activation for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)

        # Softmax output
        x = self.softmax(x)
        return x