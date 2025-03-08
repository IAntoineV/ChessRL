import torch.nn as nn


# Critic Network (adapted to 19 input channels) => TO UPDATE for better achitecture
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.conv1 = nn.Conv2d(19, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)
