import torch
import torch.nn as nn

class MFFCNN(nn.Module):
    def __init__(self):
        super(MFFCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc4 = nn.Linear(64, 3)
        self.w = nn.Parameter(torch.Tensor([0.7, 0.9]))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        x = self.conv1(x)
        y = self.conv2(y)

        x = torch.flatten(x, 1)
        y = torch.flatten(y, 1)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        x = x * w1 + y * w2

        x = self.fc1(x)
        x = self.fc4(x)
        return x

