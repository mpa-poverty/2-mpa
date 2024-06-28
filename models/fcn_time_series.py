# @MDC, MARBEC, 2023

import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, num_channels, filter_size, output_size=1):
        super(FCN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=filter_size, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling over time
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


