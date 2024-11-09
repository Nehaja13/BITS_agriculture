# utils/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define ResNet9 architecture, or use any architecture that matches the loaded model
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Add more layers as necessary to match model architecture
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)

# Export ResNet9 for use in other parts of the application
__all__ = ["ResNet9"]
