"""No medicalNET or anything yet - just a simple test of an 18.9M param model as a first test to get real, actual numbers/benchmarks"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, groups=8):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
        nn.GroupNorm(groups, out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleCNN3D(nn.Module):
    def __init__(self, base_channels=64, dropout=0.5):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            conv_block(1, c),
            conv_block(c, c * 2),
            conv_block(c * 2, c * 4),
            conv_block(c * 4, c * 8),
            conv_block(c * 8, c * 8),
            conv_block(c * 8, c * 8),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c * 8, c * 2)
        self.fc2 = nn.Linear(c * 2, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1)