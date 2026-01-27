"""
ResNet Frontend for ECG Feature Extraction

Adapted from torchvision ResNet for 1D ECG signals.
Extracts features from raw ECG waveforms.
"""

import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    """1D Basic Block for ResNet"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1dFrontend(nn.Module):
    """
    ResNet Frontend for ECG signals

    Args:
        block: BasicBlock1d
        layers: List of number of blocks in each layer
        in_channels: Number of input channels (1 for single-lead ECG)
        base_channels: Base number of channels (64 for ResNet18/34)

    Output:
        (B, 512, 313) for 5000-length input
    """

    def __init__(self, block, layers, in_channels=1, base_channels=64):
        super().__init__()
        self.in_channels = base_channels
        self.out_channels = 512  # Final output channels

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=15,
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 5000) - Single-lead ECG signal

        Returns:
            (B, 512, 313) - Feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18_frontend(in_channels=1):
    """ResNet18 frontend for ECG"""
    return ResNet1dFrontend(BasicBlock1d, [2, 2, 2, 2], in_channels=in_channels)


def resnet34_frontend(in_channels=1):
    """ResNet34 frontend for ECG"""
    return ResNet1dFrontend(BasicBlock1d, [3, 4, 6, 3], in_channels=in_channels)


# ============================================================================
# ABLATION HOOK: ResNet Architecture
# ============================================================================
# To add different ResNet variants (e.g., ResNet50, custom depths):
# 1. Define new layer configurations
# 2. Add factory functions here
# 3. Update encoder.py to accept resnet_type parameter
# ============================================================================
