"""
标准 ResNet1D 前端特征提取器

用于 MVCSE-MSSATE 的前端特征提取。
使用完整的 ResNet18 架构（4个stage），输出512通道。

输入: (B, 1, L) - 单导联信号
输出: (B, 512, L//16) - 特征序列
"""

import torch
import torch.nn as nn
from typing import Optional


class BasicBlock(nn.Module):
    """ResNet BasicBlock (用于 ResNet18/34)"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck (用于 ResNet50/101/152)"""
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, self.expansion * out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetFrontend(nn.Module):
    """
    标准 ResNet1D 前端（单导联版本）。

    完整的 ResNet 架构，4个stage，与 MELP/MERL 的 ResNet18 结构一致。

    架构:
    - stem: Conv1d(1, 64, k=7, s=2) + BN + ReLU
    - layer1: 2x BasicBlock, 64 channels, stride=1
    - layer2: 2x BasicBlock, 128 channels, stride=2
    - layer3: 2x BasicBlock, 256 channels, stride=2
    - layer4: 2x BasicBlock, 512 channels, stride=2

    下采样比例: 16x
    - stem: 2x (stride=2 conv)
    - layer2: 2x
    - layer3: 2x
    - layer4: 2x

    输入: (B, 1, 5000) - 单导联ECG
    输出: (B, 512, 313) - 特征序列 (5000 / 16 ≈ 313)
    """

    def __init__(
        self,
        in_channels: int = 1,
        block: type = BasicBlock,
        num_blocks: list = [2, 2, 2, 2],
        base_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = base_channels

        # Stem: 初始卷积 (与MELP一致，不使用MaxPool)
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        # 下采样 2x: 5000 -> 2500

        # 4个ResNet层 (完整的标准ResNet)
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        # 2500 -> 2500
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        # 2500 -> 1250
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)
        # 1250 -> 625
        self.layer4 = self._make_layer(block, base_channels * 8, num_blocks[3], stride=2)
        # 625 -> 313

        self.out_channels = base_channels * 8 * block.expansion  # 512 for BasicBlock
        self.downsample_factor = 16

        self._init_weights()

    def _make_layer(self, block: type, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) - 单导联ECG信号

        Returns:
            (B, 512, L') - 特征序列，L' ≈ L // 16
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_output_length(self, input_length: int) -> int:
        """计算输出序列长度"""
        L = input_length
        # stem: stride=2, k=7, p=3 -> (L + 6 - 7) // 2 + 1 = (L - 1) // 2 + 1
        L = (L - 1) // 2 + 1
        # layer2: stride=2
        L = (L - 1) // 2 + 1
        # layer3: stride=2
        L = (L - 1) // 2 + 1
        # layer4: stride=2
        L = (L - 1) // 2 + 1
        return L

    def extra_repr(self) -> str:
        return f'out_channels={self.out_channels}, downsample_factor={self.downsample_factor}'


# 预定义配置
def resnet18_frontend(in_channels: int = 1) -> ResNetFrontend:
    """
    标准 ResNet18 前端（单导联）。

    输入: (B, 1, 5000)
    输出: (B, 512, 313)
    """
    return ResNetFrontend(
        in_channels=in_channels,
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2]
    )


def resnet34_frontend(in_channels: int = 1) -> ResNetFrontend:
    """标准 ResNet34 前端（单导联）"""
    return ResNetFrontend(
        in_channels=in_channels,
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3]
    )


def resnet50_frontend(in_channels: int = 1) -> ResNetFrontend:
    """
    标准 ResNet50 前端（单导联）。

    输入: (B, 1, 5000)
    输出: (B, 2048, 313)
    """
    return ResNetFrontend(
        in_channels=in_channels,
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3]
    )
