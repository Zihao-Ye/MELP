"""
基础模块：DropPath, FeedForward, SEBlock, ECABlock

用于构建MVCSE-MSSATE编码器的基础组件。
"""

import torch
import torch.nn as nn
import math


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    随机丢弃整个残差分支，用于正则化。
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation.

    标准的Transformer FFN: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for inter-group channel attention.

    用于学习导联组间的协同效应。
    输入格式: (B, N, C) - batch, sequence, channels

    流程:
    1. Squeeze: 全局平均池化 (B, N, C) -> (B, C)
    2. Excitation: FC -> ReLU -> FC -> Sigmoid -> (B, C)
    3. Scale: (B, N, C) * (B, 1, C) -> (B, N, C)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) - batch, sequence, channels

        Returns:
            Attention-weighted x: (B, N, C)
        """
        # (B, N, C) format - squeeze over N dimension (sequence)
        y = x.mean(dim=1)  # (B, C) - global average pooling over sequence
        y = self.fc(y)  # (B, C) - channel attention weights
        return x * y.unsqueeze(1)  # (B, N, C) * (B, 1, C) -> (B, N, C)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block.

    使用1D卷积代替全连接层，参数更少，效率更高。
    输入格式: (B, N, C) - batch, sequence, channels

    自适应kernel size计算: k = |log2(C) / gamma + b|_odd
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        self.channels = channels

        # 自适应计算kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) - batch, sequence, channels

        Returns:
            Attention-weighted x: (B, N, C)
        """
        # (B, N, C) format
        # Global average pooling over sequence dimension
        y = x.mean(dim=1)  # (B, C)
        # Reshape for conv1d: (B, 1, C)
        y = y.unsqueeze(1)
        # Apply 1D conv for channel attention
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        return x * y  # (B, N, C) * (B, 1, C) -> (B, N, C)
