"""
Patch Embedding模块

使用标准ResNet18前端进行层级特征提取，然后多尺度切分生成tokens。
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, List, Dict

from .resnet_frontend import resnet18_frontend


class MultiScalePatchEmbedding(nn.Module):
    """
    多尺度Patch Embedding（标准ResNet18前端版本）。

    架构:
    1. 标准ResNet18前端: (B, 1, L) -> (B, 512, L//16)
       - 完整4个stage的ResNet18
       - 下采样16倍: 5000 -> 313

    2. 多尺度切分: 对ResNet特征进行多尺度切分
       - Scale 1: kernel=4, stride=4 -> 78 tokens
       - Scale 2: kernel=8, stride=8 -> 39 tokens
       - Scale 3: kernel=16, stride=16 -> 19 tokens

    输入: (B, 1, L) - 单导联信号
    输出: List[(B, N1, D), (B, N2, D), (B, N3, D)] - 三个尺度的tokens
    """
    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 5000,
        patch_configs: Optional[List[Dict]] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # 标准ResNet18前端 (完整4个stage，输出512通道)
        self.resnet_frontend = resnet18_frontend(in_channels=1)
        resnet_out_channels = self.resnet_frontend.out_channels  # 512
        resnet_out_length = self.resnet_frontend.get_output_length(seq_len)  # 313 for 5000

        # 默认多尺度配置（适配ResNet输出）
        if patch_configs is None:
            patch_configs = [
                {'kernel_size': 4, 'stride': 4},    # 78 tokens (313/4)
                {'kernel_size': 8, 'stride': 8},    # 39 tokens (313/8)
                {'kernel_size': 16, 'stride': 16},  # 19 tokens (313/16)
            ]

        self.num_scales = len(patch_configs)
        self.patch_configs = patch_configs

        # 每个尺度的patch embedding层
        self.patch_embeds = nn.ModuleList()
        self.num_patches_list = []

        for cfg in patch_configs:
            kernel_size = cfg['kernel_size']
            stride = cfg['stride']
            num_patches = (resnet_out_length - kernel_size) // stride + 1
            self.num_patches_list.append(num_patches)

            self.patch_embeds.append(
                nn.Conv1d(
                    in_channels=resnet_out_channels,  # 512
                    out_channels=embed_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False
                )
            )

        # 每个尺度的位置编码
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
            for num_patches in self.num_patches_list
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 1, L) - 单导联信号

        Returns:
            List of embeddings for each scale: [(B, N1, D), (B, N2, D), (B, N3, D)]
        """
        # ResNet前端特征提取
        x = self.resnet_frontend(x)  # (B, 512, L//16)

        # 多尺度切分
        outputs = []
        for i, patch_embed in enumerate(self.patch_embeds):
            out = patch_embed(x)  # (B, D, N)
            out = rearrange(out, 'b d n -> b n d')  # (B, N, D)
            out = out + self.pos_embeds[i]
            outputs.append(out)

        return outputs

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, seq_len={self.seq_len}, '
            f'num_scales={self.num_scales}, num_patches={self.num_patches_list}'
        )
