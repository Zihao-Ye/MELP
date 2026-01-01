"""
Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE)

多尺度时序编码器，对MVCSE输出的融合特征进行深度时序建模。
"""

import torch
import torch.nn as nn
from typing import List

from .transformer import TransformerBlock


class MSSATEEncoder(nn.Module):
    """
    Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE) 时序编码器。

    对MVCSE输出的多尺度融合特征进行深度时序建模。

    特点:
    1. 每个尺度独立的Transformer encoder
    2. 相对位置编码增强平移鲁棒性
    3. 方案B配置: 2层Transformer（导联级已有6层）

    处理流程:
    输入: List[(B, N1, 4D), (B, N2, 4D), (B, N3, 4D)] - MVCSE输出
    每个尺度独立通过Transformer blocks
    输出: List[(B, N1, 4D), (B, N2, 4D), (B, N3, 4D)]
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_groups: int = 4,
        num_scales: int = 3,
        depth: int = 2,  # 方案B: 2层
        num_heads: int = 8,
        dim_head: int = 32,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
        attn_dropout: float = 0.,
        drop_path: float = 0.1,
        max_len: int = 512,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        self.depth = depth
        self.input_dim = num_groups * embed_dim  # 4组拼接后的维度: 4 * 256 = 1024

        # 每个尺度独立的Transformer encoder
        self.scale_encoders = nn.ModuleList()

        # 渐进式drop_path
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]

        for scale_idx in range(num_scales):
            encoder_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=self.input_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path_rates[i],
                    max_len=max_len,
                    use_relative_pos=use_relative_pos
                )
                for i in range(depth)
            ])
            self.scale_encoders.append(encoder_blocks)

        # 每个尺度的LayerNorm
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(self.input_dim) for _ in range(num_scales)
        ])

    def forward(self, fused_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            fused_features: List of (B, N, 4*D) tensors for each scale
                - Scale 1: (B, 200, 1024)
                - Scale 2: (B, 100, 1024)
                - Scale 3: (B, 50, 1024)

        Returns:
            List of encoded features for each scale:
                - Scale 1: (B, 200, 1024)
                - Scale 2: (B, 100, 1024)
                - Scale 3: (B, 50, 1024)
        """
        outputs = []

        for scale_idx, feat in enumerate(fused_features):
            x = feat

            # 通过该尺度的Transformer blocks
            for block in self.scale_encoders[scale_idx]:
                x = block(x)

            # LayerNorm
            x = self.scale_norms[scale_idx](x)
            outputs.append(x)

        return outputs

    def extra_repr(self) -> str:
        return (
            f'num_scales={self.num_scales}, input_dim={self.input_dim}, '
            f'depth={self.depth}'
        )
