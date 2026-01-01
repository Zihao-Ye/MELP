"""
Transformer模块

包含:
- TransformerBlock: 标准Transformer Block，支持相对位置编码
- LeadTransformer: 导联级Transformer，对每个导联独立处理时序
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional

from .base_modules import DropPath, FeedForward
from .attention import RelativeAttention


class TransformerBlock(nn.Module):
    """
    Transformer Block with optional relative positional encoding.

    Pre-norm架构:
    x -> Norm -> Attention -> Residual -> Norm -> FFN -> Residual

    支持两种注意力模式:
    1. RelativeAttention: 带相对位置编码，用于时序建模
    2. StandardAttention: 标准多头注意力，用于非序列数据
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        drop_path: float = 0.,
        max_len: int = 512,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.use_relative_pos = use_relative_pos

        if use_relative_pos:
            self.attn = RelativeAttention(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                attn_dropout=attn_dropout,
                max_len=max_len
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True
            )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)

        Returns:
            (B, N, D)
        """
        if self.use_relative_pos:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + self.drop_path(attn_out)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LeadTransformer(nn.Module):
    """
    导联级Transformer。

    对每个导联的时序tokens独立处理，使用共享权重。
    学习单导联内的时序依赖关系（P波-QRS-ST-T波的时序模式）。

    特点:
    1. 共享权重: 所有导联使用同一组Transformer参数，参数高效
    2. 独立处理: 不混合不同导联的信息，保留导联特异性
    3. 相对位置编码: 增强对心跳平移的鲁棒性

    处理流程:
    输入: (B, num_leads, N, D)
    重排: (B*num_leads, N, D) - 展平导联维度
    Transformer blocks (depth层)
    重排: (B, num_leads, N, D)
    输出: (B, num_leads, N, D)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        dim_head: int = 64,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
        attn_dropout: float = 0.,
        drop_path: float = 0.1,
        max_len: int = 512,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # 渐进式drop_path
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # 共享权重的Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
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

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_leads, N, D)

        Returns:
            (B, num_leads, N, D)
        """
        B, L, N, D = x.shape  # L = num_leads

        # 重排: 将所有导联展平到batch维度
        x = rearrange(x, 'b l n d -> (b l) n d')  # (B*L, N, D)

        # 通过共享权重的Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 重排回原始形状
        x = rearrange(x, '(b l) n d -> b l n d', b=B, l=L)  # (B, L, N, D)

        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, depth={self.depth}'
