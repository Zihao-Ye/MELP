"""
位置编码模块

实现相对位置编码，用于增强模型对心跳在时间轴上平移的鲁棒性。
"""

import torch
import torch.nn as nn


class RelativePositionalEncoding(nn.Module):
    """
    可学习的相对位置编码。

    相比于绝对位置编码，相对位置编码能够：
    1. 更好地泛化到不同长度的序列
    2. 增强对心跳在时间轴上平移的鲁棒性
    3. 捕捉位置之间的相对关系而非绝对位置

    实现细节:
    - 维护一个 (2*max_len-1, num_heads) 的相对位置偏置表
    - 位置i到位置j的相对位置为 i-j，范围 [-(max_len-1), max_len-1]
    - 通过偏移映射到 [0, 2*max_len-2]
    """
    def __init__(self, num_heads: int, max_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len

        # 可学习的相对位置嵌入表
        # 索引范围: [0, 2*max_len-2]，对应相对位置 [-(max_len-1), max_len-1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 预计算相对位置索引
        coords = torch.arange(max_len)
        relative_coords = coords[:, None] - coords[None, :]  # (max_len, max_len)
        relative_coords += max_len - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        获取指定序列长度的相对位置偏置矩阵。

        Args:
            seq_len: 序列长度

        Returns:
            relative_position_bias: (seq_len, seq_len, num_heads)
        """
        # 获取当前序列长度的相对位置索引
        relative_position_index = self.relative_position_index[:seq_len, :seq_len]

        # 查表获取相对位置偏置
        # 使用 reshape 代替 view，因为高级索引后张量可能不连续
        relative_position_bias = self.relative_position_bias_table[
            relative_position_index.reshape(-1)
        ].reshape(seq_len, seq_len, -1)

        return relative_position_bias  # (seq_len, seq_len, num_heads)

    def extra_repr(self) -> str:
        return f'num_heads={self.num_heads}, max_len={self.max_len}'
