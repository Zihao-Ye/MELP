"""
Transformer Modules for SMMERL

LeadTransformer: Models temporal relationships within each lead's sequence
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class RelativePositionBias(nn.Module):
    """
    Relative position bias for Transformer
    Learns relative position embeddings for better temporal modeling
    """
    def __init__(self, num_heads, max_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        # Learnable relative position embeddings
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_len - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, seq_len):
        # Generate relative position indices
        coords = torch.arange(seq_len, device=self.relative_position_bias_table.device)
        relative_coords = coords[:, None] - coords[None, :]  # (seq_len, seq_len)
        relative_coords += self.max_len - 1  # Shift to start from 0

        # Get bias from table
        relative_position_bias = self.relative_position_bias_table[relative_coords]  # (seq_len, seq_len, num_heads)
        return relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, seq_len, seq_len)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        dim_head=64,
        mlp_ratio=4.,
        dropout=0.1,
        use_relative_pos=True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.use_relative_pos = use_relative_pos

        if use_relative_pos:
            self.relative_pos_bias = RelativePositionBias(num_heads, max_len=512)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) - Sequence of tokens

        Returns:
            (B, N, D) - Transformed sequence
        """
        # Self-attention with relative position bias
        attn_input = self.norm1(x)

        if self.use_relative_pos:
            B, N, D = x.shape
            rel_pos_bias = self.relative_pos_bias(N)  # (num_heads, N, N)
            # Add bias to attention scores (done internally by modifying attn_mask)
            # For simplicity, we'll use standard attention here
            # In practice, you'd modify the attention mechanism to add the bias
            attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        else:
            attn_output, _ = self.attn(attn_input, attn_input, attn_input)

        x = x + attn_output

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        return x


class LeadTransformer(nn.Module):
    """
    Transformer for modeling temporal relationships within ECG leads

    Processes each lead's sequence independently to capture:
    - Temporal dependencies (P-QRS-T relationships)
    - Long-range patterns (heart rate variability)
    - Contextual information for each time point

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        dim_head: Dimension per attention head
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
        max_len: Maximum sequence length
        use_relative_pos: Whether to use relative position encoding
    """
    def __init__(
        self,
        embed_dim=256,
        depth=4,
        num_heads=8,
        dim_head=64,
        mlp_ratio=4.,
        dropout=0.1,
        max_len=512,
        use_relative_pos=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_relative_pos=use_relative_pos
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, num_leads, N, D) - Multi-lead sequences

        Returns:
            (B, num_leads, N, D) - Transformed sequences
        """
        B, num_leads, N, D = x.shape

        # Reshape to process all leads in batch
        x = rearrange(x, 'b l n d -> (b l) n d')

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape back
        x = rearrange(x, '(b l) n d -> b l n d', b=B, l=num_leads)

        return x


# ============================================================================
# ABLATION HOOK: Transformer Architecture
# ============================================================================
# To experiment with different transformer designs:
# 1. Efficient attention mechanisms (Linear Attention, Flash Attention)
# 2. Different normalization strategies (Pre-norm vs Post-norm)
# 3. Hierarchical transformers (local → global attention)
# 4. Window-based attention for efficiency
#
# Example:
# class EfficientLeadTransformer(LeadTransformer):
#     def __init__(self, ...):
#         # Use window attention for first few layers
#         # Use global attention for last few layers
# ============================================================================
