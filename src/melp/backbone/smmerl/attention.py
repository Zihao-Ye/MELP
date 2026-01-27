"""
Attention Pooling Modules for Multi-scale Feature Extraction

Extracts wave/beat/rhythm features from full-sequence temporal representations
using learnable query-based attention pooling.
"""

import torch
import torch.nn as nn
import math


class MultiScaleAttentionPooler(nn.Module):
    """
    Multi-scale attention pooling using learnable queries

    Extracts features at different temporal scales:
    - Wave (30 queries): Fine-grained waveform features (P/QRS/T waves)
    - Beat (10 queries): Heart beat-level features
    - Rhythm (3 queries): Overall rhythm patterns

    Args:
        seq_len: Input sequence length (e.g., 313)
        embed_dim: Embedding dimension
        n_wave_queries: Number of wave-level queries (default: 30)
        n_beat_queries: Number of beat-level queries (default: 10)
        n_rhythm_queries: Number of rhythm-level queries (default: 3)
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(
        self,
        seq_len=313,
        embed_dim=256,
        n_wave_queries=30,
        n_beat_queries=10,
        n_rhythm_queries=3,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_wave_queries = n_wave_queries
        self.n_beat_queries = n_beat_queries
        self.n_rhythm_queries = n_rhythm_queries

        # Learnable queries for each scale
        self.wave_queries = nn.Parameter(
            torch.randn(1, n_wave_queries, embed_dim) * 0.02
        )
        self.beat_queries = nn.Parameter(
            torch.randn(1, n_beat_queries, embed_dim) * 0.02
        )
        self.rhythm_queries = nn.Parameter(
            torch.randn(1, n_rhythm_queries, embed_dim) * 0.02
        )

        # Positional encoding for input sequence
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, embed_dim) * 0.02
        )

        # Cross-attention modules for each scale
        self.wave_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.beat_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rhythm_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.wave_norm = nn.LayerNorm(embed_dim)
        self.beat_norm = nn.LayerNorm(embed_dim)
        self.rhythm_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, embed_dim) - Full sequence from LeadTransformer

        Returns:
            dict with:
                'wave': (B, n_wave_queries, embed_dim)
                'beat': (B, n_beat_queries, embed_dim)
                'rhythm': (B, n_rhythm_queries, embed_dim)
                'wave_attn': (B, n_wave_queries, seq_len) - attention weights
                'beat_attn': (B, n_beat_queries, seq_len)
                'rhythm_attn': (B, n_rhythm_queries, seq_len)
        """
        B, N, D = x.shape

        # Add positional encoding to input sequence
        x = x + self.pos_encoding[:, :N, :]

        # Expand queries for batch
        wave_q = self.wave_queries.expand(B, -1, -1)
        beat_q = self.beat_queries.expand(B, -1, -1)
        rhythm_q = self.rhythm_queries.expand(B, -1, -1)

        # Cross-attention: queries attend to full sequence
        wave_feat, wave_attn = self.wave_attn(
            wave_q, x, x, average_attn_weights=True
        )
        beat_feat, beat_attn = self.beat_attn(
            beat_q, x, x, average_attn_weights=True
        )
        rhythm_feat, rhythm_attn = self.rhythm_attn(
            rhythm_q, x, x, average_attn_weights=True
        )

        # Normalize
        wave_feat = self.wave_norm(wave_feat)
        beat_feat = self.beat_norm(beat_feat)
        rhythm_feat = self.rhythm_norm(rhythm_feat)

        return {
            'wave': wave_feat,
            'beat': beat_feat,
            'rhythm': rhythm_feat,
            'wave_attn': wave_attn,
            'beat_attn': beat_attn,
            'rhythm_attn': rhythm_attn
        }


class AttentionPooling(nn.Module):
    """
    Simple attention pooling with learnable queries

    Used for final pooling from sequence to vector.

    Args:
        seq_len: Input sequence length
        embed_dim: Embedding dimension
        num_queries: Number of query tokens (default: 1 for single vector output)
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        seq_len,
        embed_dim,
        num_queries=1,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, embed_dim) * 0.02
        )

        # Cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) - Input sequence

        Returns:
            pooled: (B, num_queries, D) or (B, D) if num_queries=1
            attn_weights: (B, num_queries, N)
        """
        B, N, D = x.shape

        # Add positional encoding
        x = x + self.pos_encoding[:, :N, :]

        # Expand queries
        queries = self.queries.expand(B, -1, -1)

        # Cross-attention
        pooled, attn_weights = self.attn(
            queries, x, x, average_attn_weights=True
        )

        pooled = self.norm(pooled)

        # Squeeze if single query
        if self.num_queries == 1:
            pooled = pooled.squeeze(1)

        return pooled, attn_weights


# ============================================================================
# ABLATION HOOK: Pooling Strategies
# ============================================================================
# To experiment with different pooling methods:
#
# 1. Hierarchical pooling (313 → 100 → 30 → 10 → 3)
# 2. Convolutional pooling (stride-based downsampling)
# 3. Learnable soft binning
# 4. Different numbers of queries per scale
# 5. Shared vs independent attention for different scales
#
# Example:
# class HierarchicalPooler(nn.Module):
#     def __init__(self, ...):
#         self.stage1 = AttentionPooling(313, embed_dim, num_queries=100)
#         self.stage2_wave = AttentionPooling(100, embed_dim, num_queries=30)
#         self.stage2_beat = AttentionPooling(100, embed_dim, num_queries=10)
#         self.stage2_rhythm = AttentionPooling(100, embed_dim, num_queries=3)
# ============================================================================
