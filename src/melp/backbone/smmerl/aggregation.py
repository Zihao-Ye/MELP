"""
Cross-Lead Aggregation Modules

Aggregates information across multiple ECG leads, with support for:
- LISA anatomical grouping (Lateral, Inferior, Septal, Anterior, aVR)
- Limb/Chest grouping
- No grouping (all 12 leads together)
"""

import torch
import torch.nn as nn
from einops import rearrange


class CrossLeadAggregation(nn.Module):
    """
    Aggregates features across multiple leads using attention

    Reduces the lead dimension while preserving temporal structure.
    For example: (B, num_leads, N, D) → (B, N, D)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-lead attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Learnable aggregation query
        self.agg_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, num_leads, N, D) - Multi-lead sequences

        Returns:
            aggregated: (B, N, D) - Aggregated sequence
            attn_weights: (B, N, num_leads) - Attention weights showing lead importance
        """
        B, num_leads, N, D = x.shape

        # Process each time step independently
        # Reshape: (B, num_leads, N, D) → (B*N, num_leads, D)
        x = rearrange(x, 'b l n d -> (b n) l d')

        # Expand query for all time steps
        query = self.agg_query.expand(B * N, -1, -1)  # (B*N, 1, D)

        # Cross-attention: single query attends to all leads
        aggregated, attn_weights = self.attn(
            query, x, x, average_attn_weights=True
        )  # aggregated: (B*N, 1, D), attn_weights: (B*N, 1, num_leads)

        # Reshape back
        aggregated = rearrange(aggregated, '(b n) 1 d -> b n d', b=B, n=N)
        attn_weights = rearrange(attn_weights, '(b n) 1 l -> b n l', b=B, n=N)

        aggregated = self.norm(aggregated)

        return aggregated, attn_weights


class LISAGroupAggregation(nn.Module):
    """
    LISA (Lead-based Interpretable Spatial Aggregation) Grouping

    Groups ECG leads based on cardiac anatomy:
    - Lateral (侧壁): I, aVL, V5, V6 - observes lateral wall
    - Inferior (下壁): II, III, aVF - observes inferior wall
    - Septal (间隔): V1, V2 - observes interventricular septum
    - Anterior (前壁): V3, V4 - observes anterior wall
    - aVR: Reference lead, provides cavity view

    Each group is processed independently, then fused.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads per group
        dropout: Dropout rate
    """

    # Lead indices (standard 12-lead order: I, II, III, aVR, aVL, aVF, V1-V6)
    LATERAL_LEADS = [0, 4, 10, 11]   # I, aVL, V5, V6
    INFERIOR_LEADS = [1, 2, 5]        # II, III, aVF
    SEPTAL_LEADS = [6, 7]             # V1, V2
    ANTERIOR_LEADS = [8, 9]           # V3, V4
    AVR_LEADS = [3]                   # aVR

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Independent aggregation for each anatomical group
        self.lateral_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)
        self.inferior_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)
        self.septal_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)
        self.anterior_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)
        self.avr_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)

        # Fusion layer: combine 5 groups into single representation
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 12, N, D) - 12-lead sequences

        Returns:
            fused: (B, N, D) - Fused representation
            group_attns: dict - Attention weights for each group
        """
        # Extract each anatomical group
        lateral = x[:, self.LATERAL_LEADS, :, :]    # (B, 4, N, D)
        inferior = x[:, self.INFERIOR_LEADS, :, :]  # (B, 3, N, D)
        septal = x[:, self.SEPTAL_LEADS, :, :]      # (B, 2, N, D)
        anterior = x[:, self.ANTERIOR_LEADS, :, :]  # (B, 2, N, D)
        avr = x[:, self.AVR_LEADS, :, :]            # (B, 1, N, D)

        # Aggregate each group independently
        lateral_agg, lateral_attn = self.lateral_agg(lateral)    # (B, N, D)
        inferior_agg, inferior_attn = self.inferior_agg(inferior)
        septal_agg, septal_attn = self.septal_agg(septal)
        anterior_agg, anterior_attn = self.anterior_agg(anterior)
        avr_agg, avr_attn = self.avr_agg(avr)

        # Concatenate and fuse
        concat = torch.cat([
            lateral_agg, inferior_agg, septal_agg, anterior_agg, avr_agg
        ], dim=-1)  # (B, N, 5*D)

        fused = self.fusion(concat)  # (B, N, D)

        # Store attention weights for interpretability
        group_attns = {
            'lateral': lateral_attn,
            'inferior': inferior_attn,
            'septal': septal_attn,
            'anterior': anterior_attn,
            'avr': avr_attn
        }

        return fused, group_attns


class LimbChestGroupAggregation(nn.Module):
    """
    Limb/Chest Lead Grouping

    Simpler grouping strategy:
    - Limb leads: I, II, III, aVR, aVL, aVF (6 leads)
    - Chest leads: V1-V6 (6 leads)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads per group
        dropout: Dropout rate
    """

    LIMB_LEADS = [0, 1, 2, 3, 4, 5]      # I, II, III, aVR, aVL, aVF
    CHEST_LEADS = [6, 7, 8, 9, 10, 11]   # V1-V6

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Independent aggregation for each group
        self.limb_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)
        self.chest_agg = CrossLeadAggregation(embed_dim, num_heads, dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 12, N, D) - 12-lead sequences

        Returns:
            fused: (B, N, D) - Fused representation
            group_attns: dict - Attention weights for each group
        """
        # Extract groups
        limb = x[:, self.LIMB_LEADS, :, :]    # (B, 6, N, D)
        chest = x[:, self.CHEST_LEADS, :, :]  # (B, 6, N, D)

        # Aggregate
        limb_agg, limb_attn = self.limb_agg(limb)
        chest_agg, chest_attn = self.chest_agg(chest)

        # Fuse
        concat = torch.cat([limb_agg, chest_agg], dim=-1)  # (B, N, 2*D)
        fused = self.fusion(concat)  # (B, N, D)

        group_attns = {
            'limb': limb_attn,
            'chest': chest_attn
        }

        return fused, group_attns


# ============================================================================
# ABLATION HOOK: Lead Grouping Strategies
# ============================================================================
# To experiment with different grouping strategies:
#
# 1. Different anatomical groupings (e.g., anterior/posterior/lateral)
# 2. Data-driven grouping (learn which leads to group together)
# 3. Task-specific grouping (different groupings for different tasks)
# 4. Hierarchical grouping (first group by limb/chest, then by anatomy)
#
# Example:
# class LearnableGroupAggregation(nn.Module):
#     def __init__(self, ...):
#         # Learn soft assignment of leads to groups
#         self.group_assignment = nn.Parameter(torch.randn(12, num_groups))
# ============================================================================
