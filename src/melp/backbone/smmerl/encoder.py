"""
SMMERL Encoder: Main Architecture

Improved ECG encoder with the following pipeline:
1. ResNet frontend: Extract features from raw ECG (B, 12, 5000) → (B, 12, 313, 512)
2. Projection: (B, 12, 313, 512) → (B, 12, 313, 256)
3. LeadTransformer: Model temporal relationships on full sequence
4. Multi-scale extraction: Extract wave/beat/rhythm features
5. Lead grouping: LISA/Limb-Chest/None
6. Cross-lead aggregation: Aggregate across leads
7. Final pooling: Sequence → vector
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from einops import rearrange

from .resnet_frontend import resnet18_frontend, resnet34_frontend
from .transformer import LeadTransformer
from .attention import MultiScaleAttentionPooler, AttentionPooling
from .aggregation import (
    CrossLeadAggregation,
    LISAGroupAggregation,
    LimbChestGroupAggregation
)


class SMMERLEncoder(nn.Module):
    """
    SMMERL: Simplified Multi-scale Multi-lead ECG Representation Learning

    Key improvements over previous architectures:
    - Temporal modeling on full sequence (313 tokens) before multi-scale extraction
    - Cleaner architecture without excessive ablation parameters
    - Better separation of concerns

    Args:
        embed_dim: Embedding dimension (default: 256)
        seq_len: Input ECG sequence length (default: 5000)
        num_leads: Number of ECG leads (default: 12)

        # ResNet frontend
        resnet_type: 'resnet18' or 'resnet34' (default: 'resnet18')

        # LeadTransformer
        lead_transformer_depth: Number of transformer layers (default: 4)
        lead_transformer_heads: Number of attention heads (default: 8)
        lead_transformer_dim_head: Dimension per head (default: 64)
        lead_transformer_mlp_ratio: MLP expansion ratio (default: 4.0)
        lead_transformer_dropout: Dropout rate (default: 0.1)

        # Multi-scale pooling
        n_wave_queries: Number of wave-level queries (default: 30)
        n_beat_queries: Number of beat-level queries (default: 10)
        n_rhythm_queries: Number of rhythm-level queries (default: 3)
        pooler_heads: Number of attention heads in pooler (default: 8)

        # Lead grouping
        lead_group_strategy: 'none', 'limb_chest', or 'lisa' (default: 'lisa')

        # Output
        output_dim: Output dimension (default: None, uses embed_dim)
        pool_type: Final pooling type - 'mean' or 'attn' (default: 'mean')
    """

    def __init__(
        self,
        # Basic parameters
        embed_dim: int = 256,
        seq_len: int = 5000,
        num_leads: int = 12,
        # ResNet frontend
        resnet_type: str = 'resnet18',
        # LeadTransformer
        lead_transformer_depth: int = 4,
        lead_transformer_heads: int = 8,
        lead_transformer_dim_head: int = 64,
        lead_transformer_mlp_ratio: float = 4.0,
        lead_transformer_dropout: float = 0.1,
        # Multi-scale pooling
        n_wave_queries: int = 30,
        n_beat_queries: int = 10,
        n_rhythm_queries: int = 3,
        pooler_heads: int = 8,
        # Lead grouping
        lead_group_strategy: str = 'lisa',
        # Output
        output_dim: Optional[int] = None,
        pool_type: str = 'mean',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_leads = num_leads
        self.lead_group_strategy = lead_group_strategy
        self.pool_type = pool_type
        self.output_dim = output_dim or embed_dim

        # ========== 1. ResNet Frontend ==========
        if resnet_type == 'resnet18':
            self.resnet = resnet18_frontend(in_channels=1)
        elif resnet_type == 'resnet34':
            self.resnet = resnet34_frontend(in_channels=1)
        else:
            raise ValueError(f"Unknown resnet_type: {resnet_type}")

        resnet_out_channels = self.resnet.out_channels  # 512
        resnet_out_len = 313  # For 5000-length input

        # ========== 2. Projection Layer ==========
        self.resnet_proj = nn.Sequential(
            nn.Linear(resnet_out_channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ========== 3. LeadTransformer ==========
        self.lead_transformer = LeadTransformer(
            embed_dim=embed_dim,
            depth=lead_transformer_depth,
            num_heads=lead_transformer_heads,
            dim_head=lead_transformer_dim_head,
            mlp_ratio=lead_transformer_mlp_ratio,
            dropout=lead_transformer_dropout,
            max_len=resnet_out_len,
            use_relative_pos=True
        )

        # ========== 4. Multi-scale Attention Pooler ==========
        self.multi_scale_pooler = MultiScaleAttentionPooler(
            seq_len=resnet_out_len,
            embed_dim=embed_dim,
            n_wave_queries=n_wave_queries,
            n_beat_queries=n_beat_queries,
            n_rhythm_queries=n_rhythm_queries,
            num_heads=pooler_heads,
            dropout=lead_transformer_dropout
        )

        # ========== 5. Lead Grouping & Aggregation ==========
        if lead_group_strategy == 'lisa':
            self.wave_aggregator = LISAGroupAggregation(embed_dim)
            self.beat_aggregator = LISAGroupAggregation(embed_dim)
            self.rhythm_aggregator = LISAGroupAggregation(embed_dim)
        elif lead_group_strategy == 'limb_chest':
            self.wave_aggregator = LimbChestGroupAggregation(embed_dim)
            self.beat_aggregator = LimbChestGroupAggregation(embed_dim)
            self.rhythm_aggregator = LimbChestGroupAggregation(embed_dim)
        elif lead_group_strategy == 'none':
            self.wave_aggregator = CrossLeadAggregation(embed_dim)
            self.beat_aggregator = CrossLeadAggregation(embed_dim)
            self.rhythm_aggregator = CrossLeadAggregation(embed_dim)
        else:
            raise ValueError(f"Unknown lead_group_strategy: {lead_group_strategy}")

        # ========== 6. Final Pooling ==========
        if pool_type == 'attn':
            self.wave_final_pool = AttentionPooling(n_wave_queries, embed_dim, num_queries=1)
            self.beat_final_pool = AttentionPooling(n_beat_queries, embed_dim, num_queries=1)
            self.rhythm_final_pool = AttentionPooling(n_rhythm_queries, embed_dim, num_queries=1)
        # else: use mean pooling (no module needed)

        # ========== 7. Output Projection ==========
        if self.output_dim != embed_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single-vector output (for contrastive learning)

        Args:
            x: (B, 12, 5000) - 12-lead ECG signals

        Returns:
            output: (B, output_dim) - Single vector representation
        """
        # Get multi-scale features
        multi_scale_output = self.forward_multiscale(x)

        # Average across scales
        wave_pooled = multi_scale_output['wave']
        beat_pooled = multi_scale_output['beat']
        rhythm_pooled = multi_scale_output['rhythm']

        output = (wave_pooled + beat_pooled + rhythm_pooled) / 3

        return output

    def forward_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features

        Args:
            x: (B, 12, 5000) - 12-lead ECG signals

        Returns:
            dict with:
                'wave': (B, output_dim) - Wave-level features
                'beat': (B, output_dim) - Beat-level features
                'rhythm': (B, output_dim) - Rhythm-level features
        """
        B = x.shape[0]

        # ========== Stage 1: ResNet Feature Extraction ==========
        all_lead_feats = []
        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]  # (B, 1, 5000)
            lead_feat = self.resnet(lead_signal)  # (B, 512, 313)
            lead_feat = rearrange(lead_feat, 'b c n -> b n c')  # (B, 313, 512)
            lead_feat = self.resnet_proj(lead_feat)  # (B, 313, 256)
            all_lead_feats.append(lead_feat)

        all_leads = torch.stack(all_lead_feats, dim=1)  # (B, 12, 313, 256)

        # ========== Stage 2: Full-Sequence Temporal Modeling ==========
        all_leads = self.lead_transformer(all_leads)  # (B, 12, 313, 256)

        # ========== Stage 3: Multi-scale Feature Extraction ==========
        all_wave_feats = []
        all_beat_feats = []
        all_rhythm_feats = []

        for lead_idx in range(self.num_leads):
            lead_seq = all_leads[:, lead_idx, :, :]  # (B, 313, 256)

            # Extract multi-scale features from temporally-modeled sequence
            pooler_output = self.multi_scale_pooler(lead_seq)

            all_wave_feats.append(pooler_output['wave'])      # (B, 30, 256)
            all_beat_feats.append(pooler_output['beat'])      # (B, 10, 256)
            all_rhythm_feats.append(pooler_output['rhythm'])  # (B, 3, 256)

        # Stack: (B, 12, N, 256)
        wave_stacked = torch.stack(all_wave_feats, dim=1)    # (B, 12, 30, 256)
        beat_stacked = torch.stack(all_beat_feats, dim=1)    # (B, 12, 10, 256)
        rhythm_stacked = torch.stack(all_rhythm_feats, dim=1)  # (B, 12, 3, 256)

        # ========== Stage 4: Cross-Lead Aggregation ==========
        wave_agg, _ = self.wave_aggregator(wave_stacked)    # (B, 30, 256)
        beat_agg, _ = self.beat_aggregator(beat_stacked)    # (B, 10, 256)
        rhythm_agg, _ = self.rhythm_aggregator(rhythm_stacked)  # (B, 3, 256)

        # ========== Stage 5: Final Pooling ==========
        if self.pool_type == 'mean':
            wave_pooled = wave_agg.mean(dim=1)      # (B, 256)
            beat_pooled = beat_agg.mean(dim=1)      # (B, 256)
            rhythm_pooled = rhythm_agg.mean(dim=1)  # (B, 256)
        elif self.pool_type == 'attn':
            wave_pooled, _ = self.wave_final_pool(wave_agg)
            beat_pooled, _ = self.beat_final_pool(beat_agg)
            rhythm_pooled, _ = self.rhythm_final_pool(rhythm_agg)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # ========== Stage 6: Output Projection ==========
        wave_pooled = self.output_proj(wave_pooled)
        beat_pooled = self.output_proj(beat_pooled)
        rhythm_pooled = self.output_proj(rhythm_pooled)

        return {
            'wave': wave_pooled,
            'beat': beat_pooled,
            'rhythm': rhythm_pooled
        }

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass returning intermediate features for analysis

        Args:
            x: (B, 12, 5000) - 12-lead ECG signals

        Returns:
            dict with intermediate features and attention weights
        """
        B = x.shape[0]

        # Stage 1: ResNet
        all_lead_feats = []
        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]
            lead_feat = self.resnet(lead_signal)
            lead_feat = rearrange(lead_feat, 'b c n -> b n c')
            lead_feat = self.resnet_proj(lead_feat)
            all_lead_feats.append(lead_feat)

        all_leads = torch.stack(all_lead_feats, dim=1)

        # Stage 2: LeadTransformer
        all_leads_transformed = self.lead_transformer(all_leads)

        # Stage 3: Multi-scale extraction (with attention weights)
        all_wave_feats, all_beat_feats, all_rhythm_feats = [], [], []
        all_wave_attns, all_beat_attns, all_rhythm_attns = [], [], []

        for lead_idx in range(self.num_leads):
            lead_seq = all_leads_transformed[:, lead_idx, :, :]
            pooler_output = self.multi_scale_pooler(lead_seq)

            all_wave_feats.append(pooler_output['wave'])
            all_beat_feats.append(pooler_output['beat'])
            all_rhythm_feats.append(pooler_output['rhythm'])
            all_wave_attns.append(pooler_output['wave_attn'])
            all_beat_attns.append(pooler_output['beat_attn'])
            all_rhythm_attns.append(pooler_output['rhythm_attn'])

        wave_stacked = torch.stack(all_wave_feats, dim=1)
        beat_stacked = torch.stack(all_beat_feats, dim=1)
        rhythm_stacked = torch.stack(all_rhythm_feats, dim=1)

        # Stage 4: Aggregation (with attention weights)
        wave_agg, wave_group_attns = self.wave_aggregator(wave_stacked)
        beat_agg, beat_group_attns = self.beat_aggregator(beat_stacked)
        rhythm_agg, rhythm_group_attns = self.rhythm_aggregator(rhythm_stacked)

        # Stage 5: Final pooling
        if self.pool_type == 'mean':
            wave_pooled = wave_agg.mean(dim=1)
            beat_pooled = beat_agg.mean(dim=1)
            rhythm_pooled = rhythm_agg.mean(dim=1)
            final_pool_attns = None
        else:
            wave_pooled, wave_pool_attn = self.wave_final_pool(wave_agg)
            beat_pooled, beat_pool_attn = self.beat_final_pool(beat_agg)
            rhythm_pooled, rhythm_pool_attn = self.rhythm_final_pool(rhythm_agg)
            final_pool_attns = {
                'wave': wave_pool_attn,
                'beat': beat_pool_attn,
                'rhythm': rhythm_pool_attn
            }

        return {
            'resnet_features': all_leads,
            'transformer_features': all_leads_transformed,
            'wave_features': wave_stacked,
            'beat_features': beat_stacked,
            'rhythm_features': rhythm_stacked,
            'wave_aggregated': wave_agg,
            'beat_aggregated': beat_agg,
            'rhythm_aggregated': rhythm_agg,
            'wave_pooled': wave_pooled,
            'beat_pooled': beat_pooled,
            'rhythm_pooled': rhythm_pooled,
            'multi_scale_attns': {
                'wave': torch.stack(all_wave_attns, dim=1),
                'beat': torch.stack(all_beat_attns, dim=1),
                'rhythm': torch.stack(all_rhythm_attns, dim=1)
            },
            'group_attns': {
                'wave': wave_group_attns,
                'beat': beat_group_attns,
                'rhythm': rhythm_group_attns
            },
            'final_pool_attns': final_pool_attns
        }


# ============================================================================
# ABLATION HOOK: Overall Architecture
# ============================================================================
# Key ablation experiments to consider:
#
# 1. Multi-scale vs Single-scale
#    - Disable multi-scale pooler, use single global pooling
#
# 2. Lead grouping strategies
#    - Compare LISA vs Limb/Chest vs None
#
# 3. Temporal modeling depth
#    - Vary lead_transformer_depth (2, 4, 6, 8 layers)
#
# 4. Pooling strategies
#    - Mean vs Attention pooling
#    - Different numbers of queries
#
# 5. ResNet variants
#    - ResNet18 vs ResNet34 vs custom depths
#
# To add ablation:
# - Add boolean flags or enum parameters
# - Implement alternative paths in forward()
# - Keep code clean with clear if/else branches
# ============================================================================


# ============================================================================
# Factory Functions for Different Model Sizes
# ============================================================================

def smmerl_tiny(
    seq_len: int = 5000,
    lead_group_strategy: str = 'lisa',
    pool_type: str = 'mean',
    output_dim: int = 256
) -> SMMERLEncoder:
    """
    SMMERL-Tiny: Lightweight model (~8M parameters)

    Args:
        seq_len: Input ECG sequence length
        lead_group_strategy: Lead grouping strategy ('none', 'limb_chest', 'lisa')
        pool_type: Final pooling type ('mean', 'attn')
        output_dim: Output embedding dimension

    Returns:
        SMMERLEncoder instance
    """
    return SMMERLEncoder(
        embed_dim=128,
        seq_len=seq_len,
        lead_transformer_depth=2,
        lead_transformer_heads=4,
        n_wave_queries=20,
        n_beat_queries=8,
        n_rhythm_queries=2,
        lead_group_strategy=lead_group_strategy,
        pool_type=pool_type,
        output_dim=output_dim
    )


def smmerl_small(
    seq_len: int = 5000,
    lead_group_strategy: str = 'lisa',
    pool_type: str = 'mean',
    output_dim: int = 256
) -> SMMERLEncoder:
    """
    SMMERL-Small: Small model (~18M parameters)

    Args:
        seq_len: Input ECG sequence length
        lead_group_strategy: Lead grouping strategy ('none', 'limb_chest', 'lisa')
        pool_type: Final pooling type ('mean', 'attn')
        output_dim: Output embedding dimension

    Returns:
        SMMERLEncoder instance
    """
    return SMMERLEncoder(
        embed_dim=192,
        seq_len=seq_len,
        lead_transformer_depth=4,
        lead_transformer_heads=6,
        n_wave_queries=25,
        n_beat_queries=9,
        n_rhythm_queries=3,
        lead_group_strategy=lead_group_strategy,
        pool_type=pool_type,
        output_dim=output_dim
    )


def smmerl_base(
    seq_len: int = 5000,
    lead_group_strategy: str = 'lisa',
    pool_type: str = 'mean',
    output_dim: int = 256
) -> SMMERLEncoder:
    """
    SMMERL-Base: Base model (~35M parameters)

    Args:
        seq_len: Input ECG sequence length
        lead_group_strategy: Lead grouping strategy ('none', 'limb_chest', 'lisa')
        pool_type: Final pooling type ('mean', 'attn')
        output_dim: Output embedding dimension

    Returns:
        SMMERLEncoder instance
    """
    return SMMERLEncoder(
        embed_dim=256,
        seq_len=seq_len,
        lead_transformer_depth=6,
        lead_transformer_heads=8,
        n_wave_queries=30,
        n_beat_queries=10,
        n_rhythm_queries=3,
        lead_group_strategy=lead_group_strategy,
        pool_type=pool_type,
        output_dim=output_dim
    )


def smmerl_large(
    seq_len: int = 5000,
    lead_group_strategy: str = 'lisa',
    pool_type: str = 'mean',
    output_dim: int = 256
) -> SMMERLEncoder:
    """
    SMMERL-Large: Large model (~65M parameters)

    Args:
        seq_len: Input ECG sequence length
        lead_group_strategy: Lead grouping strategy ('none', 'limb_chest', 'lisa')
        pool_type: Final pooling type ('mean', 'attn')
        output_dim: Output embedding dimension

    Returns:
        SMMERLEncoder instance
    """
    return SMMERLEncoder(
        embed_dim=384,
        seq_len=seq_len,
        lead_transformer_depth=8,
        lead_transformer_heads=12,
        n_wave_queries=40,
        n_beat_queries=12,
        n_rhythm_queries=4,
        lead_group_strategy=lead_group_strategy,
        pool_type=pool_type,
        output_dim=output_dim
    )
