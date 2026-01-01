"""
Multi-Scale Patch Embedding ECG Encoder

移除LISA分组，12导联统一处理，专注验证多尺度Patch的效果。

架构:
1. 每个导联独立: 标准ResNet18(4个stage,512通道) + 多尺度Patch Embedding (全局共享权重)
2. LeadTransformer: 单导联时序建模 (全局共享权重)
3. CrossLeadAggregation: 12导联聚合为1个表示
4. 可选通道注意力

数据流:
- 输入: (B, 12, 5000)
- ResNet18: (B, 1, 5000) → (B, 512, 313) per lead
- Multi-Scale Patch: 78/39/19 tokens per scale
- Output: List[(B, N, embed_dim)] per scale
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .patch_embedding import MultiScalePatchEmbedding
from .transformer import LeadTransformer
from .attention import CrossLeadAggregation
from .base_modules import SEBlock, ECABlock


class MVCSEEncoder(nn.Module):
    """
    简化版 ECG 空间编码器 (无分组)。

    处理流程:
    1. 12导联各自独立进行 ResNet18 + 多尺度Patch Embedding (全局共享权重)
    2. LeadTransformer 学习单导联内的时序依赖 (全局共享权重)
    3. CrossLeadAggregation 聚合12导联信息
    4. 可选的通道注意力增强

    输入: (B, 12, L)
    输出: Dict containing 'fused_features': List[(B, N, D)] per scale
    """

    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 5000,
        num_leads: int = 12,
        patch_configs: Optional[List[Dict]] = None,
        # 导联级Transformer参数
        lead_transformer_depth: int = 2,  # 简化: 从6层减到2层
        lead_transformer_heads: int = 4,
        lead_transformer_dim_head: int = 64,
        lead_transformer_mlp_ratio: float = 4.,
        lead_transformer_dropout: float = 0.1,
        lead_transformer_drop_path: float = 0.1,
        # Cross-Lead聚合参数
        cross_lead_depth: int = 1,
        cross_lead_heads: int = 4,
        cross_lead_dropout: float = 0.1,
        # 通道注意力参数
        channel_attention: str = 'se',  # 'se', 'eca', 'none'
        reduction: int = 4,
        # 其他
        max_len: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_leads = num_leads

        # 默认patch配置（适配ResNet输出）
        if patch_configs is None:
            patch_configs = [
                {'kernel_size': 4, 'stride': 4},    # 78 tokens
                {'kernel_size': 8, 'stride': 8},    # 39 tokens
                {'kernel_size': 16, 'stride': 16},  # 19 tokens
            ]
        self.num_scales = len(patch_configs)
        self.patch_configs = patch_configs

        # 全局共享的PatchEmbedding（内含ResNet18前端）
        self.patch_embed = MultiScalePatchEmbedding(
            embed_dim=embed_dim,
            seq_len=seq_len,
            patch_configs=patch_configs
        )

        # 计算每个尺度的token数
        self.num_patches_list = self.patch_embed.num_patches_list

        # 全局共享的LeadTransformer (所有导联、所有尺度共享)
        self.lead_transformer = LeadTransformer(
            embed_dim=embed_dim,
            depth=lead_transformer_depth,
            num_heads=lead_transformer_heads,
            dim_head=lead_transformer_dim_head,
            mlp_ratio=lead_transformer_mlp_ratio,
            dropout=lead_transformer_dropout,
            drop_path=lead_transformer_drop_path,
            max_len=max_len,
            use_relative_pos=True
        )

        # 全局共享的CrossLeadAggregation (所有尺度共享)
        self.cross_lead_agg = CrossLeadAggregation(
            embed_dim=embed_dim,
            num_heads=cross_lead_heads,
            dim_head=lead_transformer_dim_head,
            mlp_ratio=lead_transformer_mlp_ratio,
            dropout=cross_lead_dropout,
            depth=cross_lead_depth
        )

        # 可选的通道注意力
        if channel_attention == 'se':
            self.channel_attn = SEBlock(embed_dim, reduction)
        elif channel_attention == 'eca':
            self.channel_attn = ECABlock(embed_dim)
        else:
            self.channel_attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            Dict containing:
            - 'fused_features': List[(B, N, D)] per scale - 融合后的多尺度特征
            - 'lead_attn_weights': List[(B, N, 12)] per scale - 导联注意力权重
        """
        B = x.shape[0]

        # 收集所有导联的多尺度特征
        # scale_features[scale_idx] = list of (B, N, D) for each lead
        scale_features = [[] for _ in range(self.num_scales)]

        for lead_idx in range(self.num_leads):
            # 提取单导联信号
            lead_signal = x[:, lead_idx:lead_idx+1, :]  # (B, 1, L)

            # ResNet + 多尺度Patch Embedding（共享权重）
            lead_scales = self.patch_embed(lead_signal)  # [(B, N1, D), (B, N2, D), (B, N3, D)]

            for s, feat in enumerate(lead_scales):
                scale_features[s].append(feat)

        # 对每个尺度进行处理
        fused_features = []
        lead_attn_weights = []

        for s in range(self.num_scales):
            # 堆叠12导联: (B, 12, N, D)
            stacked = torch.stack(scale_features[s], dim=1)

            # LeadTransformer: 学习单导联内时序依赖
            stacked = self.lead_transformer(stacked)  # (B, 12, N, D)

            # CrossLeadAggregation: 12导联聚合为1个
            aggregated, attn = self.cross_lead_agg(stacked)  # (B, N, D)

            # 可选的通道注意力
            aggregated = self.channel_attn(aggregated)

            fused_features.append(aggregated)
            if attn is not None:
                lead_attn_weights.append(attn)

        return {
            'fused_features': fused_features,  # [(B, N1, D), (B, N2, D), (B, N3, D)]
            'lead_attn_weights': lead_attn_weights
        }

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, num_leads={self.num_leads}, '
            f'num_scales={self.num_scales}'
        )
