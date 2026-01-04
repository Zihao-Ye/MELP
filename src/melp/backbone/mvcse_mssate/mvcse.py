"""
Multi-Scale Patch Embedding ECG Encoder

支持两种模式:
1. 无分组: 12导联统一处理
2. 分组模式: 肢体导联(I,II,III,aVR,aVL,aVF) + 胸导联(V1-V6) 分别处理后融合

架构:
1. 每个导联独立: 标准ResNet18(4个stage,512通道) + 多尺度Patch Embedding (全局共享权重)
2. LeadTransformer: 单导联时序建模 (分组时每组独立,否则全局共享)
3. CrossLeadAggregation: 导联聚合 (分组时6→1, 否则12→1)
4. 组间融合: 拼接+投影 (仅分组模式)
5. 可选通道注意力

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
    ECG 空间编码器，支持导联分组。

    处理流程:
    1. 12导联各自独立进行 ResNet18 + 多尺度Patch Embedding (全局共享权重)
    2. LeadTransformer 学习单导联内的时序依赖
       - 无分组: 全局共享权重
       - 分组: 肢体组/胸导联组各自独立
    3. CrossLeadAggregation 聚合导联信息
       - 无分组: 12导联→1
       - 分组: 每组6导联→1，然后拼接投影
    4. 可选的通道注意力增强

    输入: (B, 12, L)
    输出: Dict containing 'fused_features': List[(B, N, D)] per scale
    """

    # 导联分组定义 (假设顺序: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
    LIMB_LEADS = [0, 1, 2, 3, 4, 5]      # I, II, III, aVR, aVL, aVF
    CHEST_LEADS = [6, 7, 8, 9, 10, 11]   # V1-V6

    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 5000,
        num_leads: int = 12,
        patch_configs: Optional[List[Dict]] = None,
        # 导联级Transformer参数
        lead_transformer_depth: int = 2,
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
        # 导联分组参数
        use_lead_groups: bool = False,  # 是否启用肢体/胸导联分组
        # 其他
        max_len: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_leads = num_leads
        self.use_lead_groups = use_lead_groups

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

        if use_lead_groups:
            # ========== 分组模式: 肢体组 + 胸导联组 ==========
            # 肢体组 LeadTransformer
            self.limb_transformer = LeadTransformer(
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
            # 胸导联组 LeadTransformer
            self.chest_transformer = LeadTransformer(
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
            # 肢体组聚合 (6→1)
            self.limb_agg = CrossLeadAggregation(
                embed_dim=embed_dim,
                num_heads=cross_lead_heads,
                dim_head=lead_transformer_dim_head,
                mlp_ratio=lead_transformer_mlp_ratio,
                dropout=cross_lead_dropout,
                depth=cross_lead_depth
            )
            # 胸导联组聚合 (6→1)
            self.chest_agg = CrossLeadAggregation(
                embed_dim=embed_dim,
                num_heads=cross_lead_heads,
                dim_head=lead_transformer_dim_head,
                mlp_ratio=lead_transformer_mlp_ratio,
                dropout=cross_lead_dropout,
                depth=cross_lead_depth
            )
            # 组间融合: 拼接后投影
            self.group_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        else:
            # ========== 无分组模式: 12导联统一处理 ==========
            # 全局共享的LeadTransformer
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
            # 全局共享的CrossLeadAggregation
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
            - 'lead_attn_weights': List[(B, N, 6/12)] per scale - 导联注意力权重
            - 'limb_attn_weights': (仅分组模式) 肢体组注意力权重
            - 'chest_attn_weights': (仅分组模式) 胸导联组注意力权重
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
        limb_attn_weights = []
        chest_attn_weights = []

        for s in range(self.num_scales):
            if self.use_lead_groups:
                # ========== 分组模式 ==========
                # 分离肢体导联和胸导联
                limb_feats = [scale_features[s][i] for i in self.LIMB_LEADS]
                chest_feats = [scale_features[s][i] for i in self.CHEST_LEADS]

                # 堆叠: (B, 6, N, D)
                limb_stacked = torch.stack(limb_feats, dim=1)
                chest_stacked = torch.stack(chest_feats, dim=1)

                # 各组独立的LeadTransformer
                limb_stacked = self.limb_transformer(limb_stacked)   # (B, 6, N, D)
                chest_stacked = self.chest_transformer(chest_stacked) # (B, 6, N, D)

                # 各组独立聚合: 6→1
                limb_agg, limb_attn = self.limb_agg(limb_stacked)     # (B, N, D)
                chest_agg, chest_attn = self.chest_agg(chest_stacked) # (B, N, D)

                # 组间融合: 拼接后投影
                concat_feat = torch.cat([limb_agg, chest_agg], dim=-1)  # (B, N, 2D)
                aggregated = self.group_fusion(concat_feat)  # (B, N, D)

                # 保存注意力权重
                if limb_attn is not None:
                    limb_attn_weights.append(limb_attn)
                if chest_attn is not None:
                    chest_attn_weights.append(chest_attn)

            else:
                # ========== 无分组模式 ==========
                # 堆叠12导联: (B, 12, N, D)
                stacked = torch.stack(scale_features[s], dim=1)

                # LeadTransformer: 学习单导联内时序依赖
                stacked = self.lead_transformer(stacked)  # (B, 12, N, D)

                # CrossLeadAggregation: 12导联聚合为1个
                aggregated, attn = self.cross_lead_agg(stacked)  # (B, N, D)

                if attn is not None:
                    lead_attn_weights.append(attn)

            # 可选的通道注意力
            aggregated = self.channel_attn(aggregated)
            fused_features.append(aggregated)

        result = {
            'fused_features': fused_features,  # [(B, N1, D), (B, N2, D), (B, N3, D)]
        }

        if self.use_lead_groups:
            result['limb_attn_weights'] = limb_attn_weights
            result['chest_attn_weights'] = chest_attn_weights
        else:
            result['lead_attn_weights'] = lead_attn_weights

        return result

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, num_leads={self.num_leads}, '
            f'num_scales={self.num_scales}, use_lead_groups={self.use_lead_groups}'
        )
