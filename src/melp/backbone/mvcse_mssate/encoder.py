"""
多尺度 ECG 编码器 (简化版)

移除LISA分组，专注验证多尺度Patch的效果。
整合简化版MVCSE和MS-SATE模块。

新增: HierarchicalMVCSEMSSATEEncoder
使用可学习query的HierarchicalPooler替代硬切分，支持波段/心拍/节律三级建模。
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .mvcse import MVCSEEncoder
from .mssate import MSSATEEncoder
from .resnet_frontend import resnet18_frontend
from .attention import HierarchicalECGPooler, AttentionalPooler
from .transformer import LeadTransformer
from einops import rearrange


class AttentionPool1d(nn.Module):
    """
    1D Attention Pooling (参考MERL实现)

    使用CLS token作为query，对序列做cross-attention来聚合信息。

    Args:
        seq_len: 序列长度
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        output_dim: 输出维度（默认与embed_dim相同）
    """
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        num_heads: int = 4,
        output_dim: int = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        # 位置编码 (CLS + 序列)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, seq_len + 1, embed_dim) * 0.02
        )
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Multi-head attention
        self.mhsa = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # 输出投影
        self.proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, D) - 序列特征

        Returns:
            pooled: (B, output_dim) - 聚合后的特征
            attn_weights: (B, 1, N) - 注意力权重
        """
        B, N, D = x.shape

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # 添加位置编码
        x = x + self.positional_embedding[:, :N+1, :]

        # CLS token作为query，整个序列作为key/value
        pooled, attn_weights = self.mhsa(
            x[:, :1, :],  # query: CLS token
            x,            # key: 全序列
            x,            # value: 全序列
            average_attn_weights=True
        )

        # 投影
        pooled = self.proj(pooled.squeeze(1))  # (B, output_dim)

        # 返回对序列的注意力权重（去掉CLS自己）
        return pooled, attn_weights[:, :, 1:]


class MVCSEMSSATEEncoder(nn.Module):
    """
    ECG 编码器，支持导联分组。

    架构概述:
    1. MVCSE: 空间编码
       - ResNet18前端特征提取
       - 多尺度Patch Embedding
       - LeadTransformer (无分组时全局共享，分组时各组独立)
       - CrossLeadAggregation (无分组: 12→1，分组: 6→1后拼接融合)
       - 可选SE/ECA通道注意力

    2. MS-SATE: 时序编码
       - 每个尺度独立的Transformer
       - 相对位置编码

    3. 特征聚合:
       - Mean/CLS/Concat pooling
       - 投影到输出维度
    """
    def __init__(
        self,
        # MVCSE参数
        embed_dim: int = 256,
        seq_len: int = 5000,
        num_leads: int = 12,
        patch_configs: Optional[List[Dict]] = None,
        lead_transformer_depth: int = 2,
        lead_transformer_heads: int = 4,
        lead_transformer_dim_head: int = 64,
        lead_transformer_mlp_ratio: float = 4.,
        lead_transformer_dropout: float = 0.1,
        lead_transformer_drop_path: float = 0.1,
        cross_lead_depth: int = 1,
        cross_lead_heads: int = 4,
        channel_attention: str = 'se',
        reduction: int = 4,
        # 导联分组参数
        use_lead_groups: bool = False,  # 是否启用肢体/胸导联分组
        # MS-SATE参数
        mssate_depth: int = 2,
        mssate_num_heads: int = 8,
        mssate_dim_head: int = 32,
        mssate_mlp_ratio: float = 4.,
        mssate_dropout: float = 0.1,
        mssate_drop_path: float = 0.1,
        # 输出参数
        output_dim: Optional[int] = None,
        pool_type: str = 'attn',  # 'mean', 'attn', 'concat_mean'
        pool_heads: int = 4,      # attention pooling 的头数
        # 其他
        max_len: int = 512,
        use_relative_pos: bool = True
    ):
        super().__init__()

        self.num_scales = 3  # 默认3个尺度
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        self.pool_heads = pool_heads
        self.use_lead_groups = use_lead_groups

        # 默认patch配置（适配ResNet输出）
        if patch_configs is None:
            patch_configs = [
                {'kernel_size': 4, 'stride': 4},    # 78 tokens
                {'kernel_size': 8, 'stride': 8},    # 39 tokens
                {'kernel_size': 16, 'stride': 16},  # 19 tokens
            ]

        self.num_scales = len(patch_configs)

        # MVCSE模块
        self.mvcse = MVCSEEncoder(
            embed_dim=embed_dim,
            seq_len=seq_len,
            num_leads=num_leads,
            patch_configs=patch_configs,
            lead_transformer_depth=lead_transformer_depth,
            lead_transformer_heads=lead_transformer_heads,
            lead_transformer_dim_head=lead_transformer_dim_head,
            lead_transformer_mlp_ratio=lead_transformer_mlp_ratio,
            lead_transformer_dropout=lead_transformer_dropout,
            lead_transformer_drop_path=lead_transformer_drop_path,
            cross_lead_depth=cross_lead_depth,
            cross_lead_heads=cross_lead_heads,
            channel_attention=channel_attention,
            reduction=reduction,
            use_lead_groups=use_lead_groups,  # 传递分组参数
            max_len=max_len
        )

        # MS-SATE模块 (输入维度现在是 embed_dim，不是 4*embed_dim)
        self.mssate = MSSATEEncoder(
            embed_dim=embed_dim,
            num_groups=1,  # 无分组，相当于1组
            num_scales=self.num_scales,
            depth=mssate_depth,
            num_heads=mssate_num_heads,
            dim_head=mssate_dim_head,
            mlp_ratio=mssate_mlp_ratio,
            dropout=mssate_dropout,
            drop_path=mssate_drop_path,
            max_len=max_len,
            use_relative_pos=use_relative_pos
        )

        # 获取每个尺度的序列长度
        self.num_patches_list = self.mvcse.num_patches_list  # [78, 39, 19]

        # 多尺度特征融合
        if pool_type == 'concat_mean' or pool_type == 'concat_attn':
            fusion_input_dim = embed_dim * self.num_scales  # 3 * 256 = 768
        else:
            fusion_input_dim = embed_dim  # 256

        # 最终输出投影
        self.output_dim = output_dim if output_dim is not None else fusion_input_dim

        if self.output_dim != fusion_input_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(fusion_input_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.output_proj = nn.Identity()

        # Attention Pooling (每个尺度独立，因为序列长度不同)
        if pool_type == 'attn' or pool_type == 'concat_attn':
            self.attn_pools = nn.ModuleList([
                AttentionPool1d(
                    seq_len=num_patches,
                    embed_dim=embed_dim,
                    num_heads=pool_heads,
                    output_dim=embed_dim
                )
                for num_patches in self.num_patches_list
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            output: (B, output_dim) - 最终特征表示
        """
        # MVCSE: 空间编码
        mvcse_output = self.mvcse(x)
        fused_features = mvcse_output['fused_features']  # [(B, N1, D), ...]

        # MS-SATE: 时序编码
        encoded_features = self.mssate(fused_features)

        # 特征聚合
        if self.pool_type == 'mean':
            # 对每个尺度取平均，然后对尺度取平均
            pooled = [feat.mean(dim=1) for feat in encoded_features]  # List of (B, D)
            output = torch.stack(pooled, dim=0).mean(dim=0)  # (B, D)

        elif self.pool_type == 'attn':
            # Attention pooling: 每个尺度独立做attention pooling，然后平均
            pooled = []
            for i, feat in enumerate(encoded_features):
                p, _ = self.attn_pools[i](feat)  # (B, D)
                pooled.append(p)
            output = torch.stack(pooled, dim=0).mean(dim=0)  # (B, D)

        elif self.pool_type == 'concat_mean':
            # 每个尺度取平均后拼接
            pooled = [feat.mean(dim=1) for feat in encoded_features]  # List of (B, D)
            output = torch.cat(pooled, dim=-1)  # (B, num_scales * D)

        elif self.pool_type == 'concat_attn':
            # 每个尺度做attention pooling后拼接
            pooled = []
            for i, feat in enumerate(encoded_features):
                p, _ = self.attn_pools[i](feat)  # (B, D)
                pooled.append(p)
            output = torch.cat(pooled, dim=-1)  # (B, num_scales * D)

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # 输出投影
        output = self.output_proj(output)

        return output

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        返回中间特征，用于可视化和分析。

        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            Dict containing various intermediate features
        """
        # MVCSE: 空间编码
        mvcse_output = self.mvcse(x)
        fused_features = mvcse_output['fused_features']
        lead_attn_weights = mvcse_output['lead_attn_weights']

        # MS-SATE: 时序编码
        encoded_features = self.mssate(fused_features)

        # 聚合特征 (根据pool_type)
        pool_attn_weights = []
        if self.pool_type == 'attn' or self.pool_type == 'concat_attn':
            pooled = []
            for i, feat in enumerate(encoded_features):
                p, attn = self.attn_pools[i](feat)
                pooled.append(p)
                pool_attn_weights.append(attn)
        else:
            pooled = [feat.mean(dim=1) for feat in encoded_features]

        if self.pool_type in ['mean', 'attn']:
            final_output = torch.stack(pooled, dim=0).mean(dim=0)
        else:  # concat_mean, concat_attn
            final_output = torch.cat(pooled, dim=-1)

        final_output = self.output_proj(final_output)

        return {
            'fused_features': fused_features,
            'lead_attn_weights': lead_attn_weights,
            'encoded_features': encoded_features,
            'pooled_features': pooled,
            'pool_attn_weights': pool_attn_weights,  # attention pooling的权重
            'output': final_output
        }

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, output_dim={self.output_dim}, '
            f'pool_type={self.pool_type}, use_lead_groups={self.use_lead_groups}'
        )


class HierarchicalMVCSEMSSATEEncoder(nn.Module):
    """
    层级式ECG编码器 (使用可学习Query)，支持多种导联分组策略。

    与原版MVCSEMSSATEEncoder的区别:
    1. 使用HierarchicalECGPooler替代Conv1d硬切分
    2. 支持波段(wave)/心拍(beat)/节律(rhythm)三级特征建模
    3. 可学习的query通过attention自动决定关注哪些位置

    架构:
    1. ResNet18前端: (B, 12, L) -> 12×(B, 512, 313)
    2. HierarchicalECGPooler: 对每个导联提取三级特征
       - wave: (B, n_wave, D) 波段级
       - beat: (B, n_beat, D) 心拍级
       - rhythm: (B, n_rhythm, D) 节律级
    3. LeadTransformer: 单导联内时序建模
       - 无分组: 每层级1个共享Transformer
       - LISA分组: 每层级5个独立Transformer
    4. CrossLeadAggregation: 导联聚合
       - 无分组: 12→1
       - LISA分组: 每组独立聚合，然后融合

    支持的分组策略 (lead_group_strategy):
    - 'none': 不分组，12导联统一处理
    - 'limb_chest': 肢体导联(6) + 胸导联(6)
    - 'lisa': LISA解剖分组 - Lateral(4) + Inferior(3) + Septal(2) + Anterior(2) + aVR(1)

    LISA分组基于心脏解剖位置:
    - Lateral (侧壁): I, aVL, V5, V6 - 观察心脏侧壁
    - Inferior (下壁): II, III, aVF - 观察心脏下壁
    - Septal (间隔): V1, V2 - 观察室间隔
    - Anterior (前壁): V3, V4 - 观察心脏前壁
    - aVR: 心腔内视角，综合参考导联
    """

    # ==================== 导联分组定义 ====================
    # 导联顺序: I(0), II(1), III(2), aVR(3), aVL(4), aVF(5), V1(6)-V6(11)

    # 策略1: 肢体/胸导联分组 (2组)
    LIMB_LEADS = [0, 1, 2, 3, 4, 5]      # I, II, III, aVR, aVL, aVF
    CHEST_LEADS = [6, 7, 8, 9, 10, 11]   # V1-V6

    # 策略2: LISA解剖分组 (5组)
    LATERAL_LEADS = [0, 4, 10, 11]   # I, aVL, V5, V6 - 侧壁
    INFERIOR_LEADS = [1, 2, 5]        # II, III, aVF - 下壁
    SEPTAL_LEADS = [6, 7]             # V1, V2 - 间隔
    ANTERIOR_LEADS = [8, 9]           # V3, V4 - 前壁
    AVR_LEADS = [3]                   # aVR - 参考

    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 5000,
        num_leads: int = 12,
        # HierarchicalPooler参数
        n_wave_queries: int = 30,
        n_beat_queries: int = 10,
        n_rhythm_queries: int = 3,
        pooler_heads: int = 8,
        pooler_dropout: float = 0.1,
        pooler_fusion: str = 'separate',  # 'concat', 'hierarchical', 'separate'
        # LeadTransformer参数
        lead_transformer_depth: int = 2,
        lead_transformer_heads: int = 4,
        lead_transformer_dim_head: int = 64,
        lead_transformer_mlp_ratio: float = 4.,
        lead_transformer_dropout: float = 0.1,
        lead_transformer_drop_path: float = 0.1,
        # 导联分组参数
        lead_group_strategy: str = 'none',  # 'none', 'limb_chest', 'lisa'
        use_lead_groups: bool = False,  # 兼容旧参数，如果为True则使用limb_chest
        # 输出参数
        output_dim: Optional[int] = None,
        pool_type: str = 'mean',  # 'mean', 'attn'
        # 其他
        max_len: int = 512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_leads = num_leads
        self.pooler_fusion = pooler_fusion

        # 处理分组策略（兼容旧参数）
        if use_lead_groups and lead_group_strategy == 'none':
            lead_group_strategy = 'limb_chest'
        self.lead_group_strategy = lead_group_strategy

        # ResNet18前端 (全局共享)
        self.resnet = resnet18_frontend(in_channels=1)
        resnet_out_dim = self.resnet.out_channels  # 512

        # 层级Pooler (全局共享，所有导联用同一个)
        self.hierarchical_pooler = HierarchicalECGPooler(
            embed_dim=embed_dim,
            context_dim=resnet_out_dim,
            n_head=pooler_heads,
            n_wave_queries=n_wave_queries,
            n_beat_queries=n_beat_queries,
            n_rhythm_queries=n_rhythm_queries,
            dropout=pooler_dropout,
            fusion_type=pooler_fusion
        )

        # 记录query数量
        self.n_wave_queries = n_wave_queries
        self.n_beat_queries = n_beat_queries
        self.n_rhythm_queries = n_rhythm_queries

        # 跨导联聚合模块
        from .attention import CrossLeadAggregation

        # LeadTransformer 和 CrossLeadAggregation
        if pooler_fusion == 'separate':
            if lead_group_strategy == 'lisa':
                # ========== LISA分组: 每层级×5组 = 15个Transformer + 15个Aggregation ==========
                # Wave层级 - 5组: lateral, inferior, septal, anterior, avr
                self.wave_lateral_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_inferior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_septal_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_anterior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_avr_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_lateral_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_inferior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_septal_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_anterior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_avr_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 5, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

                # Beat层级 - 5组
                self.beat_lateral_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_inferior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_septal_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_anterior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_avr_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_lateral_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_inferior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_septal_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_anterior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_avr_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 5, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

                # Rhythm层级 - 5组
                self.rhythm_lateral_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_inferior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_septal_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_anterior_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_avr_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_lateral_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_inferior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_septal_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_anterior_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_avr_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 5, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

            elif lead_group_strategy == 'limb_chest':
                # ========== 肢体/胸导联分组: 每层级×2组 = 6个Transformer + 6个Aggregation ==========
                # Wave层级
                self.wave_limb_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_chest_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_limb_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_chest_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.wave_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

                # Beat层级
                self.beat_limb_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_chest_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_limb_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_chest_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

                # Rhythm层级
                self.rhythm_limb_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_chest_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_limb_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_chest_agg = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_group_fusion = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU()
                )

            else:
                # ========== 无分组模式: 每层级1个Transformer + 1个Aggregation ==========
                self.wave_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.beat_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.rhythm_transformer = LeadTransformer(
                    embed_dim=embed_dim, depth=lead_transformer_depth,
                    num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                    mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                    drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
                )
                self.wave_cross_lead = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.beat_cross_lead = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
                self.rhythm_cross_lead = CrossLeadAggregation(embed_dim=embed_dim, depth=1)
        else:
            # 统一的Transformer处理融合后的特征 (暂不支持分组)
            self.unified_transformer = LeadTransformer(
                embed_dim=embed_dim, depth=lead_transformer_depth,
                num_heads=lead_transformer_heads, dim_head=lead_transformer_dim_head,
                mlp_ratio=lead_transformer_mlp_ratio, dropout=lead_transformer_dropout,
                drop_path=lead_transformer_drop_path, max_len=max_len, use_relative_pos=True
            )
            self.unified_cross_lead = CrossLeadAggregation(embed_dim=embed_dim, depth=1)

        # 最终输出
        self.pool_type = pool_type
        if pooler_fusion == 'separate':
            fusion_dim = embed_dim * 3 if pool_type == 'concat' else embed_dim
        else:
            fusion_dim = embed_dim

        self.output_dim = output_dim or fusion_dim
        if self.output_dim != fusion_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(fusion_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            output: (B, output_dim) - 最终特征表示
        """
        B = x.shape[0]

        # 收集所有导联的层级特征
        all_wave_feats = []
        all_beat_feats = []
        all_rhythm_feats = []

        for lead_idx in range(self.num_leads):
            # 提取单导联
            lead_signal = x[:, lead_idx:lead_idx+1, :]  # (B, 1, L)

            # ResNet前端
            lead_feat = self.resnet(lead_signal)  # (B, 512, 313)
            lead_feat = rearrange(lead_feat, 'b c n -> b n c')  # (B, 313, 512)

            # HierarchicalPooler
            pooler_out = self.hierarchical_pooler(lead_feat)

            all_wave_feats.append(pooler_out['wave_features'])    # (B, n_wave, D)
            all_beat_feats.append(pooler_out['beat_features'])    # (B, n_beat, D)
            all_rhythm_feats.append(pooler_out['rhythm_features']) # (B, n_rhythm, D)

        # 堆叠12导联: (B, 12, N, D)
        wave_stacked = torch.stack(all_wave_feats, dim=1)    # (B, 12, n_wave, D)
        beat_stacked = torch.stack(all_beat_feats, dim=1)    # (B, 12, n_beat, D)
        rhythm_stacked = torch.stack(all_rhythm_feats, dim=1) # (B, 12, n_rhythm, D)

        if self.pooler_fusion == 'separate':
            if self.lead_group_strategy == 'lisa':
                # ========== LISA分组模式 (5组) ==========
                # 按LISA解剖分组提取导联
                wave_lateral = wave_stacked[:, self.LATERAL_LEADS, :, :]    # (B, 4, n_wave, D)
                wave_inferior = wave_stacked[:, self.INFERIOR_LEADS, :, :]  # (B, 3, n_wave, D)
                wave_septal = wave_stacked[:, self.SEPTAL_LEADS, :, :]      # (B, 2, n_wave, D)
                wave_anterior = wave_stacked[:, self.ANTERIOR_LEADS, :, :]  # (B, 2, n_wave, D)
                wave_avr = wave_stacked[:, self.AVR_LEADS, :, :]            # (B, 1, n_wave, D)

                beat_lateral = beat_stacked[:, self.LATERAL_LEADS, :, :]
                beat_inferior = beat_stacked[:, self.INFERIOR_LEADS, :, :]
                beat_septal = beat_stacked[:, self.SEPTAL_LEADS, :, :]
                beat_anterior = beat_stacked[:, self.ANTERIOR_LEADS, :, :]
                beat_avr = beat_stacked[:, self.AVR_LEADS, :, :]

                rhythm_lateral = rhythm_stacked[:, self.LATERAL_LEADS, :, :]
                rhythm_inferior = rhythm_stacked[:, self.INFERIOR_LEADS, :, :]
                rhythm_septal = rhythm_stacked[:, self.SEPTAL_LEADS, :, :]
                rhythm_anterior = rhythm_stacked[:, self.ANTERIOR_LEADS, :, :]
                rhythm_avr = rhythm_stacked[:, self.AVR_LEADS, :, :]

                # Wave层级: 5组独立处理后融合
                wave_lateral = self.wave_lateral_transformer(wave_lateral)
                wave_inferior = self.wave_inferior_transformer(wave_inferior)
                wave_septal = self.wave_septal_transformer(wave_septal)
                wave_anterior = self.wave_anterior_transformer(wave_anterior)
                wave_avr = self.wave_avr_transformer(wave_avr)

                wave_lateral_agg, _ = self.wave_lateral_agg(wave_lateral)
                wave_inferior_agg, _ = self.wave_inferior_agg(wave_inferior)
                wave_septal_agg, _ = self.wave_septal_agg(wave_septal)
                wave_anterior_agg, _ = self.wave_anterior_agg(wave_anterior)
                wave_avr_agg, _ = self.wave_avr_agg(wave_avr)

                wave_agg = self.wave_group_fusion(torch.cat([
                    wave_lateral_agg, wave_inferior_agg, wave_septal_agg,
                    wave_anterior_agg, wave_avr_agg
                ], dim=-1))

                # Beat层级: 5组独立处理后融合
                beat_lateral = self.beat_lateral_transformer(beat_lateral)
                beat_inferior = self.beat_inferior_transformer(beat_inferior)
                beat_septal = self.beat_septal_transformer(beat_septal)
                beat_anterior = self.beat_anterior_transformer(beat_anterior)
                beat_avr = self.beat_avr_transformer(beat_avr)

                beat_lateral_agg, _ = self.beat_lateral_agg(beat_lateral)
                beat_inferior_agg, _ = self.beat_inferior_agg(beat_inferior)
                beat_septal_agg, _ = self.beat_septal_agg(beat_septal)
                beat_anterior_agg, _ = self.beat_anterior_agg(beat_anterior)
                beat_avr_agg, _ = self.beat_avr_agg(beat_avr)

                beat_agg = self.beat_group_fusion(torch.cat([
                    beat_lateral_agg, beat_inferior_agg, beat_septal_agg,
                    beat_anterior_agg, beat_avr_agg
                ], dim=-1))

                # Rhythm层级: 5组独立处理后融合
                rhythm_lateral = self.rhythm_lateral_transformer(rhythm_lateral)
                rhythm_inferior = self.rhythm_inferior_transformer(rhythm_inferior)
                rhythm_septal = self.rhythm_septal_transformer(rhythm_septal)
                rhythm_anterior = self.rhythm_anterior_transformer(rhythm_anterior)
                rhythm_avr = self.rhythm_avr_transformer(rhythm_avr)

                rhythm_lateral_agg, _ = self.rhythm_lateral_agg(rhythm_lateral)
                rhythm_inferior_agg, _ = self.rhythm_inferior_agg(rhythm_inferior)
                rhythm_septal_agg, _ = self.rhythm_septal_agg(rhythm_septal)
                rhythm_anterior_agg, _ = self.rhythm_anterior_agg(rhythm_anterior)
                rhythm_avr_agg, _ = self.rhythm_avr_agg(rhythm_avr)

                rhythm_agg = self.rhythm_group_fusion(torch.cat([
                    rhythm_lateral_agg, rhythm_inferior_agg, rhythm_septal_agg,
                    rhythm_anterior_agg, rhythm_avr_agg
                ], dim=-1))

            elif self.lead_group_strategy == 'limb_chest':
                # ========== 肢体/胸导联分组模式 (2组) ==========
                # 分离肢体导联和胸导联
                wave_limb = wave_stacked[:, self.LIMB_LEADS, :, :]      # (B, 6, n_wave, D)
                wave_chest = wave_stacked[:, self.CHEST_LEADS, :, :]    # (B, 6, n_wave, D)
                beat_limb = beat_stacked[:, self.LIMB_LEADS, :, :]      # (B, 6, n_beat, D)
                beat_chest = beat_stacked[:, self.CHEST_LEADS, :, :]    # (B, 6, n_beat, D)
                rhythm_limb = rhythm_stacked[:, self.LIMB_LEADS, :, :]  # (B, 6, n_rhythm, D)
                rhythm_chest = rhythm_stacked[:, self.CHEST_LEADS, :, :]# (B, 6, n_rhythm, D)

                # Wave层级: 各组独立处理后融合
                wave_limb = self.wave_limb_transformer(wave_limb)
                wave_chest = self.wave_chest_transformer(wave_chest)
                wave_limb_agg, _ = self.wave_limb_agg(wave_limb)        # (B, n_wave, D)
                wave_chest_agg, _ = self.wave_chest_agg(wave_chest)     # (B, n_wave, D)
                wave_agg = self.wave_group_fusion(torch.cat([wave_limb_agg, wave_chest_agg], dim=-1))

                # Beat层级
                beat_limb = self.beat_limb_transformer(beat_limb)
                beat_chest = self.beat_chest_transformer(beat_chest)
                beat_limb_agg, _ = self.beat_limb_agg(beat_limb)
                beat_chest_agg, _ = self.beat_chest_agg(beat_chest)
                beat_agg = self.beat_group_fusion(torch.cat([beat_limb_agg, beat_chest_agg], dim=-1))

                # Rhythm层级
                rhythm_limb = self.rhythm_limb_transformer(rhythm_limb)
                rhythm_chest = self.rhythm_chest_transformer(rhythm_chest)
                rhythm_limb_agg, _ = self.rhythm_limb_agg(rhythm_limb)
                rhythm_chest_agg, _ = self.rhythm_chest_agg(rhythm_chest)
                rhythm_agg = self.rhythm_group_fusion(torch.cat([rhythm_limb_agg, rhythm_chest_agg], dim=-1))

            else:
                # ========== 无分组模式 ==========
                # LeadTransformer: 时序建模
                wave_stacked = self.wave_transformer(wave_stacked)    # (B, 12, n_wave, D)
                beat_stacked = self.beat_transformer(beat_stacked)    # (B, 12, n_beat, D)
                rhythm_stacked = self.rhythm_transformer(rhythm_stacked)  # (B, 12, n_rhythm, D)

                # CrossLeadAggregation: 12导联聚合
                wave_agg, _ = self.wave_cross_lead(wave_stacked)    # (B, n_wave, D)
                beat_agg, _ = self.beat_cross_lead(beat_stacked)    # (B, n_beat, D)
                rhythm_agg, _ = self.rhythm_cross_lead(rhythm_stacked)  # (B, n_rhythm, D)

            # 特征池化
            if self.pool_type == 'mean':
                wave_pooled = wave_agg.mean(dim=1)    # (B, D)
                beat_pooled = beat_agg.mean(dim=1)    # (B, D)
                rhythm_pooled = rhythm_agg.mean(dim=1)  # (B, D)
                output = (wave_pooled + beat_pooled + rhythm_pooled) / 3
            else:  # concat then project
                wave_pooled = wave_agg.mean(dim=1)
                beat_pooled = beat_agg.mean(dim=1)
                rhythm_pooled = rhythm_agg.mean(dim=1)
                output = torch.cat([wave_pooled, beat_pooled, rhythm_pooled], dim=-1)

        else:
            # concat或hierarchical模式 (暂不支持分组)
            fused_stacked = torch.cat([wave_stacked, beat_stacked, rhythm_stacked], dim=2)
            fused_stacked = self.unified_transformer(fused_stacked)
            fused_agg, _ = self.unified_cross_lead(fused_stacked)
            output = fused_agg.mean(dim=1)

        output = self.output_proj(output)
        return output

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        返回中间特征和注意力权重，用于可视化分析。

        Returns:
            Dict containing:
            - 'wave_features': 波段级特征
            - 'beat_features': 心拍级特征
            - 'rhythm_features': 节律级特征
            - 'wave_attn': 波段级attention权重
            - 'beat_attn': 心拍级attention权重
            - 'rhythm_attn': 节律级attention权重
            - 'output': 最终输出
        """
        B = x.shape[0]

        all_wave_feats = []
        all_beat_feats = []
        all_rhythm_feats = []
        all_wave_attn = []
        all_beat_attn = []
        all_rhythm_attn = []

        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]
            lead_feat = self.resnet(lead_signal)
            lead_feat = rearrange(lead_feat, 'b c n -> b n c')

            pooler_out = self.hierarchical_pooler(lead_feat)

            all_wave_feats.append(pooler_out['wave_features'])
            all_beat_feats.append(pooler_out['beat_features'])
            all_rhythm_feats.append(pooler_out['rhythm_features'])
            all_wave_attn.append(pooler_out['wave_attn'])
            all_beat_attn.append(pooler_out['beat_attn'])
            all_rhythm_attn.append(pooler_out['rhythm_attn'])

        # 堆叠
        wave_stacked = torch.stack(all_wave_feats, dim=1)
        beat_stacked = torch.stack(all_beat_feats, dim=1)
        rhythm_stacked = torch.stack(all_rhythm_feats, dim=1)
        wave_attn = torch.stack(all_wave_attn, dim=1)    # (B, 12, n_wave, 313)
        beat_attn = torch.stack(all_beat_attn, dim=1)
        rhythm_attn = torch.stack(all_rhythm_attn, dim=1)

        # 处理...
        if self.pooler_fusion == 'separate':
            wave_stacked = self.wave_transformer(wave_stacked)
            beat_stacked = self.beat_transformer(beat_stacked)
            rhythm_stacked = self.rhythm_transformer(rhythm_stacked)

            wave_agg, _ = self.wave_cross_lead(wave_stacked)
            beat_agg, _ = self.beat_cross_lead(beat_stacked)
            rhythm_agg, _ = self.rhythm_cross_lead(rhythm_stacked)

            wave_pooled = wave_agg.mean(dim=1)
            beat_pooled = beat_agg.mean(dim=1)
            rhythm_pooled = rhythm_agg.mean(dim=1)
            output = (wave_pooled + beat_pooled + rhythm_pooled) / 3
        else:
            fused_stacked = torch.cat([wave_stacked, beat_stacked, rhythm_stacked], dim=2)
            fused_stacked = self.unified_transformer(fused_stacked)
            fused_agg, _ = self.unified_cross_lead(fused_stacked)
            output = fused_agg.mean(dim=1)

        output = self.output_proj(output)

        return {
            'wave_features': wave_stacked,
            'beat_features': beat_stacked,
            'rhythm_features': rhythm_stacked,
            'wave_attn': wave_attn,
            'beat_attn': beat_attn,
            'rhythm_attn': rhythm_attn,
            'wave_pooled': wave_pooled if self.pooler_fusion == 'separate' else None,
            'beat_pooled': beat_pooled if self.pooler_fusion == 'separate' else None,
            'rhythm_pooled': rhythm_pooled if self.pooler_fusion == 'separate' else None,
            'output': output
        }

    def forward_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回三个尺度的池化特征，用于MultiScaleClipLoss。

        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            Dict containing:
            - 'wave': (B, D) 波段级池化特征
            - 'beat': (B, D) 心拍级池化特征
            - 'rhythm': (B, D) 节律级池化特征
        """
        B = x.shape[0]

        all_wave_feats = []
        all_beat_feats = []
        all_rhythm_feats = []

        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx:lead_idx+1, :]
            lead_feat = self.resnet(lead_signal)
            lead_feat = rearrange(lead_feat, 'b c n -> b n c')

            pooler_out = self.hierarchical_pooler(lead_feat)

            all_wave_feats.append(pooler_out['wave_features'])
            all_beat_feats.append(pooler_out['beat_features'])
            all_rhythm_feats.append(pooler_out['rhythm_features'])

        # 堆叠12导联
        wave_stacked = torch.stack(all_wave_feats, dim=1)
        beat_stacked = torch.stack(all_beat_feats, dim=1)
        rhythm_stacked = torch.stack(all_rhythm_feats, dim=1)

        # LeadTransformer + CrossLeadAggregation
        if self.pooler_fusion == 'separate':
            if self.lead_group_strategy == 'lisa':
                # ========== LISA分组模式 (5组) ==========
                wave_lateral = wave_stacked[:, self.LATERAL_LEADS, :, :]
                wave_inferior = wave_stacked[:, self.INFERIOR_LEADS, :, :]
                wave_septal = wave_stacked[:, self.SEPTAL_LEADS, :, :]
                wave_anterior = wave_stacked[:, self.ANTERIOR_LEADS, :, :]
                wave_avr = wave_stacked[:, self.AVR_LEADS, :, :]

                beat_lateral = beat_stacked[:, self.LATERAL_LEADS, :, :]
                beat_inferior = beat_stacked[:, self.INFERIOR_LEADS, :, :]
                beat_septal = beat_stacked[:, self.SEPTAL_LEADS, :, :]
                beat_anterior = beat_stacked[:, self.ANTERIOR_LEADS, :, :]
                beat_avr = beat_stacked[:, self.AVR_LEADS, :, :]

                rhythm_lateral = rhythm_stacked[:, self.LATERAL_LEADS, :, :]
                rhythm_inferior = rhythm_stacked[:, self.INFERIOR_LEADS, :, :]
                rhythm_septal = rhythm_stacked[:, self.SEPTAL_LEADS, :, :]
                rhythm_anterior = rhythm_stacked[:, self.ANTERIOR_LEADS, :, :]
                rhythm_avr = rhythm_stacked[:, self.AVR_LEADS, :, :]

                # Wave层级
                wave_lateral = self.wave_lateral_transformer(wave_lateral)
                wave_inferior = self.wave_inferior_transformer(wave_inferior)
                wave_septal = self.wave_septal_transformer(wave_septal)
                wave_anterior = self.wave_anterior_transformer(wave_anterior)
                wave_avr = self.wave_avr_transformer(wave_avr)

                wave_lateral_agg, _ = self.wave_lateral_agg(wave_lateral)
                wave_inferior_agg, _ = self.wave_inferior_agg(wave_inferior)
                wave_septal_agg, _ = self.wave_septal_agg(wave_septal)
                wave_anterior_agg, _ = self.wave_anterior_agg(wave_anterior)
                wave_avr_agg, _ = self.wave_avr_agg(wave_avr)

                wave_agg = self.wave_group_fusion(torch.cat([
                    wave_lateral_agg, wave_inferior_agg, wave_septal_agg,
                    wave_anterior_agg, wave_avr_agg
                ], dim=-1))

                # Beat层级
                beat_lateral = self.beat_lateral_transformer(beat_lateral)
                beat_inferior = self.beat_inferior_transformer(beat_inferior)
                beat_septal = self.beat_septal_transformer(beat_septal)
                beat_anterior = self.beat_anterior_transformer(beat_anterior)
                beat_avr = self.beat_avr_transformer(beat_avr)

                beat_lateral_agg, _ = self.beat_lateral_agg(beat_lateral)
                beat_inferior_agg, _ = self.beat_inferior_agg(beat_inferior)
                beat_septal_agg, _ = self.beat_septal_agg(beat_septal)
                beat_anterior_agg, _ = self.beat_anterior_agg(beat_anterior)
                beat_avr_agg, _ = self.beat_avr_agg(beat_avr)

                beat_agg = self.beat_group_fusion(torch.cat([
                    beat_lateral_agg, beat_inferior_agg, beat_septal_agg,
                    beat_anterior_agg, beat_avr_agg
                ], dim=-1))

                # Rhythm层级
                rhythm_lateral = self.rhythm_lateral_transformer(rhythm_lateral)
                rhythm_inferior = self.rhythm_inferior_transformer(rhythm_inferior)
                rhythm_septal = self.rhythm_septal_transformer(rhythm_septal)
                rhythm_anterior = self.rhythm_anterior_transformer(rhythm_anterior)
                rhythm_avr = self.rhythm_avr_transformer(rhythm_avr)

                rhythm_lateral_agg, _ = self.rhythm_lateral_agg(rhythm_lateral)
                rhythm_inferior_agg, _ = self.rhythm_inferior_agg(rhythm_inferior)
                rhythm_septal_agg, _ = self.rhythm_septal_agg(rhythm_septal)
                rhythm_anterior_agg, _ = self.rhythm_anterior_agg(rhythm_anterior)
                rhythm_avr_agg, _ = self.rhythm_avr_agg(rhythm_avr)

                rhythm_agg = self.rhythm_group_fusion(torch.cat([
                    rhythm_lateral_agg, rhythm_inferior_agg, rhythm_septal_agg,
                    rhythm_anterior_agg, rhythm_avr_agg
                ], dim=-1))

            elif self.lead_group_strategy == 'limb_chest':
                # ========== 肢体/胸导联分组模式 (2组) ==========
                wave_limb = wave_stacked[:, self.LIMB_LEADS, :, :]
                wave_chest = wave_stacked[:, self.CHEST_LEADS, :, :]
                beat_limb = beat_stacked[:, self.LIMB_LEADS, :, :]
                beat_chest = beat_stacked[:, self.CHEST_LEADS, :, :]
                rhythm_limb = rhythm_stacked[:, self.LIMB_LEADS, :, :]
                rhythm_chest = rhythm_stacked[:, self.CHEST_LEADS, :, :]

                # Wave层级
                wave_limb = self.wave_limb_transformer(wave_limb)
                wave_chest = self.wave_chest_transformer(wave_chest)
                wave_limb_agg, _ = self.wave_limb_agg(wave_limb)
                wave_chest_agg, _ = self.wave_chest_agg(wave_chest)
                wave_agg = self.wave_group_fusion(torch.cat([wave_limb_agg, wave_chest_agg], dim=-1))

                # Beat层级
                beat_limb = self.beat_limb_transformer(beat_limb)
                beat_chest = self.beat_chest_transformer(beat_chest)
                beat_limb_agg, _ = self.beat_limb_agg(beat_limb)
                beat_chest_agg, _ = self.beat_chest_agg(beat_chest)
                beat_agg = self.beat_group_fusion(torch.cat([beat_limb_agg, beat_chest_agg], dim=-1))

                # Rhythm层级
                rhythm_limb = self.rhythm_limb_transformer(rhythm_limb)
                rhythm_chest = self.rhythm_chest_transformer(rhythm_chest)
                rhythm_limb_agg, _ = self.rhythm_limb_agg(rhythm_limb)
                rhythm_chest_agg, _ = self.rhythm_chest_agg(rhythm_chest)
                rhythm_agg = self.rhythm_group_fusion(torch.cat([rhythm_limb_agg, rhythm_chest_agg], dim=-1))

            else:
                # ========== 无分组模式 ==========
                wave_stacked = self.wave_transformer(wave_stacked)
                beat_stacked = self.beat_transformer(beat_stacked)
                rhythm_stacked = self.rhythm_transformer(rhythm_stacked)

                wave_agg, _ = self.wave_cross_lead(wave_stacked)
                beat_agg, _ = self.beat_cross_lead(beat_stacked)
                rhythm_agg, _ = self.rhythm_cross_lead(rhythm_stacked)

            # 池化: (B, N, D) -> (B, D)
            wave_pooled = wave_agg.mean(dim=1)
            beat_pooled = beat_agg.mean(dim=1)
            rhythm_pooled = rhythm_agg.mean(dim=1)
        else:
            # 非separate模式暂不支持
            raise NotImplementedError("forward_multiscale only supports 'separate' fusion mode")

        return {
            'wave': wave_pooled,
            'beat': beat_pooled,
            'rhythm': rhythm_pooled
        }

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, num_leads={self.num_leads}, '
            f'queries=(wave={self.n_wave_queries}, beat={self.n_beat_queries}, '
            f'rhythm={self.n_rhythm_queries}), fusion={self.pooler_fusion}, '
            f'lead_group_strategy={self.lead_group_strategy}'
        )


# ============================================================================
# 预定义模型配置
# ============================================================================

def mvcse_mssate_tiny(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    Tiny版本：适合快速实验

    配置:
    - embed_dim: 128
    - lead_transformer_depth: 1
    - mssate_depth: 1
    - 预计参数量: ~6M
    """
    return MVCSEMSSATEEncoder(
        embed_dim=128,
        seq_len=seq_len,
        lead_transformer_depth=1,
        lead_transformer_heads=4,
        lead_transformer_dim_head=32,
        lead_transformer_mlp_ratio=2.,
        mssate_depth=1,
        mssate_num_heads=4,
        mssate_dim_head=32,
        mssate_mlp_ratio=2.,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_small(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    Small版本：平衡性能和效率

    配置:
    - embed_dim: 192
    - lead_transformer_depth: 2
    - mssate_depth: 2
    - 预计参数量: ~15M
    """
    return MVCSEMSSATEEncoder(
        embed_dim=192,
        seq_len=seq_len,
        lead_transformer_depth=2,
        lead_transformer_heads=4,
        lead_transformer_dim_head=48,
        lead_transformer_mlp_ratio=3.,
        mssate_depth=2,
        mssate_num_heads=6,
        mssate_dim_head=32,
        mssate_mlp_ratio=3.,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_base(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    Base版本：标准配置

    配置:
    - embed_dim: 256
    - lead_transformer_depth: 2
    - mssate_depth: 2
    - 预计参数量: ~25M
    """
    return MVCSEMSSATEEncoder(
        embed_dim=256,
        seq_len=seq_len,
        lead_transformer_depth=2,
        lead_transformer_heads=4,
        lead_transformer_dim_head=64,
        lead_transformer_mlp_ratio=4.,
        mssate_depth=2,
        mssate_num_heads=8,
        mssate_dim_head=32,
        mssate_mlp_ratio=4.,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_large(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    Large版本：更大容量

    配置:
    - embed_dim: 384
    - lead_transformer_depth: 3
    - mssate_depth: 2
    - 预计参数量: ~45M
    """
    return MVCSEMSSATEEncoder(
        embed_dim=384,
        seq_len=seq_len,
        lead_transformer_depth=3,
        lead_transformer_heads=6,
        lead_transformer_dim_head=64,
        lead_transformer_mlp_ratio=4.,
        mssate_depth=2,
        mssate_num_heads=12,
        mssate_dim_head=32,
        mssate_mlp_ratio=4.,
        output_dim=output_dim,
        **kwargs
    )


# ============================================================================
# Hierarchical版本配置 (使用可学习Query)
# ============================================================================

def hierarchical_mvcse_mssate_base(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    层级版Base：使用可学习Query替代硬切分

    特点:
    - wave/beat/rhythm三级特征建模
    - 可学习的attention自动决定关注位置
    - 无需手工设计patch大小

    配置:
    - embed_dim: 256
    - n_wave_queries: 30 (波段级，对应P/QRS/T)
    - n_beat_queries: 10 (心拍级，对应完整心跳)
    - n_rhythm_queries: 3 (节律级，对应整体节律)
    """
    return HierarchicalMVCSEMSSATEEncoder(
        embed_dim=256,
        seq_len=seq_len,
        n_wave_queries=30,
        n_beat_queries=10,
        n_rhythm_queries=3,
        pooler_heads=8,
        pooler_fusion='separate',
        lead_transformer_depth=2,
        lead_transformer_heads=4,
        lead_transformer_dim_head=64,
        lead_transformer_mlp_ratio=4.,
        output_dim=output_dim,
        **kwargs
    )


def hierarchical_mvcse_mssate_small(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    层级版Small：轻量级配置

    配置:
    - embed_dim: 192
    - n_wave_queries: 20
    - n_beat_queries: 8
    - n_rhythm_queries: 3
    """
    return HierarchicalMVCSEMSSATEEncoder(
        embed_dim=192,
        seq_len=seq_len,
        n_wave_queries=20,
        n_beat_queries=8,
        n_rhythm_queries=3,
        pooler_heads=6,
        pooler_fusion='separate',
        lead_transformer_depth=2,
        lead_transformer_heads=4,
        lead_transformer_dim_head=48,
        lead_transformer_mlp_ratio=3.,
        output_dim=output_dim,
        **kwargs
    )


def hierarchical_mvcse_mssate_large(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """
    层级版Large：更大容量

    配置:
    - embed_dim: 384
    - n_wave_queries: 40
    - n_beat_queries: 12
    - n_rhythm_queries: 4
    """
    return HierarchicalMVCSEMSSATEEncoder(
        embed_dim=384,
        seq_len=seq_len,
        n_wave_queries=40,
        n_beat_queries=12,
        n_rhythm_queries=4,
        pooler_heads=12,
        pooler_fusion='separate',
        lead_transformer_depth=3,
        lead_transformer_heads=6,
        lead_transformer_dim_head=64,
        lead_transformer_mlp_ratio=4.,
        output_dim=output_dim,
        **kwargs
    )
