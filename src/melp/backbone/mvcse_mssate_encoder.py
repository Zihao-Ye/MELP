"""
Multi-View Cardiac Spatial Encoding (MVCSE) + Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE)

基于方案3.1和3.2实现的ECG编码器:
- MVCSE: 基于LISA解剖分组原则，将12导联分为4组，引入空间先验
- MS-SATE: 多尺度Patch Embedding + 相对位置编码，捕捉从波形级到节律级的多尺度特征

Reference:
- 方案.pdf 3.1节 Multi-View Cardiac Spatial Encoding (MVCSE)
- 方案.pdf 3.2节 Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List, Dict
import math


# ============================================================================
# 基础模块
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class FeedForward(nn.Module):
    """MLP Module with GELU activation."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
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


# ============================================================================
# SE-Block & ECA-Block for Inter-group Channel Attention
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for inter-group channel attention.
    用于学习导联组间的协同效应。

    注意：此模块专门用于处理 (B, N, C) 格式的输入，
    其中 N 是序列长度/token数，C 是通道数。
    """
    def __init__(self, channels: int, reduction: int = 4, input_format: str = 'BNC'):
        super().__init__()
        self.channels = channels
        self.input_format = input_format
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - batch, sequence, channels

        Returns:
            Attention-weighted x: (B, N, C)
        """
        if self.input_format == 'BNC':
            # (B, N, C) format - squeeze over N dimension (sequence)
            b, n, c = x.shape
            y = x.mean(dim=1)  # (B, C) - global average pooling over sequence
            y = self.fc(y)  # (B, C) - channel attention weights
            return x * y.unsqueeze(1)  # (B, N, C) * (B, 1, C) -> (B, N, C)
        else:
            # (B, C, L) format - squeeze over L dimension
            b, c, l = x.shape
            y = x.mean(dim=-1)  # (B, C)
            y = self.fc(y)  # (B, C)
            return x * y.unsqueeze(-1)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention Block.
    使用1D卷积代替全连接层，参数更少。

    注意：此模块专门用于处理 (B, N, C) 格式的输入。
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1, input_format: str = 'BNC'):
        super().__init__()
        self.channels = channels
        self.input_format = input_format

        # 自适应计算kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - batch, sequence, channels

        Returns:
            Attention-weighted x: (B, N, C)
        """
        if self.input_format == 'BNC':
            # (B, N, C) format
            b, n, c = x.shape
            # Global average pooling over sequence dimension
            y = x.mean(dim=1)  # (B, C)
            # Reshape for conv1d: (B, 1, C)
            y = y.unsqueeze(1)
            # Apply 1D conv for channel attention
            y = self.conv(y)  # (B, 1, C)
            y = self.sigmoid(y)  # (B, 1, C)
            return x * y  # (B, N, C) * (B, 1, C) -> (B, N, C)
        else:
            # (B, C, L) format
            b, c, l = x.shape
            y = x.mean(dim=-1, keepdim=True)  # (B, C, 1)
            y = y.transpose(1, 2)  # (B, 1, C)
            y = self.conv(y)  # (B, 1, C)
            y = self.sigmoid(y).transpose(1, 2)  # (B, C, 1)
            return x * y


# ============================================================================
# Relative Positional Encoding for Shift-Adaptive
# ============================================================================

class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码，增强模型对心跳在时间轴上平移的鲁棒性。
    """
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # 可学习的相对位置嵌入
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_len - 1, dim)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 注册相对位置索引
        coords = torch.arange(max_len)
        relative_coords = coords[:, None] - coords[None, :]  # (L, L)
        relative_coords += max_len - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns:
            relative_position_bias: (seq_len, seq_len, dim)
        """
        relative_position_index = self.relative_position_index[:seq_len, :seq_len]
        # 使用 reshape 代替 view，因为高级索引后张量可能不连续
        relative_position_bias = self.relative_position_bias_table[relative_position_index.reshape(-1)].reshape(
            seq_len, seq_len, -1
        )
        return relative_position_bias


class RelativeAttention(nn.Module):
    """
    带相对位置编码的多头自注意力。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        max_len: int = 512
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(attn_dropout)

        # 相对位置编码
        self.rel_pos = RelativePositionalEncoding(num_heads, max_len)

    def forward(self, x):
        b, n, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, N, N)

        # 添加相对位置偏置
        rel_pos_bias = self.rel_pos(n)  # (N, N, H)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        dots = dots + rel_pos_bias

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Transformer Block with relative positional encoding."""
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
            self.use_mha = True

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_relative_pos = use_relative_pos

    def forward(self, x):
        if self.use_relative_pos:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Multi-Scale Patch Embedding (MS-SATE核心组件)
# ============================================================================

class MultiScalePatchEmbedding(nn.Module):
    """
    多尺度Patch Embedding模块。

    并行使用三个不同卷积核大小和步长的卷积层：
    - Scale 1 (Short): 捕捉高频细节（QRS切迹、P波形态），对应儿童窄QRS波
    - Scale 2 (Medium): 捕捉波群特征（ST段形态）
    - Scale 3 (Long): 捕捉节律特征（RR间期），对应心率变异性
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        seq_len: int = 5000,
        patch_configs: Optional[List[Dict]] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 默认多尺度配置
        if patch_configs is None:
            patch_configs = [
                {'kernel_size': 25, 'stride': 25},   # Short: ~50ms @ 500Hz, 200 tokens
                {'kernel_size': 50, 'stride': 50},   # Medium: ~100ms @ 500Hz, 100 tokens
                {'kernel_size': 100, 'stride': 100}, # Long: ~200ms @ 500Hz, 50 tokens
            ]

        self.num_scales = len(patch_configs)
        self.patch_configs = patch_configs

        # 每个尺度的patch embedding层
        self.patch_embeds = nn.ModuleList()
        self.num_patches_list = []

        for cfg in patch_configs:
            kernel_size = cfg['kernel_size']
            stride = cfg['stride']
            num_patches = (seq_len - kernel_size) // stride + 1
            self.num_patches_list.append(num_patches)

            self.patch_embeds.append(
                nn.Conv1d(
                    in_channels,
                    embed_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False
                )
            )

        # 每个尺度的位置编码
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
            for num_patches in self.num_patches_list
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, C, L) - 输入信号

        Returns:
            List of embeddings for each scale: [(B, N1, D), (B, N2, D), (B, N3, D)]
        """
        outputs = []
        for i, patch_embed in enumerate(self.patch_embeds):
            # Patch embedding
            out = patch_embed(x)  # (B, D, N)
            out = rearrange(out, 'b d n -> b n d')  # (B, N, D)

            # 添加位置编码
            out = out + self.pos_embeds[i]
            outputs.append(out)

        return outputs


# ============================================================================
# Multi-View Cardiac Spatial Encoding (MVCSE)
# ============================================================================

class MVCSEEncoder(nn.Module):
    """
    Multi-View Cardiac Spatial Encoding (MVCSE).

    基于LISA（Lateral, Inferior, Septal, Anterior）解剖分组原则：
    - 侧壁组 (Lateral): I, aVL, V5, V6, aVR - 反映左室高侧及侧壁
    - 下壁组 (Inferior): II, III, aVF - 反映右冠状动脉供血区
    - 间隔组 (Septal): V1, V2 - 反映室间隔及右室
    - 前壁组 (Anterior): V3, V4 - 反映前壁

    标准12导联顺序: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """

    # 导联分组索引（基于标准12导联顺序）
    LEAD_GROUPS = {
        'lateral': [0, 4, 10, 11, 3],  # I, aVL, V5, V6, aVR
        'inferior': [1, 2, 5],          # II, III, aVF
        'septal': [6, 7],               # V1, V2
        'anterior': [8, 9],             # V3, V4
    }

    GROUP_NAMES = ['lateral', 'inferior', 'septal', 'anterior']

    def __init__(
        self,
        embed_dim: int = 256,
        seq_len: int = 5000,
        patch_configs: Optional[List[Dict]] = None,
        channel_attention: str = 'se',  # 'se' or 'eca'
        reduction: int = 4
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = len(self.GROUP_NAMES)

        # 每组导联的数量
        self.group_channels = {
            name: len(indices) for name, indices in self.LEAD_GROUPS.items()
        }

        # 每组导联共享权重的多尺度Patch Embedding
        self.group_patch_embeds = nn.ModuleDict()
        for name, indices in self.LEAD_GROUPS.items():
            num_leads = len(indices)
            self.group_patch_embeds[name] = MultiScalePatchEmbedding(
                in_channels=num_leads,
                embed_dim=embed_dim,
                seq_len=seq_len,
                patch_configs=patch_configs
            )

        # 获取尺度数量
        if patch_configs is None:
            self.num_scales = 3
        else:
            self.num_scales = len(patch_configs)

        # 组间通道注意力（对每个尺度独立）
        # 输入格式为 (B, N, C)，即 (batch, sequence, channels)
        if channel_attention == 'se':
            self.inter_group_attention = nn.ModuleList([
                SEBlock(self.num_groups * embed_dim, reduction, input_format='BNC')
                for _ in range(self.num_scales)
            ])
        elif channel_attention == 'eca':
            self.inter_group_attention = nn.ModuleList([
                ECABlock(self.num_groups * embed_dim, input_format='BNC')
                for _ in range(self.num_scales)
            ])
        else:
            self.inter_group_attention = nn.ModuleList([
                nn.Identity() for _ in range(self.num_scales)
            ])

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            Dict containing:
            - 'group_features': List of group features for each scale
            - 'fused_features': List of fused features after inter-group attention
        """
        batch_size = x.shape[0]

        # 按组提取特征
        group_features_by_scale = [[] for _ in range(self.num_scales)]

        for name in self.GROUP_NAMES:
            indices = self.LEAD_GROUPS[name]
            group_signal = x[:, indices, :]  # (B, num_leads_in_group, L)

            # 多尺度Patch Embedding
            scale_features = self.group_patch_embeds[name](group_signal)

            for scale_idx, feat in enumerate(scale_features):
                group_features_by_scale[scale_idx].append(feat)

        # 对每个尺度进行组间注意力融合
        fused_features = []
        for scale_idx in range(self.num_scales):
            # 获取该尺度下所有组的特征
            group_feats = group_features_by_scale[scale_idx]  # List of (B, N, D)

            # 在通道维度拼接
            # 注意：不同组可能有不同的N（因为patch_embed相同，所以N应该相同）
            concat_feat = torch.cat(group_feats, dim=-1)  # (B, N, 4*D)

            # 组间通道注意力
            fused = self.inter_group_attention[scale_idx](concat_feat)  # (B, N, 4*D)
            fused_features.append(fused)

        return {
            'group_features': group_features_by_scale,
            'fused_features': fused_features
        }


# ============================================================================
# Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE)
# ============================================================================

class MSSATEEncoder(nn.Module):
    """
    Multi-Scale Shift-Adaptive Temporal Encoding (MS-SATE).

    将三种尺度的Token输入到独立的Transformer encoder中，
    使用相对位置编码增强模型对心跳时间轴平移的鲁棒性。
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_groups: int = 4,
        num_scales: int = 3,
        depth: int = 6,
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
        self.input_dim = num_groups * embed_dim  # 4组拼接后的维度

        # 每个尺度独立的Transformer encoder
        self.scale_encoders = nn.ModuleList()

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

        Returns:
            List of encoded features for each scale: [(B, N1, 4*D), (B, N2, 4*D), (B, N3, 4*D)]
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


# ============================================================================
# 完整的MVCSE-MSSATE ECG编码器
# ============================================================================

class MVCSEMSSATEEncoder(nn.Module):
    """
    完整的ECG编码器，整合MVCSE和MS-SATE模块。

    架构:
    1. MVCSE: 基于解剖分组的空间编码
       - 将12导联分为4组（侧壁、下壁、间隔、前壁）
       - 每组独立进行多尺度Patch Embedding
       - 组间通道注意力学习协同效应

    2. MS-SATE: 多尺度时序编码
       - 三个尺度独立的Transformer encoder
       - 相对位置编码增强平移鲁棒性

    3. 特征聚合: 将多尺度特征融合为最终表示
    """
    def __init__(
        self,
        # MVCSE参数
        embed_dim: int = 256,
        seq_len: int = 5000,
        patch_configs: Optional[List[Dict]] = None,
        channel_attention: str = 'se',
        reduction: int = 4,
        # MS-SATE参数
        depth: int = 6,
        num_heads: int = 8,
        dim_head: int = 32,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
        attn_dropout: float = 0.,
        drop_path: float = 0.1,
        max_len: int = 512,
        use_relative_pos: bool = True,
        # 输出参数
        output_dim: Optional[int] = None,
        pool_type: str = 'mean'  # 'mean', 'cls', 'concat_mean'
    ):
        super().__init__()

        # 默认patch配置
        if patch_configs is None:
            patch_configs = [
                {'kernel_size': 25, 'stride': 25},   # Short: 200 tokens
                {'kernel_size': 50, 'stride': 50},   # Medium: 100 tokens
                {'kernel_size': 100, 'stride': 100}, # Long: 50 tokens
            ]

        self.num_scales = len(patch_configs)
        self.num_groups = 4
        self.embed_dim = embed_dim
        self.fused_dim = self.num_groups * embed_dim  # 4 * 256 = 1024
        self.pool_type = pool_type

        # MVCSE模块
        self.mvcse = MVCSEEncoder(
            embed_dim=embed_dim,
            seq_len=seq_len,
            patch_configs=patch_configs,
            channel_attention=channel_attention,
            reduction=reduction
        )

        # MS-SATE模块
        self.mssate = MSSATEEncoder(
            embed_dim=embed_dim,
            num_groups=self.num_groups,
            num_scales=self.num_scales,
            depth=depth,
            num_heads=num_heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
            max_len=max_len,
            use_relative_pos=use_relative_pos
        )

        # 多尺度特征融合
        if pool_type == 'concat_mean':
            fusion_input_dim = self.fused_dim * self.num_scales  # 3 * 1024 = 3072
        else:
            fusion_input_dim = self.fused_dim  # 1024

        # 最终输出投影
        self.output_dim = output_dim if output_dim is not None else fusion_input_dim

        if self.output_dim != fusion_input_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(fusion_input_dim, self.output_dim),
                nn.LayerNorm(self.output_dim)
            )
        else:
            self.output_proj = nn.Identity()

        # CLS token用于分类（如果使用cls pooling）
        if pool_type == 'cls':
            self.cls_token = nn.ParameterList([
                nn.Parameter(torch.randn(1, 1, self.fused_dim) * 0.02)
                for _ in range(self.num_scales)
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
        fused_features = mvcse_output['fused_features']

        # 如果使用CLS pooling，添加CLS token
        if self.pool_type == 'cls':
            batch_size = x.shape[0]
            for i in range(self.num_scales):
                cls_tokens = self.cls_token[i].expand(batch_size, -1, -1)
                fused_features[i] = torch.cat([cls_tokens, fused_features[i]], dim=1)

        # MS-SATE: 时序编码
        encoded_features = self.mssate(fused_features)

        # 特征聚合
        if self.pool_type == 'mean':
            # 对每个尺度取平均，然后对尺度取平均
            pooled = [feat.mean(dim=1) for feat in encoded_features]  # List of (B, fused_dim)
            output = torch.stack(pooled, dim=0).mean(dim=0)  # (B, fused_dim)

        elif self.pool_type == 'cls':
            # 取每个尺度的CLS token，然后平均
            pooled = [feat[:, 0, :] for feat in encoded_features]  # List of (B, fused_dim)
            output = torch.stack(pooled, dim=0).mean(dim=0)  # (B, fused_dim)

        elif self.pool_type == 'concat_mean':
            # 每个尺度取平均后拼接
            pooled = [feat.mean(dim=1) for feat in encoded_features]  # List of (B, fused_dim)
            output = torch.cat(pooled, dim=-1)  # (B, num_scales * fused_dim)

        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # 输出投影
        output = self.output_proj(output)

        return output

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回中间特征，用于可视化和分析。

        Args:
            x: (B, 12, L) - 12导联ECG信号

        Returns:
            Dict containing various intermediate features
        """
        # MVCSE: 空间编码
        mvcse_output = self.mvcse(x)

        # MS-SATE: 时序编码
        encoded_features = self.mssate(mvcse_output['fused_features'])

        # 聚合特征
        pooled = [feat.mean(dim=1) for feat in encoded_features]
        final_output = torch.stack(pooled, dim=0).mean(dim=0)
        final_output = self.output_proj(final_output)

        return {
            'group_features': mvcse_output['group_features'],
            'fused_features': mvcse_output['fused_features'],
            'encoded_features': encoded_features,
            'pooled_features': pooled,
            'output': final_output
        }


# ============================================================================
# 预定义模型配置
# ============================================================================

def mvcse_mssate_tiny(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """Tiny版本：适合快速实验"""
    return MVCSEMSSATEEncoder(
        embed_dim=128,
        seq_len=seq_len,
        depth=4,
        num_heads=4,
        dim_head=32,
        mlp_ratio=2.,
        dropout=0.1,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_small(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """Small版本：平衡性能和效率"""
    return MVCSEMSSATEEncoder(
        embed_dim=192,
        seq_len=seq_len,
        depth=6,
        num_heads=6,
        dim_head=32,
        mlp_ratio=3.,
        dropout=0.1,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_base(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """Base版本：标准配置"""
    return MVCSEMSSATEEncoder(
        embed_dim=256,
        seq_len=seq_len,
        depth=6,
        num_heads=8,
        dim_head=32,
        mlp_ratio=4.,
        dropout=0.1,
        output_dim=output_dim,
        **kwargs
    )


def mvcse_mssate_large(seq_len: int = 5000, output_dim: int = 256, **kwargs):
    """Large版本：更大容量"""
    return MVCSEMSSATEEncoder(
        embed_dim=384,
        seq_len=seq_len,
        depth=8,
        num_heads=12,
        dim_head=64,
        mlp_ratio=4.,
        dropout=0.1,
        output_dim=output_dim,
        **kwargs
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    # 测试模型
    print("Testing MVCSE-MSSATE Encoder...")

    # 创建模型
    model = mvcse_mssate_base(seq_len=5000, output_dim=256)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 12, 5000)  # (B, 12导联, 采样点)

    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # 测试详细特征输出
        features = model.forward_features(x)
        print("\nIntermediate features:")
        for k, v in features.items():
            if isinstance(v, list):
                print(f"  {k}: List of {len(v)} tensors")
                for i, t in enumerate(v):
                    if isinstance(t, list):
                        print(f"    Scale {i}: {len(t)} groups")
                    else:
                        print(f"    Scale {i}: {t.shape}")
            else:
                print(f"  {k}: {v.shape}")

    print("\nTest passed!")
