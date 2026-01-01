"""
注意力模块

包含:
- RelativeAttention: 带相对位置编码的多头自注意力
- CrossLeadAggregation: 跨导联注意力聚合（使用CLS token）
- AttentionalPooler: 可学习query的soft pooling（类似MELP）
- HierarchicalECGPooler: 层级ECG特征池化（波段/心拍/节律）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, List, Dict, Any

from .positional_encoding import RelativePositionalEncoding
from .base_modules import DropPath, FeedForward


class RelativeAttention(nn.Module):
    """
    带相对位置编码的多头自注意力。

    将相对位置偏置加到注意力分数上，增强模型对时序平移的鲁棒性。

    计算流程:
    1. Q, K, V 线性投影
    2. Attention scores: Q @ K^T / sqrt(d)
    3. 加上相对位置偏置
    4. Softmax -> Dropout -> @ V
    5. 输出投影
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - batch, sequence, dimension

        Returns:
            (B, N, D)
        """
        b, n, _ = x.shape

        # QKV投影
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, N, N)

        # 添加相对位置偏置
        rel_pos_bias = self.rel_pos(n)  # (N, N, H)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
        dots = dots + rel_pos_bias

        # Softmax和Dropout
        attn = F.softmax(dots, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossLeadAggregation(nn.Module):
    """
    跨导联注意力聚合模块。

    使用CLS token聚合组内所有导联的信息。
    导联间通过Self-Attention互相交互，CLS token收集聚合信息。

    特点:
    1. CLS token作为"代理"收集所有导联信息
    2. 导联间充分交互，可学习协同关系
    3. 对每个时间步独立执行，保留时序信息

    处理流程:
    输入: (B, num_leads, N, D)
    重排: (B*N, num_leads, D) - 每个时间步独立
    添加CLS: (B*N, 1+num_leads, D)
    Self-Attention (depth层)
    取CLS: (B*N, D)
    输出: (B, N, D)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dim_head: int = 64,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
        attn_dropout: float = 0.,
        drop_path: float = 0.,
        depth: int = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # 可学习的CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks (不使用相对位置编码，因为导联没有顺序关系)
        self.blocks = nn.ModuleList()
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]

        for i in range(depth):
            self.blocks.append(
                CrossLeadTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path_rates[i]
                )
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, num_leads, N, D)

        Returns:
            out: (B, N, D) - 聚合后的特征
            attn_weights: (B, N, num_leads) - 最后一层CLS对各导联的注意力权重
        """
        B, L, N, D = x.shape  # L = num_leads

        # 重排: 对每个时间步独立处理
        x = rearrange(x, 'b l n d -> (b n) l d')  # (B*N, L, D)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B * N, -1, -1)  # (B*N, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B*N, 1+L, D)

        # 通过Transformer blocks
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x)  # attn_weights: (B*N, H, 1+L, 1+L)

        x = self.norm(x)

        # 取CLS token作为聚合结果
        out = x[:, 0, :]  # (B*N, D)
        out = rearrange(out, '(b n) d -> b n d', b=B)  # (B, N, D)

        # 提取CLS对各导联的注意力权重（最后一层，平均所有头）
        if attn_weights is not None:
            # attn_weights: (B*N, H, 1+L, 1+L)
            # 取CLS行（第0行），去掉CLS列（第0列）
            cls_attn = attn_weights[:, :, 0, 1:]  # (B*N, H, L)
            cls_attn = cls_attn.mean(dim=1)  # (B*N, L) 平均所有头
            cls_attn = rearrange(cls_attn, '(b n) l -> b n l', b=B)  # (B, N, L)
        else:
            cls_attn = None

        return out, cls_attn


class CrossLeadTransformerBlock(nn.Module):
    """
    跨导联Transformer Block。

    不使用相对位置编码，因为导联之间没有顺序关系。
    返回注意力权重用于可视化和分析。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dim_head: int = 64,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        drop_path: float = 0.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

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

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) - L includes CLS token

        Returns:
            x: (B, L, D)
            attn_weights: (B, H, L, L)
        """
        # Self-attention with returned weights
        normed = self.norm1(x)
        attn_out, attn_weights = self._attention(normed)
        x = x + self.drop_path(attn_out)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weights

    def _attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准多头自注意力，返回注意力权重"""
        b, n, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, N, N)
        attn = F.softmax(dots, dim=-1)
        attn_weights = attn  # 保存用于返回
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn_weights


class AttentionalPooler(nn.Module):
    """
    可学习Query的Attentional Pooler (参考MELP实现)。

    与硬切分(Conv1d)不同，使用可学习的query向量通过cross-attention
    从序列中"软提取"特征，让模型自动学习该关注哪些位置。

    工作原理:
    1. 维护n_queries个可学习的query向量
    2. 输入序列作为key/value
    3. cross-attention让每个query提取它关注的信息
    4. 输出n_queries个特征向量

    与硬切分对比:
    - 硬切分: 固定窗口，按位置切分，无法适应不同心电图
    - 软切分: 可学习query，通过attention决定关注哪些位置
    """
    def __init__(
        self,
        d_model: int,
        context_dim: Optional[int] = None,
        n_head: int = 8,
        n_queries: int = 16,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: query/output的维度
            context_dim: key/value的维度，默认与d_model相同
            n_head: 注意力头数
            n_queries: 可学习query的数量（决定输出序列长度）
            dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        context_dim = context_dim or d_model

        # 可学习的query向量
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # Layer normalization
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(context_dim)

        # Cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, context_dim) - 输入序列 (e.g., ResNet输出的313个时间步)

        Returns:
            out: (B, n_queries, d_model) - 池化后的特征
            attn_weights: (B, n_queries, N) - 注意力权重，用于可视化
        """
        B = x.shape[0]

        # 准备query
        q = self.ln_q(self.query)  # (n_queries, d_model)
        q = q.unsqueeze(0).expand(B, -1, -1)  # (B, n_queries, d_model)

        # 准备key/value
        k = v = self.ln_k(x)  # (B, N, context_dim)

        # Cross-attention
        out, attn_weights = self.attn(q, k, v, average_attn_weights=True)
        # out: (B, n_queries, d_model)
        # attn_weights: (B, n_queries, N)

        # 输出投影
        out = self.out_proj(out)

        return out, attn_weights

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_queries={self.n_queries}'


class HierarchicalECGPooler(nn.Module):
    """
    层级ECG特征池化器。

    使用可学习的query在多个层级提取ECG特征:
    1. 波段级 (wave): 细粒度特征，对应P/QRS/T等波形
    2. 心拍级 (beat): 中粒度特征，对应完整心拍周期
    3. 节律级 (rhythm): 粗粒度特征，对应整体节律模式

    设计原理:
    - 313个时间步，每步约32ms (10s ECG, 500Hz采样)
    - wave: ~30 queries，每个覆盖约2-3个时间步(60-100ms)，对应单个波
    - beat: ~10 queries，每个覆盖约30个时间步(1s)，对应一个心拍
    - rhythm: ~3 queries，每个覆盖约100个时间步(3s+)，对应节律模式

    所有层级的query都是可学习的，通过attention自动学习关注哪些位置。
    """
    def __init__(
        self,
        embed_dim: int = 256,
        context_dim: Optional[int] = None,
        n_head: int = 8,
        # 各层级query数量
        n_wave_queries: int = 30,    # 波段级
        n_beat_queries: int = 10,    # 心拍级
        n_rhythm_queries: int = 3,   # 节律级
        dropout: float = 0.1,
        # 层级融合方式
        fusion_type: str = 'concat'  # 'concat', 'hierarchical', 'separate'
    ):
        """
        Args:
            embed_dim: 输出特征维度
            context_dim: 输入特征维度（ResNet输出），默认512
            n_head: 注意力头数
            n_wave_queries: 波段级query数量
            n_beat_queries: 心拍级query数量
            n_rhythm_queries: 节律级query数量
            dropout: dropout率
            fusion_type: 层级特征融合方式
                - 'concat': 拼接所有层级特征
                - 'hierarchical': 层级式聚合 (wave→beat→rhythm)
                - 'separate': 分别返回各层级特征
        """
        super().__init__()
        self.embed_dim = embed_dim
        context_dim = context_dim or 512  # ResNet18输出512通道
        self.fusion_type = fusion_type

        self.n_wave_queries = n_wave_queries
        self.n_beat_queries = n_beat_queries
        self.n_rhythm_queries = n_rhythm_queries

        # 波段级Pooler (细粒度)
        self.wave_pooler = AttentionalPooler(
            d_model=embed_dim,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_wave_queries,
            dropout=dropout
        )

        # 心拍级Pooler (中粒度)
        self.beat_pooler = AttentionalPooler(
            d_model=embed_dim,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_beat_queries,
            dropout=dropout
        )

        # 节律级Pooler (粗粒度)
        self.rhythm_pooler = AttentionalPooler(
            d_model=embed_dim,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_rhythm_queries,
            dropout=dropout
        )

        # 层级融合
        if fusion_type == 'hierarchical':
            # wave → beat 聚合
            self.wave_to_beat = AttentionalPooler(
                d_model=embed_dim,
                context_dim=embed_dim,
                n_head=n_head,
                n_queries=n_beat_queries,
                dropout=dropout
            )
            # beat → rhythm 聚合
            self.beat_to_rhythm = AttentionalPooler(
                d_model=embed_dim,
                context_dim=embed_dim,
                n_head=n_head,
                n_queries=n_rhythm_queries,
                dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: (B, N, context_dim) - ResNet输出特征 (e.g., B, 313, 512)

        Returns:
            Dict containing:
            - 'wave_features': (B, n_wave, D) 波段级特征
            - 'beat_features': (B, n_beat, D) 心拍级特征
            - 'rhythm_features': (B, n_rhythm, D) 节律级特征
            - 'wave_attn': (B, n_wave, N) 波段级注意力权重
            - 'beat_attn': (B, n_beat, N) 心拍级注意力权重
            - 'rhythm_attn': (B, n_rhythm, N) 节律级注意力权重
            - 'fused_features': 融合后的特征（根据fusion_type不同）
        """
        # 并行提取三个层级的特征
        wave_feat, wave_attn = self.wave_pooler(x)      # (B, 30, D)
        beat_feat, beat_attn = self.beat_pooler(x)      # (B, 10, D)
        rhythm_feat, rhythm_attn = self.rhythm_pooler(x)  # (B, 3, D)

        result = {
            'wave_features': wave_feat,
            'beat_features': beat_feat,
            'rhythm_features': rhythm_feat,
            'wave_attn': wave_attn,
            'beat_attn': beat_attn,
            'rhythm_attn': rhythm_attn,
        }

        # 特征融合
        if self.fusion_type == 'concat':
            # 简单拼接: (B, n_wave + n_beat + n_rhythm, D)
            result['fused_features'] = torch.cat(
                [wave_feat, beat_feat, rhythm_feat], dim=1
            )

        elif self.fusion_type == 'hierarchical':
            # 层级聚合: wave → beat → rhythm
            # wave features聚合到beat level
            wave_to_beat, _ = self.wave_to_beat(wave_feat)  # (B, n_beat, D)
            beat_combined = beat_feat + wave_to_beat  # 残差连接

            # beat features聚合到rhythm level
            beat_to_rhythm, _ = self.beat_to_rhythm(beat_combined)  # (B, n_rhythm, D)
            rhythm_combined = rhythm_feat + beat_to_rhythm

            result['fused_features'] = rhythm_combined  # (B, n_rhythm, D)
            result['beat_features_combined'] = beat_combined
            result['rhythm_features_combined'] = rhythm_combined

        else:  # 'separate'
            # 分别返回，不做融合
            result['fused_features'] = [wave_feat, beat_feat, rhythm_feat]

        return result

    def get_pooled_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取单一的pooled输出向量，用于对比学习等场景。

        Args:
            x: (B, N, context_dim) - ResNet输出特征

        Returns:
            (B, embed_dim) - 最终的表示向量
        """
        result = self.forward(x)

        if self.fusion_type == 'concat':
            # 对concat后的特征取平均
            return result['fused_features'].mean(dim=1)  # (B, D)
        elif self.fusion_type == 'hierarchical':
            # 取rhythm level的均值
            return result['rhythm_features_combined'].mean(dim=1)  # (B, D)
        else:
            # separate模式，取三个层级的均值的均值
            all_mean = torch.stack([
                result['wave_features'].mean(dim=1),
                result['beat_features'].mean(dim=1),
                result['rhythm_features'].mean(dim=1)
            ], dim=0).mean(dim=0)  # (B, D)
            return all_mean

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, '
            f'queries=(wave={self.n_wave_queries}, beat={self.n_beat_queries}, '
            f'rhythm={self.n_rhythm_queries}), fusion={self.fusion_type}'
        )
