"""
多尺度诊断生成器

基于ECG的三尺度特征（wave/beat/rhythm）生成诊断文本。

设计特点:
1. 尺度自适应Cross-Attention: 对不同尺度分别做Cross-Attention，然后门控融合
2. 动态尺度选择: 生成不同内容时自动关注不同尺度
   - 生成形态描述（如"T波倒置"）时关注wave尺度
   - 生成传导描述（如"一度房室阻滞"）时关注beat尺度
   - 生成节律描述（如"房颤"）时关注rhythm尺度
3. 可视化: 记录每个生成位置的尺度注意力权重，便于分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class ScaleAdaptiveCrossAttention(nn.Module):
    """
    尺度自适应Cross-Attention

    对每个尺度分别做Cross-Attention，然后用门控机制动态融合。
    生成不同类型的诊断描述时，模型会自动学习关注不同的尺度。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_scale_gate: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_scale_gate = use_scale_gate

        # 三个尺度的Cross-Attention (共享Q投影，各尺度独立KV投影)
        self.q_proj = nn.Linear(d_model, d_model)

        # Wave尺度的KV投影
        self.wave_k_proj = nn.Linear(d_model, d_model)
        self.wave_v_proj = nn.Linear(d_model, d_model)

        # Beat尺度的KV投影
        self.beat_k_proj = nn.Linear(d_model, d_model)
        self.beat_v_proj = nn.Linear(d_model, d_model)

        # Rhythm尺度的KV投影
        self.rhythm_k_proj = nn.Linear(d_model, d_model)
        self.rhythm_v_proj = nn.Linear(d_model, d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # 尺度门控网络
        if use_scale_gate:
            self.scale_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 3),  # 3个尺度
            )

        self.dropout = nn.Dropout(dropout)
        self.scale = (d_model // n_heads) ** -0.5

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """标准的scaled dot-product attention"""
        B, L, D = q.shape
        H = self.n_heads
        head_dim = D // H

        # 重塑为多头
        q = q.view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, head_dim)
        k = k.view(B, -1, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)
        v = v.view(B, -1, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, S)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        wave_seq: torch.Tensor,
        beat_seq: torch.Tensor,
        rhythm_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (B, L, D) 当前解码器hidden states
            wave_seq: (B, N_wave, D) wave尺度序列特征
            beat_seq: (B, N_beat, D) beat尺度序列特征
            rhythm_seq: (B, N_rhythm, D) rhythm尺度序列特征

        Returns:
            output: (B, L, D) 融合后的输出
            scale_weights: (B, L, 3) 各位置的尺度权重 (如果use_scale_gate=True)
        """
        B, L, D = hidden_states.shape

        # 共享的Query
        q = self.q_proj(hidden_states)

        # 各尺度的Cross-Attention
        # Wave
        wave_k = self.wave_k_proj(wave_seq)
        wave_v = self.wave_v_proj(wave_seq)
        wave_out = self._attention(q, wave_k, wave_v)  # (B, L, D)

        # Beat
        beat_k = self.beat_k_proj(beat_seq)
        beat_v = self.beat_v_proj(beat_seq)
        beat_out = self._attention(q, beat_k, beat_v)  # (B, L, D)

        # Rhythm
        rhythm_k = self.rhythm_k_proj(rhythm_seq)
        rhythm_v = self.rhythm_v_proj(rhythm_seq)
        rhythm_out = self._attention(q, rhythm_k, rhythm_v)  # (B, L, D)

        # 尺度门控融合
        if self.use_scale_gate:
            # 基于当前hidden state计算尺度权重
            scale_logits = self.scale_gate(hidden_states)  # (B, L, 3)
            scale_weights = F.softmax(scale_logits, dim=-1)  # (B, L, 3)

            # 加权融合
            output = (
                scale_weights[:, :, 0:1] * wave_out +
                scale_weights[:, :, 1:2] * beat_out +
                scale_weights[:, :, 2:3] * rhythm_out
            )
        else:
            # 简单平均
            output = (wave_out + beat_out + rhythm_out) / 3
            scale_weights = None

        output = self.out_proj(output)

        return output, scale_weights


class DiagnosisDecoderLayer(nn.Module):
    """
    诊断解码器层

    结构:
    1. Masked Self-Attention (因果)
    2. Scale-Adaptive Cross-Attention
    3. FFN
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_scale_gate: bool = True
    ):
        super().__init__()

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d_model)

        # Scale-Adaptive Cross-Attention
        self.cross_attn = ScaleAdaptiveCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_scale_gate=use_scale_gate
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # FFN
        mlp_hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        wave_seq: torch.Tensor,
        beat_seq: torch.Tensor,
        rhythm_seq: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D) 输入hidden states
            wave_seq, beat_seq, rhythm_seq: ECG尺度特征
            causal_mask: (L, L) 因果attention mask

        Returns:
            x: (B, L, D) 输出hidden states
            scale_weights: (B, L, 3) 尺度权重
        """
        # 1. Self-Attention (Pre-Norm)
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + self.dropout(x)

        # 2. Cross-Attention (Pre-Norm)
        residual = x
        x = self.cross_attn_norm(x)
        cross_out, scale_weights = self.cross_attn(x, wave_seq, beat_seq, rhythm_seq)
        x = residual + self.dropout(cross_out)

        # 3. FFN (Pre-Norm)
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x, scale_weights


class MultiScaleDiagnosisGenerator(nn.Module):
    """
    多尺度诊断生成器

    基于ECG的三尺度序列特征（wave/beat/rhythm）自回归生成诊断文本。
    使用尺度自适应Cross-Attention动态选择关注不同尺度。
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
        d_model: int = 768,
        ecg_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_scale_gate: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 101,
        eos_token_id: int = 102
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # ECG特征投影 (从ecg_dim投影到d_model)
        self.wave_proj = nn.Sequential(
            nn.Linear(ecg_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.beat_proj = nn.Sequential(
            nn.Linear(ecg_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(ecg_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 尺度位置编码 (用于区分三个尺度)
        self.scale_pos_embedding = nn.Embedding(3, d_model)

        # Decoder Layers
        self.layers = nn.ModuleList([
            DiagnosisDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_scale_gate=use_scale_gate
            )
            for _ in range(n_layers)
        ])

        # 输出层
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # 因果mask
        self.register_buffer(
            'causal_mask',
            self._build_causal_mask(max_seq_len)
        )

        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.scale_pos_embedding.weight, std=0.02)

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        """构建因果attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _prepare_ecg_features(
        self,
        wave_seq: torch.Tensor,
        beat_seq: torch.Tensor,
        rhythm_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        准备ECG特征，投影到decoder维度并添加尺度位置编码

        Args:
            wave_seq: (B, N_wave, ecg_dim)
            beat_seq: (B, N_beat, ecg_dim)
            rhythm_seq: (B, N_rhythm, ecg_dim)

        Returns:
            投影后的特征，每个 (B, N, d_model)
        """
        device = wave_seq.device

        # 投影
        wave_feat = self.wave_proj(wave_seq)    # (B, N_wave, d_model)
        beat_feat = self.beat_proj(beat_seq)    # (B, N_beat, d_model)
        rhythm_feat = self.rhythm_proj(rhythm_seq)  # (B, N_rhythm, d_model)

        # 添加尺度位置编码
        wave_scale_emb = self.scale_pos_embedding(
            torch.zeros(wave_feat.shape[1], dtype=torch.long, device=device)
        )
        beat_scale_emb = self.scale_pos_embedding(
            torch.ones(beat_feat.shape[1], dtype=torch.long, device=device)
        )
        rhythm_scale_emb = self.scale_pos_embedding(
            torch.full((rhythm_feat.shape[1],), 2, dtype=torch.long, device=device)
        )

        wave_feat = wave_feat + wave_scale_emb.unsqueeze(0)
        beat_feat = beat_feat + beat_scale_emb.unsqueeze(0)
        rhythm_feat = rhythm_feat + rhythm_scale_emb.unsqueeze(0)

        return wave_feat, beat_feat, rhythm_feat

    def forward(
        self,
        input_ids: torch.Tensor,
        wave_seq: torch.Tensor,
        beat_seq: torch.Tensor,
        rhythm_seq: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播

        Args:
            input_ids: (B, L) 输入token IDs
            wave_seq: (B, N_wave, ecg_dim) wave序列特征
            beat_seq: (B, N_beat, ecg_dim) beat序列特征
            rhythm_seq: (B, N_rhythm, ecg_dim) rhythm序列特征
            labels: (B, L) 目标token IDs (可选，用于计算loss)

        Returns:
            Dict containing:
            - logits: (B, L, vocab_size)
            - loss: scalar (如果提供labels)
            - scale_weights: (B, L, 3) 平均尺度权重
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Token + Position Embedding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # 准备ECG特征
        wave_feat, beat_feat, rhythm_feat = self._prepare_ecg_features(
            wave_seq, beat_seq, rhythm_seq
        )

        # 因果mask
        causal_mask = self.causal_mask[:L, :L]

        # 通过Decoder层
        all_scale_weights = []
        for layer in self.layers:
            x, scale_weights = layer(
                x, wave_feat, beat_feat, rhythm_feat,
                causal_mask=causal_mask
            )
            if scale_weights is not None:
                all_scale_weights.append(scale_weights)

        # 输出
        x = self.ln_final(x)
        logits = self.output_proj(x)

        result = {'logits': logits}

        # 平均尺度权重
        if all_scale_weights:
            avg_scale_weights = torch.stack(all_scale_weights, dim=0).mean(dim=0)
            result['scale_weights'] = avg_scale_weights

        # 计算loss
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
            result['loss'] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        wave_seq: torch.Tensor,
        beat_seq: torch.Tensor,
        rhythm_seq: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        自回归生成诊断文本

        Args:
            wave_seq, beat_seq, rhythm_seq: ECG尺度特征
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Top-p (nucleus) 采样
            do_sample: 是否采样（False则用greedy）

        Returns:
            generated_ids: (B, L) 生成的token IDs
            scale_weights_history: (B, L, 3) 每个位置的尺度权重
        """
        B = wave_seq.shape[0]
        device = wave_seq.device

        # 准备ECG特征
        wave_feat, beat_feat, rhythm_feat = self._prepare_ecg_features(
            wave_seq, beat_seq, rhythm_seq
        )

        # 初始化: [BOS]
        generated = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        scale_weights_history = []

        for _ in range(max_length - 1):
            L = generated.shape[1]

            # Embedding
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            x = self.token_embedding(generated) + self.position_embedding(positions)

            # 因果mask
            causal_mask = self.causal_mask[:L, :L]

            # 通过Decoder
            for layer in self.layers:
                x, scale_weights = layer(
                    x, wave_feat, beat_feat, rhythm_feat,
                    causal_mask=causal_mask
                )

            # 获取最后一个位置的logits
            x = self.ln_final(x)
            logits = self.output_proj(x[:, -1, :])  # (B, vocab_size)

            # 采样
            if do_sample:
                logits = logits / temperature

                # Top-k
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_vals[:, -1:]] = float('-inf')

                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # 记录尺度权重
            if scale_weights is not None:
                scale_weights_history.append(scale_weights[:, -1, :])

            # 检查是否全部生成EOS
            if (next_token == self.eos_token_id).all():
                break

        # 整理尺度权重历史
        if scale_weights_history:
            scale_weights_history = torch.stack(scale_weights_history, dim=1)
        else:
            scale_weights_history = None

        return generated, scale_weights_history
