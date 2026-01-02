"""
MVCSE-MSSATE 预训练模型

基于方案3.1和3.2实现的ECG-文本多模态预训练模型:
- MVCSE: Multi-View Cardiac Spatial Encoding
- MS-SATE: Multi-Scale Shift-Adaptive Temporal Encoding
- DiagSim-Weighted Loss: 诊断语义相似度加权的对比学习损失

与MERL保持一致的优化器和损失函数设计。
"""

import torch
import ipdb
import yaml
import math
import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from sklearn.metrics import roc_auc_score

from melp.backbone.mvcse_mssate_encoder import (
    MVCSEMSSATEEncoder,
    mvcse_mssate_tiny,
    mvcse_mssate_small,
    mvcse_mssate_base,
    mvcse_mssate_large
)
from melp.backbone.mvcse_mssate import (
    HierarchicalMVCSEMSSATEEncoder,
    hierarchical_mvcse_mssate_small,
    hierarchical_mvcse_mssate_base,
    hierarchical_mvcse_mssate_large
)
from melp.models.base_pretrain_model import BasePretrainModel
from melp.utils.openclip_loss import (
    ClipLoss, SoftClipLoss, LearnableSoftClipLoss,
    MultiScaleClipLoss, MultiScaleSemanticEnhancedClipLoss,
    ComplementaryMultiScaleClipLoss, DifferentiatedMultiScaleClipLoss,
    SymmetricMultiScaleClipLoss
)
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


class ECGGuidedTextAttention(nn.Module):
    """
    ECG引导的文本注意力模块

    用ECG特征作为Query，从文本序列中提取对应的语义信息。
    核心思想：ECG特征本身携带尺度信息，用它引导文本提取更有针对性。

    - wave_feat作为Query → 自然关注文本中波形相关描述（如"ST段抬高"）
    - rhythm_feat作为Query → 自然关注文本中节律相关描述（如"窦性心律"）

    Args:
        text_dim: 文本编码器输出维度 (如BERT为768)
        ecg_dim: ECG特征维度（也是输出维度）
        num_heads: Cross-Attention的头数
    """

    def __init__(
        self,
        text_dim: int = 768,
        ecg_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.text_dim = text_dim
        self.ecg_dim = ecg_dim

        # 文本序列投影到ECG维度空间（在CrossAttn之前）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, ecg_dim),
            nn.LayerNorm(ecg_dim)
        )

        # Cross-Attention层（Q和KV维度相同，都是ecg_dim）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ecg_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(
        self,
        ecg_feats: Dict[str, torch.Tensor],
        text_seq: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ecg_feats: Dict containing 'wave', 'beat', 'rhythm', each (B, ecg_dim)
            text_seq: (B, L, text_dim) 文本序列特征
            attention_mask: (B, L) 注意力掩码，1表示有效，0表示padding

        Returns:
            Dict containing:
            - wave: (B, ecg_dim) 波形尺度文本特征
            - beat: (B, ecg_dim) 心拍尺度文本特征
            - rhythm: (B, ecg_dim) 节律尺度文本特征
        """
        # 投影文本序列到ECG维度空间
        text_seq_proj = self.text_proj(text_seq)  # (B, L, ecg_dim)

        # 用ECG特征作为Query (B, ecg_dim) -> (B, 1, ecg_dim)
        wave_q = ecg_feats['wave'].unsqueeze(1)
        beat_q = ecg_feats['beat'].unsqueeze(1)
        rhythm_q = ecg_feats['rhythm'].unsqueeze(1)

        # 处理attention_mask: True表示忽略该位置
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Cross-Attention: ECG特征作为Query去查询文本
        text_wave, _ = self.cross_attn(
            wave_q, text_seq_proj, text_seq_proj,
            key_padding_mask=key_padding_mask
        )
        text_beat, _ = self.cross_attn(
            beat_q, text_seq_proj, text_seq_proj,
            key_padding_mask=key_padding_mask
        )
        text_rhythm, _ = self.cross_attn(
            rhythm_q, text_seq_proj, text_seq_proj,
            key_padding_mask=key_padding_mask
        )

        # (B, 1, ecg_dim) -> (B, ecg_dim)
        return {
            'wave': text_wave.squeeze(1),
            'beat': text_beat.squeeze(1),
            'rhythm': text_rhythm.squeeze(1)
        }


class MVCSEMSSATEModel(BasePretrainModel):
    """
    MVCSE-MSSATE预训练模型

    特点:
    1. 使用MVCSE进行空间编码（解剖分组 + 组间注意力）
    2. 使用MS-SATE进行多尺度时序编码
    3. 支持可学习的软标签对比损失
    4. 与MERL保持一致的优化器设计
    """

    def __init__(
        self,
        # ECG编码器参数
        ecg_encoder_name: str = "mvcse_mssate_base",
        embed_dim: int = 256,
        seq_len: int = 5000,
        # 新架构参数（方案B）
        lead_transformer_depth: int = 6,  # 导联级Transformer层数
        lead_transformer_heads: int = 4,
        cross_lead_depth: int = 1,        # Cross-Lead聚合层数
        mssate_depth: int = 2,            # MS-SATE层数
        mssate_num_heads: int = 8,
        channel_attention: str = 'se',
        use_relative_pos: bool = True,
        # 文本编码器参数
        text_encoder_name: str = "ncbi/MedCPT-Query-Encoder",
        num_freeze_layers: int = 6,
        # 共享参数
        shared_emb_dim: int = 256,
        num_leads: int = 12,
        # 对比学习参数
        init_logit_scale: float = np.log(1 / 0.07),
        # 可学习相似度参数
        use_learnable_sim: bool = True,
        init_sim_alpha: float = 0.0,
        init_threshold: float = 3.0,
        init_soft_weight: float = -3.0,
        # 优化器参数
        lr: float = 2e-4,
        weight_decay: float = 0.2,
        # 验证数据集
        val_dataset_list: List = None,
        *args,
        **kwargs
    ):
        if val_dataset_list is None:
            val_dataset_list = [
                "ptbxl_super_class", "ptbxl_sub_class",
                "ptbxl_form", "ptbxl_rhythm",
                "icbeb", "chapman"
            ]

        self.num_freeze_layers = num_freeze_layers
        self.use_learnable_sim = use_learnable_sim
        self.ecg_encoder_name = ecg_encoder_name
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        # 新架构参数
        self.lead_transformer_depth = lead_transformer_depth
        self.lead_transformer_heads = lead_transformer_heads
        self.cross_lead_depth = cross_lead_depth
        self.mssate_depth = mssate_depth
        self.mssate_num_heads = mssate_num_heads
        self.channel_attention = channel_attention
        self.use_relative_pos = use_relative_pos

        super().__init__(
            ecg_encoder_name=ecg_encoder_name,
            text_encoder_name=text_encoder_name,
            shared_emb_dim=shared_emb_dim,
            num_leads=num_leads,
            lr=lr,
            weight_decay=weight_decay,
            *args,
            **kwargs
        )

        self.save_hyperparameters()
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256
        self.val_dataset_list = val_dataset_list

        # 初始化编码器
        self.init_ecg_encoder()
        self.init_text_encoder()

        # 对比学习温度参数
        lshape = []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)

        # 可学习相似度参数
        if self.use_learnable_sim:
            self.sim_alpha = nn.Parameter(torch.tensor(init_sim_alpha))
            self.learnable_threshold = nn.Parameter(torch.tensor(init_threshold))
            self.learnable_soft_weight = nn.Parameter(torch.tensor(init_soft_weight))

        # 加载prompt和标签配置
        with open(PROMPT_PATH, 'r') as f:
            self.prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.all_labels = list(self.prompt_dict.keys())

        with open(DATASET_LABELS_PATH, 'r') as f:
            self.dataset_class_names = yaml.load(f, Loader=yaml.FullLoader)

    def init_ecg_encoder(self):
        """初始化MVCSE-MSSATE ECG编码器"""
        # 判断是否使用多尺度层级encoder
        self.use_multiscale = self.ecg_encoder_name.startswith('hierarchical_')

        if self.use_multiscale:
            # ========== 多尺度层级Encoder (wave/beat/rhythm) ==========
            if self.ecg_encoder_name == 'hierarchical_mvcse_mssate_small':
                self.ecg_encoder = hierarchical_mvcse_mssate_small(
                    seq_len=self.seq_len,
                    output_dim=self.embed_dim
                )
            elif self.ecg_encoder_name == 'hierarchical_mvcse_mssate_base':
                self.ecg_encoder = hierarchical_mvcse_mssate_base(
                    seq_len=self.seq_len,
                    output_dim=self.embed_dim
                )
            elif self.ecg_encoder_name == 'hierarchical_mvcse_mssate_large':
                self.ecg_encoder = hierarchical_mvcse_mssate_large(
                    seq_len=self.seq_len,
                    output_dim=self.embed_dim
                )
            else:
                raise ValueError(f"Unknown hierarchical encoder: {self.ecg_encoder_name}")

            # 多尺度encoder的输出维度
            self.ecg_out_dim = self.ecg_encoder.embed_dim

            # 不需要单独的proj_e，投影在MultiScaleClipLoss内部做

        else:
            # ========== 原有的单尺度Encoder ==========
            if self.ecg_encoder_name == 'mvcse_mssate_tiny':
                self.ecg_encoder = mvcse_mssate_tiny(
                    seq_len=self.seq_len,
                    output_dim=self.proj_out,
                    channel_attention=self.channel_attention,
                    use_relative_pos=self.use_relative_pos
                )
            elif self.ecg_encoder_name == 'mvcse_mssate_small':
                self.ecg_encoder = mvcse_mssate_small(
                    seq_len=self.seq_len,
                    output_dim=self.proj_out,
                    channel_attention=self.channel_attention,
                    use_relative_pos=self.use_relative_pos
                )
            elif self.ecg_encoder_name == 'mvcse_mssate_base':
                self.ecg_encoder = mvcse_mssate_base(
                    seq_len=self.seq_len,
                    output_dim=self.proj_out,
                    channel_attention=self.channel_attention,
                    use_relative_pos=self.use_relative_pos
                )
            elif self.ecg_encoder_name == 'mvcse_mssate_large':
                self.ecg_encoder = mvcse_mssate_large(
                    seq_len=self.seq_len,
                    output_dim=self.proj_out,
                    channel_attention=self.channel_attention,
                    use_relative_pos=self.use_relative_pos
                )
            else:
                # 自定义配置
                self.ecg_encoder = MVCSEMSSATEEncoder(
                    embed_dim=self.embed_dim,
                    seq_len=self.seq_len,
                    lead_transformer_depth=self.lead_transformer_depth,
                    lead_transformer_heads=self.lead_transformer_heads,
                    cross_lead_depth=self.cross_lead_depth,
                    mssate_depth=self.mssate_depth,
                    mssate_num_heads=self.mssate_num_heads,
                    channel_attention=self.channel_attention,
                    use_relative_pos=self.use_relative_pos,
                    output_dim=self.proj_out
                )

            # 获取ECG编码器输出维度
            self.ecg_out_dim = self.ecg_encoder.output_dim

            # ECG 投影层 (单尺度使用)
            self.proj_e = nn.Sequential(
                nn.Linear(self.ecg_out_dim, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )

    def init_text_encoder(self):
        """初始化文本编码器（与MERL保持一致）"""
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            self.lm_model = AutoModel.from_pretrained(self.text_encoder_name)

            # 冻结前几层
            for layer_idx in range(self.num_freeze_layers):
                for param in list(self.lm_model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

            text_encoder_hidden_dim = 768

        elif self.text_encoder_name == "google/flan-t5-small":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True)
            text_encoder_hidden_dim = 512

        elif self.text_encoder_name == "google/flan-t5-base":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True)
            text_encoder_hidden_dim = 768

        else:
            raise NotImplementedError(f"Unknown text encoder: {self.text_encoder_name}")

        self.text_encoder_hidden_dim = text_encoder_hidden_dim

        if self.use_multiscale:
            # ========== ECG引导的多尺度对齐 ==========
            # 用ECG特征作为Query，从文本序列中提取对应的语义信息

            # ECG侧的投影头（从encoder输出投影到对齐空间）
            self.ecg_wave_proj = nn.Sequential(
                nn.Linear(self.ecg_out_dim, self.proj_out),
                nn.LayerNorm(self.proj_out)
            )
            self.ecg_beat_proj = nn.Sequential(
                nn.Linear(self.ecg_out_dim, self.proj_out),
                nn.LayerNorm(self.proj_out)
            )
            self.ecg_rhythm_proj = nn.Sequential(
                nn.Linear(self.ecg_out_dim, self.proj_out),
                nn.LayerNorm(self.proj_out)
            )

            # ECG引导的文本注意力模块
            self.ecg_guided_text_attn = ECGGuidedTextAttention(
                text_dim=text_encoder_hidden_dim,  # 768
                ecg_dim=self.proj_out,             # 256
                num_heads=8,
                dropout=0.1
            )

            # 简单的ClipLoss（三个尺度共用，在shared_step中分别调用）
            self.clip_loss = ClipLoss(
                local_loss=True,
                gather_with_grad=True,
                cache_labels=True,
                rank=0,
                world_size=1,
            )
        else:
            # 单尺度模式：原有投影
            self.proj_t = nn.Sequential(
                nn.Linear(text_encoder_hidden_dim, self.proj_hidden),
                nn.GELU(),
                nn.Linear(self.proj_hidden, self.proj_out),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

    def _tokenize(self, text):
        """文本分词"""
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        return tokenizer_output

    def encode_ecg(self, ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码ECG信号

        Args:
            ecg: (B, 12, L) - 12导联ECG信号

        Returns:
            多尺度模式:
            - wave: (B, D) 波段级特征
            - beat: (B, D) 心拍级特征
            - rhythm: (B, D) 节律级特征

            单尺度模式:
            - proj_ecg_emb: 投影后的ECG嵌入
            - ecg_emb: 原始嵌入
        """
        if self.use_multiscale:
            # 多尺度模式：返回三个层级的特征
            ecg_feats = self.ecg_encoder.forward_multiscale(ecg)
            return {
                'wave': ecg_feats['wave'],      # (B, D)
                'beat': ecg_feats['beat'],      # (B, D)
                'rhythm': ecg_feats['rhythm']   # (B, D)
            }
        else:
            # 单尺度模式：原有逻辑
            ecg_emb = self.ecg_encoder(ecg)  # (B, output_dim)
            proj_ecg_emb = self.proj_e(ecg_emb)  # (B, proj_out)
            proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)
            return {
                'proj_ecg_emb': proj_ecg_emb,
                'ecg_emb': ecg_emb
            }

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码文本（仅用于单尺度模式）

        注意：多尺度模式下，文本编码需要ECG特征引导，请直接使用shared_step中的流程。

        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码

        Returns:
            - proj_text_emb: 投影后的文本嵌入
            - text_emb: 原始LM嵌入
        """
        if self.use_multiscale:
            raise RuntimeError(
                "encode_text() is not supported in multiscale mode. "
                "Text encoding requires ECG features for guidance. "
                "Use shared_step() for training or get_text_emb() for inference."
            )

        # ========== 单尺度模式：原有逻辑 ==========
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            text_emb = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output

        elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            sequence_output = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

            eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")

            batch_size, _, hidden_size = sequence_output.shape
            text_emb = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

        else:
            raise NotImplementedError(f"Unknown text encoder: {self.text_encoder_name}")

        proj_text_emb = self.proj_t(text_emb)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        return {
            'proj_text_emb': proj_text_emb,
            'text_emb': text_emb
        }

    @torch.no_grad()
    def ext_ecg_emb(self, ecg: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
        """
        提取ECG嵌入（用于推理）

        Args:
            ecg: (B, 12, L)
            mode: 多尺度模式下的融合方式 ('concat', 'mean', 'rhythm')

        Returns:
            proj_ecg_emb: (B, 3*proj_out) for concat, (B, proj_out) for mean/rhythm
        """
        if self.use_multiscale:
            # 多尺度模式：获取三个层级特征并投影
            ecg_feats = self.ecg_encoder.forward_multiscale(ecg)

            # 投影 + 归一化
            wave_z = F.normalize(self.ecg_wave_proj(ecg_feats['wave']), dim=-1)
            beat_z = F.normalize(self.ecg_beat_proj(ecg_feats['beat']), dim=-1)
            rhythm_z = F.normalize(self.ecg_rhythm_proj(ecg_feats['rhythm']), dim=-1)

            if mode == 'concat':
                proj_ecg_emb = torch.cat([wave_z, beat_z, rhythm_z], dim=-1)
            elif mode == 'mean':
                proj_ecg_emb = (wave_z + beat_z + rhythm_z) / 3
            elif mode == 'rhythm':
                proj_ecg_emb = rhythm_z
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            # 单尺度模式
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)
            proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)
        return proj_ecg_emb

    @torch.no_grad()
    def get_text_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
        """
        获取文本嵌入（用于推理/zero-shot分类）

        对于多尺度模式：
        - 训练时使用ECG引导的注意力机制提取尺度相关特征
        - 推理时使用mean pooling + text_proj（因为没有ECG来引导）
        - 训练过程中text_proj已学会将文本映射到ECG空间

        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码
            mode: 融合方式 ('concat', 'mean', 'rhythm')

        Returns:
            text_emb: 多尺度模式下(B, 3*proj_out)或单尺度模式下(B, proj_out)
        """
        if self.use_multiscale:
            # 多尺度模式：使用mean pooling + text_proj
            if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
                lm_output = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_seq = lm_output.last_hidden_state  # (B, L, 768)

            elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
                text_seq = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

            else:
                raise NotImplementedError(f"Unknown text encoder: {self.text_encoder_name}")

            # Mean pooling: 对文本序列做加权平均
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(text_seq.size()).float()
            sum_embeddings = torch.sum(text_seq * attention_mask_expanded, dim=1)
            sum_mask = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask  # (B, text_dim)

            # 通过text_proj映射到ECG空间（复用ecg_guided_text_attn中的text_proj）
            text_z = self.ecg_guided_text_attn.text_proj(mean_pooled)  # (B, proj_out)
            text_z = F.normalize(text_z, dim=-1)

            # 根据mode处理（推理时三个尺度使用相同的文本表示）
            if mode == 'concat':
                # 复制三次以匹配ECG的concat格式
                text_emb = torch.cat([text_z, text_z, text_z], dim=-1)
            elif mode == 'mean':
                text_emb = text_z
            elif mode == 'rhythm':
                text_emb = text_z
            else:
                raise ValueError(f"Unknown mode: {mode}")

        else:
            # 单尺度模式
            if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
                text_emb = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output

            elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
                sequence_output = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

                eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")

                batch_size, _, hidden_size = sequence_output.shape
                text_emb = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

            else:
                raise NotImplementedError(f"Unknown text encoder: {self.text_encoder_name}")

            text_emb = self.proj_t(text_emb)

        return text_emb

    @torch.no_grad()
    def get_class_emd(self, class_name: List[str]) -> torch.Tensor:
        """获取类别嵌入（用于zero-shot分类）"""
        zeroshot_weights = []

        for texts in class_name:
            texts = texts.lower()
            texts = [texts]
            texts = self._tokenize(texts)

            class_embeddings = self.get_text_emb(
                texts['input_ids'].type_as(self.logit_scale).long(),
                texts['attention_mask'].type_as(self.logit_scale).long(),
            )

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    @torch.no_grad()
    def compute_text_similarity(self, texts: List[str]) -> torch.Tensor:
        """计算文本相似度矩阵"""
        tokenized = self._tokenize(texts)
        input_ids = tokenized['input_ids'].type_as(self.logit_scale).long()
        attention_mask = tokenized['attention_mask'].type_as(self.logit_scale).long()

        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            text_embeddings = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output

        elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            sequence_output = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

            eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
            batch_size, _, hidden_size = sequence_output.shape
            text_embeddings = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        similarity_matrix = text_embeddings @ text_embeddings.T

        return similarity_matrix

    def compute_hybrid_similarity(
        self,
        proj_text_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> tuple:
        """
        计算混合相似度矩阵

        hybrid_sim = α × learnable_sim + (1-α) × fixed_sim
        """
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()

            # Gather embeddings
            gathered_proj = [torch.zeros_like(proj_text_emb) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_proj, proj_text_emb)
            all_proj_text_emb = torch.cat(gathered_proj, dim=0)

            text_emb_contig = text_emb.contiguous()
            gathered_text = [torch.zeros_like(text_emb_contig) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_text, text_emb_contig)
            all_text_emb = torch.cat(gathered_text, dim=0)

            learnable_sim = proj_text_emb @ all_proj_text_emb.T
            fixed_emb_local = F.normalize(text_emb, dim=-1)
            fixed_emb_all = F.normalize(all_text_emb, dim=-1)
            fixed_sim = fixed_emb_local @ fixed_emb_all.T
        else:
            learnable_sim = proj_text_emb @ proj_text_emb.T
            fixed_emb = F.normalize(text_emb, dim=-1)
            fixed_sim = fixed_emb @ fixed_emb.T

        alpha = torch.sigmoid(self.sim_alpha)
        hybrid_sim = alpha * learnable_sim + (1 - alpha) * fixed_sim

        return hybrid_sim, alpha

    def shared_step(self, batch, batch_idx):
        """训练步骤"""
        # ECG编码
        ecg_output = self.encode_ecg(batch['ecg'])

        if self.use_multiscale:
            # ========== ECG引导的多尺度对比学习 ==========
            # 1. ECG侧：投影 + L2归一化
            ecg_wave = F.normalize(self.ecg_wave_proj(ecg_output['wave']), dim=-1)
            ecg_beat = F.normalize(self.ecg_beat_proj(ecg_output['beat']), dim=-1)
            ecg_rhythm = F.normalize(self.ecg_rhythm_proj(ecg_output['rhythm']), dim=-1)

            # 2. 文本序列编码
            tokenized_input = self._tokenize(batch['report'])
            input_ids = tokenized_input['input_ids'].type_as(batch['ecg']).long()
            attention_mask = tokenized_input['attention_mask'].type_as(batch['ecg']).long()

            if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
                lm_output = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_seq = lm_output.last_hidden_state  # (B, L, 768)
            else:
                text_seq = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

            # 3. 用ECG特征作为Query，从文本中提取对应语义
            ecg_feats = {
                'wave': ecg_wave,
                'beat': ecg_beat,
                'rhythm': ecg_rhythm
            }
            text_feats = self.ecg_guided_text_attn(ecg_feats, text_seq, attention_mask)

            # 4. 文本特征归一化
            text_wave = F.normalize(text_feats['wave'], dim=-1)
            text_beat = F.normalize(text_feats['beat'], dim=-1)
            text_rhythm = F.normalize(text_feats['rhythm'], dim=-1)

            # 5. 更新分布式参数
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            self.clip_loss.rank = rank
            self.clip_loss.world_size = world_size

            # 6. 三个尺度的ClipLoss相加
            logit_scale = self.logit_scale.exp()
            loss_wave = self.clip_loss(ecg_wave, text_wave, logit_scale)
            loss_beat = self.clip_loss(ecg_beat, text_beat, logit_scale)
            loss_rhythm = self.clip_loss(ecg_rhythm, text_rhythm, logit_scale)

            total_loss = loss_wave + loss_beat + loss_rhythm

            loss_dict = {
                'loss': total_loss,
                'loss_wave': loss_wave,
                'loss_beat': loss_beat,
                'loss_rhythm': loss_rhythm,
                'logit_scale': logit_scale,
            }
        else:
            # ========== 单尺度对比学习（原有逻辑）==========
            tokenized_input = self._tokenize(batch['report'])
            input_ids = tokenized_input['input_ids'].type_as(batch['ecg']).long()
            attention_mask = tokenized_input['attention_mask'].type_as(batch['ecg']).long()
            text_output = self.encode_text(input_ids, attention_mask)

            loss_fn = ClipLoss(
                local_loss=True,
                gather_with_grad=True,
                cache_labels=True,
                rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                use_horovod=False
            )

            cma_loss = loss_fn(
                ecg_output['proj_ecg_emb'],
                text_output['proj_text_emb'],
                self.logit_scale.exp()
            )

            loss_dict = {
                'loss': cma_loss,
                'cma_loss': cma_loss,
            }

        metrics_dict = {}
        return loss_dict, metrics_dict

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """限制logit_scale范围"""
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))

    def on_validation_epoch_start(self, *args, **kwargs):
        """验证开始时计算类别嵌入"""
        val_prompts = [self.prompt_dict[i] for i in self.all_labels]
        self.zeroshot_weights = self.get_class_emd(val_prompts)
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """验证步骤"""
        cur_dataset_name = self.val_dataset_list[dataloader_idx]
        class_names = self.dataset_class_names[cur_dataset_name]
        indices = [self.all_labels.index(i) for i in class_names]
        cur_zeroshot_weights = self.zeroshot_weights[:, indices]

        ecg_emb = self.ext_ecg_emb(batch['ecg'])
        ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)
        cur_logits = torch.matmul(ecg_emb, cur_zeroshot_weights)

        self.val_step_outputs.append({
            'dataloader_idx': dataloader_idx,
            'logits': cur_logits,
            'label': batch['label']
        })

    def on_validation_epoch_end(self, *args, **kwargs):
        """验证结束时计算指标"""
        logits_dict = defaultdict(list)
        labels_dict = defaultdict(list)

        for output in self.val_step_outputs:
            dataloader_idx = output['dataloader_idx']
            logits = output['logits']
            labels = output['label']
            logits_dict[dataloader_idx].append(logits)
            labels_dict[dataloader_idx].append(labels)

        dataset_aurocs = []
        for k in logits_dict.keys():
            logits = torch.cat(logits_dict[k], dim=0).float().cpu().numpy()
            labels = torch.cat(labels_dict[k], dim=0).float().cpu().numpy()

            assert logits.shape[1] == labels.shape[1], "Number of classes mismatch"

            num_labels = logits.shape[1]
            AUROCs = []
            for i in range(num_labels):
                if len(np.unique(labels[:, i])) == 1:
                    continue
                AUROCs.append(roc_auc_score(
                    labels[:, i], logits[:, i],
                    average='macro', multi_class='ovo'
                ))

            dataset_name = self.val_dataset_list[k]
            self.log(
                f'val/{dataset_name}_AUROC',
                np.mean(AUROCs),
                on_epoch=True, prog_bar=False, sync_dist=True
            )
            dataset_aurocs.append(np.mean(AUROCs))

        self.log(
            f'val/mean_AUROC',
            np.mean(dataset_aurocs),
            on_epoch=True, prog_bar=True, sync_dist=True
        )
