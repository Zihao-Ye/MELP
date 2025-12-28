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
from melp.models.base_pretrain_model import BasePretrainModel
from melp.utils.openclip_loss import ClipLoss, SoftClipLoss, LearnableSoftClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


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
        encoder_depth: int = 6,
        num_heads: int = 8,
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
        self.encoder_depth = encoder_depth
        self.num_heads = num_heads
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
        # 根据名称选择模型大小
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
                depth=self.encoder_depth,
                num_heads=self.num_heads,
                channel_attention=self.channel_attention,
                use_relative_pos=self.use_relative_pos,
                output_dim=self.proj_out
            )

        # 获取ECG编码器输出维度
        self.ecg_out_dim = self.ecg_encoder.output_dim

        # ECG 投影层 (与 MERL ViT 分支一致，与文本侧 proj_t 对称)
        self.proj_e = nn.Sequential(
            nn.Linear(self.ecg_out_dim, self.proj_hidden),
            nn.BatchNorm1d(self.proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden, self.proj_out),
            nn.BatchNorm1d(self.proj_out),
        )

        # UMA (Unimodal Alignment) 的双投影层
        self.linear1 = nn.Linear(self.ecg_out_dim, self.proj_out, bias=False)
        self.linear2 = nn.Linear(self.ecg_out_dim, self.proj_out, bias=False)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

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

        # 文本投影层
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
            Dict containing:
            - proj_ecg_emb: 投影后的ECG嵌入 (用于CMA)
            - ecg_emb1, ecg_emb2: UMA的双视图嵌入
        """
        # 通过MVCSE-MSSATE编码器
        ecg_emb = self.ecg_encoder(ecg)  # (B, output_dim)

        # 通过投影层 (与 MERL ViT 分支一致)
        proj_ecg_emb = self.proj_e(ecg_emb)  # (B, proj_out)
        proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)

        # UMA的双投影
        ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
        ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        return {
            'proj_ecg_emb': proj_ecg_emb,
            'ecg_emb1': ecg_emb1,
            'ecg_emb2': ecg_emb2,
            'ecg_emb': ecg_emb  # 原始嵌入
        }

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码文本

        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码

        Returns:
            Dict containing:
            - proj_text_emb: 投影后的文本嵌入
            - text_emb: 原始LM嵌入
        """
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

        proj_text_emb = self.proj_t(text_emb)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        return {
            'proj_text_emb': proj_text_emb,
            'text_emb': text_emb
        }

    @torch.no_grad()
    def ext_ecg_emb(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        提取ECG嵌入（用于推理）

        Args:
            ecg: (B, 12, L)

        Returns:
            proj_ecg_emb: (B, proj_out)
        """
        ecg_emb = self.ecg_encoder(ecg)
        proj_ecg_emb = self.proj_e(ecg_emb)
        proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)
        return proj_ecg_emb

    @torch.no_grad()
    def get_text_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """获取文本嵌入（用于推理）"""
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

        # 文本编码
        tokenized_input = self._tokenize(batch['report'])
        input_ids = tokenized_input['input_ids'].type_as(batch['ecg']).long()
        attention_mask = tokenized_input['attention_mask'].type_as(batch['ecg']).long()
        text_output = self.encode_text(input_ids, attention_mask)

        # 标准CLIP损失函数
        loss_fn = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            use_horovod=False
        )

        # ========== 使用原始 ClipLoss ==========
        cma_loss = loss_fn(
            ecg_output['proj_ecg_emb'],
            text_output['proj_text_emb'],
            self.logit_scale.exp()
        )

        # # ========== 可学习软标签损失 (已注释) ==========
        # if self.use_learnable_sim:
        #     hybrid_sim, alpha = self.compute_hybrid_similarity(
        #         text_output['proj_text_emb'],
        #         text_output['text_emb']
        #     )
        #
        #     learnable_soft_loss_fn = LearnableSoftClipLoss(
        #         local_loss=True,
        #         gather_with_grad=True,
        #         rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        #         world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
        #         use_horovod=False
        #     )
        #
        #     soft_cma_loss = learnable_soft_loss_fn(
        #         image_features=ecg_output['proj_ecg_emb'],
        #         text_features=text_output['proj_text_emb'],
        #         logit_scale=self.logit_scale.exp(),
        #         sim_matrix=hybrid_sim,
        #         threshold=self.learnable_threshold,
        #         soft_weight=self.learnable_soft_weight,
        #     )
        # else:
        #     text_sim_matrix = self.compute_text_similarity(batch['report'])
        #
        #     soft_clip_loss_fn = SoftClipLoss(
        #         similarity_threshold=0.95,
        #         soft_positive_weight=0.05,
        #         local_loss=True,
        #         gather_with_grad=True,
        #         cache_labels=False,
        #         rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        #         world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
        #         use_horovod=False
        #     )
        #
        #     soft_cma_loss = soft_clip_loss_fn(
        #         image_features=ecg_output['proj_ecg_emb'],
        #         text_features=text_output['proj_text_emb'],
        #         logit_scale=self.logit_scale.exp(),
        #         text_sim_matrix=text_sim_matrix
        #     )

        # UMA损失
        uma_loss = loss_fn(
            ecg_output['ecg_emb1'],
            ecg_output['ecg_emb2'],
            torch.tensor(1 / 0.07)
        )

        loss_dict = {
            'loss': cma_loss + uma_loss,
            'cma_loss': cma_loss,
            'uma_loss': uma_loss
        }

        # # 记录可学习参数的当前值 (用于监控训练过程)
        # if self.use_learnable_sim:
        #     loss_dict['sim_alpha'] = alpha.detach()
        #     loss_dict['threshold'] = torch.sigmoid(self.learnable_threshold).detach()
        #     loss_dict['soft_weight'] = torch.sigmoid(self.learnable_soft_weight).detach()

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
