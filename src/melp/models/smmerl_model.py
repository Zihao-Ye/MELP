"""
SMMERL Model: PyTorch Lightning Module for Training

Implements ECG-Text contrastive learning with:
- SMMERL encoder for ECG signals
- Text encoder (MedCPT/T5) for diagnostic reports
- Multi-scale CLIP loss (wave/beat/rhythm)
- Zero-shot evaluation on downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from typing import List, Dict, Optional
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from melp.backbone.smmerl import (
    smmerl_tiny,
    smmerl_small,
    smmerl_base,
    smmerl_large
)
from melp.models.base_pretrain_model import BasePretrainModel
from melp.utils.openclip_loss import ClipLoss, LearnableSoftClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


class SMMERLModel(BasePretrainModel):
    """
    SMMERL Pretraining Model

    Simplified and improved ECG-text contrastive learning model.

    Key features:
    - Multi-scale ECG encoding (wave/beat/rhythm)
    - Flexible text encoder (MedCPT/T5)
    - Multi-scale CLIP loss with optional soft labels
    - Zero-shot evaluation on multiple datasets

    Args:
        # ECG Encoder
        ecg_encoder_name: Model size - 'smmerl_tiny/small/base/large'
        embed_dim: Embedding dimension (default: 256)
        seq_len: Input ECG length (default: 5000)
        lead_group_strategy: 'none', 'limb_chest', or 'lisa' (default: 'lisa')
        pool_type: Final pooling - 'mean' or 'attn' (default: 'mean')

        # Text Encoder
        text_encoder_name: Text model name (default: 'ncbi/MedCPT-Query-Encoder')
        num_freeze_layers: Number of frozen layers in text encoder (default: 6)

        # Projection
        shared_emb_dim: Shared embedding dimension for CLIP (default: 256)

        # Loss
        use_soft_labels: Use soft labels for rhythm scale (default: True)
        init_logit_scale: Initial temperature parameter (default: log(1/0.07))

        # Optimization
        lr: Learning rate (default: 2e-4)
        weight_decay: Weight decay (default: 0.2)
        max_epochs: Maximum training epochs (default: 100)

        # Validation
        val_dataset_list: List of validation datasets
    """

    def __init__(
        self,
        # ECG Encoder
        ecg_encoder_name: str = 'smmerl_base',
        embed_dim: int = 256,
        seq_len: int = 5000,
        lead_group_strategy: str = 'lisa',
        pool_type: str = 'mean',
        # Text Encoder
        text_encoder_name: str = 'ncbi/MedCPT-Query-Encoder',
        num_freeze_layers: int = 6,
        # Projection
        shared_emb_dim: int = 256,
        # Loss
        use_soft_labels: bool = True,
        init_logit_scale: float = np.log(1 / 0.07),
        # Optimization
        lr: float = 2e-4,
        weight_decay: float = 0.2,
        max_epochs: int = 100,
        # Validation
        val_dataset_list: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        # Set validation datasets
        if val_dataset_list is None:
            val_dataset_list = [
                "ptbxl_super_class", "ptbxl_sub_class",
                "ptbxl_form", "ptbxl_rhythm",
                "icbeb", "chapman"
            ]

        self.ecg_encoder_name = ecg_encoder_name
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.lead_group_strategy = lead_group_strategy
        self.pool_type = pool_type
        self.num_freeze_layers = num_freeze_layers
        self.use_soft_labels = use_soft_labels

        super().__init__(
            ecg_encoder_name=ecg_encoder_name,
            text_encoder_name=text_encoder_name,
            shared_emb_dim=shared_emb_dim,
            lr=lr,
            weight_decay=weight_decay,
            *args,
            **kwargs
        )

        self.save_hyperparameters()
        self.val_dataset_list = val_dataset_list
        self.proj_out = shared_emb_dim

        # Initialize encoders
        self.init_ecg_encoder()
        self.init_text_encoder()

        # Temperature parameter for CLIP loss
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        # Soft label parameters (for rhythm scale)
        if self.use_soft_labels:
            self.learnable_threshold = nn.Parameter(torch.tensor(3.0))
            self.learnable_soft_weight = nn.Parameter(torch.tensor(-3.0))

        # Load prompt and label configurations
        with open(PROMPT_PATH, 'r') as f:
            self.prompt_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.all_labels = list(self.prompt_dict.keys())

        with open(DATASET_LABELS_PATH, 'r') as f:
            self.dataset_class_names = yaml.load(f, Loader=yaml.FullLoader)

    def init_ecg_encoder(self):
        """Initialize SMMERL ECG encoder"""
        encoder_factory = {
            'smmerl_tiny': smmerl_tiny,
            'smmerl_small': smmerl_small,
            'smmerl_base': smmerl_base,
            'smmerl_large': smmerl_large,
        }

        if self.ecg_encoder_name not in encoder_factory:
            raise ValueError(
                f"Unknown ecg_encoder_name: {self.ecg_encoder_name}. "
                f"Choose from: {list(encoder_factory.keys())}"
            )

        self.ecg_encoder = encoder_factory[self.ecg_encoder_name](
            seq_len=self.seq_len,
            output_dim=self.embed_dim,
            lead_group_strategy=self.lead_group_strategy,
            pool_type=self.pool_type
        )

    def init_text_encoder(self):
        """Initialize text encoder and projection layers"""
        # Load text encoder
        if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
            self.lm_model = AutoModel.from_pretrained(self.text_encoder_name)

            # Freeze early layers
            for layer_idx in range(self.num_freeze_layers):
                for param in list(self.lm_model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False

            text_encoder_hidden_dim = 768

        elif self.text_encoder_name == "google/flan-t5-small":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True
            )
            text_encoder_hidden_dim = 512

        elif self.text_encoder_name == "google/flan-t5-base":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True
            )
            text_encoder_hidden_dim = 768

        else:
            raise NotImplementedError(
                f"Unknown text encoder: {self.text_encoder_name}"
            )

        self.text_encoder_hidden_dim = text_encoder_hidden_dim

        # ECG Projection layers for each scale
        self.wave_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )
        self.beat_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )

        # Text Projection layers for each scale
        self.text_wave_proj = nn.Sequential(
            nn.Linear(text_encoder_hidden_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )
        self.text_beat_proj = nn.Sequential(
            nn.Linear(text_encoder_hidden_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )
        self.text_rhythm_proj = nn.Sequential(
            nn.Linear(text_encoder_hidden_dim, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )

        # CLIP loss modules
        self.clip_loss = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=0,
            world_size=1,
        )

        if self.use_soft_labels:
            self.rhythm_soft_clip_loss = LearnableSoftClipLoss(
                local_loss=True,
                gather_with_grad=True,
                rank=0,
                world_size=1,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

    def _tokenize(self, text):
        """Tokenize text inputs"""
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
        Encode ECG signals to multi-scale features

        Args:
            ecg: (B, 12, 5000) - 12-lead ECG signals

        Returns:
            dict with:
                'wave': (B, embed_dim) - Wave-level features
                'beat': (B, embed_dim) - Beat-level features
                'rhythm': (B, embed_dim) - Rhythm-level features
        """
        return self.ecg_encoder.forward_multiscale(ecg)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode text to embeddings

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            dict with 'text_emb': (B, text_encoder_hidden_dim)
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

        else:
            raise NotImplementedError(f"Unknown text encoder: {self.text_encoder_name}")

        return {'text_emb': text_emb}

    def shared_step(self, batch, batch_idx):
        """
        Shared training/validation step with multi-scale CLIP loss

        Args:
            batch: Dictionary with 'ecg' and 'text' keys
            batch_idx: Batch index

        Returns:
            loss_dict: Dictionary of losses
        """
        ecg = batch['ecg']  # (B, 12, 5000)
        text = batch['report']  # List of strings

        # Encode ECG to multi-scale features
        ecg_features = self.encode_ecg(ecg)
        wave_ecg = ecg_features['wave']    # (B, embed_dim)
        beat_ecg = ecg_features['beat']    # (B, embed_dim)
        rhythm_ecg = ecg_features['rhythm']  # (B, embed_dim)

        # Tokenize and encode text
        tokenizer_output = self._tokenize(text)
        input_ids = tokenizer_output['input_ids'].to(ecg.device)
        attention_mask = tokenizer_output['attention_mask'].to(ecg.device)

        text_output = self.encode_text(input_ids, attention_mask)
        text_emb = text_output['text_emb']  # (B, text_encoder_hidden_dim)

        # Project ECG to each scale + L2 normalize
        wave_ecg = F.normalize(self.wave_proj(wave_ecg), dim=-1)
        beat_ecg = F.normalize(self.beat_proj(beat_ecg), dim=-1)
        rhythm_ecg = F.normalize(self.rhythm_proj(rhythm_ecg), dim=-1)

        # Project text to each scale + L2 normalize
        wave_text = F.normalize(self.text_wave_proj(text_emb), dim=-1)
        beat_text = F.normalize(self.text_beat_proj(text_emb), dim=-1)
        rhythm_text = F.normalize(self.text_rhythm_proj(text_emb), dim=-1)

        # Compute CLIP losses
        logit_scale = self.logit_scale.exp()

        # Wave and beat scales: standard CLIP loss
        wave_loss = self.clip_loss(wave_ecg, wave_text, logit_scale)
        beat_loss = self.clip_loss(beat_ecg, beat_text, logit_scale)

        # Rhythm scale: soft CLIP loss if enabled
        if self.use_soft_labels:
            # Compute semantic similarity matrix using text embeddings
            with torch.no_grad():
                # Normalize original text embeddings
                text_emb_norm = F.normalize(text_emb, dim=-1)

                # Handle distributed training
                world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                if world_size > 1:
                    # Gather text embeddings from all GPUs
                    gathered_text_emb = [torch.zeros_like(text_emb_norm) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_text_emb, text_emb_norm.contiguous())
                    all_text_emb_norm = torch.cat(gathered_text_emb, dim=0)
                    # Compute similarity: (local_batch, global_batch)
                    text_sim_matrix = text_emb_norm @ all_text_emb_norm.T
                else:
                    # Single GPU: (batch, batch)
                    text_sim_matrix = text_emb_norm @ text_emb_norm.T

            # Update distributed parameters for soft clip loss
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self.rhythm_soft_clip_loss.rank = rank
            self.rhythm_soft_clip_loss.world_size = world_size

            rhythm_loss = self.rhythm_soft_clip_loss(
                rhythm_ecg, rhythm_text, logit_scale,
                sim_matrix=text_sim_matrix,
                threshold=self.learnable_threshold,
                soft_weight=self.learnable_soft_weight
            )
        else:
            rhythm_loss = self.clip_loss(rhythm_ecg, rhythm_text, logit_scale)

        # Total loss
        total_loss = wave_loss + beat_loss + rhythm_loss

        loss_dict = {
            'loss': total_loss,
            'wave_loss': wave_loss,
            'beat_loss': beat_loss,
            'rhythm_loss': rhythm_loss,
            'logit_scale': logit_scale
        }

        if self.use_soft_labels:
            loss_dict['threshold'] = self.learnable_threshold
            loss_dict['soft_weight'] = self.learnable_soft_weight

        return loss_dict

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step"""
        loss_dict = self.shared_step(batch, batch_idx)

        # Log losses
        self.log('train/loss', loss_dict['loss'], prog_bar=True, sync_dist=True)
        self.log('train/wave_loss', loss_dict['wave_loss'], sync_dist=True)
        self.log('train/beat_loss', loss_dict['beat_loss'], sync_dist=True)
        self.log('train/rhythm_loss', loss_dict['rhythm_loss'], sync_dist=True)
        self.log('train/logit_scale', loss_dict['logit_scale'], sync_dist=True)

        if self.use_soft_labels:
            self.log('train/threshold', loss_dict['threshold'], sync_dist=True)
            self.log('train/soft_weight', loss_dict['soft_weight'], sync_dist=True)

        return loss_dict['loss']

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Clamp logit_scale to prevent overflow"""
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))

    # ========================================================================
    # Helper Methods for Inference
    # ========================================================================

    def ext_ecg_emb(self, ecg: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
        """
        Extract ECG embeddings (with projection and normalization)

        Args:
            ecg: (B, 12, 5000) - ECG signals
            mode: Multi-scale fusion mode
                - 'concat': Concatenate all scales (B, 3*proj_out)
                - 'mean': Average all scales (B, proj_out)
                - 'rhythm': Only rhythm scale (B, proj_out)

        Returns:
            embeddings: (B, 3*proj_out) for concat, (B, proj_out) for mean/rhythm
        """
        with torch.no_grad():
            ecg_features = self.encode_ecg(ecg)

            # Project and normalize each scale
            wave_z = F.normalize(self.wave_proj(ecg_features['wave']), dim=-1)
            beat_z = F.normalize(self.beat_proj(ecg_features['beat']), dim=-1)
            rhythm_z = F.normalize(self.rhythm_proj(ecg_features['rhythm']), dim=-1)

            if mode == 'concat':
                # Concatenate all scales (like mvcse_mssate default)
                emb = torch.cat([wave_z, beat_z, rhythm_z], dim=-1)
            elif mode == 'mean':
                # Average all scales
                emb = (wave_z + beat_z + rhythm_z) / 3
            elif mode == 'rhythm':
                # Only rhythm scale
                emb = rhythm_z
            else:
                raise ValueError(f"Unknown mode: {mode}")

            return emb

    def get_text_emb(self, text: List[str], mode: str = 'concat') -> torch.Tensor:
        """
        Get text embeddings with multi-scale fusion

        Args:
            text: List of text strings
            mode: Multi-scale fusion mode
                - 'concat': Concatenate all scales (B, 3*proj_out)
                - 'mean': Average all scales (B, proj_out)
                - 'rhythm': Only rhythm scale (B, proj_out)

        Returns:
            embeddings: (B, 3*proj_out) for concat, (B, proj_out) for mean/rhythm
        """
        with torch.no_grad():
            tokenizer_output = self._tokenize(text)
            input_ids = tokenizer_output['input_ids'].to(self.device)
            attention_mask = tokenizer_output['attention_mask'].to(self.device)

            text_output = self.encode_text(input_ids, attention_mask)
            text_emb = text_output['text_emb']

            # Project to each scale and normalize
            text_wave_z = F.normalize(self.text_wave_proj(text_emb), dim=-1)
            text_beat_z = F.normalize(self.text_beat_proj(text_emb), dim=-1)
            text_rhythm_z = F.normalize(self.text_rhythm_proj(text_emb), dim=-1)

            if mode == 'concat':
                # Concatenate all scales (like mvcse_mssate default)
                emb = torch.cat([text_wave_z, text_beat_z, text_rhythm_z], dim=-1)
            elif mode == 'mean':
                # Average all scales
                emb = (text_wave_z + text_beat_z + text_rhythm_z) / 3
            elif mode == 'rhythm':
                # Only rhythm scale
                emb = text_rhythm_z
            else:
                raise ValueError(f"Unknown mode: {mode}")

            return emb

    def get_class_emd(self, class_names: List[str], mode: str = 'concat') -> torch.Tensor:
        """
        Get class embeddings from class names using prompts

        Args:
            class_names: List of class names (e.g., ['NORM', 'MI', 'STTC'])
            mode: Multi-scale fusion mode ('concat', 'mean', 'rhythm')

        Returns:
            class_embeddings: (num_classes, 3*proj_out) for concat, (num_classes, proj_out) for mean/rhythm
        """
        # Generate prompts for each class
        prompts = []
        for class_name in class_names:
            if class_name in self.prompt_dict:
                prompt = self.prompt_dict[class_name]
            else:
                # Fallback: use class name directly
                prompt = f"ECG shows {class_name}"
            prompts.append(prompt)

        # Get text embeddings
        class_embeddings = self.get_text_emb(prompts, mode=mode)

        return class_embeddings

    # ========================================================================
    # Validation Methods (Zero-shot Evaluation)
    # ========================================================================

    def on_validation_epoch_start(self):
        """Compute class embeddings for all validation datasets"""
        # Compute class embeddings for all labels (concat mode, like mvcse_mssate)
        val_prompts = [self.prompt_dict[i] for i in self.all_labels]

        # Only compute concat mode for validation (merged multi-scale)
        self.zeroshot_weights = self.get_class_emd(val_prompts, mode='concat').T  # (3*proj_out, num_classes)

        # Initialize storage for validation outputs
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Zero-shot evaluation on validation datasets (concat mode like mvcse_mssate)

        Args:
            batch: Dictionary with 'ecg' and 'label'
            batch_idx: Batch index
            dataloader_idx: Dataloader index (identifies which validation dataset)
        """
        # Get current dataset name and class names
        cur_dataset_name = self.val_dataset_list[dataloader_idx]
        class_names = self.dataset_class_names[cur_dataset_name]

        # Get indices of relevant classes in the full label list
        indices = [self.all_labels.index(i) for i in class_names]

        # Extract ECG embeddings (concat mode: concatenate three scales)
        ecg = batch['ecg']
        ecg_emb = self.ext_ecg_emb(ecg, mode='concat')  # (B, 3*proj_out)

        # Get relevant class embeddings for this dataset
        cur_zeroshot_weights = self.zeroshot_weights[:, indices]  # (3*proj_out, num_classes)

        # Compute logits
        logits = torch.matmul(ecg_emb, cur_zeroshot_weights)  # (B, num_classes)

        # Store outputs for later processing
        self.val_step_outputs.append({
            'dataloader_idx': dataloader_idx,
            'logits': logits,
            'label': batch['label']
        })

    def on_validation_epoch_end(self):
        """Compute AUROC metrics for all validation datasets (like mvcse_mssate)"""
        # Group outputs by dataloader_idx
        logits_dict = defaultdict(list)
        labels_dict = defaultdict(list)

        for output in self.val_step_outputs:
            dataloader_idx = output['dataloader_idx']
            logits = output['logits']
            labels = output['label']
            logits_dict[dataloader_idx].append(logits)
            labels_dict[dataloader_idx].append(labels)

        # Collect dataset AUROCs for computing mean
        dataset_aurocs = []

        # Compute AUROC for each dataset
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
            mean_auroc = np.mean(AUROCs)

            # Log dataset AUROC
            self.log(
                f'val/{dataset_name}_AUROC',
                mean_auroc,
                on_epoch=True, prog_bar=False, sync_dist=True
            )
            dataset_aurocs.append(mean_auroc)

        # Log overall mean AUROC
        self.log(
            f'val/mean_AUROC',
            np.mean(dataset_aurocs),
            on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Clear storage
        self.val_step_outputs.clear()


# ============================================================================
# ABLATION HOOKS: Model-Level Experiments
# ============================================================================
# Key ablation experiments to consider:
#
# 1. Multi-scale vs Single-scale
#    - Add flag: use_multiscale (bool)
#    - If False: only use 'all' scale, skip wave/beat/rhythm
#
# 2. Soft labels vs Hard labels
#    - Already implemented via use_soft_labels parameter
#    - Compare rhythm_soft_clip_loss vs standard clip_loss
#
# 3. Text encoder comparison
#    - MedCPT vs T5-small vs T5-base
#    - Already supported via text_encoder_name parameter
#
# 4. Freezing strategies
#    - Vary num_freeze_layers (0, 3, 6, 9, 12)
#    - Test full fine-tuning vs partial freezing
#
# 5. Loss weighting
#    - Add learnable weights for wave/beat/rhythm losses
#    - Example: total_loss = w1*wave + w2*beat + w3*rhythm
#
# 6. Temperature initialization
#    - Vary init_logit_scale (log(1/0.07), log(1/0.1), etc.)
#
# 7. Projection dimension
#    - Vary shared_emb_dim (128, 256, 512)
#
# To add ablation:
# - Add parameters to __init__()
# - Modify shared_step() or other methods accordingly
# - Keep code clean with clear if/else branches
# ============================================================================


# ============================================================================
# Usage Examples
# ============================================================================
"""
# Basic usage with default settings
model = SMMERLModel(
    ecg_encoder_name='smmerl_base',
    text_encoder_name='ncbi/MedCPT-Query-Encoder',
    lr=2e-4,
    max_epochs=100
)

# Extract ECG embeddings (default: concat mode)
ecg = torch.randn(8, 12, 5000)  # Batch of 8 ECG signals
ecg_emb = model.ext_ecg_emb(ecg)  # (8, 768) - concat mode: 3*256
ecg_emb_mean = model.ext_ecg_emb(ecg, mode='mean')  # (8, 256) - mean mode
ecg_emb_rhythm = model.ext_ecg_emb(ecg, mode='rhythm')  # (8, 256) - rhythm only

# Multi-scale embeddings (before projection)
ecg_features = model.encode_ecg(ecg)
wave_emb = ecg_features['wave']    # (8, 256)
beat_emb = ecg_features['beat']    # (8, 256)
rhythm_emb = ecg_features['rhythm']  # (8, 256)

# Get text embeddings (default: concat mode)
texts = ["Normal sinus rhythm", "Myocardial infarction"]
text_emb = model.get_text_emb(texts)  # (2, 768) - concat mode: 3*256
text_emb_mean = model.get_text_emb(texts, mode='mean')  # (2, 256) - mean mode

# Zero-shot classification
class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
class_embeddings = model.get_class_emd(class_names)  # (5, 768) - concat mode
logits = ecg_emb @ class_embeddings.T  # (8, 5)
predictions = logits.argmax(dim=-1)  # (8,)

# Different model sizes
tiny_model = SMMERLModel(ecg_encoder_name='smmerl_tiny')    # ~8M params
small_model = SMMERLModel(ecg_encoder_name='smmerl_small')  # ~18M params
base_model = SMMERLModel(ecg_encoder_name='smmerl_base')    # ~35M params
large_model = SMMERLModel(ecg_encoder_name='smmerl_large')  # ~65M params

# Different lead grouping strategies
lisa_model = SMMERLModel(lead_group_strategy='lisa')          # 5 anatomical groups
limb_chest_model = SMMERLModel(lead_group_strategy='limb_chest')  # 2 groups
no_group_model = SMMERLModel(lead_group_strategy='none')      # No grouping

# Ablation: Disable soft labels
hard_label_model = SMMERLModel(use_soft_labels=False)

# Ablation: Different text encoders
medcpt_model = SMMERLModel(text_encoder_name='ncbi/MedCPT-Query-Encoder')
t5_small_model = SMMERLModel(text_encoder_name='google/flan-t5-small')
t5_base_model = SMMERLModel(text_encoder_name='google/flan-t5-base')

# Ablation: Different freezing strategies
full_finetune = SMMERLModel(num_freeze_layers=0)   # Fine-tune all layers
partial_freeze = SMMERLModel(num_freeze_layers=6)  # Freeze first 6 layers
heavy_freeze = SMMERLModel(num_freeze_layers=9)    # Freeze first 9 layers
"""
