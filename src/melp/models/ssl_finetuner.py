from typing import Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import AUROC, F1Score, Precision, Recall
from einops import rearrange


class SSLFineTuner(LightningModule):
    """Finetunes a self-supervised learning backbone using the standard evaluation protocol of a linear layer
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 256,
        num_classes: int = 2,
        epochs: int = 100,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        decay_epochs: Tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 1e-5,
        use_ecg_patch: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.use_ecg_patch = use_ecg_patch

        self.backbone = backbone
        # Freeze the backbone for linear probing
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        self.train_auc = AUROC(task="multilabel", num_labels=num_classes)
        self.val_auc = AUROC(task="multilabel", num_labels=num_classes)
        self.test_auc = AUROC(task="multilabel", num_labels=num_classes)

        # # F1分数 (macro和micro)
        # self.train_f1_macro = F1Score(task="multilabel", num_labels=num_classes, average='macro')
        # self.train_f1_micro = F1Score(task="multilabel", num_labels=num_classes, average='micro')
        # self.val_f1_macro = F1Score(task="multilabel", num_labels=num_classes, average='macro')
        # self.val_f1_micro = F1Score(task="multilabel", num_labels=num_classes, average='micro')
        # self.test_f1_macro = F1Score(task="multilabel", num_labels=num_classes, average='macro')
        # self.test_f1_micro = F1Score(task="multilabel", num_labels=num_classes, average='micro')

        # # === 新增：per-label 指标（仅用于 val/test epoch end）===
        # self.val_per_label_metrics = nn.ModuleDict({
        #     'auc': AUROC(task="multilabel", num_labels=num_classes, average=None),
        #     'f1': F1Score(task="multilabel", num_labels=num_classes, average=None),
        #     'precision': Precision(task="multilabel", num_labels=num_classes, average=None),
        #     'recall': Recall(task="multilabel", num_labels=num_classes, average=None),
        # })

        # self.test_per_label_metrics = nn.ModuleDict({
        #     'auc': AUROC(task="multilabel", num_labels=num_classes, average=None),
        #     'f1': F1Score(task="multilabel", num_labels=num_classes, average=None),
        #     'precision': Precision(task="multilabel", num_labels=num_classes, average=None),
        #     'recall': Recall(task="multilabel", num_labels=num_classes, average=None),
        # })

        # # 缓存 logits 和 targets 用于 epoch-end 计算（可选，但更稳定）
        # self.val_outputs = []
        # self.test_outputs = []

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        # auc = self.train_auc(logits.softmax(-1), y.long())  # 旧方案：使用softmax（不适合多标签分类）
        auc = self.train_auc(logits, y.long())  # 新方案：直接使用logits，与pretrain保持一致

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auc_step", auc, prog_bar=True)
        self.log("train_auc_epoch", self.train_auc)

        # # 多标签预测：sigmoid激活
        # probs = torch.sigmoid(logits)
        # pred = (probs > 0.5).float()

        # # 计算指标
        # auc = self.train_auc(probs, y.long())
        # f1_macro = self.train_f1_macro(pred, y.long())
        # f1_micro = self.train_f1_micro(pred, y.long())

        # self.log("train_loss", loss, prog_bar=True)
        # self.log("train_auc_step", auc, prog_bar=True)
        # self.log("train_auc_epoch", self.train_auc)
        # self.log("train_f1_macro", f1_macro, prog_bar=True)
        # self.log("train_f1_micro", f1_micro, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        # self.val_auc(logits.softmax(-1), y.long())  # 旧方案：使用softmax（不适合多标签分类）
        self.val_auc(logits, y.long())  # 新方案：直接使用logits，与pretrain保持一致

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_auc", self.val_auc)

        # # 多标签预测：sigmoid激活
        # probs = torch.sigmoid(logits)
        # pred = (probs > 0.5).float()

        # # 更新验证指标
        # self.val_auc(probs, y.long())
        # self.val_f1_macro(pred, y.long())
        # self.val_f1_micro(pred, y.long())

        # self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        # self.log("val_auc", self.val_auc, prog_bar=True)
        # self.log("val_f1_macro", self.val_f1_macro, prog_bar=True)
        # self.log("val_f1_micro", self.val_f1_micro, prog_bar=False)

        #  # 缓存用于 epoch-end per-label 计算（避免 metric 内部 reset 问题）
        # self.val_outputs.append((logits.detach(), y.detach()))

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        # self.test_auc(logits.softmax(-1), y.long())  # 旧方案：使用softmax（不适合多标签分类）
        self.test_auc(logits, y.long())  # 新方案：直接使用logits，与pretrain保持一致

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_auc", self.test_auc)

        # # 多标签预测：sigmoid激活
        # probs = torch.sigmoid(logits)
        # pred = (probs > 0.5).float()

        # # 更新测试指标
        # self.test_auc(probs, y.long())
        # self.test_f1_macro(pred, y.long())
        # self.test_f1_micro(pred, y.long())

        # self.log("test_loss", loss, sync_dist=True)
        # self.log("test_auc", self.test_auc)
        # self.log("test_f1_macro", self.test_f1_macro)
        # self.log("test_f1_micro", self.test_f1_micro)

        # self.test_outputs.append((logits.detach(), y.detach()))

        return loss

    # def on_validation_epoch_end(self):
    #     # Log overall metrics
    #     self.log("val_auc", self.val_auc, prog_bar=True)
    #     self.log("val_f1_macro", self.val_f1_macro, prog_bar=True)
    #     self.log("val_f1_micro", self.val_f1_micro, prog_bar=False)

    #     # Compute and log per-label metrics
    #     if self.val_outputs:
    #         logits = torch.cat([out[0] for out in self.val_outputs], dim=0)
    #         targets = torch.cat([out[1] for out in self.val_outputs], dim=0)

    #         probs = torch.sigmoid(logits)
    #         pred = (probs > 0.5).float()

    #         # AUC 需要 probs/logits，其他需要 pred
    #         per_label_auc = self.val_per_label_metrics['auc'](probs, targets.long())
    #         per_label_f1 = self.val_per_label_metrics['f1'](pred, targets.long())
    #         per_label_prec = self.val_per_label_metrics['precision'](pred, targets.long())
    #         per_label_rec = self.val_per_label_metrics['recall'](pred, targets.long())

    #         # Log each label's metrics (e.g., val_auc_label_0, val_f1_label_1, ...)
    #         num_labels = per_label_auc.shape[0]
    #         for i in range(num_labels):
    #             self.log(f"val_auc_label_{i}", per_label_auc[i], sync_dist=True)
    #             self.log(f"val_f1_label_{i}", per_label_f1[i], sync_dist=True)
    #             self.log(f"val_precision_label_{i}", per_label_prec[i], sync_dist=True)
    #             self.log(f"val_recall_label_{i}", per_label_rec[i], sync_dist=True)

    #         # Clear cache
    #         self.val_outputs.clear()

    # def on_test_epoch_end(self):
    #     self.log("test_auc", self.test_auc)
    #     self.log("test_f1_macro", self.test_f1_macro)
    #     self.log("test_f1_micro", self.test_f1_micro)

    #     if self.test_outputs:
    #         logits = torch.cat([out[0] for out in self.test_outputs], dim=0)
    #         targets = torch.cat([out[1] for out in self.test_outputs], dim=0)

    #         probs = torch.sigmoid(logits)
    #         pred = (probs > 0.5).float()

    #         per_label_auc = self.test_per_label_metrics['auc'](probs, targets.long())
    #         per_label_f1 = self.test_per_label_metrics['f1'](pred, targets.long())
    #         per_label_prec = self.test_per_label_metrics['precision'](pred, targets.long())
    #         per_label_rec = self.test_per_label_metrics['recall'](pred, targets.long())

    #         num_labels = per_label_auc.shape[0]
    #         for i in range(num_labels):
    #             self.log(f"test_auc_label_{i}", per_label_auc[i])
    #             self.log(f"test_f1_label_{i}", per_label_f1[i])
    #             self.log(f"test_precision_label_{i}", per_label_prec[i])
    #             self.log(f"test_recall_label_{i}", per_label_rec[i])

    #         self.test_outputs.clear()

    def shared_step(self, batch):
        # Extract features from the backbone
        with torch.no_grad():
            if self.use_ecg_patch:
                ecg_patch = batch["ecg_patch"]
                ecg_patch = rearrange(ecg_patch, 'B N (A T) -> B N A T', T=96)
                # Do attention only on visible ECG patches ...
                mask = ecg_patch.sum(-1) != 0
                t_indices = batch["t_indices"]
                feats = self.backbone.ext_ecg_emb(ecg_patch, mask, t_indices)
            else:
                ecg = batch["ecg"]
                y = batch["label"]
                feats = self.backbone.ext_ecg_emb(ecg)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]