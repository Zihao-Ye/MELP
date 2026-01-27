"""
SMMERL 微调脚本

基于预训练的SMMERL模型进行下游任务微调。

支持两种微调模式:
1. Linear Probing: 冻结backbone，只训练分类头
2. Full Fine-tuning: 微调整个模型（可选小学习率更新backbone）

使用方法:
CUDA_VISIBLE_DEVICES=0 python main_finetune_smmerl.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 1.0 \
    --ckpt_path /path/to/pretrain/checkpoint.ckpt \
    --num_devices 1 \
    --finetune_mode linear_probe
"""

import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from melp.datasets.finetune_datamodule import ECGDataModule
from melp.models.smmerl_model import SMMERLModel
from melp.models.ssl_finetuner import SSLFineTuner
from melp.paths import ROOT_PATH as REPO_ROOT_DIR
from melp.paths import RAW_DATA_PATH

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"smmerl_finetune_{hparams.dataset_name}_{hparams.finetune_mode}_{extension}"

    ckpt_dir = os.path.expanduser(
        f"~/autodl-tmp/logs/smmerl_finetune/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_auc",
            dirpath=ckpt_dir,
            save_last=False,
            mode="max",
            save_top_k=1,
            auto_insert_metric_name=True
        ),
        EarlyStopping(
            monitor="val_auc",
            min_delta=0,
            patience=5,
            verbose=False,
            mode="max"
        ),
    ]

    logger_dir = os.path.expanduser("~/autodl-tmp/logs/smmerl_finetune")
    os.makedirs(logger_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project="smmerl_finetune",
        save_dir=logger_dir,
        name=extension
    )

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        deterministic=True,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true" if hparams.num_devices > 1 else "auto",
        precision=32,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/{extension}/exp_logs")

    datamodule = ECGDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_name=hparams.dataset_name,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        train_data_pct=hparams.train_data_pct
    )

    hparams.num_classes = datamodule.train_dataloader().dataset.num_classes
    hparams.training_steps_per_epoch = (
        len(datamodule.train_dataloader()) //
        hparams.accumulate_grad_batches //
        hparams.num_devices
    )

    # 加载预训练模型
    if hparams.ckpt_path:
        print(f"Loading pretrained model from: {hparams.ckpt_path}")
        pretrain_model = SMMERLModel.load_from_checkpoint(hparams.ckpt_path)
    else:
        print("No checkpoint provided, using randomly initialized model")
        pretrain_model = SMMERLModel(
            ecg_encoder_name=hparams.ecg_encoder_name,
            text_encoder_name=hparams.text_encoder_name,
            shared_emb_dim=hparams.shared_emb_dim,
            lead_group_strategy=hparams.lead_group_strategy,
        )

    # SMMERL输出维度为shared_emb_dim
    hparams.in_features = 3*pretrain_model.proj_out

    print("\n" + "="*60)
    print("Finetuning Configuration:")
    print("="*60)
    for key, value in sorted(vars(hparams).items()):
        print(f"{key:30s}: {value}")
    print("="*60 + "\n")

    # 根据微调模式选择不同的策略
    if hparams.finetune_mode == "linear_probe":
        # Linear Probing: 使用SSLFineTuner，冻结backbone
        model = SSLFineTuner(
            backbone=pretrain_model,
            in_features=hparams.in_features,
            num_classes=hparams.num_classes,
            epochs=hparams.max_epochs,
            dropout=hparams.dropout,
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
            scheduler_type=hparams.scheduler_type,
            final_lr=hparams.final_lr,
        )

    elif hparams.finetune_mode == "full_finetune":
        # Full Fine-tuning: 使用自定义的全微调模型
        model = SMMERLFineTuner(
            backbone=pretrain_model,
            in_features=hparams.in_features,
            num_classes=hparams.num_classes,
            epochs=hparams.max_epochs,
            dropout=hparams.dropout,
            lr=hparams.lr,
            backbone_lr_scale=hparams.backbone_lr_scale,
            weight_decay=hparams.weight_decay,
            scheduler_type=hparams.scheduler_type,
            final_lr=hparams.final_lr,
        )

    else:
        raise ValueError(f"Unknown finetune_mode: {hparams.finetune_mode}")

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


class SMMERLFineTuner(SSLFineTuner):
    """
    SMMERL全微调模型

    与SSLFineTuner的区别:
    - 不完全冻结backbone，使用小学习率微调
    - 支持分层学习率
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 256,
        num_classes: int = 2,
        epochs: int = 100,
        dropout: float = 0.0,
        lr: float = 1e-3,
        backbone_lr_scale: float = 0.1,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        decay_epochs: tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 1e-5,
        *args,
        **kwargs
    ) -> None:
        # 调用父类初始化，但不冻结backbone
        super().__init__(
            backbone=backbone,
            in_features=in_features,
            num_classes=num_classes,
            epochs=epochs,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            decay_epochs=decay_epochs,
            gamma=gamma,
            final_lr=final_lr,
            *args,
            **kwargs
        )

        self.backbone_lr_scale = backbone_lr_scale

        # 解冻backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self) -> None:
        # 不调用父类的eval()，保持backbone为train模式
        self.backbone.train()

    def shared_step(self, batch):
        """重写shared_step，不使用torch.no_grad()"""
        ecg = batch["ecg"]
        y = batch["label"]

        # 直接调用backbone（不使用no_grad）
        feats = self.backbone.ext_ecg_emb(ecg)
        feats = feats.view(feats.size(0), -1)

        logits = self.linear_layer(feats)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        """分层学习率优化器配置"""
        # Backbone参数使用较小的学习率
        backbone_params = list(self.backbone.parameters())

        # 分类头参数使用正常学习率
        head_params = list(self.linear_layer.parameters())

        param_groups = [
            {
                'params': backbone_params,
                'lr': self.lr * self.backbone_lr_scale,
                'name': 'backbone'
            },
            {
                'params': head_params,
                'lr': self.lr,
                'name': 'head'
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        # 设置scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.decay_epochs, gamma=self.gamma
            )
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr
            )

        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = ArgumentParser(description="Finetune SMMERL Model.")

    # 模型参数
    parser.add_argument("--ecg_encoder_name", type=str, default="smmerl_base",
                        choices=["smmerl_tiny", "smmerl_small",
                                 "smmerl_base", "smmerl_large"],
                        help="ECG encoder name (used when no ckpt_path provided)")
    parser.add_argument("--text_encoder_name", type=str, default="ncbi/MedCPT-Query-Encoder",
                        help="Text encoder name (used when no ckpt_path provided)")
    parser.add_argument("--shared_emb_dim", type=int, default=256,
                        help="Shared embedding dimension")
    parser.add_argument("--lead_group_strategy", type=str, default="lisa",
                        choices=["none", "limb_chest", "lisa"],
                        help="Lead grouping strategy (used when no ckpt_path provided)")

    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="ptbxl_super_class",
                        choices=["ptbxl_super_class", "ptbxl_sub_class",
                                 "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman", "anzhen"])

    # 微调模式
    parser.add_argument("--finetune_mode", type=str, default="linear_probe",
                        choices=["linear_probe", "full_finetune"],
                        help="Finetuning mode: linear_probe or full_finetune")

    # 训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="Learning rate scale for backbone (only for full_finetune)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["cosine", "step"])
    parser.add_argument("--final_lr", type=float, default=1e-5)

    # 检查点
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Path to pretrained checkpoint")

    # 其他
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_features", type=int, default=256)

    hparams = parser.parse_args()

    # 设置随机种子
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)

    main(hparams)

