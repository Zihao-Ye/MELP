"""
SMMERL 预训练脚本

Simplified Multi-scale Multi-lead ECG Representation Learning

架构特点:
1. ResNet前端 → LeadTransformer (全序列) → 多尺度提取 → 导联聚合
2. 多尺度CLIP损失 (wave/beat/rhythm)
3. 可选的软标签学习 (rhythm尺度)
4. 零样本评估

使用方法:
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain_smmerl.py \
    --num_devices 4 --train_data_pct 1 \
    --ecg_encoder_name smmerl_base \
    --lead_group_strategy lisa \
    --lr 2e-4 --batch_size 32 --max_epochs 100
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

from melp.datasets.pretrain_datamodule import ECGTextDataModule
from melp.models.smmerl_model import SMMERLModel
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
    extension = f"smmerl_{hparams.ecg_encoder_name}_{extension}"

    ckpt_dir = os.path.expanduser(
        f"~/autodl-tmp/logs/smmerl/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val/mean_AUROC",
            dirpath=ckpt_dir,
            save_last=False,
            mode="max",
            save_top_k=2,
            auto_insert_metric_name=True
        ),
        EarlyStopping(
            monitor="val/mean_AUROC",
            min_delta=0,
            patience=5,
            verbose=True,
            mode="max"
        ),
    ]

    logger_dir = os.path.expanduser("~/autodl-tmp/logs/smmerl")
    os.makedirs(logger_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project="smmerl",
        save_dir=logger_dir,
        name=extension
    )

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision=32,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.expanduser(
        f"~/autodl-tmp/logs/smmerl/exp_logs/{extension}")

    datamodule = ECGTextDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_list=["mimic-iv-ecg"],
        val_dataset_list=hparams.val_dataset_list,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        train_data_pct=hparams.train_data_pct
    )

    model = SMMERLModel(
        # ECG Encoder
        ecg_encoder_name=hparams.ecg_encoder_name,
        embed_dim=hparams.embed_dim,
        seq_len=hparams.seq_len,
        lead_group_strategy=hparams.lead_group_strategy,
        pool_type=hparams.pool_type,
        # Text Encoder
        text_encoder_name=hparams.text_encoder_name,
        num_freeze_layers=hparams.num_freeze_layers,
        # Projection
        shared_emb_dim=hparams.shared_emb_dim,
        # Loss
        use_soft_labels=hparams.use_soft_labels,
        init_logit_scale=hparams.init_logit_scale,
        # Optimization
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        max_epochs=hparams.max_epochs,
        # Validation
        val_dataset_list=hparams.val_dataset_list,
    )

    model.training_steps_per_epoch = (
        len(datamodule.train_dataloader()) //
        hparams.accumulate_grad_batches //
        hparams.num_devices
    )

    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    for key, value in sorted(vars(hparams).items()):
        print(f"{key:30s}: {value}")
    print("="*60 + "\n")

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretrain SMMERL Model.")

    # Model Selection
    parser.add_argument("--ecg_encoder_name", type=str, default="smmerl_base",
                        choices=["smmerl_tiny", "smmerl_small",
                                 "smmerl_base", "smmerl_large"])
    parser.add_argument("--text_encoder_name", type=str, default="ncbi/MedCPT-Query-Encoder",
                        choices=["ncbi/MedCPT-Query-Encoder",
                                 "google/flan-t5-small", "google/flan-t5-base"])

    # ECG Encoder Parameters
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (default: 256)")
    parser.add_argument("--seq_len", type=int, default=5000,
                        help="Input sequence length (default: 5000)")
    parser.add_argument("--lead_group_strategy", type=str, default="lisa",
                        choices=["none", "limb_chest", "lisa"],
                        help="Lead grouping strategy")
    parser.add_argument("--pool_type", type=str, default="mean",
                        choices=["mean", "attn"],
                        help="Final pooling type")

    # Text Encoder Parameters
    parser.add_argument("--num_freeze_layers", type=int, default=6,
                        help="Number of frozen layers in text encoder")
    parser.add_argument("--shared_emb_dim", type=int, default=256,
                        help="Shared embedding dimension")

    # Loss Parameters
    parser.add_argument("--use_soft_labels", action="store_true", default=True,
                        help="Use soft labels for rhythm scale")
    parser.add_argument("--no_soft_labels", action="store_false", dest="use_soft_labels",
                        help="Disable soft labels")
    parser.add_argument("--init_logit_scale", type=float, default=np.log(1 / 0.07),
                        help="Initial logit scale (temperature)")

    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.2)

    # Validation Datasets
    parser.add_argument("--val_dataset_list", type=str, nargs="+",
                        default=["ptbxl_super_class", "ptbxl_sub_class",
                                 "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman"])

    hparams = parser.parse_args()

    # Set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)

    main(hparams)
