"""
MVCSE-MSSATE 预训练脚本

支持两种架构:
1. 单尺度架构 (mvcse_mssate_*): 原有Conv1d硬切分方案
2. 多尺度层级架构 (hierarchical_mvcse_mssate_*): 可学习Query软切分方案
   - wave/beat/rhythm三级特征建模
   - 每个层级独立和文本对齐
   - 使用MultiScaleClipLoss

使用方法:
# 单尺度架构（原有方案）
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain_mvcse_mssate.py \
    --num_devices 4 --train_data_pct 1 \
    --ecg_encoder_name mvcse_mssate_base \
    --lr 2e-4 --batch_size 64 --max_epochs 100

# 多尺度层级架构（新方案，推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain_mvcse_mssate.py \
    --num_devices 4 --train_data_pct 1 \
    --ecg_encoder_name hierarchical_mvcse_mssate_base \
    --lr 2e-4 --batch_size 64 --max_epochs 100
"""

import ipdb
from pprint import pprint
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
from melp.models.mvcse_mssate_model import MVCSEMSSATEModel
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
    extension = f"mvcse_mssate_{hparams.ecg_encoder_name}_{extension}"

    ckpt_dir = os.path.expanduser(
        f"~/autodl-tmp/logs/mvcse_mssate/ckpts/{extension}")
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

    logger_dir = os.path.expanduser("~/autodl-tmp/logs/mvcse_mssate")
    os.makedirs(logger_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project="mvcse_mssate",
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
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.expanduser(
        f"~/autodl-tmp/logs/mvcse_mssate/exp_logs/{extension}")

    datamodule = ECGTextDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_list=["mimic-iv-ecg"],
        val_dataset_list=hparams.val_dataset_list,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        train_data_pct=hparams.train_data_pct
    )

    model = MVCSEMSSATEModel(
        # ECG编码器参数
        ecg_encoder_name=hparams.ecg_encoder_name,
        embed_dim=hparams.embed_dim,
        seq_len=hparams.seq_len,
        # 新架构参数（方案B）
        lead_transformer_depth=hparams.lead_transformer_depth,
        lead_transformer_heads=hparams.lead_transformer_heads,
        cross_lead_depth=hparams.cross_lead_depth,
        mssate_depth=hparams.mssate_depth,
        mssate_num_heads=hparams.mssate_num_heads,
        channel_attention=hparams.channel_attention,
        use_relative_pos=hparams.use_relative_pos,
        # 文本编码器参数
        text_encoder_name=hparams.text_encoder_name,
        num_freeze_layers=hparams.num_freeze_layers,
        # 共享参数
        shared_emb_dim=hparams.shared_emb_dim,
        # 对比学习参数
        use_learnable_sim=hparams.use_learnable_sim,
        # 优化器参数
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
        max_epochs=hparams.max_epochs,
        # 验证数据集
        val_dataset_list=hparams.val_dataset_list,
    )

    model.training_steps_per_epoch = (
        len(datamodule.train_dataloader()) //
        hparams.accumulate_grad_batches //
        hparams.num_devices
    )

    pprint(vars(hparams))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretrain MVCSE-MSSATE Model.")

    # 模型选择
    parser.add_argument("--ecg_encoder_name", type=str, default="mvcse_mssate_base",
                        choices=["mvcse_mssate_tiny", "mvcse_mssate_small",
                                 "mvcse_mssate_base", "mvcse_mssate_large",
                                 "hierarchical_mvcse_mssate_small",
                                 "hierarchical_mvcse_mssate_base",
                                 "hierarchical_mvcse_mssate_large",
                                 "custom"])
    parser.add_argument("--text_encoder_name", type=str, default="ncbi/MedCPT-Query-Encoder",
                        choices=["ncbi/MedCPT-Query-Encoder",
                                 "google/flan-t5-small", "google/flan-t5-base"])

    # ECG编码器参数
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension for MVCSE")
    parser.add_argument("--seq_len", type=int, default=5000,
                        help="Input sequence length")

    # 新架构参数（方案B）
    parser.add_argument("--lead_transformer_depth", type=int, default=6,
                        help="Number of lead-level transformer layers (per group)")
    parser.add_argument("--lead_transformer_heads", type=int, default=4,
                        help="Number of attention heads in lead transformer")
    parser.add_argument("--cross_lead_depth", type=int, default=1,
                        help="Number of cross-lead aggregation layers")
    parser.add_argument("--mssate_depth", type=int, default=2,
                        help="Number of MS-SATE transformer layers per scale")
    parser.add_argument("--mssate_num_heads", type=int, default=8,
                        help="Number of attention heads in MS-SATE")

    parser.add_argument("--channel_attention", type=str, default="se",
                        choices=["se", "eca", "none"],
                        help="Inter-group channel attention type")
    parser.add_argument("--use_relative_pos", action="store_true", default=True,
                        help="Use relative positional encoding")
    parser.add_argument("--no_relative_pos", action="store_false", dest="use_relative_pos",
                        help="Disable relative positional encoding")

    # 文本编码器参数
    parser.add_argument("--num_freeze_layers", type=int, default=6,
                        help="Number of frozen layers in text encoder")
    parser.add_argument("--shared_emb_dim", type=int, default=256,
                        help="Shared embedding dimension")

    # 对比学习参数
    parser.add_argument("--use_learnable_sim", action="store_true", default=True,
                        help="Use learnable similarity for soft labels")
    parser.add_argument("--no_learnable_sim", action="store_false", dest="use_learnable_sim",
                        help="Disable learnable similarity")

    # 训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.2)

    # 验证数据集
    parser.add_argument("--val_dataset_list", type=str, nargs="+",
                        default=["ptbxl_super_class", "ptbxl_sub_class",
                                 "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman"])

    hparams = parser.parse_args()

    # 设置随机种子
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)

    main(hparams)
