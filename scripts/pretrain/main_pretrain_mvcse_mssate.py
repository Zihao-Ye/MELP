"""
MVCSE-MSSATE 预训练脚本

基于方案3.1和3.2的ECG-文本多模态预训练:
- MVCSE: Multi-View Cardiac Spatial Encoding (解剖分组空间编码)
- MS-SATE: Multi-Scale Shift-Adaptive Temporal Encoding (多尺度时序编码)
- DiagSim-Weighted Loss: 诊断语义相似度加权对比学习损失

使用方法:
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain_mvcse_mssate.py \
    --num_devices 4 --train_data_pct 1 \
    --text_encoder_name ncbi/MedCPT-Query-Encoder \
    --lr 2e-4 --batch_size 64 --max_epochs 100 \
    --ecg_encoder_name mvcse_mssate_base \
    --channel_attention se --use_relative_pos
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
        encoder_depth=hparams.encoder_depth,
        num_heads=hparams.num_heads,
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
                                 "mvcse_mssate_base", "mvcse_mssate_large", "custom"])
    parser.add_argument("--text_encoder_name", type=str, default="ncbi/MedCPT-Query-Encoder",
                        choices=["ncbi/MedCPT-Query-Encoder",
                                 "google/flan-t5-small", "google/flan-t5-base"])

    # ECG编码器参数
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension for MVCSE")
    parser.add_argument("--seq_len", type=int, default=5000,
                        help="Input sequence length")
    parser.add_argument("--encoder_depth", type=int, default=6,
                        help="Number of transformer layers per scale")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
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
