#!/bin/bash

# MVCSE-MSSATE 微调实验脚本
# 使用方法: bash mvcse_mssate_finetune.sh

# ============ 配置区域 ============
# 设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/mvcse_mssate/ckpts/mvcse_mssate_hierarchical_mvcse_mssate_base_2026_01_06_13_59_00/epoch=1-step=5824.ckpt"
# 微调模式: linear_probe (冻结backbone) 或 full_finetune (全微调)
FINETUNE_MODE="linear_probe"
# ==================================

echo "开始 MVCSE-MSSATE 批量微调实验..."
echo "使用checkpoint: $CKPT_PATH"
echo "微调模式: $FINETUNE_MODE"
echo "总实验数: 21 (7个数据集 × 3个训练比例)"
echo "开始时间: $(date)"
echo "=================================="

# anzhen 数据集实验
echo "数据集: anzhen, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name anzhen \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: anzhen, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name anzhen \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: anzhen, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name anzhen \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# ptbxl_super_class 数据集实验
echo "数据集: ptbxl_super_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_super_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_super_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# ptbxl_sub_class 数据集实验
echo "数据集: ptbxl_sub_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_sub_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_sub_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# ptbxl_form 数据集实验
echo "数据集: ptbxl_form, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_form \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_form, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_form \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_form, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_form \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# ptbxl_rhythm 数据集实验
echo "数据集: ptbxl_rhythm, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_rhythm, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: ptbxl_rhythm, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# icbeb 数据集实验
echo "数据集: icbeb, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: icbeb, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name icbeb \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: icbeb, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name icbeb \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

# chapman 数据集实验
echo "数据集: chapman, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name chapman \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: chapman, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name chapman \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "数据集: chapman, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune_mvcse_mssate.py \
    --dataset_name chapman \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices 4

echo "=================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
