#!/bin/bash

# SMMERL 微调实验脚本
# 使用方法: bash smmerl_finetune.sh

# ============ 配置区域 ============
# 设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/smmerl/ckpts/smmerl_smmerl_base_2026_01_27_05_11_50/epoch=0-step=292.ckpt"
# 微调模式: linear_probe (冻结backbone) 或 full_finetune (全微调)
FINETUNE_MODE="linear_probe"
# GPU配置
CUDA_DEVICES="0,1,2,3"
NUM_DEVICES=4
# ==================================

echo "开始 SMMERL 批量微调实验..."
echo "使用checkpoint: $CKPT_PATH"
echo "微调模式: $FINETUNE_MODE"
echo "总实验数: 21 (7个数据集 × 3个训练比例)"
echo "开始时间: $(date)"
echo "=================================="

# icbeb 数据集实验
echo "数据集: icbeb, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: icbeb, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name icbeb \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: icbeb, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name icbeb \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# chapman 数据集实验
echo "数据集: chapman, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name chapman \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: chapman, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name chapman \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: chapman, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name chapman \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# ptbxl_super_class 数据集实验
echo "数据集: ptbxl_super_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_super_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_super_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_super_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# ptbxl_sub_class 数据集实验
echo "数据集: ptbxl_sub_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_sub_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_sub_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_sub_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# ptbxl_form 数据集实验
echo "数据集: ptbxl_form, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_form \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_form, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_form \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_form, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_form \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# ptbxl_rhythm 数据集实验
echo "数据集: ptbxl_rhythm, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_rhythm, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: ptbxl_rhythm, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name ptbxl_rhythm \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

# anzhen 数据集实验
echo "数据集: anzhen, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name anzhen \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: anzhen, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name anzhen \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "数据集: anzhen, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_finetune_smmerl.py \
    --dataset_name anzhen \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --finetune_mode $FINETUNE_MODE \
    --num_devices $NUM_DEVICES

echo "=================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
