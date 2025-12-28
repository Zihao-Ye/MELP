#!/bin/bash

# 连续运行多个微调实验的脚本
# 使用方法: bash run_experiments.sh

# ============ 配置区域 ============
# 在这里设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/melp/ckpts/melp_merl_2025_12_25_21_28_04/epoch=2-step=8736.ckpt"
# ==================================

echo "开始批量微调实验..."
echo "使用checkpoint: $CKPT_PATH"
echo "总实验数: 21 (7个数据集 × 3个训练比例)"
echo "开始时间: $(date)"
echo "=================================="

# anzhen 数据集实验
echo "数据集: anzhen, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name anzhen \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: anzhen, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name anzhen \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: anzhen, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name anzhen \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# ptbxl_super_class 数据集实验
echo "数据集: ptbxl_super_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_super_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_super_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_super_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_super_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_super_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# ptbxl_sub_class 数据集实验
echo "数据集: ptbxl_sub_class, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_sub_class \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_sub_class, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_sub_class \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_sub_class, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_sub_class \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# ptbxl_form 数据集实验
echo "数据集: ptbxl_form, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_form \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_form, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_form \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_form, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_form \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# ptbxl_rhythm 数据集实验
echo "数据集: ptbxl_rhythm, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_rhythm \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_rhythm, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_rhythm \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: ptbxl_rhythm, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name ptbxl_rhythm \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# icbeb 数据集实验
echo "数据集: icbeb, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: icbeb, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name icbeb \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: icbeb, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name icbeb \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

# chapman 数据集实验
echo "数据集: chapman, 训练比例: 0.01"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name chapman \
    --train_data_pct 0.01 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: chapman, 训练比例: 0.1"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name chapman \
    --train_data_pct 0.1 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "数据集: chapman, 训练比例: 1.0"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
    --model_name merl --dataset_name chapman \
    --train_data_pct 1.0 \
    --ckpt_path $CKPT_PATH \
    --num_devices 4

echo "=================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
