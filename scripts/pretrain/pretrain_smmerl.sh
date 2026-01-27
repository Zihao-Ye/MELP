#!/bin/bash

# SMMERL 预训练脚本
# 使用方法: bash pretrain_smmerl.sh

# ============ 配置区域 ============
# 模型大小选择: smmerl_tiny, smmerl_small, smmerl_base, smmerl_large
ECG_ENCODER_NAME="smmerl_base"

# 导联分组策略: none, limb_chest, lisa
LEAD_GROUP_STRATEGY="lisa"

# 池化类型: mean, attn
POOL_TYPE="mean"

# 文本编码器选择
# ncbi/MedCPT-Query-Encoder (推荐)
# google/flan-t5-small
# google/flan-t5-base
TEXT_ENCODER_NAME="ncbi/MedCPT-Query-Encoder"

# 软标签开关 (用于rhythm尺度)
USE_SOFT_LABELS=true

# GPU配置
NUM_DEVICES=4
CUDA_DEVICES="0,1,2,3"

# 训练参数
BATCH_SIZE=32
ACCUMULATE_GRAD_BATCHES=2
MAX_EPOCHS=100
LR=2e-4
# ==================================

echo "=============================================="
echo "SMMERL 预训练"
echo "=============================================="
echo "ECG Encoder: $ECG_ENCODER_NAME"
echo "Lead Group Strategy: $LEAD_GROUP_STRATEGY"
echo "Pool Type: $POOL_TYPE"
echo "Text Encoder: $TEXT_ENCODER_NAME"
echo "Use Soft Labels: $USE_SOFT_LABELS"
echo "Batch Size: $BATCH_SIZE x $ACCUMULATE_GRAD_BATCHES (acc) x $NUM_DEVICES (gpus) = $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * NUM_DEVICES))"
echo "Learning Rate: $LR"
echo "Max Epochs: $MAX_EPOCHS"
echo "开始时间: $(date)"
echo "=============================================="

cd /root/MELP/scripts/pretrain

# 根据USE_SOFT_LABELS设置参数
if [ "$USE_SOFT_LABELS" = true ]; then
    SOFT_LABELS_ARG="--use_soft_labels"
else
    SOFT_LABELS_ARG="--no_soft_labels"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_pretrain_smmerl.py \
    --num_devices $NUM_DEVICES \
    --train_data_pct 1 \
    --ecg_encoder_name $ECG_ENCODER_NAME \
    --lead_group_strategy $LEAD_GROUP_STRATEGY \
    --pool_type $POOL_TYPE \
    --text_encoder_name $TEXT_ENCODER_NAME \
    $SOFT_LABELS_ARG \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs $MAX_EPOCHS

echo ""
echo "=============================================="
echo "预训练完成!"
echo "结束时间: $(date)"
echo "=============================================="

/usr/bin/shutdown