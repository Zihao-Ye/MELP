#!/bin/bash

# MVCSE-MSSATE 预训练脚本
# 使用方法: bash pretrain_mvcse_mssate.sh

# ============ 配置区域 ============
# 导联分组策略: none, limb_chest, lisa
LEAD_GROUP_STRATEGY="lisa"

# 多尺度Pooler开关 (消融实验)
# true: 使用多尺度 (wave/beat/rhythm)
# false: 不使用多尺度 (消融实验)
USE_MULTISCALE_POOLER=true

# ResNet前端选择 (消融实验)
# resnet18: 标准ResNet18 (默认)
# resnet34: 更深的ResNet34
RESNET_TYPE="resnet34"

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
echo "MVCSE-MSSATE 预训练"
echo "=============================================="
echo "Lead Group Strategy: $LEAD_GROUP_STRATEGY"
echo "Use Multiscale Pooler: $USE_MULTISCALE_POOLER"
echo "ResNet Type: $RESNET_TYPE"
echo "Batch Size: $BATCH_SIZE x $ACCUMULATE_GRAD_BATCHES (acc) x $NUM_DEVICES (gpus) = $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * NUM_DEVICES))"
echo "Learning Rate: $LR"
echo "Max Epochs: $MAX_EPOCHS"
echo "开始时间: $(date)"
echo "=============================================="

cd /root/MELP/scripts/pretrain

# 根据USE_MULTISCALE_POOLER设置参数
if [ "$USE_MULTISCALE_POOLER" = true ]; then
    MULTISCALE_ARG="--use_multiscale_pooler"
else
    MULTISCALE_ARG="--no_multiscale_pooler"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python main_pretrain_mvcse_mssate.py \
    --num_devices $NUM_DEVICES \
    --train_data_pct 1 \
    --ecg_encoder_name hierarchical_mvcse_mssate_base \
    --lead_group_strategy $LEAD_GROUP_STRATEGY \
    $MULTISCALE_ARG \
    --resnet_type $RESNET_TYPE \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --no_diagnosis_generator

echo ""
echo "=============================================="
echo "预训练完成!"
echo "结束时间: $(date)"
echo "=============================================="
