#!/bin/bash

# MERL Zero-shot 评估脚本
# 使用方法: bash merl_zeroshot.sh

# ============ 配置区域 ============
# 设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/melp/ckpts/melp_merl_2026_01_09_15_09_26/epoch=5-step=17472.ckpt"

# 是否计算置信区间 (会增加运行时间)
COMPUTE_CI=false

# 是否保存结果到CSV
SAVE_RESULTS=true
# ==================================

echo "=============================================="
echo "MERL Zero-shot 评估"
echo "=============================================="
echo "Checkpoint: $CKPT_PATH"
echo "Compute CI: $COMPUTE_CI"
echo "Save Results: $SAVE_RESULTS"
echo "开始时间: $(date)"
echo "=============================================="

# 构建命令参数
CMD="python test_zeroshot.py \
    --model_name merl \
    --ckpt_path $CKPT_PATH \
    --batch_size 128 \
    --num_workers 4 \
    --test_sets ptbxl_super_class ptbxl_sub_class ptbxl_form ptbxl_rhythm icbeb chapman"

if [ "$COMPUTE_CI" = true ]; then
    CMD="$CMD --compute_ci"
fi

if [ "$SAVE_RESULTS" = true ]; then
    CMD="$CMD --save_results"
fi

# 运行评估
echo ""
echo "执行命令: $CMD"
echo ""

CUDA_VISIBLE_DEVICES=0 $CMD

echo ""
echo "=============================================="
echo "评估完成!"
echo "结束时间: $(date)"
echo "=============================================="
