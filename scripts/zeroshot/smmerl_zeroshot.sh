#!/bin/bash

# SMMERL Zero-shot 评估脚本
# 使用方法: bash smmerl_zeroshot.sh

# ============ 配置区域 ============
# 设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/smmerl/ckpts/smmerl_smmerl_base_2026_01_27_05_43_13/epoch=1-step=5824.ckpt"

# 多尺度融合模式: concat (拼接三个尺度), mean (平均三个尺度), rhythm (仅节律尺度)
MODE="concat"

# 是否计算置信区间 (会增加运行时间)
COMPUTE_CI=false

# 是否保存结果到CSV
SAVE_RESULTS=true
# ==================================

echo "=============================================="
echo "SMMERL Zero-shot 评估"
echo "=============================================="
echo "Checkpoint: $CKPT_PATH"
echo "Mode: $MODE"
echo "Compute CI: $COMPUTE_CI"
echo "Save Results: $SAVE_RESULTS"
echo "开始时间: $(date)"
echo "=============================================="

# 构建命令参数
CMD="python test_zeroshot_smmerl.py \
    --ckpt_path $CKPT_PATH \
    --mode $MODE \
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
