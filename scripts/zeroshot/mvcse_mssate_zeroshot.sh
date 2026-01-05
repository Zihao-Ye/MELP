#!/bin/bash

# MVCSE-MSSATE Zero-shot 评估脚本
# 使用方法: bash mvcse_mssate_zeroshot.sh

# ============ 配置区域 ============
# 设置预训练模型的checkpoint路径
CKPT_PATH="~/autodl-tmp/logs/mvcse_mssate/ckpts/mvcse_mssate_hierarchical_mvcse_mssate_base_2026_01_05_07_54_30/epoch=1-step=5824.ckpt"

# 多尺度融合模式: concat (拼接三个尺度), mean (三个尺度平均), rhythm (仅使用rhythm尺度)
FUSION_MODE="concat"

# 是否计算置信区间 (会增加运行时间)
COMPUTE_CI=true

# 是否保存结果到CSV
SAVE_RESULTS=true
# ==================================

echo "=============================================="
echo "MVCSE-MSSATE Zero-shot 评估"
echo "=============================================="
echo "Checkpoint: $CKPT_PATH"
echo "Fusion Mode: $FUSION_MODE"
echo "Compute CI: $COMPUTE_CI"
echo "Save Results: $SAVE_RESULTS"
echo "开始时间: $(date)"
echo "=============================================="

# 构建命令参数
CMD="python test_zeroshot_mvcse_mssate.py \
    --ckpt_path $CKPT_PATH \
    --fusion_mode $FUSION_MODE \
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
