"""
MVCSE-MSSATE ECG编码器 - 兼容性导入模块

此文件保留用于向后兼容。新代码请直接从 mvcse_mssate 包导入:

```python
from melp.backbone.mvcse_mssate import (
    MVCSEMSSATEEncoder,
    mvcse_mssate_base,
    mvcse_mssate_tiny,
    mvcse_mssate_small,
    mvcse_mssate_large,
)
```

架构说明:
- MVCSE: 基于LISA解剖分组的空间编码
- MS-SATE: 多尺度时序编码

方案B配置（默认）:
- 导联级Transformer: 6层
- MS-SATE Transformer: 2层
"""

# 从新模块导入所有组件
from .mvcse_mssate import (
    # 基础模块
    DropPath,
    FeedForward,
    SEBlock,
    ECABlock,
    # 位置编码
    RelativePositionalEncoding,
    # 注意力模块
    RelativeAttention,
    CrossLeadAggregation,
    CrossLeadTransformerBlock,
    # Transformer模块
    TransformerBlock,
    LeadTransformer,
    # Patch Embedding
    MultiScalePatchEmbedding,
    # ResNet Frontend
    BasicBlock,
    Bottleneck,
    ResNetFrontend,
    resnet18_frontend,
    resnet34_frontend,
    resnet50_frontend,
    # 编码器
    MVCSEEncoder,
    MSSATEEncoder,
    MVCSEMSSATEEncoder,
    # 预定义配置
    mvcse_mssate_tiny,
    mvcse_mssate_small,
    mvcse_mssate_base,
    mvcse_mssate_large,
)

__all__ = [
    # 基础模块
    'DropPath',
    'FeedForward',
    'SEBlock',
    'ECABlock',
    # 位置编码
    'RelativePositionalEncoding',
    # 注意力模块
    'RelativeAttention',
    'CrossLeadAggregation',
    'CrossLeadTransformerBlock',
    # Transformer模块
    'TransformerBlock',
    'LeadTransformer',
    # Patch Embedding
    'MultiScalePatchEmbedding',
    # ResNet Frontend
    'BasicBlock',
    'Bottleneck',
    'ResNetFrontend',
    'resnet18_frontend',
    'resnet34_frontend',
    'resnet50_frontend',
    # 编码器
    'MVCSEEncoder',
    'MSSATEEncoder',
    'MVCSEMSSATEEncoder',
    # 预定义配置
    'mvcse_mssate_tiny',
    'mvcse_mssate_small',
    'mvcse_mssate_base',
    'mvcse_mssate_large',
]


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    import torch

    print("Testing MVCSE-MSSATE Encoder (New Architecture)...")
    print("=" * 60)

    # 创建模型
    model = mvcse_mssate_base(seq_len=5000, output_dim=256)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 12, 5000)  # (B, 12导联, 采样点)

    with torch.no_grad():
        output = model(x)
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # 测试详细特征输出
        features = model.forward_features(x)
        print("\nIntermediate features:")
        for k, v in features.items():
            if isinstance(v, list):
                print(f"  {k}: List of {len(v)} items")
                for i, t in enumerate(v):
                    if isinstance(t, torch.Tensor):
                        print(f"    [{i}]: {t.shape}")
            elif isinstance(v, dict):
                print(f"  {k}: Dict with {len(v)} groups")
                for group_name, group_v in v.items():
                    if isinstance(group_v, list) and len(group_v) > 0:
                        print(f"    {group_name}: {len(group_v)} scales, first shape: {group_v[0].shape if isinstance(group_v[0], torch.Tensor) else 'N/A'}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")

    print("\n" + "=" * 60)
    print("Test passed!")
