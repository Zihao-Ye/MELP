"""
多尺度 ECG 编码器模块

Multi-Scale Patch Embedding + Temporal Encoding

使用标准ResNet18（4个stage，512通道输出）作为前端特征提取器。
移除LISA分组，专注验证多尺度Patch的效果。

主要组件:
- MVCSEEncoder: 空间编码器（标准ResNet18 + 多尺度Patch + 导联级Transformer + Cross-Lead聚合）
- MSSATEEncoder: 时序编码器（多尺度Transformer）
- MVCSEMSSATEEncoder: 完整编码器

预定义配置:
- mvcse_mssate_tiny: ~8M参数，快速实验
- mvcse_mssate_small: ~18M参数，平衡性能
- mvcse_mssate_base: ~30M参数，标准配置
- mvcse_mssate_large: ~50M参数，大容量

使用示例:
```python
from melp.backbone.mvcse_mssate import mvcse_mssate_base

# 创建编码器
encoder = mvcse_mssate_base(seq_len=5000, output_dim=256)

# 前向传播
x = torch.randn(2, 12, 5000)  # (batch, 12导联, 采样点)
output = encoder(x)  # (2, 256)

# 获取中间特征
features = encoder.forward_features(x)
```
"""

# 基础模块
from .base_modules import (
    DropPath,
    FeedForward,
    SEBlock,
    ECABlock
)

# 位置编码
from .positional_encoding import RelativePositionalEncoding

# 注意力模块
from .attention import (
    RelativeAttention,
    CrossLeadAggregation,
    CrossLeadTransformerBlock,
    AttentionalPooler,
    HierarchicalECGPooler
)

# Transformer模块
from .transformer import (
    TransformerBlock,
    LeadTransformer
)

# Patch Embedding
from .patch_embedding import MultiScalePatchEmbedding

# ResNet Frontend
from .resnet_frontend import (
    BasicBlock,
    Bottleneck,
    ResNetFrontend,
    resnet18_frontend,
    resnet34_frontend,
    resnet50_frontend
)

# 编码器
from .mvcse import MVCSEEncoder
from .mssate import MSSATEEncoder
from .encoder import (
    AttentionPool1d,
    MVCSEMSSATEEncoder,
    HierarchicalMVCSEMSSATEEncoder,
    mvcse_mssate_tiny,
    mvcse_mssate_small,
    mvcse_mssate_base,
    mvcse_mssate_large,
    hierarchical_mvcse_mssate_small,
    hierarchical_mvcse_mssate_base,
    hierarchical_mvcse_mssate_large
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
    'AttentionalPooler',
    'HierarchicalECGPooler',
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
    # Pooling
    'AttentionPool1d',
    # 编码器
    'MVCSEEncoder',
    'MSSATEEncoder',
    'MVCSEMSSATEEncoder',
    'HierarchicalMVCSEMSSATEEncoder',
    # 预定义配置 (Conv1d硬切分版)
    'mvcse_mssate_tiny',
    'mvcse_mssate_small',
    'mvcse_mssate_base',
    'mvcse_mssate_large',
    # 预定义配置 (Query软切分版)
    'hierarchical_mvcse_mssate_small',
    'hierarchical_mvcse_mssate_base',
    'hierarchical_mvcse_mssate_large',
]
