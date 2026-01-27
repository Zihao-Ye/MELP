"""
SMMERL: Simplified Multi-scale Multi-lead ECG Representation Learning

A clean, modular implementation of the improved ECG encoder architecture:
1. ResNet frontend for feature extraction
2. Full-sequence LeadTransformer for temporal modeling
3. Multi-scale attention pooling (wave/beat/rhythm)
4. LISA anatomical grouping
5. Cross-lead aggregation

Key improvements over MVCSE-MSSATE:
- Temporal modeling on full sequence (313 tokens) before multi-scale extraction
- Cleaner architecture without excessive ablation parameters
- Better separation of concerns
- Easier to understand and modify
"""

from .encoder import (
    SMMERLEncoder,
    smmerl_tiny,
    smmerl_small,
    smmerl_base,
    smmerl_large
)

__all__ = [
    'SMMERLEncoder',
    'smmerl_tiny',
    'smmerl_small',
    'smmerl_base',
    'smmerl_large',
]
