"""
DiffiT Model Architectures

This module contains all DiffiT model architectures with exact algorithm preservation.
"""

from .unet import UShapedNetwork
# from .latent import LatentDiffiTNetwork  # TODO: Implement when needed
from .components import (
    TMSA,
    DiffiTBlock,
    DiffiTResBlock,
    ResBlockGroup,
    TimeEmbedding,
    LabelEmbedding,
    LayerNormalization,
    MLP,
    Tokenizer,
    Head,
    Downsample,
    Upsample,
    SinusoidalPositionEmbeddings,
)

__all__ = [
    # Main models
    "UShapedNetwork",
    # "LatentDiffiTNetwork",  # TODO: Implement when needed
    
    # Core components
    "TMSA",
    "DiffiTBlock", 
    "DiffiTResBlock",
    "ResBlockGroup",
    "TimeEmbedding",
    "LabelEmbedding",
    "LayerNormalization",
    "MLP",
    "Tokenizer",
    "Head",
    "Downsample",
    "Upsample",
    "SinusoidalPositionEmbeddings",
]
