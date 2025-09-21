"""
DiffiT Model Components

Core building blocks for DiffiT architectures with exact algorithm preservation.
"""

from .layers import LayerNormalization, MLP
from .embeddings import TimeEmbedding, LabelEmbedding, SinusoidalPositionEmbeddings
from .attention import TMSA
from .blocks import DiffiTBlock, DiffiTResBlock, ResBlockGroup
from .spatial import Tokenizer, Head, Downsample, Upsample

__all__ = [
    # Basic layers
    "LayerNormalization",
    "MLP",
    
    # Embeddings
    "TimeEmbedding",
    "LabelEmbedding", 
    "SinusoidalPositionEmbeddings",
    
    # Attention
    "TMSA",
    
    # Blocks
    "DiffiTBlock",
    "DiffiTResBlock",
    "ResBlockGroup",
    
    # Spatial operations
    "Tokenizer",
    "Head",
    "Downsample",
    "Upsample",
]
