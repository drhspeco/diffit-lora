"""
DiffiT Transformer Blocks

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch
import torch.nn as nn

from .layers import LayerNormalization, MLP
from .attention import TMSA
from .embeddings import TimeEmbedding, LabelEmbedding


class DiffiTBlock(nn.Module):
    """
    Core DiffiT transformer block
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int,
                 img_size: int, label_size: int = None):
        super().__init__()
        self.ln = LayerNormalization()
        self.tmsa = TMSA(d_model, num_heads, dropout, img_size)
        self.mlp = MLP(img_size, d_ff)
        self.time_embedding = TimeEmbedding(d_model, img_size * img_size)

        # Only for latent model
        if label_size is not None:
            self.label_size = label_size
            self.label_embedding = LabelEmbedding(label_size, d_model)

    def forward(self, xs, t, l=None):
        """
        Forward pass through DiffiT block
        
        Args:
            xs: Input spatial features
            t: Time step
            l: Label (optional, for conditional generation)
            
        Returns:
            Processed features with residual connections
        """
        xt = self.time_embedding(t)
        tmsa_comb = xt

        if l is not None:
            tmsa_comb += self.label_embedding(l)

        xs1 = self.tmsa(self.ln(xs), tmsa_comb) + xs
        xs2 = self.mlp(self.ln(xs1)) + xs1

        return xs2


class DiffiTResBlock(nn.Module):
    """
    DiffiT residual block with convolution and transformer
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, d_model: int, num_heads: int,
                 dropout: float, d_ff: int, img_size: int, device, label_size: int = None):
        super().__init__()
        self.device = device
        self.seq_len = img_size * img_size

        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, padding=1)
        self.swish = nn.SiLU()
        self.group_norm = nn.GroupNorm(num_groups=in_channels // 4, num_channels=in_channels)
        self.diffit_block = DiffiTBlock(out_channels, num_heads, dropout, d_ff, img_size, label_size)

    def forward(self, xs, t, l=None):
        """
        Forward pass through DiffiT residual block
        
        Args:
            xs: Input features
            t: Time step
            l: Label (optional)
            
        Returns:
            Features processed through conv + transformer with residual connection
        """
        xs_1 = self.conv3x3(self.swish(self.group_norm(xs)))
        xs = xs + self.diffit_block(xs_1, t, l)
        return xs


class ResBlockGroup(nn.Module):
    """
    Group of residual blocks
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int, L: int,
                 in_channels: int, out_channels: int, img_size: int, device, label_size: int = None):
        super().__init__()
        self.L = L
        self.diffit_res_block = DiffiTResBlock(in_channels, out_channels, d_model, num_heads,
                                              dropout, d_ff, img_size, device, label_size)

    def forward(self, x, t, l=None):
        """
        Forward pass through L repeated residual blocks
        
        Args:
            x: Input features
            t: Time step
            l: Label (optional)
            
        Returns:
            Features processed through L residual blocks
        """
        for _ in range(self.L):
            x = self.diffit_res_block(x, t, l)
        return x
