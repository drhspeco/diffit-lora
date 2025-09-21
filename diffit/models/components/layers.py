"""
Basic Neural Network Layers for DiffiT

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormalization(nn.Module):
    """
    Custom layer normalization implementation
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplies
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation
    
    EXACT preservation from original implementation.
    """
    
    def __init__(self, img_size: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(img_size, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, img_size)

    def forward(self, x):
        out_linear_1 = self.linear_1(x)
        out_gelu = self.gelu(out_linear_1)
        out_linear_2 = self.linear_2(out_gelu)
        return out_linear_2
