"""
Core LoRA Implementation

EXACT preservation of original algorithms from diffit_blockwise_lora_finetuning.py
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for Linear layers
    Implements: output = W*x + (alpha/r)*(B@A*x)
    where W is frozen, A and B are trainable low-rank matrices
    
    EXACT preservation from original implementation.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base layer
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA parameters: A (input_dim x r), B (r x output_dim)
        self.lora_A = nn.Parameter(torch.randn(base.in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # For easy access to original dimensions
        self.in_features = base.in_features
        self.out_features = base.out_features

    def reset_parameters(self):
        """
        Initialize LoRA parameters
        
        EXACT preservation from original implementation.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass: base_output + lora_adaptation
        
        EXACT preservation from original implementation.
        """
        base_out = self.base(x)

        # LoRA adaptation: x -> A -> dropout -> B -> scale
        lora_out = self.dropout(x @ self.lora_A) @ self.lora_B * self.scaling

        return base_out + lora_out

    def fuse(self):
        """
        Fuse LoRA weights into base layer for deployment
        
        EXACT preservation from original implementation.
        """
        with torch.no_grad():
            # Compute LoRA weight: (A @ B) * scaling
            # A: (in_features, r), B: (r, out_features) -> (in_features, out_features)
            lora_weight = (self.lora_A @ self.lora_B) * self.scaling
            # Add to base weight (which is (out_features, in_features))
            self.base.weight.data += lora_weight.T

        # Zero out LoRA parameters to avoid double counting
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
