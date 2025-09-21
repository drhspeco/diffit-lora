"""
Diffusion Utilities

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch


def extract(a, t, x_shape):
    """
    Extract values from tensor a at timesteps t
    
    EXACT preservation from original implementation.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def causal_mask(size, device=None):
    """
    Create a causal mask for attention mechanism
    
    EXACT preservation from original implementation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int).to(device)
    return mask == 0
