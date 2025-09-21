"""
Device Management Utilities

Device detection and management for DiffiT models.
"""

import torch
import torch.nn as nn
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device for computation
    
    Args:
        device: Specific device string (e.g., 'cuda', 'cpu'), or None for auto-detection
        
    Returns:
        PyTorch device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def fix_device_mismatch(model: nn.Module, target_device: Optional[torch.device] = None) -> int:
    """
    Fix device mismatches in model parameters
    
    Args:
        model: PyTorch model to fix
        target_device: Target device (auto-detect if None)
        
    Returns:
        Number of parameters moved
    """
    if target_device is None:
        target_device = next(model.parameters()).device
    
    fixed_count = 0
    
    for name, param in model.named_parameters():
        if param.device != target_device:
            param.data = param.data.to(target_device)
            fixed_count += 1
    
    for name, buffer in model.named_buffers():
        if buffer.device != target_device:
            buffer.data = buffer.data.to(target_device)
            fixed_count += 1
    
    return fixed_count
