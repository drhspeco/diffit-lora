"""
LoRA Utilities

EXACT preservation of original algorithms from diffit_blockwise_lora_finetuning.py
"""

import torch
import torch.nn as nn
from typing import Dict

from .core import LoRALinear


def calculate_lora_parameters(model: nn.Module) -> Dict:
    """
    Calculate LoRA parameter statistics
    
    EXACT preservation from original implementation.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and "lora_" in n)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "trainable_ratio": trainable_params / total_params * 100,
        "lora_ratio": lora_params / total_params * 100
    }


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA parameters
    
    EXACT preservation from original implementation.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.data
            state[f"{name}.lora_B"] = module.lora_B.data
    torch.save(state, path)
    print(f"💾 LoRA weights saved to {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = False):
    """
    Load LoRA parameters
    
    EXACT preservation from original implementation.
    """
    ckpt = torch.load(path, map_location="cpu")
    misses = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in ckpt:
                module.lora_A.data = ckpt[f"{name}.lora_A"]
            else:
                misses.append(f"{name}.lora_A")

            if f"{name}.lora_B" in ckpt:
                module.lora_B.data = ckpt[f"{name}.lora_B"]
            else:
                misses.append(f"{name}.lora_B")

    if strict and misses:
        raise RuntimeError(f"Missing LoRA weights: {misses}")
    elif misses:
        print(f"⚠️ Warning: Missing LoRA weights: {len(misses)} parameters")

    print(f"📥 LoRA weights loaded from {path}")


def fuse_all_lora(model: nn.Module):
    """
    Fuse all LoRA weights into base model for deployment
    
    EXACT preservation from original implementation.
    """
    fused_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.fuse()
            fused_count += 1
    print(f"🔗 Fused {fused_count} LoRA modules into base weights")
