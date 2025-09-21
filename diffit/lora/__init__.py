"""
LoRA (Low-Rank Adaptation) Implementation for DiffiT

EXACT preservation of original algorithms from diffit_blockwise_lora_finetuning.py
"""

from .core import LoRALinear
from .injection import inject_blockwise_lora, rank_for_module_path, match_target_linear_name
from .config import LoRAConfig, LORA_CONFIG
from .utils import (
    save_lora_weights,
    load_lora_weights,
    fuse_all_lora,
    calculate_lora_parameters,
)

__all__ = [
    # Core LoRA implementation
    "LoRALinear",
    
    # Injection utilities
    "inject_blockwise_lora",
    "rank_for_module_path",
    "match_target_linear_name",
    
    # Configuration
    "LoRAConfig",
    "LORA_CONFIG",
    
    # Utilities
    "save_lora_weights",
    "load_lora_weights", 
    "fuse_all_lora",
    "calculate_lora_parameters",
]
