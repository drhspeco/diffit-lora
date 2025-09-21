"""
DiffiT-LoRA: Diffusion Vision Transformers with LoRA Fine-tuning

A comprehensive implementation of DiffiT (Diffusion Vision Transformers) with 
Low-Rank Adaptation (LoRA) for efficient fine-tuning.
"""

__version__ = "0.1.0"
__author__ = "DiffiT Team"
__email__ = "contact@diffit.ai"

# Core model imports
from .models import UShapedNetwork
# from .models import LatentDiffiTNetwork  # TODO: Implement when needed
from .models.components import TMSA, DiffiTBlock, TimeEmbedding

# LoRA imports
from .lora import LoRALinear, inject_blockwise_lora, LoRAConfig
from .lora import save_lora_weights, load_lora_weights, fuse_all_lora

# Diffusion imports
from .diffusion import (
    linear_beta_schedule,
    cosine_beta_schedule,
    q_sample,
    p_sample,
    p_sample_loop,
    sample,
    p_losses,
)

# Training imports
from .training import DiffiTTrainer

# Evaluation imports
from .evaluation import calculate_fid, calculate_kid, calculate_lpips
from .evaluation import visualize_samples, plot_training_metrics

# Utility imports
from .utils import load_config, save_config, setup_logging

__all__ = [
    # Models
    "UShapedNetwork",
    # "LatentDiffiTNetwork",  # TODO: Implement when needed
    "TMSA",
    "DiffiTBlock",
    "TimeEmbedding",
    
    # LoRA
    "LoRALinear",
    "inject_blockwise_lora",
    "LoRAConfig",
    "save_lora_weights",
    "load_lora_weights",
    "fuse_all_lora",
    
    # Diffusion
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "q_sample",
    "p_sample", 
    "p_sample_loop",
    "sample",
    "p_losses",
    
    # Training
    "DiffiTTrainer",
    
    # Evaluation
    "calculate_fid",
    "calculate_kid", 
    "calculate_lpips",
    "visualize_samples",
    "plot_training_metrics",
    
    # Utils
    "load_config",
    "save_config",
    "setup_logging",
]
