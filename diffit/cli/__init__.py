"""
Command Line Interface for DiffiT

Simple CLI tools for training, fine-tuning, and generation.
"""

from .train import main as train_main
from .finetune import main as finetune_main
from .generate import main as generate_main

__all__ = [
    "train_main",
    "finetune_main", 
    "generate_main",
]
