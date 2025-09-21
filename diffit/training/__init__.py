"""
Training Pipeline for DiffiT

PyTorch Lightning-based training system with exact algorithm preservation.
"""

from .trainer import DiffiTTrainer
from .callbacks import setup_callbacks
from .data import DiffiTDataModule

__all__ = [
    "DiffiTTrainer",
    "setup_callbacks", 
    "DiffiTDataModule",
]
