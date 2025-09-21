"""
LoRA Configuration

EXACT preservation of original algorithms from diffit_blockwise_lora_finetuning.py
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """
    LoRA Configuration class for type safety and validation
    """
    enabled: bool = True
    alpha: float = 1.0
    dropout: float = 0.0
    targets: list = field(default_factory=list)
    include_WK: bool = True
    encoder: Dict[str, Any] = field(default_factory=dict)
    decoder: Dict[str, Any] = field(default_factory=dict)
    ushape: Dict[str, Any] = field(default_factory=dict)
    latent: Dict[str, int] = field(default_factory=dict)
    time_embedding: Dict[str, int] = field(default_factory=dict)
    tokenizer: Dict[str, int] = field(default_factory=dict)
    head: Dict[str, int] = field(default_factory=dict)
    default_rank: int = 8
    freeze_others: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create LoRAConfig from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert LoRAConfig to dictionary"""
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": self.targets,
            "include_WK": self.include_WK,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "ushape": self.ushape,
            "latent": self.latent,
            "time_embedding": self.time_embedding,
            "tokenizer": self.tokenizer,
            "head": self.head,
            "default_rank": self.default_rank,
            "freeze_others": self.freeze_others,
        }


# Block-wise LoRA Configuration for Actual DiffiT Architecture
# EXACT preservation from original implementation
LORA_CONFIG = {
    "enabled": True,
    "alpha": 1.0,          # LoRA scaling factor
    "dropout": 0.0,        # LoRA dropout for regularization

    # Target layers for LoRA adaptation - updated for actual DiffiT architecture
    "targets": [
        # TMSA (Temporal-Spatial Multi-head Self-Attention) components
        "Wqs", "Wks", "Wvs",     # Spatial attention projections
        "Wqt", "Wkt", "Wvt",     # Temporal attention projections
        "wo",                     # Output projection
        "WK",                     # Bias projection in TMSA

        # MLP components
        "linear_1", "linear_2",   # Feed-forward layers

        # Additional linear layers in actual DiffiT
        "time_embedding_mlp.1",   # Time embedding MLP layers
        "time_embedding_mlp.3",
        "embedding_layer",        # Label embedding (for latent model)
        "linear_layer",           # Label embedding linear layer

        # Convolutional layers (treated as linear for LoRA purposes)
        "conv3x3",               # Tokenizer and Head conv layers
        "conv",                  # Downsample/Upsample conv layers
        "proj",                  # Patch embedding projections
    ],
    "include_WK": True,           # Include WK bias projection in TMSA

    # Block-specific ranks optimized for actual DiffiT architecture
    "encoder": {
        "default_rank": 8,
        "groups": {
            1: 16,  # Early encoder layers need higher adaptation capacity
            2: 12,  # Intermediate layers
            3: 8,   # Deeper layers
            4: 8    # Deepest encoder layers
        }
    },
    "decoder": {
        "default_rank": 8,
        "groups": {
            3: 8,   # Deepest decoder layers
            2: 10,  # Intermediate decoder layers
            1: 12   # Final decoder layers need good reconstruction
        }
    },
    "ushape": {
        "default_rank": 8,
        "groups": {
            1: 12,  # First U-shaped group (skip connections important)
            2: 10,  # Second U-shaped group
            3: 8    # Third U-shaped group
        }
    },
    "latent": {"default_rank": 16},       # Critical bottleneck needs high capacity
    "time_embedding": {"default_rank": 4}, # Time embeddings need less adaptation
    "tokenizer": {"default_rank": 6},     # Input tokenization
    "head": {"default_rank": 8},          # Output head

    "default_rank": 8,        # Fallback rank
    "freeze_others": True     # Freeze non-LoRA parameters
}
