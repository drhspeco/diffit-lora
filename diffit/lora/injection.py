"""
LoRA Injection Utilities

EXACT preservation of original algorithms from diffit_blockwise_lora_finetuning.py
"""

import re
import torch.nn as nn
from typing import Dict, List

from .core import LoRALinear


def match_target_linear_name(name: str, targets: List[str]) -> bool:
    """
    Check if module name matches target linear layers
    
    EXACT preservation from original implementation.
    """
    last = name.split(".")[-1]
    # Also check for partial matches in the full path for complex modules
    return last in targets or any(target in name for target in targets)


def rank_for_module_path(module_path: str, cfg: Dict) -> int:
    """
    Determine LoRA rank based on module path and configuration
    Enhanced for actual DiffiT architecture with proper component recognition
    
    EXACT preservation from original implementation.
    """

    # Time embedding components
    if "time_embedding" in module_path:
        return cfg.get("time_embedding", {}).get("default_rank", 4)

    # Tokenizer (input processing)
    if "tokenizer" in module_path:
        return cfg.get("tokenizer", {}).get("default_rank", 6)

    # Head (output processing)
    if "head" in module_path:
        return cfg.get("head", {}).get("default_rank", 8)

    # Encoder groups: diffit_res_block_group_1..4
    m = re.search(r"encoder\.diffit_res_block_group_(\d+)", module_path)
    if m:
        group_id = int(m.group(1))
        return cfg.get("encoder", {}).get("groups", {}).get(group_id,
               cfg.get("encoder", {}).get("default_rank", 8))

    # Decoder groups: diffit_res_block_group_1..3 (in decoder)
    m = re.search(r"decoder\.diffit_res_block_group_(\d+)", module_path)
    if m:
        group_id = int(m.group(1))
        return cfg.get("decoder", {}).get("groups", {}).get(group_id,
               cfg.get("decoder", {}).get("default_rank", 8))

    # U-Shaped root groups (for UShapedNetwork)
    m = re.search(r"(^|\.)diffit_res_block_group_(\d+)(\.|$)", module_path)
    if m:
        group_id = int(m.group(2))
        return cfg.get("ushape", {}).get("groups", {}).get(group_id,
               cfg.get("ushape", {}).get("default_rank", 8))

    # Latent block - critical for latent-space models
    if "latent_block" in module_path:
        return cfg.get("latent", {}).get("default_rank", 16)

    # Patch embedding and unpatch operations
    if "patch_embedding" in module_path or "unpatchify" in module_path:
        return cfg.get("latent", {}).get("default_rank", 12)

    # Label embedding (for latent-space models)
    if "label_embedding" in module_path:
        return cfg.get("time_embedding", {}).get("default_rank", 4)

    # Downsample/Upsample layers
    if "downsample" in module_path or "upsample" in module_path:
        return max(4, cfg.get("default_rank", 8) // 2)  # Lower rank for spatial operations

    # Fallback to default
    return cfg.get("default_rank", 8)


def inject_blockwise_lora(model: nn.Module, cfg: Dict):
    """
    Inject Block-wise LoRA into the actual DiffiT model
    Enhanced to handle real DiffiT architecture components

    Args:
        model: The DiffiT model to modify (UShapedNetwork or LatentDiffiTNetwork)
        cfg: Configuration dictionary with LoRA settings
        
    Returns:
        List of (module_path, rank) tuples for successfully replaced modules
        
    EXACT preservation from original implementation.
    """
    targets = cfg.get("targets", ["Wqs", "Wks", "Wvs", "Wqt", "Wkt", "Wvt", "wo"])
    include_WK = cfg.get("include_WK", False)
    if include_WK and "WK" not in targets:
        targets.append("WK")

    alpha = cfg.get("alpha", 1.0)
    dropout = cfg.get("dropout", 0.0)

    replacements = []
    skipped = []

    def replace_in(parent: nn.Module, prefix: str):
        for child_name, child_module in parent.named_children():
            full_path = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child_module, nn.Linear) and match_target_linear_name(child_name, targets):
                # Determine rank for this specific module
                rank = rank_for_module_path(full_path, cfg)

                # Skip very small layers or those with insufficient dimensions
                min_dim = min(child_module.in_features, child_module.out_features)
                if rank >= min_dim:
                    rank = max(1, min_dim // 2)  # Adjust rank to be feasible
                    if rank < 2:  # Skip if still too small
                        skipped.append((full_path, f"rank {rank} too small"))
                        continue

                try:
                    # Create LoRA wrapper
                    lora_module = LoRALinear(child_module, rank, alpha, dropout)
                    lora_module.reset_parameters()

                    # Replace the module
                    setattr(parent, child_name, lora_module)
                    replacements.append((full_path, rank))

                except Exception as e:
                    skipped.append((full_path, f"error: {e}"))

            else:
                # Recursively process children
                replace_in(child_module, full_path)

    replace_in(model, "")

    # Freeze non-LoRA parameters
    if cfg.get("freeze_others", True):
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    print(f"✅ Block-wise LoRA injection complete!")
    print(f"📊 Successfully replaced {len(replacements)} linear layers:")
    for path, rank in replacements[:10]:  # Show first 10
        print(f"   • {path} -> rank {rank}")
    if len(replacements) > 10:
        print(f"   ... and {len(replacements) - 10} more")

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} layers:")
        for path, reason in skipped[:5]:  # Show first 5 skipped
            print(f"   • {path}: {reason}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped) - 5} more skipped")

    return replacements
