"""
Utility Functions for DiffiT

Configuration management, logging, and other utilities.
"""

from .config import load_config, save_config, merge_configs
from .logging import setup_logging
from .device import get_device, fix_device_mismatch

__all__ = [
    # Configuration
    "load_config",
    "save_config", 
    "merge_configs",
    
    # Logging
    "setup_logging",
    
    # Device management
    "get_device",
    "fix_device_mismatch",
]
