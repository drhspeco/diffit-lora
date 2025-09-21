"""
Training CLI for DiffiT

Command-line interface for base model training.
"""

import argparse
import os
from ..training import DiffiTTrainer
from ..utils import setup_logging


def main():
    """Main training CLI function"""
    parser = argparse.ArgumentParser(description="Train DiffiT base model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    try:
        # Create trainer
        trainer = DiffiTTrainer(args.config)
        
        # Train model
        trainer.fit()
        
        # Test model
        trainer.test()
        
        print("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
