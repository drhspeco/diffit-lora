#!/usr/bin/env python3
"""
Complete Base Model Training Script

Train a DiffiT base model from scratch using the restructured codebase.
"""

import argparse
import sys
import os
from pathlib import Path

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffit.training import DiffiTTrainer
from diffit.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train DiffiT base model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/base_training.yaml",
        help="Path to training configuration YAML file"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/models/unet_config.yaml",
        help="Path to model configuration YAML file"
    )
    
    parser.add_argument(
        "--data-config", 
        type=str,
        default="configs/data/cifar10.yaml",
        help="Path to data configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    print("🚀 Starting DiffiT Base Model Training")
    print("=" * 50)
    print(f"Training config: {args.config}")
    print(f"Model config: {args.model_config}")
    print(f"Data config: {args.data_config}")
    print("=" * 50)
    
    try:
        # Create and run trainer
        trainer = DiffiTTrainer(args.config)
        
        print("\n📊 Model Summary:")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        
        # Train the model
        trainer.fit()
        
        # Test the model
        trainer.test()
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Checkpoints saved to: {trainer.config['training']['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
