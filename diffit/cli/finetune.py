"""
LoRA Fine-tuning CLI for DiffiT

Command-line interface for LoRA fine-tuning.
"""

import argparse
import os
from ..training import LoRAFineTuner
from ..utils import setup_logging


def main():
    """Main fine-tuning CLI function"""
    parser = argparse.ArgumentParser(description="Fine-tune DiffiT with LoRA")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to fine-tuning configuration YAML file"
    )
    
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Path to base model checkpoint (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for fine-tuned model (overrides config)"
    )
    
    parser.add_argument(
        "--save-lora-only",
        action="store_true",
        help="Save only LoRA weights instead of full checkpoint"
    )
    
    parser.add_argument(
        "--fuse-lora",
        action="store_true",
        help="Fuse LoRA weights into base model for deployment"
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
        # Create fine-tuner
        fine_tuner = LoRAFineTuner(args.config)
        
        # Override base checkpoint if provided
        if args.base_checkpoint:
            if os.path.exists(args.base_checkpoint):
                fine_tuner.load_pretrained(args.base_checkpoint)
            else:
                print(f"❌ Base checkpoint not found: {args.base_checkpoint}")
                return 1
        
        # Fine-tune model
        fine_tuner.fit()
        
        # Test fine-tuned model
        fine_tuner.test()
        
        # Save model based on options
        output_dir = args.output_dir or fine_tuner.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        if args.save_lora_only:
            # Save only LoRA weights
            lora_path = os.path.join(output_dir, "lora_weights.pth")
            fine_tuner.save_lora_weights(lora_path)
        else:
            # Save full checkpoint
            checkpoint_path = os.path.join(output_dir, "finetuned_model.ckpt")
            fine_tuner.save_model(checkpoint_path)
        
        if args.fuse_lora:
            # Fuse LoRA and save
            fine_tuner.fuse_lora()
            fused_path = os.path.join(output_dir, "fused_model.ckpt")
            fine_tuner.save_model(fused_path)
        
        print("🎉 Fine-tuning completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
