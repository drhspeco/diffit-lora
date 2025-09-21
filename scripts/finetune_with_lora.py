#!/usr/bin/env python3
"""
Complete LoRA Fine-tuning Script

Fine-tune a DiffiT model with LoRA adapters using the restructured codebase.
"""

import argparse
import sys
import os
from pathlib import Path

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffit.training import LoRAFineTuner
from diffit.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DiffiT with LoRA")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/lora_finetuning.yaml",
        help="Path to fine-tuning configuration YAML file"
    )
    
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="./weights/ImageSpaceWeights/best_model.ckpt",
        help="Path to base model checkpoint"
    )
    
    parser.add_argument(
        "--lora-config",
        type=str,
        default="configs/lora/blockwise_config.yaml",
        help="Path to LoRA configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./weights/lora_finetuned/",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--save-lora-only",
        action="store_true",
        help="Save only LoRA weights instead of full checkpoint"
    )
    
    parser.add_argument(
        "--fuse-and-save",
        action="store_true",
        help="Also save a fused model for deployment"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    print("🚀 Starting DiffiT LoRA Fine-tuning")
    print("=" * 50)
    print(f"Training config: {args.config}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"LoRA config: {args.lora_config}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Create fine-tuner
        fine_tuner = LoRAFineTuner(args.config)
        
        # Load base checkpoint if provided and exists
        if args.base_checkpoint and os.path.exists(args.base_checkpoint):
            print(f"📥 Loading base checkpoint: {args.base_checkpoint}")
            fine_tuner.load_pretrained(args.base_checkpoint)
        elif args.base_checkpoint:
            print(f"⚠️ Base checkpoint not found: {args.base_checkpoint}")
            print("Continuing with randomly initialized weights")
        
        print(f"\n📊 LoRA Statistics:")
        print(f"  • Total parameters: {fine_tuner.lora_stats['total_parameters']:,}")
        print(f"  • LoRA parameters: {fine_tuner.lora_stats['lora_parameters']:,}")
        print(f"  • Parameter efficiency: {fine_tuner.lora_stats['lora_ratio']:.2f}%")
        print(f"  • Memory savings: {100 - fine_tuner.lora_stats['lora_ratio']:.1f}%")
        
        # Fine-tune the model
        fine_tuner.fit()
        
        # Test the fine-tuned model
        fine_tuner.test()
        
        # Save models
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.save_lora_only:
            # Save only LoRA weights (small file)
            lora_path = os.path.join(args.output_dir, "lora_weights.pth")
            fine_tuner.save_lora_weights(lora_path)
            print(f"💾 LoRA weights saved to: {lora_path}")
        else:
            # Save full checkpoint with LoRA
            checkpoint_path = os.path.join(args.output_dir, "finetuned_model.ckpt")
            fine_tuner.save_model(checkpoint_path)
            print(f"💾 Full model saved to: {checkpoint_path}")
        
        if args.fuse_and_save:
            # Create fused model for deployment
            print("🔗 Creating fused model for deployment...")
            fine_tuner.fuse_lora()
            fused_path = os.path.join(args.output_dir, "fused_model.ckpt")
            fine_tuner.save_model(fused_path)
            print(f"💾 Fused model saved to: {fused_path}")
        
        print("\n🎉 LoRA fine-tuning completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        print("\n💡 Usage Tips:")
        if args.save_lora_only:
            print("  • Use LoRA weights with: load_lora_weights(base_model, 'lora_weights.pth')")
        if args.fuse_and_save:
            print("  • Fused model can be used like any standard DiffiT model")
        print("  • LoRA adapters allow switching between different fine-tuned domains")
        
    except KeyboardInterrupt:
        print("\n⏹️ Fine-tuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
