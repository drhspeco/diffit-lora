"""
Generation CLI for DiffiT

Command-line interface for sample generation.
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from ..models import UShapedNetwork
# from ..models import LatentDiffiTNetwork  # TODO: Implement when needed
from ..diffusion import sample
from ..utils import setup_logging


def save_images(images, output_dir: str, prefix: str = "sample"):
    """Save generated images to directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_array in enumerate(images):
        # Convert from numpy array to PIL Image
        # Assuming img_array is in [-1, 1] range
        img_array = (img_array + 1) / 2  # Convert to [0, 1]
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert from CHW to HWC format
        if len(img_array.shape) == 3:
            img_array = img_array.transpose(1, 2, 0)
        
        # Create PIL Image
        img = Image.fromarray(img_array)
        
        # Save image
        img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        img.save(img_path)
    
    print(f"💾 Saved {len(images)} images to {output_dir}")


def main():
    """Main generation CLI function"""
    parser = argparse.ArgumentParser(description="Generate samples with DiffiT")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["image-space", "latent-space"],
        default="image-space",
        help="Type of DiffiT model"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Size of generated images"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Number of diffusion timesteps"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_samples/",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, mps, or auto)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model dimension (default: 128)"
    )
    
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    try:
        # Load model
        print(f"📥 Loading model from {args.checkpoint}")
        
        if args.model_type == "image-space":
            # Load checkpoint and extract hyperparameters
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Create model with parameters (these should match training config)
            model = UShapedNetwork(
                learning_rate=0.001,  # Not used during inference
                d_model=args.d_model,
                num_heads=args.num_heads,
                dropout=0.1,  # Not used during inference
                d_ff=args.d_model * 2,  # Standard: 2 * d_model
                img_size=args.image_size,
                device=device,
                denoising_steps=args.timesteps,
                L1=2, L2=2, L3=2, L4=2  # Standard ResBlock group sizes
            )
            
            # Load the state dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        elif args.model_type == "latent-space":
            # TODO: Implement LatentDiffiTNetwork when needed
            raise NotImplementedError(
                "LatentDiffiTNetwork is not yet implemented. "
                "Currently only UShapedNetwork (image-space) is available. "
                "Please use model type 'image-space'."
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        
        # Generate samples
        print(f"🎨 Generating {args.num_samples} samples...")
        
        # Calculate number of batches needed
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        all_samples = []
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
                
                print(f"📊 Generating batch {batch_idx + 1}/{num_batches} ({batch_size} samples)")
                
                # Generate samples for this batch
                batch_samples = sample(
                    model=model,
                    image_size=args.image_size,
                    batch_size=batch_size,
                    channels=3,
                    timesteps=args.timesteps
                )
                
                # Take the final samples (last timestep)
                final_samples = batch_samples[-1]  # Shape: (batch_size, 3, H, W)
                all_samples.extend(final_samples)
        
        print(f"✅ Generated {len(all_samples)} samples")
        
        # Save images
        save_images(all_samples, args.output_dir, "diffit_sample")
        
        print("🎉 Generation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
