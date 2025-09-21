#!/usr/bin/env python3
"""
Complete Sample Generation Script

Generate images using trained DiffiT models.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffit.models import UShapedNetwork, LatentDiffiTNetwork
from diffit.diffusion import sample
from diffit.utils import setup_logging


def save_images(images, output_dir: str, prefix: str = "sample"):
    """Save generated images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_array in enumerate(images):
        # Convert from [-1, 1] to [0, 255]
        img_array = (img_array + 1) / 2
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
        # Convert from CHW to HWC if needed
        if len(img_array.shape) == 3 and img_array.shape[0] == 3:
            img_array = img_array.transpose(1, 2, 0)
        
        # Save image
        img = Image.fromarray(img_array)
        img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        img.save(img_path)
    
    print(f"💾 Saved {len(images)} images to {output_dir}")


def create_image_grid(images, grid_size=(4, 4), image_size=32):
    """Create a grid of images for easy viewing"""
    rows, cols = grid_size
    grid_img = np.zeros((rows * image_size, cols * image_size, 3), dtype=np.uint8)
    
    for idx, img_array in enumerate(images[:rows * cols]):
        row = idx // cols
        col = idx % cols
        
        # Convert and resize if needed
        img_array = (img_array + 1) / 2
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 3 and img_array.shape[0] == 3:
            img_array = img_array.transpose(1, 2, 0)
        
        # Place in grid
        start_row = row * image_size
        end_row = start_row + image_size
        start_col = col * image_size
        end_col = start_col + image_size
        
        grid_img[start_row:end_row, start_col:end_col] = img_array
    
    return grid_img


def main():
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
        "--timesteps",
        type=int,
        default=500,
        help="Number of diffusion timesteps"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_samples/",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--create-grid",
        action="store_true",
        help="Create an image grid for easy viewing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    print("🚀 Starting DiffiT Sample Generation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    try:
        # Load model
        print(f"📥 Loading model from {args.checkpoint}")
        
        if args.model_type == "image-space":
            model = UShapedNetwork.load_from_checkpoint(
                args.checkpoint,
                map_location=device
            )
        elif args.model_type == "latent-space":
            model = LatentDiffiTNetwork.load_from_checkpoint(
                args.checkpoint,
                map_location=device
            )
        
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
        # Generate samples
        print(f"🎨 Generating {args.num_samples} samples...")
        
        all_samples = []
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(args.batch_size, args.num_samples - len(all_samples))
                
                print(f"  📊 Batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples)")
                
                # Generate batch
                batch_samples = sample(
                    model=model,
                    image_size=args.image_size,
                    batch_size=current_batch_size,
                    channels=3,
                    timesteps=args.timesteps
                )
                
                # Get final samples (last timestep)
                final_samples = batch_samples[-1]
                all_samples.extend(final_samples)
        
        print(f"✅ Generated {len(all_samples)} samples")
        
        # Save individual images
        save_images(all_samples, args.output_dir, "diffit_sample")
        
        # Create and save grid if requested
        if args.create_grid:
            print("🖼️ Creating image grid...")
            grid_size = int(np.ceil(np.sqrt(len(all_samples))))
            grid_img = create_image_grid(
                all_samples, 
                grid_size=(grid_size, grid_size),
                image_size=args.image_size
            )
            
            grid_path = os.path.join(args.output_dir, "sample_grid.png")
            Image.fromarray(grid_img).save(grid_path)
            print(f"💾 Grid saved to: {grid_path}")
        
        print("\n🎉 Generation completed successfully!")
        print(f"📁 Images saved to: {args.output_dir}")
        
        print("\n💡 Usage Tips:")
        print("  • Use different seeds for varied results")
        print("  • Adjust timesteps for quality vs speed trade-off")
        print("  • Lower timesteps = faster but lower quality")
        print("  • Higher timesteps = slower but better quality")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
