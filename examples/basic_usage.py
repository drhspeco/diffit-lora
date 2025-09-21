#!/usr/bin/env python3
"""
Basic Usage Example for DiffiT-LoRA

Demonstrates how to use the restructured DiffiT codebase.
"""

import sys
from pathlib import Path

# Add diffit to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from diffit.models import UShapedNetwork
from diffit.lora import inject_blockwise_lora, LORA_CONFIG, save_lora_weights, load_lora_weights
from diffit.diffusion import sample
from diffit.utils import get_device


def main():
    print("🚀 DiffiT-LoRA Basic Usage Example")
    print("=" * 50)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Create a DiffiT model
    print("\n📦 Creating DiffiT model...")
    model = UShapedNetwork(
        learning_rate=0.001,
        d_model=128,
        num_heads=2,
        dropout=0.1,
        d_ff=256,
        img_size=32,
        device=device,
        denoising_steps=500,
        L1=2, L2=2, L3=2, L4=2
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # 2. Inject LoRA adapters
    print("\n🔧 Injecting LoRA adapters...")
    replacements = inject_blockwise_lora(model, LORA_CONFIG)
    
    from diffit.lora import calculate_lora_parameters
    lora_stats = calculate_lora_parameters(model)
    print(f"✅ LoRA injected: {lora_stats['lora_parameters']:,} trainable parameters")
    print(f"📈 Parameter efficiency: {lora_stats['lora_ratio']:.2f}%")
    
    # 3. Test forward pass
    print("\n🧪 Testing forward pass...")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 32, 32, device=device)
    test_timesteps = torch.randint(0, 500, (batch_size,), device=device).long()
    
    model.eval()
    with torch.no_grad():
        output = model(test_input, test_timesteps)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input: {test_input.shape}")
    print(f"   Output: {output.shape}")
    
    # 4. Save and load LoRA weights
    print("\n💾 Testing LoRA save/load...")
    lora_path = "example_lora_weights.pth"
    save_lora_weights(model, lora_path)
    
    # Create a new model and load LoRA weights
    model2 = UShapedNetwork(
        learning_rate=0.001,
        d_model=128,
        num_heads=2,
        dropout=0.1,
        d_ff=256,
        img_size=32,
        device=device,
        denoising_steps=500,
        L1=2, L2=2, L3=2, L4=2
    )
    model2 = model2.to(device)
    inject_blockwise_lora(model2, LORA_CONFIG)
    load_lora_weights(model2, lora_path)
    
    print("✅ LoRA weights saved and loaded successfully!")
    
    # 5. Generate a sample (simplified, just 5 timesteps for demo)
    print("\n🎨 Generating sample (quick demo with 5 timesteps)...")
    
    # Use a very small number of timesteps for demo
    demo_timesteps = 5
    model.eval()
    
    with torch.no_grad():
        # Start with noise
        noise = torch.randn(1, 3, 32, 32, device=device)
        x = noise
        
        # Simple denoising (not the full diffusion process)
        for t_val in [4, 3, 2, 1, 0]:
            t = torch.tensor([t_val], device=device).long()
            pred_noise = model(x, t)
            
            # Simple denoising step
            alpha = 0.1
            x = x - alpha * pred_noise
    
    # Clamp to valid range
    generated_sample = torch.clamp(x, -1.0, 1.0)
    print(f"✅ Sample generated: {generated_sample.shape}")
    
    # Clean up
    import os
    if os.path.exists(lora_path):
        os.remove(lora_path)
    
    print("\n🎉 Basic usage example completed successfully!")
    print("\n💡 Next steps:")
    print("  • Use scripts/train_base_model.py for full training")
    print("  • Use scripts/finetune_with_lora.py for LoRA fine-tuning") 
    print("  • Use scripts/generate_samples.py for sample generation")
    print("  • Customize configs/ for your specific use case")


if __name__ == "__main__":
    main()
