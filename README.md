# DiffiT-LoRA: Diffusion Vision Transformers with LoRA Fine-tuning

A professional implementation of **DiffiT (Diffusion Vision Transformers)** with **Low-Rank Adaptation (LoRA)** for efficient fine-tuning.

## 🎯 Key Features

- **🔬 State-of-the-art Architecture**: Time-aware Multi-head Self-Attention (TMSA) mechanism
- **⚡ Efficient Fine-tuning**: Block-wise LoRA with ~1-5% trainable parameters
- **🏗️ Dual Architectures**: Both Image-space (U-shaped) and Latent-space models
- **📊 Comprehensive Evaluation**: FID, KID, LPIPS, and FLOPs metrics
- **⚙️ Production Ready**: Clean modular design with configuration management

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/diffit-lora.git
cd diffit-lora

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from diffit.models import UShapedNetwork
from diffit.lora import inject_blockwise_lora, LoRAConfig
from diffit.training import DiffiTTrainer

# Load base model
model = UShapedNetwork.load_from_checkpoint("base_model.ckpt")

# Inject LoRA adapters
lora_config = LoRAConfig.from_yaml("configs/lora/blockwise_config.yaml")
inject_blockwise_lora(model, lora_config)

# Fine-tune
trainer = DiffiTTrainer(model, config="configs/training/lora_finetuning.yaml")
trainer.fit(train_loader, val_loader)
```

### Command Line Interface

```bash
# Train base model
diffit-train --config configs/training/base_training.yaml

# LoRA fine-tune
diffit-finetune --base-checkpoint base_model.ckpt \
                --config configs/lora/blockwise_config.yaml \
                --data-config configs/data/cifar10.yaml

# Generate samples
diffit-generate --checkpoint finetuned_model.ckpt \
                --num-samples 16 \
                --output samples/
```

## 📁 Project Structure

```
diffit-lora/
├── diffit/                 # Main package
│   ├── models/            # Model architectures (UNet, Latent, TMSA)
│   ├── lora/              # LoRA implementation
│   ├── diffusion/         # Diffusion algorithms  
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Metrics and visualization
│   └── cli/               # Command-line interface
├── configs/               # YAML configurations
├── scripts/               # Ready-to-use scripts
└── examples/              # Usage examples
```

## 🏗️ Architecture

### DiffiT Models
- **UShapedNetwork**: Image-space diffusion with U-Net architecture
- **LatentDiffiTNetwork**: Latent-space diffusion with patch embeddings
- **TMSA**: Time-aware Multi-head Self-Attention mechanism

### LoRA Fine-tuning
- **Block-wise LoRA**: Different ranks for different model components
- **Parameter Efficiency**: Only 1-5% of parameters trainable
- **Flexible Deployment**: Addition mode or weight fusion options

## 📊 Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Image quality and diversity
- **KID (Kernel Inception Distance)**: Robust alternative to FID
- **LPIPS**: Perceptual similarity using deep features
- **FLOPs**: Computational complexity analysis

## 🔧 Configuration

All hyperparameters are managed through YAML configuration files:

```yaml
# configs/lora/blockwise_config.yaml
lora:
  alpha: 1.0
  targets: ["Wqs", "Wks", "Wvs", "Wqt", "Wkt", "Wvt", "wo"]
  encoder:
    groups: {1: 16, 2: 12, 3: 8, 4: 8}
  decoder:
    groups: {3: 8, 2: 10, 1: 12}
```

## 🎓 Research

Based on the paper: **"DiffiT: Diffusion Vision Transformers for Image Generation"**

### Key Innovations:
1. **Time-aware Multi-head Self-Attention (TMSA)**
2. **Transformer-based Diffusion Models**
3. **Block-wise LoRA Adaptation**

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.

## 📧 Contact

For questions and support, please open an issue or contact: contact@diffit.ai
