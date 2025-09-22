# 🎨 DiffiT Generation Guide

## ✅ **LORA CHECKPOINT LOADING FIXED!**

The generation scripts now **automatically detect and handle both standard and LoRA checkpoints**.

---

## 🚀 **Quick Usage**

### **Generate from LoRA Fine-tuned Model:**
```bash
python scripts/generate_samples.py \
    --checkpoint /path/to/lora_finetuned_model.ckpt \
    --num-samples 16 \
    --output-dir ./samples/ \
    --create-grid
```

### **Generate from Standard Model:**
```bash
python scripts/generate_samples.py \
    --checkpoint /path/to/base_model.ckpt \
    --num-samples 16 \
    --output-dir ./samples/ \
    --create-grid
```

---

## 🔧 **What the Script Does Automatically:**

### **For LoRA Checkpoints:**
1. **🔍 Detects** LoRA parameters (`lora_A`, `lora_B`, `base.weight`)
2. **🏗️ Creates** base model with correct architecture
3. **🔧 Injects** LoRA adapters using the same configuration
4. **📥 Loads** the LoRA checkpoint weights
5. **✅ Ready** for generation!

### **For Standard Checkpoints:**
1. **🔍 Detects** standard parameters (`weight`, `bias`)
2. **🏗️ Creates** base model with correct architecture  
3. **📥 Loads** the standard checkpoint weights
4. **✅ Ready** for generation!

---

## 📊 **Example Output:**

```
🚀 Using device: cuda
📥 Loading model from /path/to/finetuned_model.ckpt
📊 Checkpoint type: LoRA fine-tuned
🔧 Injecting LoRA adapters for checkpoint loading...
✅ Block-wise LoRA injection complete!
📊 Successfully replaced 30 linear layers
✅ LoRA checkpoint loaded successfully!
✅ Model loaded successfully

🎨 Generating 16 samples...
  📊 Batch 1/2 (8 samples)
  📊 Batch 2/2 (8 samples)
✅ Generated 16 samples

🖼️ Creating image grid...
💾 Saved 16 images to ./samples/
💾 Grid saved to: ./samples/sample_grid.png

🎉 Generation completed successfully!
```

---

## ⚙️ **Advanced Options:**

### **Custom Model Architecture:**
```bash
python diffit/cli/generate.py \
    --checkpoint model.ckpt \
    --d-model 256 \
    --num-heads 4 \
    --num-samples 32 \
    --timesteps 100
```

### **High Quality Generation:**
```bash
python scripts/generate_samples.py \
    --checkpoint model.ckpt \
    --timesteps 1000 \      # More timesteps = better quality
    --batch-size 4 \        # Smaller batches for large models
    --num-samples 64
```

### **Fast Generation:**
```bash
python scripts/generate_samples.py \
    --checkpoint model.ckpt \
    --timesteps 50 \        # Fewer timesteps = faster
    --batch-size 16 \       # Larger batches = faster
    --num-samples 16
```

---

## 🎯 **Key Features:**

- ✅ **Automatic Detection** - No need to specify if LoRA or standard
- ✅ **Exact Algorithm Preservation** - Same math as original research
- ✅ **Professional CLI** - Easy to use with helpful output
- ✅ **Flexible Configuration** - Customize all parameters
- ✅ **Error Handling** - Clear error messages and guidance
- ✅ **Image Grid Creation** - Easy visualization of results

---

## 💡 **Tips:**

1. **Use `--create-grid`** for easy viewing of multiple samples
2. **Adjust `--timesteps`** for quality vs speed trade-off
3. **Use smaller `--batch-size`** if you run out of memory
4. **Set `--seed`** for reproducible results
5. **LoRA models** work exactly the same as standard models!

---

The generation system now **seamlessly handles both standard and LoRA checkpoints** with zero configuration required! 🎉
