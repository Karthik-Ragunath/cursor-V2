# LoRA Fine-tuning for DeepSeek-Coder on Manim Dataset

This implementation uses **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning of the DeepSeek-Coder-7B model on the bespoke-manim dataset.

## 🚀 Why LoRA?

### **Advantages over Full Fine-tuning:**
- **Memory Efficient**: Only trains ~1% of model parameters
- **Faster Training**: Reduced computational requirements
- **No Gradient Issues**: Avoids the NaN gradient problems we experienced
- **Modular**: LoRA adapters can be easily swapped or combined
- **Storage Efficient**: Adapter files are only ~50MB vs 14GB for full model

### **How LoRA Works:**
```
Original Model: W (frozen)
LoRA: W + A × B (trainable)
- A: rank × d_model matrix
- B: d_model × rank matrix  
- rank << d_model (typically 16-64)
```

## 📋 Key Components

### **1. Fixed Label Masking**
```python
# OLD (caused NaN gradients):
"labels": encoding["input_ids"].flatten()

# NEW (LoRA-compatible):
labels[:, :instruction_length] = -100  # Mask question
labels[attention_mask == 0] = -100     # Mask padding
```

### **2. LoRA Configuration**
```python
LoraConfig(
    r=16,                    # Rank (controls adapter size)
    lora_alpha=32,          # Scaling factor (typically 2*r)
    target_modules=[        # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,       # Regularization
)
```

### **3. Training Optimizations**
- **Higher Learning Rate**: `2e-4` (vs `2e-5` for full fine-tuning)
- **BF16 Precision**: More stable than FP16
- **Conservative Generation**: Nucleus sampling, repetition penalty

## 🛠️ Installation & Setup

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Key Dependencies**
- `peft>=0.7.0` - For LoRA implementation
- `transformers>=4.35.0` - Model and tokenizer
- `torch>=2.0.0` - Deep learning framework
- `bitsandbytes>=0.41.0` - Quantization support

## 🏃 Quick Start

### **1. Start LoRA Training**
```bash
python sft_training.py
```

**Or use VS Code launch configuration:**
- Press `F5` → Select **"LoRA Training: Start Fine-tuning"**

### **2. Monitor Training**
Expected logs should show:
```
Training samples: X
Validation samples: Y
trainable params: 41,943,040 || all params: 6,966,061,056 || trainable%: 0.6%
{'loss': 2.3456, 'grad_norm': 0.234, 'learning_rate': 1.8e-04, 'epoch': 0.1}
```

**✅ Healthy signs:**
- Loss: 1-8 (not 49,603!)
- grad_norm: 0.1-2.0 (not NaN!)
- Trainable%: ~0.6%

### **3. Test the Trained Model**
```bash
# Default test questions
python load_lora_model.py

# Interactive mode
python load_lora_model.py --interactive

# Single question
python load_lora_model.py --question "Create a red circle"
```

## 📁 Output Structure

After training, you'll have:
```
deepseek-coder-manim-lora/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights (~50MB)
├── tokenizer.json           # Tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
└── training_metrics.json   # Training history
```

## 🔧 Configuration Options

### **Memory Optimization**
```python
# For even more memory savings:
load_in_4bit=True          # Quantize base model
r=8                        # Smaller rank
target_modules=["q_proj", "v_proj"]  # Fewer target modules
```

### **Quality vs Speed**
```python
# Higher quality (slower):
r=32, lora_alpha=64
per_device_train_batch_size=1

# Faster (lower quality):
r=8, lora_alpha=16  
per_device_train_batch_size=4
```

## 📊 Expected Results

### **Training Metrics**
- **Initial Loss**: ~3-6 (reasonable starting point)
- **Final Loss**: ~0.5-2 (good convergence)
- **Training Time**: 2-4 hours (vs 8-12 for full fine-tuning)
- **Memory Usage**: ~40GB (vs 80GB for full fine-tuning)

### **Generation Quality**
```
Question: "Create a red circle"
Generated Code:
circle = Circle().set_color(RED)
self.play(Create(circle))
```

## 🚨 Troubleshooting

### **If training fails:**
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in training_args
per_device_train_batch_size=1

# Use 4-bit quantization
load_in_4bit=True
```

### **If generation fails:**
```bash
# Test base model first
python load_lora_model.py --question "test"

# Check adapter files exist
ls deepseek-coder-manim-lora/
```

### **Common Issues:**
1. **"No LoRA found"** → Check adapter_config.json exists
2. **Memory error** → Reduce batch size or use quantization
3. **Poor quality** → Increase rank or training epochs

## 🎯 Next Steps

### **1. Experiment with Configurations**
- Try different ranks (8, 16, 32, 64)
- Adjust target modules
- Test various learning rates

### **2. Advanced Techniques**
- **QLoRA**: 4-bit quantization + LoRA
- **Multi-LoRA**: Train multiple adapters for different tasks
- **LoRA Merging**: Combine multiple adapters

### **3. Evaluation**
- Test on held-out questions
- Compare with base model performance
- Measure code correctness

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)

---

**Happy LoRA fine-tuning! 🎉** 