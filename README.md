# DeepSeek-Coder SFT Training for Manim Code Generation

This repository contains a complete Supervised Fine-Tuning (SFT) pipeline for fine-tuning the DeepSeek-Coder model on the Bespoke-Manim dataset to generate Python code for Manim animations.

## 📋 Overview

The SFT strategy fine-tunes the DeepSeek-Coder-7B-Instruct model to generate Python code that creates mathematical animations using the Manim library. The training uses question-answer pairs where questions describe desired animations and answers are the corresponding Python code.

## 🗂️ Project Structure

```
├── sft_training.py           # Main training script
├── training_config.py        # Configuration parameters
├── test_finetuned_model.py   # Model testing and evaluation
├── monitor_training.py       # Training progress monitoring
├── requirements.txt          # Python dependencies
├── train.py                  # Original exploration script
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Training

```bash
python sft_training.py
```

### 3. Monitor Training Progress

In a separate terminal:
```bash
# Real-time monitoring
python monitor_training.py --mode live

# Plot training curves
python monitor_training.py --mode plot

# Training summary
python monitor_training.py --mode summary
```

### 4. Test the Fine-tuned Model

```bash
# Test on dataset samples
python test_finetuned_model.py --mode dataset --num_samples 10

# Interactive testing
python test_finetuned_model.py --mode interactive
```

## 📖 Step-by-Step Explanation

### Step 1: Data Preparation

The training pipeline processes the Bespoke-Manim dataset:

1. **Data Loading**: Loads questions and Python code pairs from the `bespokelabs/bespoke-manim` dataset
2. **Data Formatting**: Formats each sample as:
   ```
   ### Question:
   [question text]
   
   ### Code:
   [python code]
   ```
3. **Train/Validation Split**: Splits data 90/10 for training and validation

### Step 2: Model and Tokenizer Setup

1. **Model Loading**: Loads DeepSeek-Coder-7B-Instruct with:
   - Half precision (FP16) for memory efficiency
   - Automatic device mapping
   - Gradient checkpointing enabled

2. **Tokenizer Configuration**: 
   - Sets padding token if not present
   - Configures for causal language modeling

### Step 3: Custom Dataset Creation

The `SFTDataset` class:
- Tokenizes question-answer pairs
- Handles padding and truncation
- Sets labels equal to input_ids for causal LM training

### Step 4: Training Configuration

Key training parameters:
- **Batch Size**: 1 per device with gradient accumulation (effective batch size: 8)
- **Learning Rate**: 2e-5 with warmup
- **Epochs**: 3 (configurable)
- **Memory Optimization**: FP16, gradient checkpointing
- **Evaluation**: Every 500 steps
- **Checkpointing**: Save best model based on eval loss

### Step 5: Training Execution

The training uses Hugging Face's `Trainer` class with:
- `DataCollatorForLanguageModeling` for dynamic batching
- Automatic mixed precision
- TensorBoard logging
- Checkpoint management

### Step 6: Model Evaluation

After training:
- Tests on validation samples
- Generates code for new questions
- Analyzes generation quality and speed

## 🔧 Configuration

### Training Parameters

Edit `training_config.py` to customize:

```python
class TrainingConfig:
    # Training hyperparameters
    NUM_TRAIN_EPOCHS = 3
    LEARNING_RATE = 2e-5
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    
    # Model parameters
    MAX_LENGTH = 2048
    
    # Paths
    OUTPUT_DIR = "./deepseek-coder-manim-finetuned"
```

### Memory Optimization

For limited GPU memory:
1. Reduce `MAX_LENGTH` to 1024 or 512
2. Enable more aggressive gradient checkpointing
3. Consider using DeepSpeed ZeRO for multi-GPU setups

## 📊 Monitoring and Evaluation

### Training Monitoring

```bash
# Live monitoring (updates every 30 seconds)
python monitor_training.py --mode live --refresh_interval 30

# Generate training plots
python monitor_training.py --mode plot

# Quick summary
python monitor_training.py --mode summary
```

### Model Testing

```bash
# Test on 20 validation samples
python test_finetuned_model.py --mode dataset --num_samples 20

# Interactive testing
python test_finetuned_model.py --mode interactive

# Both modes
python test_finetuned_model.py --mode both
```

## 🎯 Expected Results

After successful training, you should see:

1. **Decreasing Loss**: Both training and validation loss should decrease
2. **Generated Code Quality**: The model should generate syntactically correct Python code
3. **Manim Integration**: Code should properly use Manim classes and methods
4. **Question Understanding**: Generated code should match the requested animation

### Sample Output

```python
# Question: "Create a simple animation showing a square moving from left to right"

# Generated Code:
from manim import *

class SquareAnimation(Scene):
    def construct(self):
        square = Square()
        square.shift(LEFT * 3)
        
        self.play(Create(square))
        self.play(square.animate.shift(RIGHT * 6))
        self.wait()
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size or max_length
   - Enable gradient checkpointing
   - Use DeepSpeed for multi-GPU training

2. **Slow Training**
   - Ensure GPU utilization is high
   - Check data loading bottlenecks
   - Consider mixed precision training

3. **Poor Generation Quality**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and formatting

### GPU Requirements

- **Minimum**: 24GB VRAM (A100, RTX 3090/4090)
- **Recommended**: 40GB+ VRAM for larger batch sizes
- **Multi-GPU**: Supported via device_map="auto"

## 📈 Advanced Features

### Custom Prompt Templates

Modify the prompt format in `training_config.py`:

```python
@classmethod
def get_prompt_template(cls):
    return "Question: {question}\n\nAnswer: {code}
```

### Learning Rate Scheduling

Add custom schedulers to the training arguments:

```python
training_args.lr_scheduler_type = "cosine"
training_args.warmup_ratio = 0.1
```

### Data Augmentation

Extend the dataset class to include:
- Question paraphrasing
- Code style variations
- Difficulty-based sampling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- [DeepSeek](https://github.com/deepseek-ai) for the base model
- [Bespoke Labs](https://github.com/bespokelabsai) for the Manim dataset
- [Hugging Face](https://huggingface.co/) for the transformers library
- [Manim Community](https://github.com/ManimCommunity/manim) for the animation library

---

For questions or issues, please open a GitHub issue or contact the maintainers.