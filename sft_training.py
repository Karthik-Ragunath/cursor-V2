import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from torch.utils.data import DataLoader
import json
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTDataset:
    """Custom dataset class for SFT training"""
    
    def __init__(self, questions: List[str], python_codes: List[str], tokenizer, max_length: int = 2048):
        self.questions = questions
        self.python_codes = python_codes
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        python_code = self.python_codes[idx]
        
        # Format the input-output pair
        # Using a format similar to instruction following
        formatted_text = f"### Question:\n{question}\n\n### Code:\n{python_code}<|EOT|>"
        
        # Create the instruction part (question + prompt)
        instruction_part = f"### Question:\n{question}\n\n### Code:\n"
        
        # Tokenize the full text and instruction part separately
        full_encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        instruction_encoding = self.tokenizer(
            instruction_part,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels: -100 for tokens we don't want to compute loss on
        labels = full_encoding["input_ids"].clone()
        instruction_length = instruction_encoding["input_ids"].shape[1]
        
        # Mask the instruction part (question + prompt) - don't compute loss on these
        labels[:, :instruction_length] = -100
        
        # Mask padding tokens - don't compute loss on these  
        labels[full_encoding["attention_mask"] == 0] = -100
        
        return {
            "input_ids": full_encoding["input_ids"].flatten(),
            "attention_mask": full_encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

def load_and_prepare_data(dataset_cache_dir: str, train_split_ratio: float = 0.9):
    """Load and prepare the bespoke-manim dataset for training"""
    logger.info("Loading bespoke-manim dataset...")
    
    # Load dataset
    ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
    
    questions = ds["train"]["question"]
    python_codes = ds["train"]["python_code"]
    
    logger.info(f"Total samples: {len(questions)}")
    
    # Split into train and validation
    total_samples = len(questions)
    train_size = int(total_samples * train_split_ratio)
    
    train_questions = questions[:10]
    train_codes = python_codes[:10]
    
    val_questions = questions[:2]
    val_codes = python_codes[:2]
    
    logger.info(f"Training samples: {len(train_questions)}")
    logger.info(f"Validation samples: {len(val_questions)}")
    
    return (train_questions, train_codes), (val_questions, val_codes)

def create_lora_config():
    """Create LoRA configuration for DeepSeek-coder model"""
    
    # DeepSeek-coder might have different layer names, let's try multiple possibilities
    possible_target_modules = [
        # Standard transformer names
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Alternative names
        ["query", "key", "value", "dense", "gate", "up", "down"],
        # Minimal working set
        ["q_proj", "v_proj"],
        # Even more minimal
        ["q_proj"]
    ]
    
    # We'll use the first set and add debugging
    target_modules = possible_target_modules[0]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Causal language modeling task
        inference_mode=False,          # Training mode
        r=8,                          # Smaller rank for stability
        lora_alpha=16,                # Smaller alpha
        lora_dropout=0.1,             # LoRA dropout for regularization
        target_modules=target_modules, # Target modules for LoRA adaptation
        bias="none",                  # Don't adapt bias terms
        use_rslora=False,             # Standard LoRA (not RS-LoRA)
    )
    
    logger.info(f"LoRA target modules: {target_modules}")
    return lora_config

def setup_model_and_tokenizer(model_cache_dir: str, model_tokenizer_dir: str):
    """Setup the model and tokenizer for LoRA training"""
    logger.info("Loading DeepSeek-coder model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir=model_tokenizer_dir
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        torch_dtype=torch.bfloat16,    # Use BF16 for better stability
        device_map="auto",             # Automatically distribute across available GPUs
        load_in_8bit=False,           # Keep full precision for LoRA
        load_in_4bit=False,           # Can enable for even more memory savings
    )
    
    # Prepare model for LoRA training
    logger.info("Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    
    # Create and apply LoRA config
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Debug: Check if any parameters actually require gradients
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    logger.info(f"Trainable parameter count: {len(trainable_params)}")
    if len(trainable_params) == 0:
        logger.error("❌ NO TRAINABLE PARAMETERS FOUND! LoRA setup failed.")
        raise RuntimeError("No trainable parameters - LoRA configuration issue")
    else:
        logger.info("✅ Trainable parameters found:")
        for i, name in enumerate(trainable_params[:5]):  # Show first 5
            logger.info(f"  {i+1}. {name}")
        if len(trainable_params) > 5:
            logger.info(f"  ... and {len(trainable_params) - 5} more")
    
    # Enable gradient checkpointing for memory efficiency
    # model.gradient_checkpointing_enable()  # This interferes with LoRA gradients!
    
    logger.info("LoRA model setup completed!")
    return model, tokenizer

def create_training_arguments(output_dir: str, num_train_epochs: int = 3):
    """Create training arguments optimized for LoRA training"""
    
    # Check if BF16 is available (more stable than FP16)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,  # Slightly larger batch for LoRA
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        learning_rate=2e-4,             # Higher LR for LoRA (typical range: 1e-4 to 3e-4)
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=250,                 # Save more frequently for LoRA
        eval_steps=250,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=50,                # Fewer warmup steps for LoRA
        bf16=use_bf16,                  # Use BF16 if available (more stable)
        fp16=not use_bf16,              # Use FP16 only if BF16 not available
        max_grad_norm=1.0,              # Gradient clipping
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],      # For monitoring
        save_total_limit=3,             # Keep more checkpoints for LoRA
        # LoRA-specific optimizations
        optim="adamw_torch",            # Stable optimizer
        lr_scheduler_type="cosine",     # Smooth learning rate decay
        save_safetensors=True,          # Use safer model format
        dataloader_num_workers=0,       # Avoid multiprocessing issues
        group_by_length=False,          # Keep False for instruction tuning
        ddp_find_unused_parameters=False,  # Optimization for DDP
    )

class SFTTrainer:
    """Custom trainer class for LoRA SFT"""
    
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        
        # Use default data collator instead of DataCollatorForLanguageModeling
        # The default collator will work with our pre-processed labels
        # self.data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer,
        #     mlm=False,  # We're doing causal LM, not masked LM
        #     pad_to_multiple_of=8,  # Optimization for tensor cores
        # )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # data_collator=self.data_collator,  # Remove this line
        )
    
    def train(self):
        """Start the LoRA training process"""
        logger.info("Starting LoRA SFT training...")
        
        # Debug: Check a sample batch before training
        logger.info("🔍 Debugging first batch...")
        try:
            sample_batch = next(iter(self.trainer.get_train_dataloader()))
            logger.info(f"Batch keys: {list(sample_batch.keys())}")
            logger.info(f"Input shape: {sample_batch['input_ids'].shape}")
            logger.info(f"Labels shape: {sample_batch['labels'].shape}")
            
            # Check if labels have valid tokens (not all -100)
            labels = sample_batch['labels']
            valid_labels = labels[labels != -100]
            logger.info(f"Valid label tokens: {len(valid_labels)} / {labels.numel()}")
            
            if len(valid_labels) == 0:
                logger.error("❌ ALL LABELS ARE MASKED! No tokens to learn from.")
                raise RuntimeError("All labels are -100 - check dataset preprocessing")
        
        except Exception as e:
            logger.error(f"Batch debugging failed: {e}")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Log training results
            logger.info(f"Training completed! Final loss: {train_result.training_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            
            # Additional debugging for gradient error
            if "does not require grad" in str(e):
                logger.error("🔍 GRADIENT ERROR DETECTED!")
                logger.error("This usually means:")
                logger.error("1. No LoRA parameters are trainable")
                logger.error("2. All labels are masked (-100)")
                logger.error("3. Model setup issue")
                
                # Check model parameters again
                trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
                logger.error(f"Trainable parameters at error time: {trainable_count}")
            
            raise e
        
        # Save the LoRA adapter
        logger.info("Saving LoRA adapter...")
        self.model.save_pretrained(self.training_args.output_dir)
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        # Save training metrics
        metrics = self.trainer.state.log_history
        with open(f"{self.training_args.output_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save LoRA configuration
        lora_config_path = f"{self.training_args.output_dir}/adapter_config.json"
        logger.info(f"LoRA configuration saved to: {lora_config_path}")
        
        logger.info("LoRA training completed successfully!")
        return self.trainer

def main():
    """Main LoRA training function"""
    
    # Configuration
    dataset_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    model_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    model_tokenizer_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    output_dir = "./deepseek-coder-manim-lora"  # Changed to LoRA-specific directory
    
    # Step 1: Load and prepare data
    (train_questions, train_codes), (val_questions, val_codes) = load_and_prepare_data(dataset_cache_dir)
    
    # Step 2: Setup model and tokenizer with LoRA
    model, tokenizer = setup_model_and_tokenizer(model_cache_dir, model_tokenizer_dir)
    
    # Step 3: Create datasets
    logger.info("Creating training datasets...")
    train_dataset = SFTDataset(train_questions, train_codes, tokenizer, max_length=2048)
    val_dataset = SFTDataset(val_questions, val_codes, tokenizer, max_length=2048)
    
    # Step 4: Validate a few samples
    logger.info("Validating dataset samples...")
    try:
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            labels = sample["labels"]
            valid_labels = labels[labels != -100]
            logger.info(f"Sample {i}: {len(valid_labels)} valid label tokens")
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return
    
    # Step 5: Create training arguments
    training_args = create_training_arguments(output_dir, num_train_epochs=3)
    
    # Step 6: Initialize trainer and start training
    sft_trainer = SFTTrainer(model, tokenizer, train_dataset, val_dataset, training_args)
    trainer = sft_trainer.train()
    
    # Step 7: Test the fine-tuned LoRA model
    logger.info("Testing the fine-tuned LoRA model...")
    test_inference(model, tokenizer, train_questions[0])

def test_inference(model, tokenizer, test_question: str):
    """Test the fine-tuned LoRA model with a sample question"""
    logger.info("Testing LoRA model inference...")
    model.eval()
    
    # Format the input
    prompt = f"### Question:\n{test_question}\n\n### Code:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with more conservative parameters for LoRA
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,         # Shorter generation for testing
                temperature=0.8,            # Slightly more conservative
                do_sample=True,
                top_p=0.9,                 # Nucleus sampling
                top_k=50,                  # Top-k sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,    # Avoid repetition
                no_repeat_ngram_size=3,    # Avoid 3-gram repetition
            )
        
        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text[len(prompt):]
        
        # Clean up generated code
        if "<|EOT|>" in generated_code:
            generated_code = generated_code.split("<|EOT|>")[0]
        
        logger.info(f"Test Question: {test_question}")
        logger.info(f"Generated Code: {generated_code.strip()}")
        
        return generated_code.strip()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.info("This might be due to model corruption from previous training.")
        logger.info("Try restarting with a fresh model if the error persists.")
        return None

if __name__ == "__main__":
    main() 