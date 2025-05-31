import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()  # For causal LM, labels = input_ids
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
    
    train_questions = questions[:train_size]
    train_codes = python_codes[:train_size]
    
    val_questions = questions[train_size:]
    val_codes = python_codes[train_size:]
    
    logger.info(f"Training samples: {len(train_questions)}")
    logger.info(f"Validation samples: {len(val_questions)}")
    
    return (train_questions, train_codes), (val_questions, val_codes)

def setup_model_and_tokenizer(model_cache_dir: str, model_tokenizer_dir: str):
    """Setup the model and tokenizer for training"""
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
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        device_map="auto"  # Automatically distribute across available GPUs
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def create_training_arguments(output_dir: str, num_train_epochs: int = 3):
    """Create training arguments for the Trainer"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,  # Small batch size for 7B model
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=True,  # Mixed precision training
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],  # For monitoring
        save_total_limit=2,  # Only keep 2 best checkpoints
    )

class SFTTrainer:
    """Custom trainer class for SFT"""
    
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        
        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
        )
    
    def train(self):
        """Start the training process"""
        logger.info("Starting SFT training...")
        
        # Train the model
        self.trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        # Save training metrics
        metrics = self.trainer.state.log_history
        with open(f"{self.training_args.output_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training completed!")
        return self.trainer

def main():
    """Main training function"""
    
    # Configuration
    dataset_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    model_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    model_tokenizer_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    output_dir = "./deepseek-coder-manim-finetuned"
    
    # Step 1: Load and prepare data
    (train_questions, train_codes), (val_questions, val_codes) = load_and_prepare_data(dataset_cache_dir)
    
    # Step 2: Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_cache_dir, model_tokenizer_dir)
    
    # Step 3: Create datasets
    logger.info("Creating training datasets...")
    train_dataset = SFTDataset(train_questions, train_codes, tokenizer, max_length=2048)
    val_dataset = SFTDataset(val_questions, val_codes, tokenizer, max_length=2048)
    
    # Step 4: Create training arguments
    training_args = create_training_arguments(output_dir, num_train_epochs=3)
    
    # Step 5: Initialize trainer and start training
    sft_trainer = SFTTrainer(model, tokenizer, train_dataset, val_dataset, training_args)
    trainer = sft_trainer.train()
    
    # Step 6: Test the fine-tuned model
    logger.info("Testing the fine-tuned model...")
    test_inference(model, tokenizer, val_questions[0])

def test_inference(model, tokenizer, test_question: str):
    """Test the fine-tuned model with a sample question"""
    model.eval()
    
    # Format the input
    prompt = f"### Question:\n{test_question}\n\n### Code:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = generated_text[len(prompt):]
    
    logger.info(f"Test Question: {test_question}")
    logger.info(f"Generated Code: {generated_code}")

if __name__ == "__main__":
    main() 