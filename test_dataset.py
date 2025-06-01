"""
Quick test to debug dataset label masking
"""

import torch
from transformers import AutoTokenizer
from sft_training import SFTDataset, load_and_prepare_data

def test_dataset():
    print("🔍 Testing SFTDataset...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    dataset_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    (train_questions, train_codes), _ = load_and_prepare_data(dataset_cache_dir)
    
    # Create dataset
    dataset = SFTDataset(train_questions, train_codes, tokenizer, max_length=2048)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first few samples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        
        print(f"Question: {train_questions[i][:100]}...")
        print(f"Code: {train_codes[i][:100]}...")
        
        # Check shapes
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        # Check label distribution
        total_tokens = labels.numel()
        masked_tokens = (labels == -100).sum().item()
        valid_tokens = total_tokens - masked_tokens
        
        print(f"Total tokens: {total_tokens}")
        print(f"Masked tokens (-100): {masked_tokens}")
        print(f"Valid tokens: {valid_tokens}")
        print(f"Valid percentage: {valid_tokens/total_tokens*100:.1f}%")
        
        if valid_tokens == 0:
            print("❌ ALL TOKENS ARE MASKED! This will cause the gradient error.")
            
            # Debug the masking logic
            print("\nDebugging masking logic...")
            
            # Re-create the sample manually to debug
            question = train_questions[i]
            python_code = train_codes[i]
            formatted_text = f"### Question:\n{question}\n\n### Code:\n{python_code}<|EOT|>"
            instruction_part = f"### Question:\n{question}\n\n### Code:\n"
            
            print(f"Formatted text length: {len(formatted_text)}")
            print(f"Instruction part length: {len(instruction_part)}")
            
            # Tokenize separately
            full_encoding = tokenizer(formatted_text, truncation=True, max_length=2048, padding="max_length", return_tensors="pt")
            instruction_encoding = tokenizer(instruction_part, truncation=True, max_length=2048, padding=False, return_tensors="pt")
            
            print(f"Full encoding length: {full_encoding['input_ids'].shape[1]}")
            print(f"Instruction encoding length: {instruction_encoding['input_ids'].shape[1]}")
            
            # Check if instruction is too long
            if instruction_encoding["input_ids"].shape[1] >= full_encoding["input_ids"].shape[1]:
                print("❌ INSTRUCTION IS AS LONG AS FULL TEXT! This masks everything.")
            
        else:
            print("✅ Valid tokens found - should work fine")

if __name__ == "__main__":
    test_dataset() 