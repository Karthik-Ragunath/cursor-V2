"""
Analyze token lengths in the bespoke-manim dataset to find optimal max_length
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_data_lengths():
    print("🔍 Analyzing token lengths in bespoke-manim dataset...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
    
    questions = ds["train"]["question"]
    python_codes = ds["train"]["python_code"]
    
    print(f"Total samples to analyze: {len(questions)}")
    
    # Analyze lengths
    lengths = []
    instruction_lengths = []
    code_lengths = []
    
    print("Analyzing samples...")
    for i, (question, python_code) in enumerate(zip(questions, python_codes)):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(questions)} samples...")
        
        # Format the same way as in SFTDataset
        formatted_text = f"### Question:\n{question}\n\n### Code:\n{python_code}<|EOT|>"
        instruction_part = f"### Question:\n{question}\n\n### Code:\n"
        code_part = f"{python_code}<|EOT|>"
        
        # Tokenize (without padding, to get actual lengths)
        full_tokens = tokenizer(formatted_text, truncation=False, padding=False)["input_ids"]
        instruction_tokens = tokenizer(instruction_part, truncation=False, padding=False)["input_ids"]
        code_tokens = tokenizer(code_part, truncation=False, padding=False)["input_ids"]
        
        lengths.append(len(full_tokens))
        instruction_lengths.append(len(instruction_tokens))
        code_lengths.append(len(code_tokens))
    
    # Convert to numpy arrays for analysis
    lengths = np.array(lengths)
    instruction_lengths = np.array(instruction_lengths)
    code_lengths = np.array(code_lengths)
    
    print(f"\n📊 TOKEN LENGTH ANALYSIS")
    print("=" * 50)
    
    # Overall statistics
    print(f"\n🔢 FULL SEQUENCE LENGTHS:")
    print(f"  Min length: {lengths.min()}")
    print(f"  Max length: {lengths.max()}")
    print(f"  Mean length: {lengths.mean():.1f}")
    print(f"  Median length: {np.median(lengths):.1f}")
    print(f"  Std deviation: {lengths.std():.1f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5]
    print(f"\n📈 PERCENTILES:")
    for p in percentiles:
        value = np.percentile(lengths, p)
        coverage = (lengths <= value).sum() / len(lengths) * 100
        print(f"  {p:4.1f}th percentile: {value:4.0f} tokens (covers {coverage:5.1f}% of data)")
    
    # Common max_length recommendations
    print(f"\n💡 RECOMMENDED MAX_LENGTH:")
    recommendations = [512, 1024, 1536, 2048, 3072, 4096]
    for max_len in recommendations:
        coverage = (lengths <= max_len).sum() / len(lengths) * 100
        truncated = len(lengths) - (lengths <= max_len).sum()
        print(f"  max_length={max_len:4d}: {coverage:5.1f}% coverage, {truncated:4d} samples truncated")
    
    # Instruction vs code breakdown
    print(f"\n📝 INSTRUCTION LENGTHS:")
    print(f"  Min: {instruction_lengths.min()}, Max: {instruction_lengths.max()}, Mean: {instruction_lengths.mean():.1f}")
    
    print(f"\n💻 CODE LENGTHS:")
    print(f"  Min: {code_lengths.min()}, Max: {code_lengths.max()}, Mean: {code_lengths.mean():.1f}")
    
    # Find outliers (very long sequences)
    long_threshold = np.percentile(lengths, 95)
    long_samples = lengths > long_threshold
    
    print(f"\n🔍 LONG SAMPLES (>{long_threshold:.0f} tokens):")
    print(f"  Count: {long_samples.sum()}")
    if long_samples.sum() > 0:
        long_indices = np.where(long_samples)[0]
        print(f"  Examples:")
        for i, idx in enumerate(long_indices[:3]):  # Show first 3
            print(f"    Sample {idx}: {lengths[idx]} tokens")
            print(f"      Question: {questions[idx][:100]}...")
            print(f"      Code length: {len(python_codes[idx])} chars")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Total Token Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Sequence Lengths')
    plt.axvline(np.median(lengths), color='red', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    plt.axvline(np.percentile(lengths, 95), color='orange', linestyle='--', label=f'95th percentile: {np.percentile(lengths, 95):.0f}')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(instruction_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Instruction Token Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Instruction Lengths')
    
    plt.subplot(2, 2, 3)
    plt.hist(code_lengths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Code Token Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Code Lengths')
    
    plt.subplot(2, 2, 4)
    # Cumulative distribution
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, cumulative)
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Distribution of Sequence Lengths')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for common max_length values
    for max_len in [512, 1024, 2048]:
        coverage = (lengths <= max_len).sum() / len(lengths) * 100
        plt.axvline(max_len, color='red', linestyle='--', alpha=0.7, label=f'{max_len}: {coverage:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('token_length_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Histogram saved as 'token_length_analysis.png'")
    
    # Optimal recommendation
    print(f"\n🎯 OPTIMAL RECOMMENDATION:")
    
    # Find the sweet spot (covers 95% with minimal waste)
    target_coverage = 95
    optimal_length = np.percentile(lengths, target_coverage)
    actual_coverage = (lengths <= optimal_length).sum() / len(lengths) * 100
    
    # Round up to nearest 128 (good for GPU efficiency)
    optimal_length_rounded = int(np.ceil(optimal_length / 128) * 128)
    actual_coverage_rounded = (lengths <= optimal_length_rounded).sum() / len(lengths) * 100
    
    print(f"  For {target_coverage}% coverage: max_length = {optimal_length:.0f}")
    print(f"  Rounded for efficiency: max_length = {optimal_length_rounded} (covers {actual_coverage_rounded:.1f}%)")
    print(f"  Current setting (2048): covers {(lengths <= 2048).sum() / len(lengths) * 100:.1f}%")
    
    if optimal_length_rounded < 2048:
        savings = 2048 - optimal_length_rounded
        print(f"  💰 Potential savings: {savings} tokens per sample ({savings/2048*100:.1f}% reduction)")
        print(f"  🚀 This means faster training and less memory usage!")
    elif optimal_length_rounded > 2048:
        increase = optimal_length_rounded - 2048
        print(f"  ⚠️  Need to increase: +{increase} tokens per sample ({increase/2048*100:.1f}% increase)")
        print(f"  📊 Current setting truncates {(lengths > 2048).sum()} samples")
    else:
        print(f"  ✅ Current setting (2048) is optimal!")

if __name__ == "__main__":
    analyze_data_lengths() 
    

# 🔢 FULL SEQUENCE LENGTHS:
#   Min length: 509
#   Max length: 3544
#   Mean length: 1309.5
#   Median length: 1251.5
#   Std deviation: 380.9