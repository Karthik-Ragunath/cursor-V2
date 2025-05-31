"""
Script to test and evaluate the fine-tuned DeepSeek-coder model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import json
import os
from typing import List, Dict
import time

class ModelTester:
    """Class to test the fine-tuned model"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the model tester
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to load the model on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading fine-tuned model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_code(self, question: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate code for a given question
        
        Args:
            question: The input question
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated code as string
        """
        # Format the prompt
        prompt = f"### Question:\n{question}\n\n### Code:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract generated code
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = generated_text[len(prompt):]
        
        # Clean up the generated code (remove <|EOT|> if present)
        if "<|EOT|>" in generated_code:
            generated_code = generated_code.split("<|EOT|>")[0]
        
        return generated_code.strip()
    
    def test_on_dataset(self, dataset_cache_dir: str, num_samples: int = 10) -> List[Dict]:
        """Test the model on a subset of the validation dataset
        
        Args:
            dataset_cache_dir: Path to the cached dataset
            num_samples: Number of samples to test on
            
        Returns:
            List of test results
        """
        print(f"Testing model on {num_samples} samples from the dataset...")
        
        # Load dataset
        ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
        questions = ds["train"]["question"]
        ground_truth_codes = ds["train"]["python_code"]
        
        # Take validation samples (last 10% of the dataset)
        total_samples = len(questions)
        val_start = int(total_samples * 0.9)
        val_questions = questions[val_start:val_start + num_samples]
        val_codes = ground_truth_codes[val_start:val_start + num_samples]
        
        results = []
        
        for i, (question, ground_truth) in enumerate(zip(val_questions, val_codes)):
            print(f"\nTesting sample {i+1}/{num_samples}")
            print(f"Question: {question[:100]}...")
            
            start_time = time.time()
            generated_code = self.generate_code(question)
            generation_time = time.time() - start_time
            
            result = {
                "sample_id": i,
                "question": question,
                "ground_truth": ground_truth,
                "generated_code": generated_code,
                "generation_time": generation_time
            }
            
            results.append(result)
            
            print(f"Generated in {generation_time:.2f}s")
            print(f"Generated code preview: {generated_code[:200]}...")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("Interactive testing mode. Type 'quit' to exit.")
        
        while True:
            question = input("\nEnter a question: ").strip()
            
            if question.lower() == 'quit':
                break
            
            if not question:
                continue
            
            print("Generating code...")
            start_time = time.time()
            generated_code = self.generate_code(question)
            generation_time = time.time() - start_time
            
            print(f"\nGenerated Code (in {generation_time:.2f}s):")
            print("-" * 50)
            print(generated_code)
            print("-" * 50)

def save_test_results(results: List[Dict], output_file: str):
    """Save test results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to {output_file}")

def analyze_results(results: List[Dict]):
    """Analyze and print test results"""
    if not results:
        print("No results to analyze")
        return
    
    print("\n=== Test Results Analysis ===")
    
    # Calculate average generation time
    avg_time = sum(r['generation_time'] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f} seconds")
    
    # Calculate average code length
    avg_code_length = sum(len(r['generated_code']) for r in results) / len(results)
    avg_gt_length = sum(len(r['ground_truth']) for r in results) / len(results)
    
    print(f"Average generated code length: {avg_code_length:.0f} characters")
    print(f"Average ground truth length: {avg_gt_length:.0f} characters")
    
    # Find the best and worst examples (by length similarity)
    length_diffs = []
    for r in results:
        diff = abs(len(r['generated_code']) - len(r['ground_truth']))
        length_diffs.append((diff, r['sample_id']))
    
    length_diffs.sort()
    best_sample = results[length_diffs[0][1]]
    worst_sample = results[length_diffs[-1][1]]
    
    print(f"\nBest example (most similar length):")
    print(f"Question: {best_sample['question'][:100]}...")
    print(f"Generated length: {len(best_sample['generated_code'])}")
    print(f"Ground truth length: {len(best_sample['ground_truth'])}")
    
    print(f"\nWorst example (most different length):")
    print(f"Question: {worst_sample['question'][:100]}...")
    print(f"Generated length: {len(worst_sample['generated_code'])}")
    print(f"Ground truth length: {len(worst_sample['ground_truth'])}")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned DeepSeek-coder model")
    parser.add_argument("--model_path", type=str, default="./deepseek-coder-manim-finetuned",
                       help="Path to the fine-tuned model")
    parser.add_argument("--mode", type=str, choices=["dataset", "interactive", "both"], default="dataset",
                       help="Testing mode")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to test on (for dataset mode)")
    parser.add_argument("--dataset_cache_dir", type=str, 
                       default="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim",
                       help="Dataset cache directory")
    parser.add_argument("--output_file", type=str, default="test_results.json",
                       help="Output file for test results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to load the model on")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    # Initialize model tester
    tester = ModelTester(args.model_path, args.device)
    
    # Run tests based on mode
    if args.mode in ["dataset", "both"]:
        results = tester.test_on_dataset(args.dataset_cache_dir, args.num_samples)
        save_test_results(results, args.output_file)
        analyze_results(results)
    
    if args.mode in ["interactive", "both"]:
        tester.interactive_test()

if __name__ == "__main__":
    main() 