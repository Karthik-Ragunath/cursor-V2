"""
Script to test and evaluate the base DeepSeek-coder model (without LoRA fine-tuning)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import json
import os
from typing import List, Dict
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModelTester:
    """Class to test the base model without fine-tuning"""
    
    def __init__(self, base_model_name: str, device: str = "auto"):
        """Initialize the base model tester
        
        Args:
            base_model_name: Name/path of the base model (e.g., "deepseek-ai/deepseek-coder-7b-instruct-v1.5")
            device: Device to load the model on
        """
        self.base_model_name = base_model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
    def load_model(self):
        """Load the base model"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            # Use cache dir from training if available
            cache_dir="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        logger.info("🎉 Base model loaded successfully!")
        
        # Print model info
        self.print_model_info()
    
    def print_model_info(self):
        """Print information about the loaded model"""
        try:
            # Print device info
            device = next(self.model.parameters()).device
            logger.info(f"Model loaded on device: {device}")
            
            # Print model size info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            
        except Exception as e:
            logger.warning(f"Could not print model info: {e}")
    
    def generate_code(self, question: str, max_new_tokens: int = 3600, temperature: float = 0.8) -> str:
        """Generate code for a given question
        
        Args:
            question: The input question
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated code as string
        """
        # Format the prompt (same as training format)
        prompt = f"### Question:\n{question}\n\n### Code:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate with standard parameters
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,                 # Nucleus sampling
                    top_k=50,                  # Top-k sampling
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,    # Avoid repetition
                    no_repeat_ngram_size=3,    # Avoid 3-gram repetition
                )
            
            # Decode and extract generated code
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated_text[len(prompt):]
            
            # Clean up generated code
            if "<|EOT|>" in generated_code:
                generated_code = generated_code.split("<|EOT|>")[0]
            
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"ERROR: Generation failed - {e}"
    
    def test_on_dataset(self, dataset_cache_dir: str, num_samples: int = 10, skip_samples: int = 0) -> List[Dict]:
        """Test the model on a subset of the dataset
        
        Args:
            dataset_cache_dir: Path to the cached dataset
            num_samples: Number of samples to test on
            skip_samples: Number of samples to skip (for testing different subsets)
            
        Returns:
            List of test results
        """
        logger.info(f"Testing base model on {num_samples} samples from the dataset...")
        
        # Load dataset
        ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
        questions = ds["train"]["question"]
        ground_truth_codes = ds["train"]["python_code"]
        
        # Take test samples
        total_samples = len(questions)
        start_idx = 0 + skip_samples
        end_idx = min(start_idx + num_samples, total_samples)
        
        test_questions = questions[start_idx:end_idx]
        test_codes = ground_truth_codes[start_idx:end_idx]
        
        logger.info(f"Testing on samples {start_idx}-{end_idx-1}")
        
        results = []
        
        for i, (question, ground_truth) in enumerate(zip(test_questions, test_codes)):
            logger.info(f"Testing sample {i+1}/{len(test_questions)}")
            print(f"Question: {question[:100]}...")
            
            start_time = time.time()
            generated_code = self.generate_code(question, max_new_tokens=3600)
            generation_time = time.time() - start_time
            
            result = {
                "sample_id": start_idx + i,
                "question": question,
                "ground_truth": ground_truth,
                "generated_code": generated_code,
                "generation_time": generation_time,
                "generated_length": len(generated_code),
                "ground_truth_length": len(ground_truth)
            }
            
            results.append(result)
            
            print(f"Generated in {generation_time:.2f}s")
            print(f"Generated code preview: {generated_code[:3600]}...")
            print("-" * 50)
        
        return results
    
    def test_default_questions(self) -> List[Dict]:
        """Test the model on default questions"""
        default_questions = [
            "Create a red circle",
            "Draw a blue square that moves to the right",
            "Make a green triangle that rotates",
            "Create text that says 'Hello World' and make it fade in",
            "Create an animation showing a line growing from left to right",
            "Draw a purple rectangle that scales up and down",
            "Create a yellow star that moves in a circle pattern",
            "Make a mathematical equation that appears and transforms"
        ]
        
        logger.info(f"Testing base model on {len(default_questions)} default questions...")
        
        results = []
        
        for i, question in enumerate(default_questions):
            print(f"\nTest {i+1}/{len(default_questions)}: {question}")
            print("=" * 60)
            
            start_time = time.time()
            generated_code = self.generate_code(question)
            generation_time = time.time() - start_time
            
            result = {
                "sample_id": f"default_{i}",
                "question": question,
                "generated_code": generated_code,
                "generation_time": generation_time,
                "generated_length": len(generated_code)
            }
            
            results.append(result)
            
            print(f"Generated Code (in {generation_time:.2f}s):")
            print(generated_code)
            print("=" * 60)
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎯 Base Model Interactive Testing Mode")
        print("Type your questions and see the base model generate Manim code!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("💬 Enter a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Generating code with base model...")
                start_time = time.time()
                generated_code = self.generate_code(question)
                generation_time = time.time() - start_time
                
                print(f"\n✨ Generated Code (in {generation_time:.2f}s):")
                print("-" * 50)
                print(generated_code)
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")

def save_test_results(results: List[Dict], output_file: str):
    """Save test results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test results saved to {output_file}")

def analyze_results(results: List[Dict]):
    """Analyze and print test results"""
    if not results:
        logger.warning("No results to analyze")
        return
    
    print("\n" + "="*60)
    print("📊 BASE MODEL TEST RESULTS ANALYSIS")
    print("="*60)
    
    # Calculate average generation time
    avg_time = sum(r['generation_time'] for r in results) / len(results)
    print(f"📈 Average generation time: {avg_time:.2f} seconds")
    
    # Calculate average code length
    avg_code_length = sum(r['generated_length'] for r in results) / len(results)
    print(f"📝 Average generated code length: {avg_code_length:.0f} characters")
    
    if 'ground_truth_length' in results[0]:
        avg_gt_length = sum(r['ground_truth_length'] for r in results) / len(results)
        print(f"📋 Average ground truth length: {avg_gt_length:.0f} characters")
        
        # Calculate length similarity
        length_ratios = []
        for r in results:
            if r['ground_truth_length'] > 0:
                ratio = r['generated_length'] / r['ground_truth_length']
                length_ratios.append(ratio)
        
        if length_ratios:
            avg_ratio = sum(length_ratios) / len(length_ratios)
            print(f"📊 Average length ratio (generated/ground_truth): {avg_ratio:.2f}")
    
    # Find fastest and slowest generation
    fastest = min(results, key=lambda x: x['generation_time'])
    slowest = max(results, key=lambda x: x['generation_time'])
    
    print(f"\n⚡ Fastest generation: {fastest['generation_time']:.2f}s")
    print(f"🐌 Slowest generation: {slowest['generation_time']:.2f}s")
    
    # Show a sample result
    print(f"\n🎯 Sample Result:")
    sample = results[0]
    print(f"Question: {sample['question'][:100]}...")
    print(f"Generated: {sample['generated_code'][:200]}...")

def main():
    parser = argparse.ArgumentParser(description="Test base DeepSeek-coder model (without fine-tuning)")
    parser.add_argument("--base_model", type=str, 
                       default="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                       help="Base model name or path")
    parser.add_argument("--mode", type=str, 
                       choices=["dataset", "interactive", "default", "all"], 
                       default="default",
                       help="Testing mode")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to test on (for dataset mode)")
    parser.add_argument("--skip_samples", type=int, default=0,
                       help="Number of samples to skip (for testing different subsets)")
    parser.add_argument("--dataset_cache_dir", type=str, 
                       default="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim",
                       help="Dataset cache directory")
    parser.add_argument("--output_file", type=str, default="base_model_test_results.json",
                       help="Output file for test results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to load the model on")
    
    args = parser.parse_args()
    
    # Initialize base model tester
    logger.info("🚀 Initializing Base Model Tester...")
    tester = BaseModelTester(args.base_model, args.device)
    
    results = []
    
    # Run tests based on mode
    if args.mode in ["default", "all"]:
        logger.info("🎯 Running default questions test...")
        default_results = tester.test_default_questions()
        results.extend(default_results)
        analyze_results(default_results)
    
    if args.mode in ["dataset", "all"]:
        logger.info("📊 Running dataset test...")
        dataset_results = tester.test_on_dataset(args.dataset_cache_dir, args.num_samples, args.skip_samples)
        results.extend(dataset_results)
        if args.mode == "dataset":  # Only analyze if not already done
            analyze_results(dataset_results)
    
    if args.mode in ["interactive", "all"]:
        tester.interactive_test()
    
    # Save results if any tests were run
    if results:
        save_test_results(results, args.output_file)
        logger.info(f"🎉 Testing completed! Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 