"""
Script to test and evaluate the LoRA fine-tuned DeepSeek-coder model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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

class LoRAModelTester:
    """Class to test the LoRA fine-tuned model"""
    
    def __init__(self, base_model_name: str, lora_adapter_path: str, device: str = "auto", checkpoint: str = None):
        """Initialize the LoRA model tester
        
        Args:
            base_model_name: Name/path of the base model (e.g., "deepseek-ai/deepseek-coder-7b-instruct-v1.5")
            lora_adapter_path: Path to the LoRA adapter directory
            device: Device to load the model on
            checkpoint: Specific checkpoint to load (e.g., "checkpoint-50", "checkpoint-6") or None for final
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.checkpoint = checkpoint
        self.model = None
        self.tokenizer = None
        
        # Determine the actual adapter path
        self.actual_adapter_path = self._resolve_adapter_path()
        
        self.load_model()
    
    def _resolve_adapter_path(self) -> str:
        """Resolve the actual adapter path based on checkpoint specification"""
        if self.checkpoint:
            # User specified a specific checkpoint
            checkpoint_path = os.path.join(self.lora_adapter_path, self.checkpoint)
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Checkpoint {self.checkpoint} not found in {self.lora_adapter_path}")
            
            # Validate checkpoint has required files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            for file in required_files:
                if not os.path.exists(os.path.join(checkpoint_path, file)):
                    raise ValueError(f"Checkpoint {self.checkpoint} missing required file: {file}")
            
            logger.info(f"🔍 Using specific checkpoint: {self.checkpoint}")
            return checkpoint_path
        else:
            # Use main directory (final/best model)
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            for file in required_files:
                if not os.path.exists(os.path.join(self.lora_adapter_path, file)):
                    raise ValueError(f"Main directory missing required file: {file}")
            
            logger.info("✅ Using FINAL/BEST LoRA adapter from main directory")
            return self.lora_adapter_path
    
    def _analyze_training_progress(self):
        """Analyze training progress and show which checkpoint we're using"""
        try:
            training_metrics_path = os.path.join(self.lora_adapter_path, "training_metrics.json")
            if os.path.exists(training_metrics_path):
                with open(training_metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Find final training metrics
                final_metrics = None
                for entry in reversed(metrics):
                    if 'train_loss' in entry and 'step' in entry:
                        final_metrics = entry
                        break
                
                if final_metrics:
                    logger.info("📊 Training Summary:")
                    logger.info(f"   Final Step: {final_metrics['step']}")
                    logger.info(f"   Final Loss: {final_metrics['train_loss']:.4f}")
                    logger.info(f"   Total Epochs: {final_metrics['epoch']}")
                    logger.info(f"   Training Time: {final_metrics['train_runtime']:.1f}s")
                
                # Show available checkpoints
                checkpoints = []
                for item in os.listdir(self.lora_adapter_path):
                    if item.startswith('checkpoint-') and os.path.isdir(os.path.join(self.lora_adapter_path, item)):
                        checkpoints.append(item)
                
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                    logger.info(f"📁 Available checkpoints: {', '.join(checkpoints)}")
                
                # Show what we're using
                if self.checkpoint:
                    logger.info(f"🎯 USING: {self.checkpoint}")
                    if self.checkpoint == checkpoints[-1]:
                        logger.info("   (This is the LATEST checkpoint)")
                    elif self.checkpoint == checkpoints[0]:
                        logger.info("   (This is the EARLIEST checkpoint)")
                    else:
                        logger.info("   (This is an INTERMEDIATE checkpoint)")
                else:
                    logger.info("🎯 USING: FINAL/BEST adapter (recommended for inference)")
                    logger.info("   (This is the polished version after training completion)")
                
        except Exception as e:
            logger.warning(f"Could not analyze training progress: {e}")
    
    def load_model(self):
        """Load the base model and LoRA adapter"""
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
        
        logger.info(f"Loading LoRA adapter from: {self.actual_adapter_path}")
        
        # Analyze training progress first
        self._analyze_training_progress()
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, self.actual_adapter_path)
        
        # Load tokenizer (try adapter path first, then fall back to main path)
        tokenizer_path = self.actual_adapter_path
        if not os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
            # Checkpoint dirs don't have tokenizer, use main directory
            tokenizer_path = self.lora_adapter_path
            logger.info("📝 Using tokenizer from main directory (checkpoints don't include tokenizer)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set model to evaluation mode
        self.model.eval()
        logger.info("🎉 LoRA model loaded successfully!")
        
        # Print model info
        self.print_model_info()
    
    def print_model_info(self):
        """Print information about the loaded model"""
        try:
            # Try to get trainable parameters info (if available)
            if hasattr(self.model, 'print_trainable_parameters'):
                logger.info("Model parameter info:")
                self.model.print_trainable_parameters()
            
            # Print device info
            device = next(self.model.parameters()).device
            logger.info(f"Model loaded on device: {device}")
            
            # Print adapter info
            if hasattr(self.model, 'peft_config'):
                config = self.model.peft_config['default']
                logger.info(f"LoRA rank: {config.r}, alpha: {config.lora_alpha}")
                logger.info(f"Target modules: {config.target_modules}")
                
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
        
        # Generate with optimized parameters for LoRA
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
        logger.info(f"Testing LoRA model on {num_samples} samples from the dataset...")
        
        # Load dataset
        ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
        questions = ds["train"]["question"]
        ground_truth_codes = ds["train"]["python_code"]
        
        # Take test samples (avoid training samples)
        total_samples = len(questions)
        # start_idx = 100 + skip_samples  # Start from sample 100 to avoid training samples
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
        
        logger.info(f"Testing LoRA model on {len(default_questions)} default questions...")
        
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
        print("\n🎯 LoRA Interactive Testing Mode")
        print("Type your questions and see the LoRA model generate Manim code!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("💬 Enter a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Generating code with LoRA model...")
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
    print("📊 LoRA TEST RESULTS ANALYSIS")
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
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned DeepSeek-coder model")
    parser.add_argument("--lora_path", type=str, 
                       default="/home/ubuntu/github/cursor-V2/deepseek-coder-manim-lora",
                       help="Path to the LoRA adapter directory")
    parser.add_argument("--base_model", type=str, 
                       default="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                       help="Base model name or path")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint to load (e.g., 'checkpoint-50', 'checkpoint-6'). If not specified, uses final/best adapter.")
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
    parser.add_argument("--output_file", type=str, default="lora_test_results.json",
                       help="Output file for test results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to load the model on")
    parser.add_argument("--list_checkpoints", action="store_true",
                       help="List available checkpoints and exit")
    
    args = parser.parse_args()
    
    # Check if LoRA adapter exists
    if not os.path.exists(args.lora_path):
        logger.error(f"LoRA adapter path {args.lora_path} does not exist")
        return
    
    # List checkpoints if requested
    if args.list_checkpoints:
        logger.info(f"📁 Analyzing LoRA directory: {args.lora_path}")
        
        # Check main directory
        main_files = ["adapter_config.json", "adapter_model.safetensors"]
        main_valid = all(os.path.exists(os.path.join(args.lora_path, f)) for f in main_files)
        
        print(f"\n🎯 MAIN DIRECTORY (RECOMMENDED FOR INFERENCE):")
        print(f"   Path: {args.lora_path}")
        print(f"   Status: {'✅ Valid' if main_valid else '❌ Invalid'}")
        if main_valid:
            print(f"   Contains: Final/best LoRA adapter after training completion")
        
        # List checkpoints
        checkpoints = []
        for item in os.listdir(args.lora_path):
            if item.startswith('checkpoint-') and os.path.isdir(os.path.join(args.lora_path, item)):
                checkpoint_path = os.path.join(args.lora_path, item)
                checkpoint_valid = all(os.path.exists(os.path.join(checkpoint_path, f)) for f in main_files)
                checkpoints.append((item, checkpoint_valid))
        
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x[0].split('-')[1]))
            print(f"\n📂 AVAILABLE CHECKPOINTS:")
            for checkpoint, valid in checkpoints:
                step = checkpoint.split('-')[1]
                status = '✅ Valid' if valid else '❌ Invalid'
                print(f"   {checkpoint}: {status} (training step {step})")
        else:
            print(f"\n📂 No checkpoints found")
        
        # Show training progress if available
        training_metrics_path = os.path.join(args.lora_path, "training_metrics.json")
        if os.path.exists(training_metrics_path):
            try:
                with open(training_metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                print(f"\n📊 TRAINING PROGRESS:")
                step_losses = [(entry['step'], entry['loss']) for entry in metrics if 'step' in entry and 'loss' in entry and entry['step'] % 10 == 0]
                for step, loss in step_losses:
                    print(f"   Step {step}: Loss {loss:.4f}")
                
                # Final summary
                final_entry = next((entry for entry in reversed(metrics) if 'train_loss' in entry), None)
                if final_entry:
                    print(f"\n🏁 FINAL RESULTS:")
                    print(f"   Final Step: {final_entry['step']}")
                    print(f"   Final Loss: {final_entry['train_loss']:.4f}")
                    print(f"   Training Time: {final_entry['train_runtime']:.1f}s")
                    
            except Exception as e:
                logger.warning(f"Could not read training metrics: {e}")
        
        print(f"\n💡 USAGE EXAMPLES:")
        print(f"   Use main directory (recommended):  --checkpoint (not specified)")
        print(f"   Use latest checkpoint:             --checkpoint checkpoint-50")
        print(f"   Use early checkpoint:              --checkpoint checkpoint-6")
        
        return
    
    # Check for required files in main directory or checkpoint
    if args.checkpoint:
        checkpoint_path = os.path.join(args.lora_path, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {args.checkpoint} not found in {args.lora_path}")
            logger.info("Use --list_checkpoints to see available checkpoints")
            return
        check_path = checkpoint_path
    else:
        check_path = args.lora_path
    
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in required_files:
        if not os.path.exists(os.path.join(check_path, file)):
            logger.error(f"Required file {file} not found in {check_path}")
            logger.info("Use --list_checkpoints to see valid options")
            return
    
    # Initialize LoRA model tester
    logger.info("🚀 Initializing LoRA Model Tester...")
    tester = LoRAModelTester(args.base_model, args.lora_path, args.device, args.checkpoint)
    
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
        # Add checkpoint info to filename if using specific checkpoint
        if args.checkpoint:
            base_name, ext = os.path.splitext(args.output_file)
            output_file = f"{base_name}_{args.checkpoint.replace('-', '_')}{ext}"
        else:
            output_file = args.output_file
            
        save_test_results(results, output_file)
        logger.info(f"🎉 Testing completed! Results saved to {output_file}")

if __name__ == "__main__":
    main() 