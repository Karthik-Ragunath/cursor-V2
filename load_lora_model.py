"""
Script to load and test a trained LoRA adapter
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lora_model(base_model_path: str, lora_adapter_path: str, device: str = "auto"):
    """Load base model and LoRA adapter"""
    
    logger.info("Loading base model...")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    logger.info("Loading tokenizer...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    
    logger.info("Loading LoRA adapter...")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    
    # Merge LoRA weights for faster inference (optional)
    # model = model.merge_and_unload()
    
    model.eval()
    logger.info("LoRA model loaded successfully!")
    
    return model, tokenizer

def generate_code(model, tokenizer, question: str, max_new_tokens: int = 256):
    """Generate code for a given question"""
    
    # Format the prompt
    prompt = f"### Question:\n{question}\n\n### Code:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode and clean
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = generated_text[len(prompt):]
    
    if "<|EOT|>" in generated_code:
        generated_code = generated_code.split("<|EOT|>")[0]
    
    return generated_code.strip()

def interactive_mode(model, tokenizer):
    """Interactive mode for testing the model"""
    
    logger.info("Entering interactive mode. Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\nEnter your question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question.strip():
                continue
            
            print("Generating code...")
            code = generate_code(model, tokenizer, question)
            print(f"\nGenerated Code:\n{code}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")

def main():
    parser = argparse.ArgumentParser(description="Load and test LoRA fine-tuned model")
    parser.add_argument(
        "--lora_path", 
        type=str, 
        default="./deepseek-coder-manim-lora",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        help="Base model name or path"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default=None,
        help="Single question to test"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Enter interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model, tokenizer = load_lora_model(args.base_model, args.lora_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Test mode
    if args.question:
        logger.info(f"Testing with question: {args.question}")
        code = generate_code(model, tokenizer, args.question)
        print(f"\nGenerated Code:\n{code}")
    
    elif args.interactive:
        interactive_mode(model, tokenizer)
    
    else:
        # Default test questions
        test_questions = [
            "Create a red circle",
            "Draw a blue square that moves to the right",
            "Make a green triangle that rotates",
            "Create text that says 'Hello World' and make it fade in"
        ]
        
        logger.info("Testing with default questions...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"Test {i}: {question}")
            print('='*50)
            
            try:
                code = generate_code(model, tokenizer, question)
                print(f"Generated Code:\n{code}")
            except Exception as e:
                logger.error(f"Failed to generate for question {i}: {e}")

if __name__ == "__main__":
    main() 