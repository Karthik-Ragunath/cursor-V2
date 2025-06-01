"""
Minimal test for LoRA setup
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def test_lora_minimal():
    print("🧪 Testing LoRA setup...")
    
    # Load tiny model for testing
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        cache_dir="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print("✅ Base model loaded")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    print("✅ Model prepared for training")
    
    # Create LoRA config - start with minimal
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # Very small rank
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj"],  # Only one module to start
        bias="none",
    )
    
    print("✅ LoRA config created")
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print("✅ LoRA applied")
    
    # Check trainable parameters
    model.print_trainable_parameters()
    
    # Count actual trainable params
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape, param.numel()))
    
    print(f"\nDetailed trainable parameters ({len(trainable_params)}):")
    total_trainable = 0
    for name, shape, count in trainable_params[:10]:  # Show first 10
        print(f"  {name}: {shape} ({count:,} params)")
        total_trainable += count
    
    if len(trainable_params) > 10:
        remaining = sum(count for _, _, count in trainable_params[10:])
        total_trainable += remaining
        print(f"  ... and {len(trainable_params) - 10} more ({remaining:,} params)")
    
    print(f"Total trainable: {total_trainable:,}")
    
    # Test simple forward pass
    print("\n🧪 Testing forward pass...")
    text = "### Question:\nCreate a red circle\n\n### Code:\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Test forward pass with gradients
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    print(f"✅ Forward pass successful, loss: {loss.item():.4f}")
    
    # Test backward pass
    print("🧪 Testing backward pass...")
    try:
        loss.backward()
        print("✅ Backward pass successful!")
        
        # Check if gradients are computed
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        print(f"✅ Gradients computed for {grad_count} parameters")
        
        if grad_count == 0:
            print("❌ NO GRADIENTS COMPUTED!")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

if __name__ == "__main__":
    test_lora_minimal() 