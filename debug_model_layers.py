"""
Debug script to inspect DeepSeek model architecture
"""

import torch
from transformers import AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_model_layers():
    """Inspect DeepSeek model to find correct layer names for LoRA"""
    
    logger.info("Loading DeepSeek model for inspection...")
    
    # Load model (small cache for quick inspection)
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Keep on CPU for inspection
    )
    
    logger.info("Model loaded! Inspecting layers...")
    
    # Find all linear layers
    linear_layers = []
    attention_layers = []
    mlp_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
            
            # Categorize by likely function
            name_lower = name.lower()
            if any(x in name_lower for x in ['q_', 'k_', 'v_', 'query', 'key', 'value', 'attn']):
                attention_layers.append(name)
            elif any(x in name_lower for x in ['mlp', 'feed', 'gate', 'up', 'down']):
                mlp_layers.append(name)
    
    print("\n" + "="*60)
    print("🔍 DEEPSEEK MODEL LAYER INSPECTION")
    print("="*60)
    
    print(f"\nTotal Linear Layers Found: {len(linear_layers)}")
    
    print(f"\n📋 First 20 Linear Layers:")
    for i, layer in enumerate(linear_layers[:20]):
        print(f"  {i+1:2d}. {layer}")
    if len(linear_layers) > 20:
        print(f"  ... and {len(linear_layers) - 20} more")
    
    print(f"\n🎯 Likely Attention Layers ({len(attention_layers)}):")
    for layer in attention_layers[:10]:
        print(f"  - {layer}")
    if len(attention_layers) > 10:
        print(f"  ... and {len(attention_layers) - 10} more")
    
    print(f"\n🧠 Likely MLP Layers ({len(mlp_layers)}):")
    for layer in mlp_layers[:10]:
        print(f"  - {layer}")
    if len(mlp_layers) > 10:
        print(f"  ... and {len(mlp_layers) - 10} more")
    
    # Try to find common patterns
    print(f"\n🔍 Layer Name Patterns:")
    
    # Extract unique endings
    endings = set()
    for layer in linear_layers:
        if '.' in layer:
            ending = layer.split('.')[-1]
            endings.add(ending)
    
    print("Common layer endings:")
    for ending in sorted(endings):
        count = sum(1 for layer in linear_layers if layer.endswith(ending))
        print(f"  - '{ending}': {count} layers")
    
    # Suggest LoRA targets
    print(f"\n💡 SUGGESTED LORA TARGET MODULES:")
    
    # Find the most common attention patterns
    attention_patterns = []
    for ending in endings:
        if any(x in ending.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            attention_patterns.append(ending)
    
    mlp_patterns = []
    for ending in endings:
        if any(x in ending.lower() for x in ['gate_proj', 'up_proj', 'down_proj']):
            mlp_patterns.append(ending)
    
    if attention_patterns:
        print("Attention modules:", attention_patterns)
    if mlp_patterns:
        print("MLP modules:", mlp_patterns)
    
    suggested_targets = attention_patterns + mlp_patterns
    if suggested_targets:
        print(f"\nRecommended target_modules list:")
        print(f"target_modules = {suggested_targets}")
    else:
        print("\n⚠️  Could not auto-detect standard patterns.")
        print("You may need to manually specify target modules.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    inspect_model_layers() 