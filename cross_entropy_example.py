import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def demonstrate_cross_entropy():
    """Demonstrate cross-entropy loss with a simple example"""
    
    print("=== Cross-Entropy Loss Example ===\n")
    
    # Step 1: Define vocabulary
    vocab = {0: "circle", 1: "=", 2: "Circle()", 3: "<EOS>"}
    print("Vocabulary:")
    for idx, token in vocab.items():
        print(f"  {idx}: '{token}'")
    
    # Step 2: Model predictions (logits)
    print(f"\nScenario: Model needs to predict next token after 'circle ='")
    print("Expected answer: 'Circle()' (index 2)")
    
    logits = torch.tensor([1.2, 0.8, 3.1, 0.5])
    print(f"\nModel logits: {logits.tolist()}")
    
    # Step 3: Convert to probabilities
    probabilities = F.softmax(logits, dim=0)
    print(f"Probabilities: {probabilities.tolist()}")
    
    for idx, (token, prob) in enumerate(zip(vocab.values(), probabilities)):
        print(f"  P('{token}') = {prob:.4f} ({prob*100:.1f}%)")
    
    # Step 4: Calculate cross-entropy loss
    correct_token_idx = 2  # "Circle()"
    target = torch.tensor(correct_token_idx)
    
    # Method 1: Manual calculation
    correct_prob = probabilities[correct_token_idx]
    manual_loss = -torch.log(correct_prob)
    
    # Method 2: PyTorch CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    pytorch_loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
    
    print(f"\nLoss Calculation:")
    print(f"Probability of correct token: {correct_prob:.4f}")
    print(f"Cross-entropy loss: -log({correct_prob:.4f}) = {manual_loss:.4f}")
    print(f"PyTorch CrossEntropyLoss: {pytorch_loss:.4f}")
    
    # Step 5: Show different scenarios
    print(f"\n=== Different Scenarios ===")
    
    scenarios = [
        ("Perfect prediction", torch.tensor([0.0, 0.0, 10.0, 0.0])),
        ("Good prediction", torch.tensor([0.5, 0.3, 4.0, 0.2])),  
        ("Random guess", torch.tensor([1.0, 1.0, 1.0, 1.0])),
        ("Wrong prediction", torch.tensor([4.0, 3.0, 0.1, 2.0]))
    ]
    
    for name, scenario_logits in scenarios:
        scenario_probs = F.softmax(scenario_logits, dim=0)
        scenario_loss = criterion(scenario_logits.unsqueeze(0), target.unsqueeze(0))
        correct_confidence = scenario_probs[correct_token_idx]
        
        print(f"\n{name}:")
        print(f"  Confidence in correct answer: {correct_confidence:.4f} ({correct_confidence*100:.1f}%)")
        print(f"  Cross-entropy loss: {scenario_loss:.4f}")

def demonstrate_sequence_loss():
    """Demonstrate how cross-entropy works across a sequence"""
    
    print(f"\n\n=== Sequence-Level Loss Example ===")
    
    # Example sequence: "circle = Circle()"
    sequence_text = ["circle", "=", "Circle()"]
    sequence_ids = [0, 1, 2]  # Token IDs
    
    print(f"Training sequence: {' '.join(sequence_text)}")
    print(f"Token IDs: {sequence_ids}")
    
    # Simulate model predictions at each position
    # Position 0: predict "=" given "circle"  
    # Position 1: predict "Circle()" given "circle ="
    
    predictions = [
        torch.tensor([0.1, 2.8, 0.5, 0.2]),  # After "circle", predict "="
        torch.tensor([0.3, 0.1, 3.2, 0.4])   # After "circle =", predict "Circle()"
    ]
    
    targets = torch.tensor([1, 2])  # Correct next tokens
    
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nPosition-by-position loss:")
    for pos, (logits, target) in enumerate(zip(predictions, targets)):
        probs = F.softmax(logits, dim=0)
        loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
        total_loss += loss
        
        print(f"  Position {pos}: Predict '{sequence_text[target]}' (ID {target})")
        print(f"    Confidence: {probs[target]:.4f} ({probs[target]*100:.1f}%)")
        print(f"    Loss: {loss:.4f}")
    
    avg_loss = total_loss / len(predictions)
    print(f"\nAverage loss across sequence: {avg_loss:.4f}")

def demonstrate_training_effect():
    """Show how cross-entropy loss guides training"""
    
    print(f"\n\n=== How Loss Guides Training ===")
    
    # Simulate training progress
    epochs = [
        ("Epoch 1 (Untrained)", torch.tensor([1.0, 1.0, 1.0, 1.0])),  # Random
        ("Epoch 5", torch.tensor([0.8, 0.9, 2.1, 0.7])),              # Learning
        ("Epoch 10", torch.tensor([0.5, 0.6, 3.0, 0.4])),             # Better
        ("Epoch 20", torch.tensor([0.2, 0.3, 4.2, 0.1]))              # Best
    ]
    
    target = torch.tensor(2)  # Correct answer is "Circle()"
    criterion = nn.CrossEntropyLoss()
    
    print("Training progress (predicting 'Circle()' after 'circle ='):")
    
    for epoch_name, logits in epochs:
        probs = F.softmax(logits, dim=0)
        loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
        confidence = probs[2]
        
        print(f"\n{epoch_name}:")
        print(f"  Confidence in correct answer: {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"  Cross-entropy loss: {loss:.4f}")
        
        # Show what gradients will do
        if loss > 1.0:
            print("  → Gradients will strongly push toward correct answer")
        elif loss > 0.5:
            print("  → Gradients will moderately adjust weights")
        else:
            print("  → Small gradient updates, model is confident")

if __name__ == "__main__":
    demonstrate_cross_entropy()
    demonstrate_sequence_loss()
    demonstrate_training_effect() 