"""
Utility script to monitor training progress and visualize metrics
"""

import json
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict
import time

def load_training_metrics(output_dir: str) -> List[Dict]:
    """Load training metrics from the output directory"""
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return []
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_training_metrics(metrics: List[Dict], output_dir: str):
    """Plot training metrics"""
    if not metrics:
        print("No metrics to plot")
        return
    
    # Extract training and eval losses
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    for entry in metrics:
        if 'train_loss' in entry:
            train_losses.append(entry['train_loss'])
            train_steps.append(entry['step'])
        
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry['step'])
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss plot
    if train_losses:
        ax1.plot(train_steps, train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        ax1.grid(True)
    
    # Evaluation loss plot
    if eval_losses:
        ax2.plot(eval_steps, eval_losses, 'r-', label='Evaluation Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Evaluation Loss Over Time')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training metrics plot saved to {os.path.join(output_dir, 'training_metrics.png')}")

def monitor_training_live(output_dir: str, refresh_interval: int = 30):
    """Monitor training progress in real-time"""
    print(f"Monitoring training progress in {output_dir}")
    print(f"Refreshing every {refresh_interval} seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Check if training is still running
            trainer_state_file = os.path.join(output_dir, "trainer_state.json")
            
            if os.path.exists(trainer_state_file):
                with open(trainer_state_file, 'r') as f:
                    state = json.load(f)
                
                print(f"\nCurrent step: {state.get('global_step', 'Unknown')}")
                print(f"Current epoch: {state.get('epoch', 'Unknown')}")
                
                if 'log_history' in state and state['log_history']:
                    latest_log = state['log_history'][-1]
                    if 'train_loss' in latest_log:
                        print(f"Latest training loss: {latest_log['train_loss']:.4f}")
                    if 'eval_loss' in latest_log:
                        print(f"Latest eval loss: {latest_log['eval_loss']:.4f}")
            else:
                print("Training not started yet or trainer_state.json not found")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring")

def print_training_summary(output_dir: str):
    """Print a summary of the training"""
    metrics = load_training_metrics(output_dir)
    
    if not metrics:
        print("No training metrics found")
        return
    
    print("=== Training Summary ===")
    
    # Find final metrics
    final_train_loss = None
    final_eval_loss = None
    
    for entry in reversed(metrics):
        if final_train_loss is None and 'train_loss' in entry:
            final_train_loss = entry['train_loss']
        if final_eval_loss is None and 'eval_loss' in entry:
            final_eval_loss = entry['eval_loss']
        
        if final_train_loss is not None and final_eval_loss is not None:
            break
    
    print(f"Final training loss: {final_train_loss:.4f}" if final_train_loss else "No final training loss found")
    print(f"Final evaluation loss: {final_eval_loss:.4f}" if final_eval_loss else "No final evaluation loss found")
    
    # Count total steps
    total_steps = max([entry.get('step', 0) for entry in metrics])
    print(f"Total training steps: {total_steps}")
    
    # Check if model was saved
    model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
    saved_files = [f for f in model_files if os.path.exists(os.path.join(output_dir, f))]
    
    if saved_files:
        print(f"Model saved successfully. Found files: {saved_files}")
    else:
        print("Warning: No model files found in output directory")

def main():
    parser = argparse.ArgumentParser(description="Monitor SFT training progress")
    parser.add_argument("--output_dir", type=str, default="./deepseek-coder-manim-finetuned",
                       help="Training output directory")
    parser.add_argument("--mode", type=str, choices=["plot", "live", "summary"], default="summary",
                       help="Monitoring mode")
    parser.add_argument("--refresh_interval", type=int, default=30,
                       help="Refresh interval for live monitoring (seconds)")
    
    args = parser.parse_args()
    
    if args.mode == "plot":
        metrics = load_training_metrics(args.output_dir)
        plot_training_metrics(metrics, args.output_dir)
    elif args.mode == "live":
        monitor_training_live(args.output_dir, args.refresh_interval)
    elif args.mode == "summary":
        print_training_summary(args.output_dir)

if __name__ == "__main__":
    main() 