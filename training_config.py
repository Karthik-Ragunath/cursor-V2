"""
Configuration file for SFT training parameters
"""

import os

class TrainingConfig:
    """Configuration class for SFT training"""
    
    # Data configuration
    DATASET_CACHE_DIR = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    MODEL_CACHE_DIR = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    MODEL_TOKENIZER_DIR = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    TRAIN_SPLIT_RATIO = 0.9
    
    # Model configuration
    MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    MAX_LENGTH = 2048
    
    # Training configuration
    OUTPUT_DIR = "./deepseek-coder-manim-finetuned"
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    
    # Memory and efficiency settings
    USE_FP16 = True
    USE_GRADIENT_CHECKPOINTING = True
    
    # Generation settings for testing
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    DO_SAMPLE = True
    
    @classmethod
    def get_prompt_template(cls):
        """Get the prompt template for formatting questions and answers"""
        return "### Question:\n{question}\n\n### Code:\n{code}<|EOT|>"
    
    @classmethod
    def get_inference_prompt_template(cls):
        """Get the prompt template for inference"""
        return "### Question:\n{question}\n\n### Code:\n" 