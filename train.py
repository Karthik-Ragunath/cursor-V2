import os
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    dataset_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/bespokelabs___bespoke-manim"
    model_cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/model/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    model_tokenizer_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai/tokenizer/models--deepseek-ai--deepseek-coder-7b-instruct-v1.5"
    ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=dataset_cache_dir)
    # print(ds)
    questions = ds["train"]["question"]
    python_code = ds["train"]["python_code"]
    print(type(questions)) # <class 'list'>
    print(type(python_code)) # <class 'list'>
    print(questions[0]) # str
    print(python_code[0]) # str
    
    # load deepseek coder-7b model
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True, cache_dir=model_tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True, cache_dir=model_cache_dir).cuda()

    # tokenize questions
    tokenized_questions = tokenizer(questions[0], return_tensors="pt", padding=True, truncation=True)
    print(tokenized_questions)
    
if __name__ == "__main__":
    main()
