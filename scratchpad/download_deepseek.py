from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim/deepseek-ai"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True, cache_dir=f"{cache_dir}/tokenizer")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True, cache_dir=f"{cache_dir}/model").cuda()