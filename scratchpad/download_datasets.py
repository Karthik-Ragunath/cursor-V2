from datasets import load_dataset

# Specify the directory where you want to download the dataset
download_directory = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/text-to-manim"

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("bespokelabs/bespoke-manim", cache_dir=download_directory)