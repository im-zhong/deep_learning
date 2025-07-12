# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/manu/project_gutenberg


from datasets import load_dataset

ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)
