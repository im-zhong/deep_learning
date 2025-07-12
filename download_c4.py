# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/allenai/c4
# https://www.tensorflow.org/datasets/catalog/c4
# https://arxiv.org/pdf/1910.10683


from datasets import load_dataset

# English only
en = load_dataset("allenai/c4", "en")
