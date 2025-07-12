# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/wikimedia/wikipedia
# https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data

# hugging face上的wikipedia数据集 只到了2023年
# kaggle上的数据集到了2025年，我估计有人传到hugging face上了，我要搜一下
# https://huggingface.co/datasets/wikimedia/structured-wikipedia 这是结构化的，不是原始的文本

from datasets import load_dataset

# 70GB
ds = load_dataset("wikimedia/wikipedia", "20231101.en")
