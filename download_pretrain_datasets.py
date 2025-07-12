# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/arxiv-community/arxiv_dataset 这个数据集很大，但是是论文数据集，先不用了
# https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl
# 太多了，就不下了。。。

from datasets import load_dataset

ds = load_dataset(
    "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
    # split="train",
    streaming=True,
)
