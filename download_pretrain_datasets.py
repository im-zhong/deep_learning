# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/arxiv-community/arxiv_dataset 这个数据集很大，但是是论文数据集，先不用了
# https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl
# 太多了，就不下了。。。

# 这个整合了好多的数据集
# https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

from datasets import load_dataset


# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/allenai/c4
# https://www.tensorflow.org/datasets/catalog/c4
# https://arxiv.org/pdf/1910.10683

# English only
# en = load_dataset("allenai/c4", "en")
#
# ds = load_dataset(

#     "flax-sentence-embeddings/stackexchange_title_best_voted_answer_jsonl",
#     # split="train",
#     # streaming=True,
# )
#

# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/FalconNet/GitHub-code-dialogs-1.2K-v0.1
# https://huggingface.co/datasets/codeparrot/github-code
# https://huggingface.co/datasets/codeparrot/github-code-clean

# from datasets import load_dataset

# https://huggingface.co/datasets/codeparrot/github-code
# 具体怎么用，参考原始的仓库
# The GitHub Code dataset is a very large dataset，全部解压所有1T
# 所以还是用clean的版本，但是clean的版本估计也非常大
# 可以使用 languages 限制下载的语言
# ds = load_dataset(
# 32GB
#     "codeparrot/github-code-clean",
#     split="train",
#     # languages=["python"]
# )
#

# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/wikimedia/wikipedia
# https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data

# hugging face上的wikipedia数据集 只到了2023年
# kaggle上的数据集到了2025年，我估计有人传到hugging face上了，我要搜一下
# https://huggingface.co/datasets/wikimedia/structured-wikipedia 这是结构化的，不是原始的文本

# from datasets import load_dataset

# 70GB
# ds = load_dataset("wikimedia/wikipedia", "20231101.en")


# ds = load_dataset(
#     "manu/project_gutenberg",
#     split="en",
#     # Lazily loads the data sample-by-sample over the internet. do not download to the disk
#     # streaming=True
# )

# 14G
# https://huggingface.co/datasets/ArmelR/stack-exchange-instruction
# ds = load_dataset("ArmelR/stack-exchange-instruction")
# ds = load_dataset("donfu/oa-stackexchange")
# 3G
# https://huggingface.co/datasets/nampdn-ai/tiny-codes
# ds = load_dataset("nampdn-ai/tiny-codes")


gutenberg_en = load_dataset(path="eminorhan/gutenberg_en", name="chunk_size_1024")
