# 2025/7/12
# zhangzhong
# https://huggingface.co/datasets/FalconNet/GitHub-code-dialogs-1.2K-v0.1
# https://huggingface.co/datasets/codeparrot/github-code
# https://huggingface.co/datasets/codeparrot/github-code-clean

from datasets import load_dataset

# https://huggingface.co/datasets/codeparrot/github-code
# 具体怎么用，参考原始的仓库
# The GitHub Code dataset is a very large dataset，全部解压所有1T
# 所以还是用clean的版本，但是clean的版本估计也非常大
# 可以使用 languages 限制下载的语言
# 32GB
ds = load_dataset(
    "codeparrot/github-code-clean",
    split="train",
    # languages=["python"]
)
