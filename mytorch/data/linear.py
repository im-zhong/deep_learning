from . import data
import torch


class SyntheticLinearRegressionData(data.DataModule):
    """Synthetic data for linear regression."""

    # 这种设计不好 我们为什么不区分数据集是train val test
    # 我们让上层取区分 因为底层生成数据的逻辑是完全一样的
    def __init__(self, w, b, num, noise=0.01):
        super().__init__()
        self.X = torch.randn((num, len(w)))
        noise = torch.randn((num, 1)) * noise
        self.y = self.X @ w.reshape((-1, 1)) + b + noise

    def __len__(self):
        return len(self.X)
