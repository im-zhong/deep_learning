# 2023/11/17
# zhangzhong
# linear layer

import torch
from torch import nn


class LinearRegressionScratch(nn.Module):
    # 这是我们的第一个模型
    def __init__(self, in_features):
        super().__init__()
        # 初始化参数
        # 默认使用N(0, 0.0.^2)
        # 但是模型的大小要怎么确定呢??
        # TODO: 你这个不对，初始化的参数的正态分布的方差应该是0.01
        self.w = torch.randn((in_features, 1), requires_grad=True)
        self.b = torch.randn((1,), requires_grad=True)
        self.net = lambda X: X @ self.w + self.b

    def parameters(self):
        """
        Returns an iterator over module parameters.

        code example:
            for name, param in self.named_parameters(recurse=recurse):
                yield param
        """
        # 返回模型的参数
        # return self.w, self.b
        yield self.w
        yield self.b

    def forward(self, X):
        return self.net(X)

# net = torch.nn.Linear(in_feature, out_feature)
# Applies a linear transformation to the incoming data: y = xA^T + b
# net.weight, net.bias

# torch.nn.LazyLinear(out_feature)

# in_feature就是输入训练数据的维数，也就是一个神经元内部的w的个数
# out_feature就是一层的神经元的个数，也就是输出向量的维数
