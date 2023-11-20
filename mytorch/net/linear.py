# 2023/11/17
# zhangzhong
# linear layer

import torch
from torch import nn
import mytorch.config
from torch import Tensor
from mytorch import func


class LinearRegressionScratch(nn.Module):
    # 这是我们的第一个模型
    def __init__(self, in_features):
        super().__init__()
        # 初始化参数
        # 默认使用N(0, 0.0.^2)
        # 但是模型的大小要怎么确定呢??
        # TODO: 你这个不对，初始化的参数的正态分布的方差应该是0.01
        default_device = mytorch.config.conf['device']
        self.w = torch.randn((in_features, 1), requires_grad=True,
                             device=torch.device(default_device))
        self.b = torch.randn((1,), requires_grad=True,
                             device=torch.device(default_device))
        # AttributeError: Can't pickle local object 'LinearRegressionScratch.__init__.<locals>.<lambda>'
        # self.net = lambda X: X @ self.w + self.b

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
        return X @ self.w + self.b

# net = torch.nn.Linear(in_feature, out_feature)
# Applies a linear transformation to the incoming data: y = xA^T + b
# net.weight, net.bias

# torch.nn.LazyLinear(out_feature)

# in_feature就是输入训练数据的维数，也就是一个神经元内部的w的个数
# out_feature就是一层的神经元的个数，也就是输出向量的维数


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.LazyLinear(1)

    def forward(self, X):
        return self.net(X)


class LinearClassifierScratch(nn.Module):

    def __init__(self, in_features, out_features, sigma=0.01):
        super().__init__()

        # 定义参数和网络结构
        # W: (in_features, out_features)
        # B: (out_features, 1)
        # y_hat = X @ W + B

        # TODO: randn how to set the std
        # 其实你可以发现我们的网络结构和linear非常像
        # 其实就是一个线性层，只不过我们的out_features是10 而不是1
        # 原来是我用错了函数 应该是 torch.normal
        default_device = mytorch.config.conf['device']
        self.w = torch.randn((in_features, out_features),
                             requires_grad=True, device=torch.device(default_device))
        self.b = torch.randn(
            (1, out_features), requires_grad=True, device=torch.device(default_device))

    def forward(self, x):
        # step1, 将x展平
        # x.shape = (batch_size, channel, height, width)
        # TODO: 我们会发现将数据展平这个操作可以发生在Dataset的预处理阶段
        # 也可以发生在网络的计算阶段
        # 那么到底放在哪里？？这是个问题
        x = torch.flatten(x, start_dim=1)
        y_hat = x @ self.w + self.b
        return y_hat

    def parameters(self):
        yield self.w
        yield self.b


class LinearClassifier(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        # A sequential container
        # Modules will be added to it in the order they are passed in the constructor
        # torch.nn.Sequential(*args: Module)
        self.net = nn.Sequential(
            # Flattens a contiguous range of dims into a tensor.
            # For use with Sequential.
            nn.Flatten(),
            nn.LazyLinear(out_features)
        )

    def forward(self, x):
        return self.net(x)


class MLPScratch(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_1, num_hidden_2, dropout_1=0.0, dropout_2=0.0) -> None:
        super().__init__()
        # we should not use this dynamic add attr to a class
        # it is bad for type checker
        # self.make_parameters_be_attributes()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        default_device: str = mytorch.config.conf['device']
        self.W1 = torch.normal(
            0, 0.01, (in_features, num_hidden_1), requires_grad=True, device=torch.device(default_device))
        self.b1 = torch.normal(0, 0.01, (1, num_hidden_1),
                               requires_grad=True, device=torch.device(default_device))
        self.W2 = torch.normal(
            0, 0.01, (num_hidden_1, num_hidden_2), requires_grad=True, device=torch.device(default_device))
        self.b2 = torch.normal(0, 0.01, (1, num_hidden_2),
                               requires_grad=True, device=torch.device(default_device))
        self.W3 = torch.normal(
            0, 0.01, (num_hidden_2, out_features), requires_grad=True, device=torch.device(default_device))
        self.b3 = torch.normal(0, 0.01, (1, out_features),
                               requires_grad=True, device=torch.device(default_device))

    def forward(self, X: Tensor) -> Tensor:
        # step 1. flatten
        x = torch.flatten(X, start_dim=1)
        h1_linear = x @ self.W1 + self.b1
        h1_dropout = func.dropout_layer(h1_linear, self.dropout_1)
        h1_relu = func.relu_layer(h1_dropout)
        h2_linear = h1_relu @ self.W2 + self.b2
        h2_dropout = func.dropout_layer(h2_linear, self.dropout_2)
        h2_relu = func.relu_layer(h2_dropout)
        y_hat = h2_relu @ self.W3 + self.b3
        return y_hat

    def parameters(self):
        yield self.W1
        yield self.b1
        yield self.W2
        yield self.b2
        yield self.W3
        yield self.b3


class MLP(nn.Module):
    def __init__(self, out_features, num_hidden_1, num_hidden_2, dropout_1, dropout_2):
        super().__init__()
        # self.make_parameters_be_attributes()
        self.out_features = out_features
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.LazyLinear(num_hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.LazyLinear(out_features)
        )

    def forward(self, X):
        return self.net(X)
