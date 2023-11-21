# 2023/9/9
# zhangzhong

import torch
import numpy as np
from torch import Tensor
from typing import Iterator
from torch import nn


class Loss:
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        raise NotImplementedError


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_hat, y):
        return torch.mean((y_hat - y)**2) / 2

# todo: weight_decay 需要 module.parameters().weights


# class MSELossWithWeightDecay(Loss):

#     def __init__(self, parameters: Iterator[Tensor], weight_decay=0.0):
#         super().__init__()
#         self.parameters = list(parameters)
#         self.weight: Tensor | None = None
#         self.weight_decay = weight_decay

#     # nn.Parameters() register the attr to the model, so we can do a lot of other useful things
#     # https://stackoverflow.com/questions/64507404/defining-named-parameters-for-a-customized-nn-module-in-pytorch
#     # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters

#     def __call__(self, y_hat, y):
#         # weight panalty = weight_decay * (w1**2 + w2**2 + ... + wn**2) / 2
#         if self.weight is None:
#             # TODO: 或者我们可以指定parameters的名字吗 那样或许可以拿到weight
#             # 这里想要拿到weight就需要使用named_parameters 还是算了
#             self.weight = self.parameters[0]
#         return torch.mean((y_hat - y)**2) / 2 + self.weight_decay * (self.weight**2).sum() / 2

# 我们应该写两个类
# 一个是CrossEntropyLoss的简单实现
# 另一个是复杂实现
# 两者计算的loss在logits in [-90, 90] 之间应该是大致相同的


class NaiveCrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_hat, y):
        """
        y_hat: shape=(batch_size, num_labels)
        y: shape=(batch_size,)
        """

        # step 1. softmax
        # p.shape = (batch_size, num_labels)
        p = self.softmax(y_hat)
        assert p.shape == y_hat.shape

        # step 2. cross entropy
        # 找到概率矩阵p中每一行对应的label的概率
        batch_size, num_labels = y_hat.shape
        pl = p[list(range(batch_size)), y]
        assert pl.shape == (batch_size,)

        loss = -torch.log(pl)
        return loss.mean()

    def softmax(self, logits):
        """
        logits: shape=(batch_size, num_labels)
        """

        exp_logits = torch.exp(logits)
        # 将每一行的exp求和，并且保持dim
        sumexp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
        # 保持dim才能保证这里的boardcasting是沿行扩展的
        p_matrix = exp_logits / sumexp_logits
        return p_matrix

# TODO: 现在为了实现RNN，CrossEntropy需要考虑三维的向量


class CrossEntropyLoss(Loss):
    def __init__(self, calculate_mean: bool = True):
        super().__init__()
        self.calculate_mean = calculate_mean

    def __call__(self, y_hat: Tensor, y: Tensor):
        """
        y_hat: shape=(batch_size, num_labels) logits, 也就是线性输出
        y: shape=(batch_size,) labels, 代表是的类别 0, 1, 2 ...
        """
        # do the logsumexp trick
        # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        # 因为y_hat是一个矩阵 shape=(batch_size, num_labels)
        # 也就是每一行都是一个样本的logits，所以我们需要沿着行的方向
        # 找到最大的logits
        # keepdim=True, c.shape=(batch_size, 1)
        # keepdim=False, c.shape=(batch_size,)

        # batch_size, num_labels = y_hat.shape

        # 在RNN实现的时候，y_hat是个三维矩阵, y是二维矩阵
        # 所以这些代码不在适用
        batch_size, *_, num_labels = y_hat.shape
        assert batch_size == y_hat.shape[0]
        assert num_labels == y_hat.shape[-1]

        y_hat = y_hat.reshape(shape=(-1, num_labels))
        y = y.reshape(-1)

        # key
        # keepdim=True 可以保证 y_hat-c 的时候 boardcasting是沿行扩展的
        # https://pytorch.org/docs/stable/generated/torch.max.html
        # Returns the maximum value of all elements in the input tensor.
        # out: (max, max_indices)
        # 这里必须要keepdim 因为之后的 logsumexp的计算需要用到c的广播
        c, _ = torch.max(y_hat, dim=1, keepdim=True)
        # c, _ = torch.max(y_hat, dim=1)
        # BUGFIX：这里如果我们输出的是三维的数组，那么batch_size显然是小了 所以我们需要重新获取batch_size
        batch_size, num_labels = y_hat.shape
        l = y_hat[list(range(batch_size)), y]

        # loss = -(p - c - torch.log(torch.sum(torch.exp(y_hat-c), dim=1)))
        # 这个公式太长了 引入几个中间变量 一步一步的写吧
        # loss = torch.log(torch.sum(torch.exp(y_hat-c), dim=1)) - p - c

        # y_hat矩阵的所有行都减去本行最大的数字
        # norm_y_hat = y_hat - c
        # # 之后所有值都取指数
        # exp_y_hat = torch.exp(norm_y_hat)
        # # 然后沿行方向求和
        # sumed_y_hat = torch.sum(exp_y_hat, dim=1, keepdim=True)
        # # 求和完成之后对每个元素求log
        # log_y_hat = torch.log(sumed_y_hat)

        logsumexp = torch.log(torch.sum(torch.exp(y_hat-c), dim=1))

        # 在计算完毕之后 在将c进行一个squeeze 因为计算的loss的时候我们不应该再广播了
        c = c.squeeze()
        # 最后 根据公式写出表达式
        loss = logsumexp - l + c

        # 取整个minibatch的loss的平均值
        # 因为我们每次的batch数量有可能不一样，就是最后一个batch可能会少一些
        # 所以为了公平对比，对整个batch的loss取平均
        if self.calculate_mean:
            return loss.mean()
        else:
            return loss


# class LossFactory:
#     def __init__(self, name: str):
#         self.name = name

#     def new(self, model: nn.Module, optimizer):
#         if
