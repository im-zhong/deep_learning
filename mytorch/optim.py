# 2023/9/10
# zhangzhong

# class Optimizer:
#     def __init__(self, params, lr, weight_decay=None):
#         self.params = params,
#         self.lr = lr
#         self.weight_decay = weight_decay

#     def step(self):
#         raise NotImplementedError

#     def zero_grad(self):
#         for param in self.params:
#             param.grad = None

import torch
from torch import nn, Tensor
from typing import Iterator
import math


def magnitude(params: list[Tensor]):
    with torch.no_grad():
        m = 0.0
        for p in params:
            if p.grad is not None:
                m += float(torch.sum(p.grad**2))
        m = math.sqrt(m)
        return m


def grad_clip(params: list[Tensor], clip=1.0):
    # 我们拿到一系列的tensor
    # 我们将所有的参数向量统统合起来 看成一个向量
    # 然后我们计算这个向量的模
    # 如果他的大小超过所给定的阈值，那么就将向量大小设置为该阈值
    # 但是切记 TIP: 向量的方向不应该发生改变
    # BUG:FIX: 不把params从迭代器中拿出来，对p.grad的修改不会生效！！！
    # params = list(params)
    with torch.no_grad():
        m = 0.0
        for p in params:
            # 卧槽 python的类型系统竟然可以检测if语句对None的判断 厉害了
            if p.grad is not None:
                m += float(torch.sum(p.grad**2))
        m = math.sqrt(m)
        if m >= clip:
            for p in params:
                if p.grad is not None:
                    p.grad *= (clip / m)


class SGD():
    # params = sequence of tensors
    # 只有这样 我们可以for遍历所有的param，然后使用梯度下降
    def __init__(self, params: Iterator[Tensor], lr, weight_decay=0.0, gradient_clip=None):
        # why this params is tuple of tuple
        # self.params_it = params
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

    def magnitude(self):
        with torch.no_grad():
            m = 0.0
            for p in self.params:
                # 实际上这里只需要计算范数 咱们用pytorch的库来实现吧 就不会有这些类型问题了
                # 或者使用if判断类型 就不会爆警告了
                if p.grad is not None:
                    m += float(torch.sum(p.grad**2))
            m = math.sqrt(m)
            return m

    # # TIP: only can call within torch.no_grad()
    # def grad_clip(self):
    #     if self.gradient_clip is not None:
    #         m = self.magnitude()
    #         if m >= self.gradient_clip:
    #             for p in self.params:
    #                 p.grad *= (self.gradient_clip / m)

    # 不对呀 这样只有我们自己的优化器可以用这个trainer类 pytorch的就用不了了
    # def set_parameters(self, params: Iterator[Tensor]) -> None:
    #     self.params = list(params)

    def step(self):
        if self.gradient_clip is not None:
            grad_clip(self.params, self.gradient_clip)

        with torch.no_grad():
            for param in self.params:
                # a leaf Variable that requires grad is being used in an in-place operation
                # param *= (1.0 - self.lr*self.weight_decay)
                # param -= self.lr * param.grad
                # param = param * (1.0 - self.lr*self.weight_decay) - \
                #     self.lr*param.grad

                # [https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf]
                # 果然是这样！！！
                # Context-manager that disabled gradient calculation.
                # Disabling gradient calculation is useful for inference,
                param *= (1.0 - self.lr * self.weight_decay)
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = None

# TODO: 实现 grad clip
# 为什么grad clip 不集成在optimizer里面呢 ?? 这显然是一个极好的位置呀??
# 我们可以在utils里面实现，参考 [https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html]
# 然后再optimizer里面添加一个clip参数即可


class OptimizerFactory:
    def __init__(self, name: str, lr: float):
        self.name = name
        self.lr = lr

    def new(self, model: nn.Module):
        # 这里其实应该用反射来构建对象 因为所有的Optimizer的的构造方式都差不太多
        if self.name == 'MySGD':
            return SGD(params=model.parameters(), lr=self.lr)
        elif self.name == 'Adam':
            return torch.optim.Adam(params=model.parameters(), lr=self.lr)
        else:
            assert False, f'not support optimizer {self.name}'
