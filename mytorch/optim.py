# 2023/9/10
# zhangzhong
# https://pytorch.org/docs/stable/optim.html
# https://pytorch.org/docs/stable/optim.html
# https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR


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

import math
from typing import Iterable, Iterator

import torch
from torch import Tensor, nn

# https://pytorch.org/docs/stable/optim.html#base-class
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler


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
                    p.grad *= clip / m


class MySGD:
    # params = sequence of tensors
    # 只有这样 我们可以for遍历所有的param，然后使用梯度下降
    def __init__(
        self, params: Iterator[Tensor], lr, weight_decay=0.0, gradient_clip=None
    ):
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
                param *= 1.0 - self.lr * self.weight_decay
                param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = None


# TODO: 实现 grad clip
# 为什么grad clip 不集成在optimizer里面呢 ?? 这显然是一个极好的位置呀??
# 我们可以在utils里面实现，参考 [https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html]
# 然后再optimizer里面添加一个clip参数即可


# params_t: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
# 咱们简化成只考虑 Iterable[torch.Tensor]
# https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
# 参考pytorch的实现
class MySGDV2(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        gradient_clip: float = 1.0,
    ) -> None:
        pass
        for param in params:
            pass


# 感觉其他的优化器都没有实现的必要啊 就是先一个Adam就行了 其他的都太重复了
class MyAdam(Optimizer):
    def __init__(self):
        pass


class WarmUpCosineScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.current_epoch = 0

    def step(self):
        pass


class TransformerScheduler(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int
    ) -> None:
        super().__init__(optimizer)
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.step_count = 0

    def step(self) -> None:
        pass
