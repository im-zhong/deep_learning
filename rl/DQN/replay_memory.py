# https://docs.python.org/3.12/library/collections.html#collections.deque
from collections import deque

import torch
from pydantic import BaseModel
from torch import Tensor

from .cart_pole import State


class Transition(BaseModel):
    state: State
    action: int
    next_state: State | None
    reward: float

    # pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'torch.Tensor'>.
    # Set `arbitrary_types_allowed=True` in the model_config to ignore this error
    # class Config:
    #     arbitrary_types_allowed = True

    @staticmethod
    def sample() -> "Transition":
        return Transition(
            state=State.sample(),
            action=1,
            next_state=State.sample(),
            reward=1.0,
        )


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        # 这里需要一个环形队列
        # Once a bounded length deque is full, when new items are added,
        # a corresponding number of items are discarded from the opposite end
        #
        # 还有一个问题，deque只能保存在内存中
        # 所以ReplayMemory中的state等Tensor也需要保存在内存中
        # 在存入memroy之前，需要to cpu
        # 然后在拿出来之后，需要 to device
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        # 最简单的方式是先生成一个1 2 3 。。。的列表
        # 然后shuffle
        # 然后取前batch_size个
        indices: Tensor = torch.randperm(n=len(self.memory))[:batch_size]
        # print(indices)
        return [self.memory[index] for index in indices]

    def __len__(self) -> int:
        return len(self.memory)


replay_memory = ReplayMemory(capacity=10000)


# 我们可以在这里做一些测试
# 首先生成一些随机的Transition
# 我们发现这样非常难以写成单元测试
# 对于每一个新的网络，我们应该采取一个模块的形式进行组织，也就是一个文件夹
# 然后在此文件夹内部，我们对模块进行拆分，并且携带一些测试代码
# 这样应该是极好的
