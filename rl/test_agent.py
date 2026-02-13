# 2026/2/13
# zhangzhong
import numpy as np
from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.distributions import Categorical
from typing import override

# np.ndarray是比较好的state的表现形式
# 对于连续的state，当然可以表示
# 对于离散的状态，我们也可以用one hot vector来表示，在模型里面放上一个embedding就行了


# Actor-Critic Agent
class AgentBase(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def sample_action(self, state: np.ndarray) -> tuple[np.int64, torch.Tensor]:
        pass

    @abstractmethod
    def policy_forward(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def value_forward(self, state: torch.Tensor) -> torch.Tensor:
        pass


class SimpleActorCritic(AgentBase):
    def __init__(self, observation_shape: int, action_shape: int):
        super().__init__()
        self.action_shape = action_shape
        self.observation_shape = observation_shape

        # 这里还是用一个简单的MLP来作为policy net吧
        # 然后用另外一个简单的MLP做value net
        # 然后让这两个网络共享参数

        # 这次的state是连续的，所以就没有state embedding了
        self.encoder = nn.Sequential(
            nn.Linear(in_features=observation_shape, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
        )
        self.policy_head = nn.Linear(in_features=32, out_features=action_shape)
        self.value_head = nn.Linear(in_features=32, out_features=1)

    @override
    def policy_forward(self, state: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.encoder(state)
        return self.policy_head(embedding)

    @override
    def value_forward(self, state: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.encoder(state)
        return self.value_head(embedding)

    @override
    def sample_action(self, state: np.ndarray) -> tuple[np.int64, torch.Tensor]:
        logits: torch.Tensor = self.policy_forward(state=torch.tensor(data=state))
        probs = Categorical(logits=logits)
        actions: torch.Tensor = probs.sample()
        # 只有log prob是需要携带梯度的
        return np.int64(actions.item()), probs.log_prob(value=actions)
