# 2024/8/14
# zhangzhong

import torch
from torch import Tensor

from .cart_pole import State
from .dqn import DQN


# 然后是一个根据当前状态 选择action的 policy函数
# 我们叫做 deday_epsilon_greedy_policy
# input: current state, policy_net(即QDN)，epsilon
# output: action
# 不过在最开始的时候，可以选择一个不decay的函数，这样好实现一些
def greedy_policy(observation: State, policy_net: DQN, epsilon: float) -> int:
    # 这个policy每次只能选择一个action而已
    # 但是state却可以穿入一个batch
    # 这样是不对的，我们应该在入参里酒表示出来，只能穿入一个state
    # 那么就需要定义一种特殊的类型 叫做 Observation
    # 这样我们就能明确的知道，这个参数只能传入一个state
    # 然后在函数内部，我们根据情况将此observation转换为对应device的tensor 穿入dqn
    if torch.rand(size=(1,)).item() < epsilon:
        return int(torch.randint(low=0, high=policy_net.action_size, size=(1,)))
    else:
        return int(
            policy_net(observation.to_tensor().to(device=policy_net.device)).argmax()
        )
