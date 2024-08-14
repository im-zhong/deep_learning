# 2024/8/14
# zhangzhong

import torch
from torch import Tensor

from .cart_pole import State
from .dqn import DQN
from .policy import greedy_policy
from .replay_memory import ReplayMemory, Transition
from .train_dqn import train


def test_replay_memory() -> None:
    replay_memory = ReplayMemory(capacity=10000)

    for _ in range(100):
        replay_memory.push(transition=Transition.sample())

    assert len(replay_memory) == 100

    transitions: list[Transition] = replay_memory.sample(batch_size=10)
    assert len(transitions) == 10

    transitions: list[Transition] = replay_memory.sample(batch_size=10)
    assert len(transitions) == 10


def test_dqn() -> None:
    observation_dim: int = 4
    action_dim: int = 2
    net = DQN(observation_size=observation_dim, action_size=action_dim)

    # randomly generate some transitions
    transitions: list[Transition] = [Transition.sample() for _ in range(10)]
    input: Tensor = torch.stack(
        tensors=[transition.state.to_tensor() for transition in transitions], dim=0
    )
    # 这里的print也很难处理啊，我们在调试的时候希望这些测试可以输出，但是在运行的时候希望没有这些输出
    print(input.shape)

    output: Tensor = net(input)
    print(output.shape)
    print(output)

    # 对于每一个输出，我们怎么拿到对应的action呢 就是argmax操作
    actions: Tensor = output.argmax(dim=1)
    print(actions)
    # actions.shape = torch.size([batch_size])
    # 也就是每个state对应一个action
    print(actions.shape)


def test_greedy_policy() -> None:
    net = DQN(observation_size=4, action_size=2)

    action = greedy_policy(observation=State.sample(), policy_net=net, epsilon=0.1)
    print(action)


# 大多数情况下，我们不会同时训练很多个模型，所以我们写好一个train开始的函数
# 这样pytest不会把这个东西视为一个测试用例，我们可以安心的执行测试用例
# 但是当我们想要进行训练的时候，就把相应的训练函数前面加上test，这样就可以执行了
def test_train_dqn() -> None:
    train()
