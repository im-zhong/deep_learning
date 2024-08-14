# 2024/8/13
# zhangzhong
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://gymnasium.farama.org/environments/classic_control/cart_pole/

# Episode
# In Reinforcement Learning (RL), an episode refers to a complete sequence
# of states, actions, and rewards that starts from an initial state and ends when a terminal state is reached
# An episode can be thought of as a single run or trial of the agent interacting with the environment

# https://docs.python.org/3.12/library/collections.html#collections.deque
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from .cart_pole import CartPole, State
from .dqn import DQN
from .policy import greedy_policy
from .replay_memory import ReplayMemory, Transition

# 看起来必须用绝对路径，否则执行某个文件的时候，因为PYTHONPATH=.
# python就会在根目录下寻找car_pole等模块，而不是在对应的子模块下面找
# 导致找不到对应的模块
# 但是这样写有一个非常大的坏处，就是我们的子模块实际上和整个项目是耦合的
# 当我们想要将子模块移动到其他地方的时候，我们就需要修改这个文件
# 这是非常非常不灵活的
# from rl.DQN.cart_pole import CartPole, State
# from rl.DQN.dqn import DQN
# from rl.DQN.policy import greedy_policy
# from rl.DQN.replay_memory import ReplayMemory, Transition


seed = 42
# env = gym.make(id="CartPole-v1")
# # Resets the environment to an initial internal state, returning an initial observation and info.
# # This method generates a new starting state often with some randomness to ensure that the agent explores the state space and
# # learns a generalised policy about the environment.
# # This randomness can be controlled with the seed parameter otherwise if the environment already has a random number generator and reset is called with seed=None, the RNG is not reset.
# # Therefore, reset should (in the typical use case) be called with a seed right after initialization and then never again.
# env.reset(seed=seed)

# if GPU is to be used


cart_pole = CartPole(seed=seed)

device: str = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
policy_net = DQN(
    observation_size=cart_pole.observation_size,
    action_size=cart_pole.action_size,
    device=device,
)
target_net = DQN(
    observation_size=cart_pole.observation_size,
    action_size=cart_pole.action_size,
    device=device,
)

# 想想看怎么和我们现有的Trainer结合
# 但是我们先不结合，先实现出来，然后再结合起来，不然比较耗费时间了，因为这个代码是很久之前写的了，一些细节估计我自己也忘了
# 而且也不是说写的有多好吧

# 总的来说分成几个模块

# 还需要一个东西， Transition
# Transition(state, action, next_state, reward)
# 每当我们模拟一次，我们就把这个Transition放入ReplayMemory中
# 现在问题来了，我们怎么定义这个TRansition呢？
# 用namedtuple 还是 BaseModel ?


# 然后是一步，就是一个时间步，我们做了什么，每个时间步之内我们都会随机选择一个策略，模拟一下
# 拿到transition之后放入replay memory中
# episode， step, 其实可以多写几个函数
# 一次训练过程就是许多个episode，而每个episode就是许多个step

# # 4
# observation_dim: int = env.observation_space.shape[0]
# # 2
# action_dim: int = env.action_space.n
# print(f"Observation dim: {observation_dim}")
# print(f"Action dim: {action_dim}")


batch_size: int = 128
greedy_rate: float = 0.9
replay_memory = ReplayMemory(capacity=10000)
discount_factor: float = 0.9
learning_rate: float = 0.001
tau: float = 0.01

optimizer = Adam(params=policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


def step():

    # 首先，当ReplayMemory中的Transition数量不够的时候，我们直接返回就行了
    if len(replay_memory) < batch_size:
        # 这里返回不了任何东西啊，是不是说明这个句子应该移动到episode中
        return

    # 否则我们就采样一些transition出来, 但是这样回造成死循环啊
    # 也就是说我们应该直接进行一个模拟才对

    # observation, done = step(observation=observation)

    samples = replay_memory.sample(batch_size=batch_size)
    # make shape of (batch_size,) to (batch_size, 1)
    # unsqueeze adds a dimension of size 1 at the specified position.
    actions: Tensor = (
        torch.tensor(
            data=[transition.action for transition in samples], dtype=torch.int64
        )
        .unsqueeze(dim=1)
        .to(device=device)
    )

    input: Tensor = torch.stack(
        tensors=[transition.state.to_tensor() for transition in samples]
    ).to(device=device)
    output: Tensor = policy_net(input).gather(dim=1, index=actions)

    # 然后我们计算另外一部分，也就是target net的部分
    # 然后两者做差，就可以计算出loss来了，然后反向传播就ok了

    # 这样写反而让计算变麻烦了，其实最简单而且最快的方式就是让这些哪怕是None的state也参与计算
    # 然后在target_output的时候把这些值mask掉，所以即使next_state是None，我们也让其参与计算
    next_states: Tensor = (
        torch.stack(
            tensors=[
                (
                    transition.next_state.to_tensor()
                    if transition.next_state
                    else State.sample().to_tensor()
                )
                for transition in samples
            ]
        )
        # .unsqueeze(dim=1)
        .to(device=device)
    )
    assert next_states.shape == (batch_size, 4)

    mask: Tensor = (
        torch.tensor(data=[transition.next_state is not None for transition in samples])
        .unsqueeze(dim=1)
        .to(device=device)
    )
    assert mask.shape == (batch_size, 1)

    rewards: Tensor = (
        torch.tensor(data=[transition.reward for transition in samples])
        .unsqueeze(dim=1)
        .to(device=device)
    )
    assert rewards.shape == (batch_size, 1)

    # 将target_output初始化为零
    target_output: Tensor = torch.zeros(size=(batch_size, 1)).to(device=device)
    assert target_output.shape == (batch_size, 1)

    # 注意这段计算不能计算梯度
    with torch.no_grad():
        a: Tensor = target_net(next_states)
        # a.max(dim=1, keepdim=True) 返回的是一个元组，第一个元素是最大值，第二个元素是最大值的索引
        b = a.max(dim=1, keepdim=True)[0]
        c = rewards + discount_factor * b
        target_output[mask] = c[mask]

        # target_output[mask, :] = rewards + discount_factor * target_net(
        #     next_states
        # ).max(dim=1, keepdim=True)

    # 同时有一部分target_output应该是零，也就是next_state是终止状态的
    # 但是我们如何可以知道next_state是终止状态呢？
    # 我们的

    # 还是用state吧，毕竟最开始的公式里面用的也是stateQ(s,a)吗，state action
    # 不过，我们需要mask掉target output中，属于terminated的部分，还是done的部分？
    # QUESTION： 这里需要区分terminated和truncated吗？
    # 不对，像memory这么大的东西，不应该存放在显存中，所以里面不应该直接存放Tensor，而是应该存在Observation
    # 然后在sample之后，我们将他们转换成tensor，在传到模型对应的device上 进行训练
    # 这样的话，我们可以用 Observation | None 来作为 next_observation的类型
    # 这样我们就知道，这个next_observation是不是None了，也可以获取正确的mask了
    # QUESTION：还有就是如果next_observation是None，那么我们就需不需要计算target_output了？
    # 不应该！因为QNet的职责是根据给定状态，预测出相应的Q值，如果是terminate状态，那么其Q值应该为零
    # 不过为了计算上方便，我们仍然将整个batch放进网络，然后通过mask限制，只复制非terminate状态的Q值就ok了
    # 别忘了讲target output初始化为零

    loss: Tensor = criterion(output, target_output)
    optimizer.zero_grad()
    loss.backward()
    # 然后我们更新参数

    optimizer.step()


def episode() -> int:

    # 为什么一会observation 一会state的
    observation, _ = cart_pole.reset()
    observation = State.from_tuple(observation)

    # 这里有两个选择，一是我们不断的迭代，直到episode结束
    # 也就是 terminated or truncated
    # 还有另外一个选择，就是规定一个最大的step
    done = False
    while not done:

        # 首先根据observation选择一个action
        # TIP：我好像懂了，实际上这个尝试和我们的训练没有关系，
        # 我们在这里不断的尝试实际上只是在和环境交互，从而不断的获取训练数据而已
        action = greedy_policy(
            observation=observation, policy_net=policy_net, epsilon=greedy_rate
        )

        # 这里我们采取一个随机策略
        # 震惊了呀，step一次只能做一个动作，那我们怎么获得多个transition呢?
        next_observation, reward, done = cart_pole.step(action=action)

        # 将transition存入replay memory中
        replay_memory.push(
            transition=Transition(
                state=observation,
                action=action,
                next_state=next_observation,
                reward=reward,
            )
        )

        step()

        # 然后我们更新target_net soft update
        # 也就是有一个参数 tau
        # target_net = tau * policy_net + (1 - tau) * target_net
        # 然后target net的更新其实很慢的，也就是tau比较小，新的policy net的参数只有一小部分传递到target net中
        # 这样做的好处是，我们的target net的参数不会一直跟着policy net的参数在变化，这样我们的target net就能更加稳定

        # # 如何遍历所有的参数呢？
        # for name, params in policy_net.named_parameters():
        #     # 说实话这里真的不知道怎么弄
        # state_dict: Return a dictionary containing references to the whole state of the module
        # exactly what I want!
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = (
                tau * policy_net_state_dict[key]
                + (1 - tau) * target_net_state_dict[key]
            )
        # load_state_dict: Copy parameters and buffers from state_dict into this module and its descendants.
        target_net.load_state_dict(state_dict=target_net_state_dict)

    return cart_pole.score


def train():

    # make target_net.weight = policy_net.weight
    target_net.load_state_dict(state_dict=policy_net.state_dict())

    max_episodes = 1000
    for i in range(max_episodes):
        score = episode()
        print(f"Episode {i}: {score}")

        # 我们怎么知道我们训练的怎么样了呢？
        # 就是通过我们的reward来判断
        # 不对，应该是我们每个episode坚持了多久作为我们的分数


# 我们怎么执行这个文件呢？
# 用单元测试的方式？就像我们以前那样，还是写一个__main__的可执行部分
# 看起来不能直接运行 不export env 在python里面不能直接运行这个脚本
# 但是吧 我还是非常希望可以区分测试和训练的
# 如果测试和训练混在一起，我们就不能随心所欲的跑单元测试
# 总是要注释掉一些实际在训练的代码
# if __name__ == "__main__":
# train()
# print("Hello, world!")
