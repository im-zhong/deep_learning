# 2024/8/13
# zhangzhong

import random

import gym
import numpy as np

seed = np.random.randint(low=1000, high=100000)

env = gym.make(
    id="FrozenLake-v1",
    # desc=["SFFF", "FHFH", "FFFH", "HFFG"],
    # desc=["SFHF", "HFFF", "GHHF", "FFFF"],
    desc=[
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHHFFF",
        "FFFHGHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFH",
    ],
    # map_name="8x8",
    is_slippery=False,
    # render_mode="human",
)
observation, info = env.reset(seed=seed)

state_count = env.observation_space.n
print(f"State count: {state_count}")

action_count = env.action_space.n
print(f"Action count: {action_count}")

max_iterations = 4096


Q = np.zeros(shape=(state_count, action_count))
V = np.zeros(shape=(state_count))
pi = np.zeros(shape=(state_count), dtype=np.int32)

learning_rate = 0.9
# 下面两个参数同时调整为0.99，同时增加iteration之后，我们就能解出来了8*8的地图
# TIP ，不行，必须两者同时调整为0.99
# 我们必须更加具有冒险精神，所以我们必须增加epsilon
# TMD，突然有不行了，调回0.9就又ok了。。。
epsilon = 0.9
# TIP
# 对于一个8*8的地图，我们增加了discount_factor之后，就能解出来了
# 说明对于一个较大的问题，我们需要更多的步数，显然就需要对步数惩罚小一些
discount_factor = 0.99


def epsilon_greedy_policy(state):
    # 其实简单来说，就是我们有一个参数控制我们是采取随机策略还是采取最优策略
    # BUG：难道是这里的问题？是我们不够随机吗？
    # if random.random() > epsilon:
    # !!! 果然是这样，如果太墨守陈规，我们就一直采取所谓的最优策略，就会导致我们无法探索到其他的状态
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])


for iteration in range(max_iterations):
    # 每次迭代都重新初始化环境
    print(f"Iteration {iteration}")
    # 注意每次的种子都必须一样
    state, info = env.reset(seed=seed)

    # # 然后我们采取一个随机策略，进入下一个状态
    # action = epsilon_greedy_policy(state=state)
    # # 然后模拟进入下一个状态
    # state, reward, terminated, truncated, info = env.step(action)

    # 只要不是碰到了结束状态，我们就一直循环
    # 但是这里我们有没有可能陷入死循环呢？
    # while not terminated and not truncated:
    terminated = False
    truncated = False
    while not terminated and not truncated:

        # 随机采取一个策略
        action = epsilon_greedy_policy(state=state)
        # BUG: 这里错了，这里应该获取到next_state，而不是state
        next_state, reward, terminated, truncated, info = env.step(action)
        # next_state, reward, done, _, _ = env.step(action)

        # update Q
        # BUG: 同样，在更新Q(s, a)的时候，应该获取next state的Q值，而不是当前的Q值
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state, :])
        )

        # y = reward + discount_factor * np.max(Q[next_state, :])
        # Q[state, action] = Q[state, action] + learning_rate * (y - Q[state, action])

        # 然后我们进入下一个状态
        state = next_state

    # 一旦一个迭代周期结束了，我们就更新一下V和pi
    for state in range(state_count):
        V[state] = np.max(Q[state, :])
        pi[state] = np.argmax(Q[state, :])
    print(pi)


# an the end, we use the pi to render the result
env = gym.make(
    id="FrozenLake-v1",
    # desc=["SFFF", "FHFH", "FFFH", "HFFG"],
    # desc=["SFHF", "HFFF", "GHHF", "FFFF"],
    desc=[
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHHFFF",
        "FFFHGHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFH",
    ],
    # map_name="8x8",
    is_slippery=False,
    render_mode="human",
)
observation, info = env.reset(seed=seed)
terminated = False
truncated = False
step = 0
while not terminated and not truncated and step < state_count:
    action = pi[observation]
    observation, reward, terminated, truncated, info = env.step(action)
    step += 1

# if observation == 15:
#     print(f"Found solution at iteration {iteration}")
#     # break
