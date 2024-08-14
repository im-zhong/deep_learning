# 2024/8/12
# zhangzhong
# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

import gymnasium as gym
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
    render_mode="human",
)
observation, info = env.reset(seed=seed)


# for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(
#         env.action_space.sample()
#     )

#     if terminated or truncated:
#         observation, info = env.reset(seed=42)


state_count = env.observation_space.n
print(f"State count: {state_count}")

action_count = env.action_space.n
print(f"Action count: {action_count}")

max_iterations = 100

# 书上的实现这里使用了zero 而不是random
# 难道是这里的问题？还真是这里的问题！为什么随机初始化会导致结果不稳定呢？
# 可能是我们随机初始化的参数又一些比较大，而且我们最大的reward其实就是1，所以影响了路径的选择
Q = np.zeros(shape=(max_iterations, state_count, action_count))
V = np.zeros(shape=(max_iterations, state_count))
pi = np.zeros(shape=(max_iterations, state_count), dtype=np.int32)

current_iteration = 1
discount_factor = 0.99

for iteration in range(1, max_iterations):
    print(f"Iteration {iteration}")
    for state in range(state_count):
        for action in range(action_count):
            # 计算Q(s, a)
            # 其实就是遍历所有的状态和动作，然后计算Q值
            for probability, next_state, reward, _ in env.P[state][action]:
                Q[iteration, state, action] += probability * (
                    reward + discount_factor * V[iteration - 1][next_state]
                )
        # 现在计算完了所有的Q，取出最大值出来，其实就是对应的V和pi了
        V[iteration, state] = np.max(Q[iteration, state, :])
        pi[iteration, state] = np.argmax(Q[iteration, state, :])
    print(f"V: {V[iteration]}")
    print(f"pi: {pi[iteration]}")

# 等所有的state都遍历完了，完了可以把pi拿出来，模拟一下
observation, info = env.reset(seed=seed)
# initial state = observation
terminated = False
truncated = False
step = 0
while not terminated and not truncated and step < state_count:
    action = pi[iteration, observation]
    observation, reward, terminated, truncated, info = env.step(action)
    step += 1

if observation == 15:
    print(f"Found solution at iteration {iteration}")
    # break


env.close()
