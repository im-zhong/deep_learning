import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

# https://gymnasium.farama.org/environments/box2d/lunar_lander/


# Initialise the environment
# ok。直接用代码打印类型事最方便的
env: gym.Env[np.ndarray, np.int64] = gym.make("LunarLander-v3")


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()
    # <class 'numpy.int64'>
    # 同时根据文档，action只有 0 1 2 3

    # print(type(action))

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'bool'> <class 'bool'> <class 'dict'>
    # observation.shape = 8, float32
    # print(
    #     type(observation), type(reward), type(terminated), type(truncated), type(info)
    # )
    # print(info)  # info: dict[str, Any]
    print(reward)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
