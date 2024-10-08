# 2024/8/12
# zhangzhong
# https://www.gymlibrary.dev/content/basic_usage/

import gymnasium as gym

env = gym.make(id="LunarLander-v2", render_mode="human")
env.action_space.seed(seed=42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    if terminated or truncated:
        observation, info = env.reset(seed=42)

env.close()
