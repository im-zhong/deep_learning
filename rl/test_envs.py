# 2026/2/13
# zhangzhong

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete


class EnvBase:
    def __init__(self, id: str) -> None:
        self.env: gym.Env[np.ndarray, np.int64] = gym.make(id=id)
        self.curr_obs = np.zeros(0)
        self.terminated = True
        # 我们自己先调用一次reset
        self.reset()

    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset()
        self.curr_obs = observation
        self.terminated = False
        return observation

    def sample_action(self) -> np.int64:
        return self.env.action_space.sample()

    def step(self, action: np.int64) -> tuple[np.ndarray, np.float64, bool]:
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.curr_obs = observation
        self.terminated = terminated or truncated
        return observation, np.float64(reward), self.terminated

    def current_observation(self) -> np.ndarray:
        return self.curr_obs

    def is_terminated(self) -> bool:
        return self.terminated

    def observation_space(self) -> tuple:
        # 我们目前可以只支持float
        # assert self.env.observation_space.dtype is float
        return self.env.observation_space.shape

    def action_space(self) -> np.int64:
        action_space = self.env.action_space
        if not isinstance(action_space, Discrete):
            raise TypeError("Action space is not Discrete")
        return np.int64(action_space.n)


class LunarLander(EnvBase):
    def __init__(self) -> None:
        super().__init__(id="LunarLander-v3")


# https://gymnasium.farama.org/environments/classic_control/cart_pole/
class CartPole(EnvBase):
    def __init__(self) -> None:
        super().__init__(id="CartPole-v1")




# 这里就可以做一些测试了
# 我感觉测试的流程都是一样的啊
def test_lunar_lander():
    lunar_lander = LunarLander()
    pass


def test_cart_pole():
    cart_pole = CartPole()
    pass
