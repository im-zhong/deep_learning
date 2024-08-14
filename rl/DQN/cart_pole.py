# 2024/8/14
# zhangzhong

import random

import gymnasium as gym
from pydantic import BaseModel
from torch import Tensor


class State(BaseModel):
    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float

    def to_tensor(self) -> Tensor:
        return Tensor(
            [
                self.cart_position,
                self.cart_velocity,
                self.pole_angle,
                self.pole_angular_velocity,
            ]
        )

    @staticmethod
    def from_tuple(observation: tuple[float, float, float, float]) -> "State":
        return State(
            cart_position=observation[0],
            cart_velocity=observation[1],
            pole_angle=observation[2],
            pole_angular_velocity=observation[3],
        )

    @staticmethod
    def sample() -> "State":
        return State(
            cart_position=random.random(),
            cart_velocity=random.random(),
            pole_angle=random.random(),
            pole_angular_velocity=random.random(),
        )


class CartPole:
    def __init__(self, seed: int = 42) -> None:
        self.env = gym.make(id="CartPole-v1")
        self.env.action_space.seed(seed=seed)
        self.observation, self.info = self.env.reset(seed=seed)
        self.env.reset(seed=seed)
        self._score = 0

    def reset(self):
        self._score = 0
        return self.env.reset()

    def step(self, action: int) -> tuple[State | None, float, bool]:
        self._score += 1
        observation, reward, terminated, truncated, _ = self.env.step(action=action)
        if terminated or truncated:
            return None, 0, True
        else:
            return (
                State.from_tuple(observation),
                float(reward),
                False,
            )

    def close(self) -> None:
        self.env.close()

    @property
    def observation_size(self) -> int:
        return self.env.observation_space.shape[0]  # type: ignore

    @property
    def action_size(self) -> int:
        return self.env.action_space.n  # type: ignore

    @property
    def score(self) -> int:
        return self._score
