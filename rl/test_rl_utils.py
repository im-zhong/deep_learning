# 2026/2/13
# zhangzhong

import numpy as np
import torch
from dataclasses import dataclass
from rl.test_agent import AgentBase
from rl.test_envs import EnvBase


# some utils in a rewards to go manner
def compute_returns(
    rewards: np.ndarray, bootstrap: float = 0.0, gamma: float = 0.99
) -> np.ndarray:
    returns: list[float] = []
    rewards_to_go: float = bootstrap
    for reward in reversed(rewards):
        rewards_to_go = reward + gamma * rewards_to_go
        returns.append(rewards_to_go)
    return np.array(list(reversed(returns)))


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap: float = 0.0,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> np.ndarray:
    assert len(rewards) == len(values)

    next_value: float = bootstrap
    gae: float = 0
    gaes: list[float] = []

    for reward, value in zip(reversed(rewards), reversed(values)):
        delta: float = reward + gamma * next_value - value
        gae = delta + gamma * lambda_ * gae
        gaes.append(gae)
        next_value = value
    return np.array(list(reversed(gaes)))


# TODO: return to compute_advantages
def compute_gaes(
    rewards: list[np.float64],
    values: list[float],
    bootstrap: float = 0.0,
    gamma: float = 0.99,
    lambda_: float = 0.99,
) -> torch.Tensor:
    assert len(rewards) == len(values)

    next_value: float = bootstrap
    gae: float = 0
    gaes: list[float] = []

    for reward, value in zip(reversed(rewards), reversed(values)):
        delta: float = reward + gamma * next_value - value
        gae = delta + gamma * lambda_ * gae
        gaes.append(gae)
        next_value = value
    return torch.tensor(list(reversed(gaes)))


# 你要怎么定义episode中的一步？
# 用一个结构体比较方便
@dataclass
class RolloutStep:
    observation: np.ndarray
    action: np.int64
    reward: np.float64
    terminated: bool
    value: np.float32
    log_prob: torch.Tensor
    entropy: torch.Tensor
    # truncated: bool

    # @property
    # def done(self) -> bool:
    #     return self.terminated or self.truncated


# contain automatic env reset 在这里重新reset一下是非常方便的
# 我们没有任何手段可以判断env是不是已经done了, 还好我封装了一下env，现在可以了
#
def sample_rollout(
    agent: AgentBase, env: EnvBase, rollout_steps: int
) -> list[RolloutStep]:

    rollout: list[RolloutStep] = []

    if env.is_terminated():
        env.reset()

    curr_state = env.current_observation()
    curr_terminated = env.is_terminated()

    for _ in range(rollout_steps):
        action, log_prob, entropy = agent.sample_action(state=curr_state)

        next_obs, reward, terminated = env.step(action=action)

        value: float = (
            agent.value_forward(state=torch.as_tensor(curr_state))
            .squeeze(dim=-1)
            .item()
        )

        rollout.append(
            RolloutStep(
                observation=curr_state,
                reward=reward,
                terminated=curr_terminated,
                log_prob=log_prob,
                action=action,
                entropy=entropy,
                value=value,
            )
        )

        curr_state = next_obs
        curr_terminated = terminated

        if terminated:
            # rollout.append(
            #     RolloutStep(
            #         observation=next_obs,
            #         reward=0.0,
            #         action=-1,
            #         terminated=True,
            #         log_prob=torch.tensor(1),
            #     )
            # )
            env.reset()
            break

    # 无论如何，我们都要把当前的状态放到rollout的最后一个位置上，
    # 这个位置只会用来记录 next state and if it is terminated
    rollout.append(
        RolloutStep(
            observation=curr_state,
            action=0,
            reward=0,
            terminated=curr_terminated,
            log_prob=torch.tensor(0),
            entropy=torch.tensor(0),
            value=torch.tensor(0),
        )
    )

    # 现在要怎么保存terminated state ？
    # 我觉得就直接保存在rollout里面就行了呗。reward设置成零
    return rollout


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: list[np.float64]
    log_probs: torch.Tensor
    entropy: torch.Tensor

    next_observation: torch.Tensor
    terminated: bool
    truncated: bool = False


# 我们加一个函数吧，来嫁接一下
# TODO-DONE: 把这个函数的返回值改成dataclass会好一些
# TODO: refactor, remove sample_rollout and only keep this
def sample_rollout_v2(agent: AgentBase, env: EnvBase, rollout_steps: int) -> Rollout:
    rollout = sample_rollout(agent, env, rollout_steps)
    observations: list[np.ndarray] = []
    rewards: list[np.float64] = []
    log_probs: list[torch.Tensor] = []
    actions: list[np.int64] = []
    terminateds: list[bool] = []
    entropys: list[torch.Tensor] = []
    for step in rollout[:-1]:
        observations.append(step.observation)
        rewards.append(step.reward)
        log_probs.append(step.log_prob)
        actions.append(step.action)
        terminateds.append(step.terminated)
        entropys.append(step.entropy)
    return Rollout(
        observations=torch.tensor(observations),
        actions=torch.tensor(actions),
        rewards=rewards,
        # BUG: 这个应该只在sample old rollout的时候用了，所以detach一下
        # 不对！为了通用，这个不应该做detach，因为后面可能detach，也可能不，
        # log_probs=torch.stack(log_probs).detach(),
        log_probs=torch.stack(log_probs),
        # 我们根本没必要返回一整个step每个state的terminate状态
        # 我们只需要返回next obs是否是terminated就行了！
        # terminateds,
        next_observation=torch.tensor(rollout[-1].observation),
        terminated=rollout[-1].terminated,
        entropy=torch.stack(entropys),
    )


@dataclass
class Rollouts:
    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    # entropy: torch.Tensor


# 实现一个combine shape？还是直接返回一个int得了？
# 但是如果我们后面有的state是图像，我们就不好处理了。所以还是实现一个combine shape吧
def combine_shape(size: int, sizes: tuple) -> tuple:
    return (size,) + sizes


# 那么这个东西和PPO是无关的，A2C也可以使用
# env
# 就这三个函数，写完，就基本上实现完了
# spinning up 里面的实现用的是numpy, 看来我也得用numpy，因为这些数据都是来自gym的，gym的api返回的就是numpy
class RolloutBuffer:
    def __init__(
        self,
        observation_shape: tuple,
        action_shape: tuple,
        size: int,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.size = size
        self.gamma = gamma
        self.lambda_ = lambda_
        pass

        self.observations: np.ndarray = np.zeros(
            combine_shape(size, observation_shape), dtype=np.float32
        )
        self.actions: np.ndarray = np.zeros(size, dtype=np.int32)
        self.rewards: np.ndarray = np.zeros(size, dtype=np.float32)
        self.log_probs: np.ndarray = np.zeros(size, dtype=np.float32)
        self.entropy: np.ndarray = np.zeros(size, dtype=np.float32)
        self.values: np.ndarray = np.zeros(size, dtype=np.float32)
        # next_observation: torch.Tensor
        # terminated: bool
        # truncated: bool = False

        # 用一个slice来表示现在的rollout的范围
        # 仍然采取[first, last)
        self.first: int = 0
        self.last: int = 0
        # 我们要计算的是returns和advantages
        self.returns: np.ndarray = np.zeros(size, dtype=np.float32)
        self.advantages: np.ndarray = np.zeros(size, dtype=np.float32)
        # TODO: 计算这个东西是需要values的

    # 我们可以像spining up一样，每次一个rollout结束的时候，就计算一下rewards和advantages
    # 只需要保留两个指针就行
    # 然后我们也可以提前用torch zeros来占位

    def push_back(self, step: RolloutStep):
        # 就是把现在的step里面的东西放到buffer里面，那我们的buffer应该是没有grad的
        # TIP: 是的，buffer里面的数据就是没有梯度的，PPO每个step的policy loss都是重新计算的

        # assert out of index
        assert self.last < self.size

        self.observations[self.last] = step.observation
        self.actions[self.last] = step.action
        self.rewards[self.last] = step.reward
        self.log_probs[self.last] = step.log_prob
        self.entropy[self.last] = step.entropy
        # TODO: 为什么PPO的实现里，这里还保存了value ?
        # 因为我们接下来计算returns和advantages需要用到
        self.values[self.last] = step.value

        self.last += 1

        pass

    # 还要定义一个函数，每次sample结束都要计算一下rewards和returns
    def finish_rollout(self, bootstrap: float):
        pass

        # 这里是需要把[self.first, self.last) 这个范围里面的rollout，计算一次returns和advantages
        # 然后更新self.first self.last开启下一rollout的计算
        first = self.first
        last = self.last

        self.returns[first:last] = compute_returns(
            rewards=self.rewards[first:last], bootstrap=bootstrap, gamma=self.gamma
        )
        self.advantages[first:last] = compute_advantages(
            rewards=self.rewards[first:last],
            values=self.values[first:last],
            bootstrap=bootstrap,
            gamma=self.gamma,
            lambda_=self.lambda_,
        )

        self.first = last
        self.last = last

    # get的逻辑就是把我们收集起来的numpy变成tensor，vectorize化
    def get(self) -> Rollouts:

        assert self.last == self.size

        # 先算一次advantages的standadize
        advantages = torch.as_tensor(data=self.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollouts = Rollouts(
            observations=torch.as_tensor(data=self.observations, dtype=torch.float32),
            actions=torch.as_tensor(data=self.actions, dtype=torch.int32),
            # rewards=torch.as_tensor(data=self.rewards, dtype=torch.float32),
            log_probs=torch.as_tensor(data=self.log_probs, dtype=torch.float32),
            # entropy=torch. #TODO： entropy要重新计算，没必要在sample的数据里面保存！
            returns=torch.as_tensor(self.returns, dtype=torch.float32),
            advantages=advantages,
        )

        # add some assert
        assert rollouts.observations.shape == combine_shape(
            self.size, self.observation_shape
        )
        assert rollouts.actions.shape == (self.size,)
        assert rollouts.log_probs.shape == (self.size,)
        assert rollouts.returns.shape == (self.size,)
        assert rollouts.advantages.shape == (self.size,)

        return rollouts

    def clear(self) -> None:
        self.first = 0
        self.last = 0

    def is_empty(self) -> bool:
        return self.first == 0 and self.last == 0

    def is_full(self) -> bool:
        return self.last == self.size

    def get_remain_capacity(self) -> int:
        return self.size - self.last
