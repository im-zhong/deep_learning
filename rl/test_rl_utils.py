# 2026/2/13
# zhangzhong

import numpy as np
import torch
from dataclasses import dataclass
from rl.test_agent import AgentBase
from rl.test_envs import EnvBase


# some utils in a rewards to go manner
def compute_returns(
    rewards: list[float], bootstrap: float = 0.0, gamma: float = 0.99
) -> torch.Tensor:
    returns: list[float] = []
    rewards_to_go: float = bootstrap
    for reward in reversed(rewards):
        rewards_to_go = reward + gamma * rewards_to_go
        returns.append(rewards_to_go)
    return torch.tensor(list(reversed(returns)))


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

        rollout.append(
            RolloutStep(
                observation=curr_state,
                reward=reward,
                terminated=curr_terminated,
                log_prob=log_prob,
                action=action,
                entropy=entropy,
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
