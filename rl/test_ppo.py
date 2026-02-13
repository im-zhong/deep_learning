# 2026/2/12
# zhangzhong
# try to impl PPO in a naive way
# https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py?utm_source=chatgpt.com
# 大概就是三百行代码

import torch
from torch import nn
from typing import Iterator
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from dataclasses import dataclass
from torch.optim import Adam
from tqdm import tqdm


class LunarLander:
    def __init__(self) -> None:
        self.env: gym.Env[np.ndarray, np.int64] = gym.make(id="LunarLander-v3")
        self.curr_obs = np.zeros(shape=(8,))
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
        return observation, reward, self.terminated

    def current_observation(self) -> np.ndarray:
        return self.curr_obs

    def is_terminated(self) -> bool:
        return self.terminated


class CartPole:

# 我们接入gymnasium的API就行了
# 有一个能用的env就行了，咱们先写PPO

# 有一个办法，咱们先写一个简单的A2C吧还是，然后要用在gymnasium上，看看效果，然后换成复杂的环境，看看效果，在看看到底是哪里出了问题？
# 是环境太难？参数不对？还是写的不对！


# some utils in a rewards to go manner
def compute_returns(
    rewards: list[float], bootstrap: float = 0.0, gamma: float = 0.99
) -> list[float]:
    returns: list[float] = []
    rewards_to_go: float = bootstrap
    for reward in reversed(rewards):
        rewards_to_go = reward + gamma * rewards_to_go
        returns.append(rewards_to_go)
    return list(reversed(returns))


def compute_gaes(
    rewards: list[np.float64],
    values: list[float],
    bootstrap: float = 0.0,
    gamma: float = 0.99,
    lambda_: float = 0.99,
) -> list[float]:
    assert len(rewards) == len(values)

    next_value: float = bootstrap
    gae: float = 0
    gaes: list[float] = []

    for reward, value in zip(reversed(rewards), reversed(values)):
        delta: float = reward + gamma * next_value - value
        gae = delta + gamma * lambda_ * gae
        gaes.append(gae)
        next_value = value

    return list(reversed(gaes))


## TODO: 注意区分gymnasium的env的terminated和truncated
# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# 试一下CartPole吧，这个比较简单，应该是能训练出来的





# 为了能更方便的切换环境，咱们最好有一套接口


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        # 这里还是用一个简单的MLP来作为policy net吧
        # 然后用另外一个简单的MLP做value net
        # 然后让这两个网络共享参数

        # 这次的state是连续的，所以就没有state embedding了
        self.encoder = nn.Sequential(
            nn.Linear(in_features=8, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
        )
        self.policy_head = nn.Linear(in_features=32, out_features=4)
        self.value_head = nn.Linear(in_features=32, out_features=1)

    def policy_forward(self, state: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.encoder(state)
        return self.policy_head(embedding)

    def value_forward(self, state: torch.Tensor) -> torch.Tensor:
        embedding: torch.Tensor = self.encoder(state)
        return self.value_head(embedding)

    def sample_action(self, state: np.ndarray) -> tuple[np.int64, torch.Tensor]:
        logits: torch.Tensor = self.policy_forward(state=torch.tensor(data=state))
        probs = Categorical(logits=logits)
        actions: torch.Tensor = probs.sample()
        # 只有log prob是需要携带梯度的
        return actions.item(), probs.log_prob(value=actions)


# 你要怎么定义episode中的一步？
# 用一个结构体比较方便
@dataclass
class RolloutStep:
    observation: np.ndarray
    action: np.int64
    reward: np.float64
    terminated: bool
    log_prob: torch.Tensor
    # truncated: bool

    # @property
    # def done(self) -> bool:
    #     return self.terminated or self.truncated


def sample_episode(agent: Agent, env: LunarLander):
    pass


# contain automatic env reset 在这里重新reset一下是非常方便的
# 我们没有任何手段可以判断env是不是已经done了, 还好我封装了一下env，现在可以了
#
def sample_rollout(
    agent: Agent, env: LunarLander, rollout_steps: int
) -> list[RolloutStep]:

    rollout: list[RolloutStep] = []

    if env.is_terminated():
        env.reset()

    curr_state = env.current_observation()
    curr_terminated = env.is_terminated()

    for _ in range(rollout_steps):
        action, log_prob = agent.sample_action(state=curr_state)

        next_obs, reward, terminated = env.step(action=action)

        rollout.append(
            RolloutStep(
                observation=curr_state,
                reward=reward,
                terminated=curr_terminated,
                log_prob=log_prob,
                action=action,
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
        )
    )

    # 现在要怎么保存terminated state ？
    # 我觉得就直接保存在rollout里面就行了呗。reward设置成零
    return rollout


# 我们加一个函数吧，来嫁接一下
# TODO: 把这个函数的返回值改成dataclass会好一些
def sample_rollout_v2(
    agent: Agent, env: LunarLander, rollout_steps: int
) -> tuple[
    torch.Tensor, torch.Tensor, list[np.float64], torch.Tensor, torch.Tensor, bool
]:
    rollout = sample_rollout(agent, env, rollout_steps)
    observations: list[np.ndarray] = []
    rewards: list[np.float64] = []
    log_probs: list[torch.Tensor] = []
    actions: list[np.int64] = []
    terminateds: list[bool] = []
    for step in rollout[:-1]:
        observations.append(step.observation)
        rewards.append(step.reward)
        log_probs.append(step.log_prob)
        actions.append(step.action)
        terminateds.append(step.terminated)
    return (
        torch.tensor(observations),
        torch.tensor(actions),
        rewards,
        # BUG: 这个应该只在sample old rollout的时候用了，所以detach一下
        torch.stack(log_probs).detach(),
        # 我们根本没必要返回一整个step每个state的terminate状态
        # 我们只需要返回next obs是否是terminated就行了！
        # terminateds,
        torch.tensor(rollout[-1].observation),
        rollout[-1].terminated,
    )


# TODO: 后面再整吧，现在先用最简单的env就行。
# 我需要一个paralle env
# 来一次性获取多个


# 什么是PPO？或者说什么是RL
# RL是一种算法
# 他可以将一个Agent，训练的适应于某种环境
class PPO:
    def __init__(
        self,
        agnet: Agent,
        env: LunarLander,
        epsilon: float = 0.2,
        max_epochs: int = 1000,
    ) -> None:
        self.agent = agnet
        self.env = env
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.optimizer = Adam(self.agent.parameters(), lr=0.001)
        self.c1: float = 0.5
        self.c2: float = 0.1

    # 这个应该拿到外面去实现
    # def sample_rollout(self):
    #     pass

    def collect_old_rollout(self):
        with torch.no_grad():
            return sample_rollout_v2(agent=self.agent, env=self.env, rollout_steps=128)

    # 还要具体实现PPO的收集训练数据的策略
    # 他是先用当前的policy sample一系列的rollout
    # 注意这个是不会进入梯度计算的

    # 我有一计
    def compute_clip_loss(self):
        pass

    def compute_value_loss(self):
        pass

    def compute_entropy_penalty_loss(self):
        pass

    def train(
        self,
    ):
        for epoch in tqdm(range(self.max_epochs)):
            self.train_iter()

    def train_iter(self):
        self.optimizer.zero_grad()

        # 我修改一下collect rollout的机制吧，我直接返回list，这样就不用在自己组织成list了
        old_observations, old_actions, old_rewards, old_log_probs, next_obs, done = (
            self.collect_old_rollout()
        )
        # 我们打印一下每次的old_rewards吧
        print("rollout rewards: ", old_rewards)

        all_step: int = len(old_rewards)

        # old_observations.shape = (all_step, 8)
        # old_actions.shape = (all_step,)
        # old_rewards.shape = len()=allsteps
        # old_log_probs.shape = (all_step,)
        # next_obs.shape = (8,)
        # done: bool
        # BUG: 我们要在一开始就计算出所有的returns和advantages，并且不要计算梯度
        # 还有一个问题就是我们返回的observations啥的都会在后面携带一个多余的state，实际上这个东西没用
        # 他只有next obs和terminated有用，我们后面的计算需要用到这些列表，难道每个都要变成 xxx[:-1]才行吗？
        # 这样也太麻烦了，所以next obs和is terminated单独返回是最合适的！

        # assert there is no grad on all the tensors
        assert not old_observations.requires_grad
        assert not old_actions.requires_grad
        assert not old_log_probs.requires_grad
        assert not next_obs.requires_grad

        with torch.no_grad():
            bootstrap: float = 0
            if not done:
                bootstrap = self.agent.value_forward(state=next_obs).item()

            old_values: list[float] = (
                self.agent.value_forward(state=old_observations)
                .squeeze(-1)
                .detach()
                .tolist()
            )
            # 这里还要考虑bootstrap
            # 我们所有的
            old_returns = torch.tensor(
                data=compute_returns(rewards=old_rewards, bootstrap=bootstrap)
            )
            old_advantages = torch.tensor(
                data=compute_gaes(
                    rewards=old_rewards, values=old_values, bootstrap=bootstrap
                )
            )
            # old_retrns.shape = (all_step,)
            # old_abvantages.shape = (all_step,)

            # TIP: 做一下advantages normalize
            old_advantages = (old_advantages - old_advantages.mean()) / (
                old_observations.std() + 1e-8
            )

            # 他们的shape是什么？

        # 到这里，两个网络上应该没有任何的梯度才对！
        assert not old_returns.requires_grad
        assert not old_advantages.requires_grad

        # 然后我们从old_rollout里面sample minibatch就行
        # 不过一个rollout肯定是连续的才行
        # 那我们也非常的简单，我们就规定一个步长，比如1
        step: int = 16
        for i in range(0, all_step, step):
            self.optimizer.zero_grad()

            # TODO: 或许一开始就应该返回list？
            # observations: list[np.ndarray] = []
            # rewards: list[np.float64] = []
            # log_probs: list[torch.Tensor] = []
            # actions: list[np.int64] = []
            # for step in old_rollout[i : i + 16]:
            #     observations.append(step.observation)
            #     rewards.append(step.reward)
            #     # TODO: 还要处理terminate
            #     log_probs.append(step.log_prob)
            #     actions.append(step.action)

            # TODO: tensor or stack, 注意梯度是否正确的传播！
            # input = torch.tensor(data=old_observations)
            logits = self.agent.policy_forward(state=old_observations[i : i + step])

            probs = Categorical(logits=logits)
            new_log_probs: torch.Tensor = probs.log_prob(
                value=torch.tensor(old_actions[i : i + step])
            )
            # TODO-DONE: collect old_log_prob
            # old_log_probs: torch.Tensor = torch.tensor(log_probs)

            # TODO: bootstrap = ?
            # TODO: 在minibatch里面是不是就不用再计算returns和advantages了？
            # values = self.agent.value_forward(state=input).squeeze(-1)
            # returns = torch.tensor(data=compute_returns(rewards=rewards))
            # advantages = torch.tensor(
            #     data=compute_gaes(rewards=rewards, values=values.detach())
            # )

            # compute clip loss
            # r_theta   = policy / old_policy
            #           = e^(lnp - lnoldp)
            r_theta = torch.exp(new_log_probs - old_log_probs[i : i + step])
            # TODO：yo！ pytroch的min还不是这样用的？
            clip_loss = -(
                torch.min(
                    r_theta * old_advantages[i : i + step],
                    torch.clip(
                        input=r_theta, min=1 - self.epsilon, max=1 + self.epsilon
                    )
                    * old_advantages[i : i + step],
                )
            ).mean()

            # 这里要重新做value net的前向过程
            values = self.agent.value_forward(
                state=old_observations[i : i + step]
            ).squeeze(-1)
            # compute value loss
            value_loss = (values - old_returns[i : i + step]).pow(2).mean()
            # print("value loss: ", value_loss.detach().item())

            # compute entropy bonus/penalty
            entropy = -probs.entropy().mean()

            loss = clip_loss + self.c1 * value_loss + self.c2 * entropy

            # backprop on loss
            loss.backward()
            self.optimizer.step()

    def inference(self):
        pass


def test_compute_returns():
    pass


def test_lunar_lander():
    env = LunarLander()
    state = env.reset()
    print("initial state: ", state)
    action = env.sample_action()
    state, reward, done = env.step(action=action)
    print(state, reward, done)


def test_sample_rollout() -> None:
    agent = Agent()
    env = LunarLander()

    rollout = sample_rollout(agent=agent, env=env, rollout_steps=1280)
    # print(len(rollout))
    # print(rollout[-1])
    # print(rollout[-2])
    # print(rollout)

    assert rollout[-1].terminated
    assert not rollout[-2].terminated

    env.reset()
    rollout = sample_rollout(agent=agent, env=env, rollout_steps=4)
    # print(len(rollout))
    # print(rollout[-1])
    # print(rollout[-2])
    assert not rollout[-1].terminated

    rollout2 = sample_rollout(agent=agent, env=env, rollout_steps=4)
    # # 第一个state应该和上面的rollout[-1]一样才对
    # print(rollout[0])
    assert not rollout[-1].terminated
    # print(rollout2[0].observation)
    # print(rollout[-1].observation)
    # assert rollout2[0].observation == rollout[-1].observation
    assert np.array_equal(rollout2[0].observation, rollout[-1].observation)


def test_ppo():
    agent = Agent()
    env = LunarLander()

    ppo = PPO(agent, env)
    ppo.train()

    # 强化学习有什么办法能观察我们的训练过程呢？
    # 输出value net loss？
