# 2026/2/13
# zhangzhong

# a configural general RL trainer
from tqdm import tqdm
from rl.test_envs import EnvBase
from rl.test_agent import AgentBase
from torch.optim import Adam
import torch
from rl.test_rl_utils import (
    Rollout,
    sample_rollout_v2,
    sample_rollout,
    compute_gaes,
    compute_returns,
    RolloutBuffer,
    Rollouts,
    RolloutStep,
)
from rl.test_envs import CartPole, LunarLander
from rl.test_agent import SimpleActorCritic
from torch.distributions import Categorical


# 不管怎么样，我们都是单线程的
# 所以没必要写一个parallel env
# 只需要写好一个过程的T个step的ppo buffer，然后在N个env上遍历就行了


class PPOTrainer:
    def __init__(
        self,
        # PPO 需要使用N个env来sample数据
        agent: AgentBase,
        envs: list[EnvBase],
        max_epochs: int,
        lr: float,
        rollout_step: int,
        gamma: float,
        lambda_: float,
        epsilon: float,
        c1: float,
        c2: float,
        iter_per_epoch: int,
        # rollout_step: int, # steps per rollout
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.max_epochs = max_epochs
        self.optimizer = Adam(params=agent.parameters(), lr=lr)
        self.rollout_step = rollout_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.ppo_buffers: list[RolloutBuffer] = [
            RolloutBuffer(
                observation_shape=envs[0].observation_space(),
                action_shape=envs[0].action_space(),
                size=rollout_step,
                gamma=gamma,
                lambda_=lambda_,
            )
            for _ in range(len(envs))
        ]
        self.iter_per_epoch = iter_per_epoch

    def inference(self) -> float:
        env = CartPole()
        with torch.no_grad():
            rollout = sample_rollout_v2(self.agent, env, rollout_steps=500)
            # 然后计算所有的reward和
            returns = sum([reward for reward in rollout.rewards])
            return returns

    def train(self) -> None:
        for epoch in tqdm(range(self.max_epochs)):
            self.train_one_epoch(epoch)
            print(f" epoch {epoch}: {self.inference()}", flush=True)
            # print(f"epoch {epoch}: {self.inference()}", flush=True)
            # print(f"epoch {epoch}: {self.inference()}", flush=True)

    # write one function to sample data from env

    def fill_buffer(self, env: EnvBase, buffer: RolloutBuffer):

        # 在fill之前，buffer应该是空的
        buffer.clear()

        while not buffer.is_full():
            # 我们要根据buffer的余量和self.rollout_step的最小值来确定这次sample多少个东西
            rollout_step = min(buffer.get_remain_capacity(), self.rollout_step)

            # 这里应该也封装成一个函数，就是用当前的env填满一个buffer
            # 应该就是sample rollout稍微改一下就行
            rollout = sample_rollout(
                agent=self.agent,
                env=env,
                rollout_steps=rollout_step,
            )

            bootstrap: float = 0.0
            if not rollout[-1].terminated:
                # 这样这里的计算就可以放到gpu上了
                bootstrap = (
                    self.agent.value_forward(
                        state=torch.as_tensor(rollout[-1].observation)
                    )
                    .squeeze(dim=-1)
                    .item()
                )

            # TODO
            # 然后把我们收集起来的数据放到ppo buffer里面就行
            # 这个应该结合到一块，
            for step in rollout[:-1]:
                buffer.push_back(step=step)
            buffer.finish_rollout(bootstrap=bootstrap)

    def sample_one_epoch_data(self):
        # 我们先收集足够的数据
        # 假设我们有N个env，每个env收集T step的数据
        for buffer, env in zip(self.ppo_buffers, self.envs):
            self.fill_buffer(env=env, buffer=buffer)

    def train_one_epoch(self, epoch: int):
        with torch.no_grad():
            self.sample_one_epoch_data()

        # 然后把buffer的数据合并成一个可以用来训练的tensor
        # 把shape从[n, T] -》 [B,]
        # 叫做 old_xxx

        for iter in range(self.iter_per_epoch):
            self.train_one_iter(
                iter,
                # data ? 我觉得还是传递一个总体的东西比较好
                # 最简单的方式就是什么都不传，我们计算好buffer之后，把buffer直接传给
                buffers=self.ppo_buffers,
            )

        # 确认一下所有人的shape

    # 我应该有一个测试的阶段，就是在每个epoch之后，看看平均的reward是多少

    def train_one_iter(self, iter: int, buffers):
        # 1. collect training data of this epoch
        # batch_states, batch_actions, batch_log_probs, batch_rewards_to_go = (
        #     collect_training_data()
        # )

        data = self.ppo_buffers[0].get()
        datas = [self.ppo_buffers[i].get() for i in range(len(self.envs))]

        observations_shape = 4
        action_shape = 2

        num_env = len(self.envs)
        T = self.rollout_step
        batch_size = num_env * T

        # 然后把这些数据传到 train_one_step里面，这里就一个for loop就行
        old_observations = torch.stack([data.observations for data in datas]).reshape(
            batch_size, -1
        )  # N, T, observation_shapex
        old_actions = torch.stack([data.actions for data in datas]).reshape(batch_size)
        old_returns = torch.stack([data.returns for data in datas]).reshape(batch_size)
        old_log_probs = torch.stack([data.log_probs for data in datas]).reshape(
            batch_size
        )
        old_advantages = torch.stack([data.advantages for data in datas]).reshape(
            batch_size
        )

        # 这里必须得加一些shape的assert
        assert old_observations.shape == (batch_size, observations_shape)
        assert old_actions.shape == (batch_size,)
        assert old_returns.shape == (batch_size,)
        assert old_log_probs.shape == (batch_size,)
        assert old_advantages.shape == (batch_size,)

        # with torch.no_grad():
        #     old_rollout = sample_rollout_v2(agent, env, self.rollout_step)

        #     # 咱们把rewards給加起来，看看训练的效果
        #     print(
        #         f"epoch: {epoch}: rewards:, {len(old_rollout.rewards)}, {sum(old_rollout.rewards)}",
        #         flush=True,
        #     )

        #     bootstrap: float = 0.0
        #     if not old_rollout.terminated:
        #         # 这样这里的计算就可以放到gpu上了
        #         bootstrap = (
        #             agent.value_forward(state=old_rollout.next_observation)
        #             .squeeze(dim=-1)
        #             .item()
        #         )

        #     # TODO: 应该重构成返回tensor
        #     old_returns = compute_returns(
        #         rewards=old_rollout.rewards, bootstrap=bootstrap, gamma=self.gamma
        #     )
        #     # 这里需要计算values，我们可以重复计算，这里不进行梯度计算，可以和PPO更好的融合
        #     old_values: torch.Tensor = agent.value_forward(
        #         state=old_rollout.observations
        #     ).squeeze(dim=-1)
        #     old_advantages = compute_gaes(
        #         old_rollout.rewards,
        #         old_values,
        #         bootstrap,
        #         gamma=self.gamma,
        #         lambda_=self.lambda_,
        #     )
        #     # advantages = returns - values
        #     # 加上一个mormalize操作
        #     old_advantages = (old_advantages - old_advantages.mean()) / (
        #         old_advantages.std() + 1e-8
        #     )
        #     assert (
        #         old_rollout.log_probs.shape == old_advantages.shape == old_values.shape
        #     )

        # loop in minibatch
        # TODO: spining up的ppo实现里面policy和value的参数是分开的，如果咱们这次还是训练不出来，可以试试把参数給分开

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
        logits = self.agent.policy_forward(state=old_observations)

        probs = Categorical(logits=logits)
        new_log_probs: torch.Tensor = probs.log_prob(old_actions)
        # TODO-DONE: collect old_log_prob
        # old_log_probs: torch.Tensor = torch.tensor(log_probs)

        # TODO: bootstrap = ?
        # TODO: 在minibatch里面是不是就不用再计算returns和advantages了？
        # values = self.agent.value_forward(state=input).squeeze(-1)
        # returns = torch.tensor(data=compute_returns(rewards=rewards))
        # advantages = torch.tensor(
        #     data=compute_gaes(rewards=rewards, values=values.detach())
        # )

        # 咱们要不用A2C的loss试一试
        # compute clip loss
        # r_theta   = policy / old_policy
        #           = e^(lnp - lnoldp)
        # Loss 1: PPO clip Loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        # # TODO：yo！ pytroch的min还不是这样用的？
        clip_loss = -(
            torch.min(
                ratio * old_advantages,
                torch.clip(input=ratio, min=1 - self.epsilon, max=1 + self.epsilon)
                * old_advantages,
            )
        ).mean()

        # Loss 2: A2C loss
        # clip_loss = -(new_log_probs * old_advantages).mean()

        # 这里要重新做value net的前向过程
        values = self.agent.value_forward(state=old_observations).squeeze(-1)
        # compute value loss
        value_loss = (values - old_returns).pow(2).mean()
        # print("value loss: ", value_loss.detach().item())

        # compute entropy bonus/penalty
        # entropy = -probs.entropy().mean()

        loss = clip_loss + self.c1 * value_loss
        # + self.c2 * entropy

        # backprop on loss
        loss.backward()
        self.optimizer.step()

        # sample_rollout_v2(agent, env, self.rollout_step)

        # 2. zero policy net grads
        # optimizer4agent.zero_grad()
        # # 应该在这里吧
        # optimizer4value.zero_grad()
        # optimizer.zero_grad()

        # 3. compute vt by using value net
        # values = value_net.forward(state=torch.stack(batch_states).squeeze(dim=-1))
        # assert values.shape == (batch_size, 1)
        # # this must use detach, to isolate loss of policy net and value net
        # # 这确保 policy loss 不会把梯度传进 value_net
        # # TODO(actor-critic):
        # # 当前 actor 和 critic 使用的是完全独立的网络参数，因此：
        # # - policy_loss.backward() 只会更新 actor
        # # - value_loss.backward() 只会更新 critic
        # # 这是在数学和工程上都成立的。
        # #
        # # ⚠️ 注意：如果未来引入 shared embedding / shared trunk（actor-critic 共享部分参数），
        # # 必须严格控制梯度流向：
        # # - policy loss 中的 advantage 必须对 value.detach()
        # # - actor 与 critic 的 backward / optimizer.step 需要明确隔离
        # # 否则 critic 会被 policy loss 的梯度“错误地拖动”，导致训练不稳定。
        # batch_vt: list[float] = values.detach().squeeze(dim=-1).tolist()
        # 然后开始计算returns和advantages

        # 4. policy compute loss and value loss
        # 大问题！我这样计算出来的loss是没有梯度的！为什么？
        # policy_loss = compute_loss(
        #     batch_log_probs, batch_rewards_to_go, batch_vt=batch_vt
        # )

        # policy_loss = -(old_rollout.log_probs * advantages).mean()
        # values = agent.value_forward(state=old_rollout.observations).squeeze(dim=-1)
        # value_loss = (values - returns).pow(2).mean()
        # entropy = old_rollout.entropy.mean()
        # loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # # 5. policy net backward propogation
        # loss.backward()
        # # 1) actor must have grads
        # # assert any(p.grad is not None for p in agent.parameters())
        # # # 2) critic must NOT get grads from policy loss
        # # assert all(p.grad is None for p in value_net.parameters())

        # # 6. optimize policy net
        # optimizer.step()

        ## 7. compute vt loss and optimize
        # BUG！不对，这个optimizer 不应该在这里
        # 经过测试，这个optimizer在哪里都不影响
        # optimizer4value.zero_grad()
        # value_loss = compute_value_loss(values, batch_rewards_to_go)
        # value_loss.backward()
        # assert any(p.grad is not None for p in value_net.parameters())
        # optimizer4value.step()

    # 为了知道我们当前模型的效果，我们必须提取一些metric
    # 一个比较简单的做法就是直接用当前的网络模型的参数sample几个episode，看看分数的分布


def test_ppo_on_cart_pole():
    env = CartPole()
    assert env.observation_space()[0] == 4
    assert env.action_space() == 2

    num_env = 8
    envs: list[EnvBase] = [CartPole() for _ in range(num_env)]

    agent = SimpleActorCritic(
        observation_shape=env.observation_space()[0], action_shape=env.action_space()
    )

    trainer = PPOTrainer(
        agent=agent,
        envs=envs,
        max_epochs=5000,
        lr=0.001,
        rollout_step=128,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        c1=0.5,
        c2=0.01,
        iter_per_epoch=8,  # PPO
        # iter_per_epoch=1,  # A2C
    )

    trainer.train()
