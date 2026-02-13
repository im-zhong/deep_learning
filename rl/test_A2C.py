# 2026/2/13
# zhangzhong

# a configural general RL trainer
from tqdm import tqdm
from rl.test_envs import EnvBase
from rl.test_agent import AgentBase
from torch.optim import Adam
import torch
from rl.test_rl_utils import Rollout, sample_rollout_v2, compute_gaes, compute_returns
from rl.test_envs import CartPole, LunarLander
from rl.test_agent import SimpleActorCritic


# TODO： 我感觉到是这个可以变成A2CTrainer，然后把train one epoch里面的collect data and sample rollout and compute returns and advantages 给抽象出来
class A2CTrainer:
    def __init__(
        self,
        agent: AgentBase,
        env: EnvBase,
        max_epochs: int,
        lr: float,
        rollout_step: int,
        gamma: float,
        lambda_: float,
    ) -> None:
        self.agent = agent
        self.env = env
        self.max_epochs = max_epochs
        self.optimizer = Adam(params=agent.parameters(), lr=lr)
        self.rollout_step = rollout_step
        self.gamma = gamma
        self.lambda_ = lambda_

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.train_one_epoch(epoch, self.env, self.agent, self.optimizer)

    # 好像还真不行，不同的算法的采样数据的方式和计算loss的方式不一样！
    # 不如把这个train one epoch扔给每个算法自己实现？
    # 其实咱总共就实现两个算法啊 A2C 和PPO 先不管了
    def train_one_epoch(
        self,
        epoch: int,
        env: EnvBase,
        agent: AgentBase,
        optimizer: torch.optim.Optimizer,
    ):
        # 1. collect training data of this epoch
        # batch_states, batch_actions, batch_log_probs, batch_rewards_to_go = (
        #     collect_training_data()
        # )

        rollout = sample_rollout_v2(agent, env, self.rollout_step)

        # 咱们把rewards給加起来，看看训练的效果
        print(
            f"epoch: {epoch}: rewards:, {len(rollout.rewards)}, {sum(rollout.rewards)}",
            flush=True,
        )

        # sample_rollout_v2(agent, env, self.rollout_step)

        # 2. zero policy net grads
        # optimizer4agent.zero_grad()
        # # 应该在这里吧
        # optimizer4value.zero_grad()
        optimizer.zero_grad()

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
        with torch.no_grad():
            bootstrap: float = 0.0
            if not rollout.terminated:
                # 这样这里的计算就可以放到gpu上了
                bootstrap = (
                    agent.value_forward(state=rollout.next_observation)
                    .squeeze(dim=-1)
                    .item()
                )

            # TODO: 应该重构成返回tensor
            returns = compute_returns(
                rewards=rollout.rewards, bootstrap=bootstrap, gamma=self.gamma
            )
            # 这里需要计算values，我们可以重复计算，这里不进行梯度计算，可以和PPO更好的融合
            values: torch.Tensor = agent.value_forward(
                state=rollout.observations
            ).squeeze(dim=-1)
            advantages = compute_gaes(
                rollout.rewards,
                values,
                bootstrap,
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            # advantages = returns - values
            # 加上一个mormalize操作
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. policy compute loss and value loss
        # 大问题！我这样计算出来的loss是没有梯度的！为什么？
        # policy_loss = compute_loss(
        #     batch_log_probs, batch_rewards_to_go, batch_vt=batch_vt
        # )
        assert rollout.log_probs.shape == advantages.shape == values.shape

        policy_loss = -(rollout.log_probs * advantages).mean()
        values = agent.value_forward(state=rollout.observations).squeeze(dim=-1)
        value_loss = (values - returns).pow(2).mean()
        entropy = rollout.entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 5. policy net backward propogation
        loss.backward()
        # 1) actor must have grads
        # assert any(p.grad is not None for p in agent.parameters())
        # # 2) critic must NOT get grads from policy loss
        # assert all(p.grad is None for p in value_net.parameters())

        # 6. optimize policy net
        optimizer.step()

        ## 7. compute vt loss and optimize
        # BUG！不对，这个optimizer 不应该在这里
        # 经过测试，这个optimizer在哪里都不影响
        # optimizer4value.zero_grad()
        # value_loss = compute_value_loss(values, batch_rewards_to_go)
        # value_loss.backward()
        # assert any(p.grad is not None for p in value_net.parameters())
        # optimizer4value.step()


def test_A2C_trainer():
    env = CartPole()
    assert env.observation_space()[0] == 4
    assert env.action_space() == 2

    agent = SimpleActorCritic(
        observation_shape=env.observation_space()[0], action_shape=env.action_space()
    )

    trainer = A2CTrainer(
        agent=agent,
        env=env,
        max_epochs=5000,
        lr=0.001,
        rollout_step=256,
        gamma=0.99,
        lambda_=0.9,
    )

    trainer.train()


def test_A2C_trainer_on_lunar_lander():
    env = LunarLander()
    assert env.observation_space()[0] == 8
    assert env.action_space() == 4

    agent = SimpleActorCritic(
        observation_shape=env.observation_space()[0], action_shape=env.action_space()
    )

    trainer = A2CTrainer(
        agent=agent,
        env=env,
        max_epochs=10000,
        lr=0.0001,
        rollout_step=256,
        gamma=0.99,
        lambda_=0.95,
    )

    trainer.train()
