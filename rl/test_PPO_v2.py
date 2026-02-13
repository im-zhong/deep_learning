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
from torch.distributions import Categorical


class PPOTrainer:
    def __init__(
        self,
        agent: AgentBase,
        env: EnvBase,
        max_epochs: int,
        lr: float,
        rollout_step: int,
        gamma: float,
        lambda_: float,
        epsilon: float,
        c1: float,
        c2: float,
    ) -> None:
        self.agent = agent
        self.env = env
        self.max_epochs = max_epochs
        self.optimizer = Adam(params=agent.parameters(), lr=lr)
        self.rollout_step = rollout_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.train_one_epoch(epoch, self.env, self.agent, self.optimizer)

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

        with torch.no_grad():
            old_rollout = sample_rollout_v2(agent, env, self.rollout_step)

            # 咱们把rewards給加起来，看看训练的效果
            print(
                f"epoch: {epoch}: rewards:, {len(old_rollout.rewards)}, {sum(old_rollout.rewards)}",
                flush=True,
            )

            bootstrap: float = 0.0
            if not old_rollout.terminated:
                # 这样这里的计算就可以放到gpu上了
                bootstrap = (
                    agent.value_forward(state=old_rollout.next_observation)
                    .squeeze(dim=-1)
                    .item()
                )

            # TODO: 应该重构成返回tensor
            old_returns = compute_returns(
                rewards=old_rollout.rewards, bootstrap=bootstrap, gamma=self.gamma
            )
            # 这里需要计算values，我们可以重复计算，这里不进行梯度计算，可以和PPO更好的融合
            old_values: torch.Tensor = agent.value_forward(
                state=old_rollout.observations
            ).squeeze(dim=-1)
            old_advantages = compute_gaes(
                old_rollout.rewards,
                old_values,
                bootstrap,
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            # advantages = returns - values
            # 加上一个mormalize操作
            old_advantages = (old_advantages - old_advantages.mean()) / (
                old_advantages.std() + 1e-8
            )
            assert (
                old_rollout.log_probs.shape == old_advantages.shape == old_values.shape
            )

        # loop in minibatch
        all_step = len(old_rollout.rewards)
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
            logits = self.agent.policy_forward(
                state=old_rollout.observations[i : i + step]
            )

            probs = Categorical(logits=logits)
            new_log_probs: torch.Tensor = probs.log_prob(
                value=torch.tensor(old_rollout.actions[i : i + step])
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
            r_theta = torch.exp(new_log_probs - old_rollout.log_probs[i : i + step])
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
                state=old_rollout.observations[i : i + step]
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


def test_ppo_on_cart_pole():
    env = CartPole()
    assert env.observation_space()[0] == 4
    assert env.action_space() == 2

    agent = SimpleActorCritic(
        observation_shape=env.observation_space()[0], action_shape=env.action_space()
    )

    trainer = PPOTrainer(
        agent=agent,
        env=env,
        max_epochs=5000,
        lr=0.0001,
        rollout_step=256,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        c1=0.5,
        c2=0.01,
    )

    trainer.train()
