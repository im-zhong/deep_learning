from torch import Tensor, nn


# QNet
# 其实我们的qnet比较简单，就是一个MLP，输入纬度是observation space的维度
# 输出就是action space的维度
class DQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        device: str = "cpu",  # cpu, cuda, mps, ...
    ) -> None:
        super().__init__()
        self._observation_size: int = observation_size
        self._action_size: int = action_size
        self._device = device

        self.layer1 = nn.Linear(
            in_features=observation_size, out_features=128, device=device
        )
        self.layer2 = nn.Linear(in_features=128, out_features=128, device=device)
        self.layer3 = nn.Linear(
            in_features=128, out_features=action_size, device=device
        )
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        x = self.relu(self.layer1(input))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    # 在pytorch中 dim表示第几个维度， 如 dim = 0 表示第一个维度
    # 像 torch.stack(tensors, dim=0) 就是把tensors中的tensor按照第一个维度堆叠起来
    # 所以我们用dim表示元素的数量是不合适呢
    # 应该用size, 或者 shape
    # shape通常是一个tuple，而且在pytorch中就是用来表示输入或者输出的向量形状的
    # 所以非常合适
    # 这里我们使用size 表示 shape中 某一个维度的 元素数量
    # 这样 dim shape size 的含义就固定下来
    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device
