# 2023/12/14
# zhangzhong
# MLP

from torch import Tensor, nn


class MLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(out_features=hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class MLP(nn.Module):
    def __init__(self, arch: list[int], output_size: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        for hidden_size in arch:
            blocks.append(MLPBlock(hidden_size=hidden_size, dropout=dropout))
        self.net = nn.Sequential(
            *blocks, nn.LazyBatchNorm1d(), nn.LazyLinear(out_features=output_size)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # if inputs is image, we should flatten it
        if len(inputs.shape) == 4:
            inputs = inputs.flatten(start_dim=1)
        return self.net(inputs)
