# 2023/11/20
# zhangzhong

import torch
from torch import Tensor, nn

from mytorch import func


class BatchNorm(nn.Module):
    def __init__(self, feature_size: int):
        self.feature_size = feature_size
        self.gamma = torch.nn.Parameter(torch.ones(size=[feature_size]))
        self.beta = torch.nn.Parameter(torch.zeros(size=[feature_size]))

    def forward(self, input: Tensor) -> Tensor:
        return func.batch_norm(input, self.gamma, self.beta)


class LinearBatchNorm(nn.Module):
    pass


class Conv2dBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gamma: nn.Parameter | None = None
        self.beta: nn.Parameter | None = None

    def forward(self, input: Tensor) -> Tensor:
        batch_size, channels, height, width = input.shape
        if self.gamma is None:
            self.gamma = nn.Parameter(
                torch.ones(
                    size=(1, channels, 1, 1), device=input.device, requires_grad=True
                )
            )
            self.beta = nn.Parameter(
                torch.zeros(
                    size=(1, channels, 1, 1), device=input.device, requires_grad=True
                )
            )

        output = func.conv2d_batch_norm(
            input=input, gamma=self.gamma, beta=self.beta
        )  # type: ignore
        return output


class LayerNorm(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.gamma = torch.nn.Parameter(torch.ones(size=[feature_size]))
        self.beta = torch.nn.Parameter(torch.zeros(size=[feature_size]))

    def forward(self, input: Tensor) -> Tensor:
        return func.layer_norm(input, self.gamma, self.beta)


class AddNorm(nn.Module):
    def __init__(self, feature_size: int, dropout: float = 0.2):
        super().__init__()
        # self.ln = LayerNorm(feature_size)
        self.ln = nn.LayerNorm(normalized_shape=feature_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: Tensor, output: Tensor) -> Tensor:
        # TODO: 感觉这里的实现好像和论文里面不一样？确认一下
        return self.ln(input + self.dropout(output))
