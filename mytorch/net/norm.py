# 2023/11/20
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch import func


class BatchNorm(nn.Module):
    def __init__(self, feature_size: int):
        self.feature_size = feature_size
        self.gamma = torch.nn.Parameter(torch.ones(size=[feature_size]))
        self.beta = torch.nn.Parameter(torch.zeros(size=[feature_size]))

    def forward(self, input: Tensor) -> Tensor:
        return func.batch_norm(input, self.gamma, self.beta)


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
        return self.ln(input + self.dropout(output))
