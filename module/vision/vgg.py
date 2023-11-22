# 2023/11/21
# zhangzhong

import torch
from torch import nn, device, Tensor
from typing import Any


class VGGBlock(nn.Module):
    def __init__(self, num_convs: int, out_channels: int):
        self.num_convs = num_convs
        self.out_channels = out_channels
        layers: list[Any] = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(
                out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(
            *layers,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class VGG(nn.Module):
    def __init__(self, arch: list[tuple[int, int]]):
        self.arch = arch
        blocks: list[VGGBlock] = []
        for num_convs, out_channels in arch:
            blocks.append(VGGBlock(num_convs=num_convs,
                          out_channels=out_channels))
        # encoder, lots of conv layers
        self.net = nn.Sequential(
            # encoder, lots of conv layers
            *blocks,

            # flatten and dense block
            nn.Flatten(),
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(out_features=1000),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


original_arch = [(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)]
