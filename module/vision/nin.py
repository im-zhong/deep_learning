# 2023/11/22
# zhangzhong

from numpy import pad
import torch
from torch import nn, Tensor


class NiNBlock(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int, padding: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            # 这里没有maxpool 因为最后一个conv层用的是avgpool
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class NiN(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            NiNBlock(out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            NiNBlock(out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            NiNBlock(out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Dropout(p=0.5),

            # output layer
            NiNBlock(out_channels=1000, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        pass

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)
