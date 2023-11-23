# 2023/11/23
# zhangzhong

import torch
from torch import nn, Tensor


class ConvBlock(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(), nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, num_convs: int, growth_rate: int) -> None:
        super().__init__()
        # pytorch可以检测到list Of Modules 吗
        # 还是换成sequential更保险一点
        convs = [ConvBlock(out_channels=growth_rate)] * num_convs
        self.net = nn.Sequential(*convs)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.net:
            y = conv(x)
            # concat x and y on the channel dim
            x = torch.cat([x, y], dim=1)
        # 最终输出是x
        return x


class TransitionBlock(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        # 1. 削减channels
        # 2. 降采样 AvgPool2d
        self.net = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )


class DenseNet(nn.Module):
    def __init__(self, arch: list[tuple[int, int]], num_classes: int) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        # structure:
        # block1
        # dense block x 4
        # last block

        block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blocks: list[DenseBlock] = []
        for i, (num_convs, growth_rate) in enumerate(arch):
            blocks.append(DenseBlock(num_convs=num_convs, growth_rate=growth_rate))

        last = nn.Sequential(
            # 相较于resnet的last block添加了前两行
            # 因为densenet最后一层是conv层
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes)
        )

        self.net = nn.Sequential(
            block1,
            *blocks,
            last
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


dense_net_arch = [(4, 32), (4, 32), (4, 32), (4, 32)]
