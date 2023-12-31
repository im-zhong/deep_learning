# 2023/11/22
# zhangzhong


from torch import nn, Tensor
import torch
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int, use_bypass: bool) -> None:
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.use_bypass = use_bypass

        self.net = nn.Sequential(
            nn.LazyConv2d(
                out_channels=channels, kernel_size=3, stride=stride, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(
                out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
        )
        self.bypass = nn.Sequential(
            nn.LazyConv2d(out_channels=channels, kernel_size=1, stride=stride),
            nn.LazyBatchNorm2d()
        ) if use_bypass else None
        self.relu = nn.ReLU()
        # print(stride)
        # print(self.net)
        # print(self)

    def forward(self, input: Tensor) -> Tensor:
        res = self.net(input)
        bypass = self.bypass(input) if self.bypass is not None else input
        output = res + bypass
        return self.relu(output)


class ResNetBlock(nn.Module):
    def __init__(self, num_residuals: int, out_channels: int, is_first_block: bool) -> None:
        super().__init__()
        self.num_residuals = num_residuals
        self.out_channels = out_channels

        # print(is_first_block)
        # self.net = nn.Sequential()

        # for i in range(num_residuals):
        #     if i == 0 and not is_first_block:
        #         # at the first residual block, we need use bypass
        #         # TIP: add_module 只会add一个模块上去
        #         self.net.add_module(name='ResNetBlock', module=ResidualBlock(
        #             channels=out_channels, stride=2, use_bypass=True))
        #     else:
        #         self.net.add_module(name='ResNetBlock', module=ResidualBlock(
        #             channels=out_channels, stride=1, use_bypass=False))
        blocks: list[ResidualBlock] = []
        for i in range(num_residuals):
            if i == 0 and not is_first_block:
                # at the first residual block, we need use bypass
                blocks.append(ResidualBlock(
                    channels=out_channels, stride=2, use_bypass=True))
            else:
                blocks.append(ResidualBlock(
                    channels=out_channels, stride=1, use_bypass=False))
        self.net = nn.Sequential(*blocks)
        # print(self.net)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class ResNet(nn.Module):
    def __init__(self, arch: list[tuple[int, int]], num_classes: int) -> None:
        super().__init__()
        self.arch: list[tuple[int, int]] = arch
        self.num_classes = num_classes

        # 我感觉如果这些临时的模块是不应该作为成员变量的
        # 因为pytorch可能会扫描到重复的module
        block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blocks: list[ResNetBlock] = []
        for i, (num_residuals, out_channels) in enumerate(arch):
            blocks.append(ResNetBlock(num_residuals=num_residuals,
                                      out_channels=out_channels, is_first_block=(i == 0)))

        # for block in blocks:
        #     print(block)
        last = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes)
        )

        self.net = nn.Sequential(
            block1,
            *blocks,
            last
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class SmallResNet(nn.Module):
    def __init__(self, arch: list[tuple[int, int]], num_classes: int) -> None:
        super().__init__()
        self.arch: list[tuple[int, int]] = arch
        self.num_classes = num_classes

        # 我感觉如果这些临时的模块是不应该作为成员变量的
        # 因为pytorch可能会扫描到重复的module
        block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blocks: list[ResNetBlock] = []
        for i, (num_residuals, out_channels) in enumerate(arch):
            blocks.append(ResNetBlock(num_residuals=num_residuals,
                                      out_channels=out_channels, is_first_block=(i == 0)))

        # for block in blocks:
        #     print(block)
        last = nn.Sequential(OrderedDict([
            ('AdaptiveMaxPool2d', nn.AdaptiveMaxPool2d(output_size=(1, 1))),
            ('Flatten', nn.Flatten()),
            ('LazyLinear1', nn.LazyLinear(out_features=1024)),
            ('BatchNorm', nn.LazyBatchNorm1d()),
            ('ReLU', nn.ReLU()),
            # ('Dropout', nn.Dropout(p=0.01)),
            ('LazyLinear2', nn.LazyLinear(out_features=num_classes))
        ]))

        self.net = nn.Sequential(OrderedDict([
            ('block1', block1),
            *[('ResNetBlock', block) for block in blocks],
            ('last', last)
        ]))

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class ResNet18(ResNet):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(arch=[(2, 64), (2, 128),
                               (2, 256), (2, 512)], num_classes=num_classes)


class SmallResNet18(SmallResNet):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(arch=[(2, 64), (2, 128),
                               (2, 256), (2, 512)], num_classes=num_classes)
