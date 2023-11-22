# 2023/11/22
# zhangzhong
# GoogLeNet

import torch
from torch import nn, Tensor


class InceptionBlock(nn.Module):
    def __init__(self, out_channels: list[tuple[int, int]]):
        super().__init__()
        self.out_channels = out_channels
        assert len(out_channels) == 4

        self.branch1 = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels[0][0], kernel_size=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels[1][0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(
                out_channels=out_channels[1][1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels[2][0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(
                out_channels=out_channels[2][1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels[3][0], kernel_size=1),
            nn.ReLU()
        )

    def forward(self, input: Tensor) -> Tensor:
        bi, ci, hi, wi = input.shape
        # concat four branches on the channel dim
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output3 = self.branch3(input)
        output4 = self.branch4(input)

        output = torch.cat(tensors=[output1, output2, output3, output4], dim=1)
        bo, co, ho, wo = output.shape
        assert bi == bo and hi == ho and wi == wo
        return output


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        )
        self.block3 = nn.Sequential(
            InceptionBlock(
                out_channels=[(64, 0), (96, 128), (16, 32), (32, 0)]),
            InceptionBlock(
                out_channels=[(128, 0), (128, 192), (32, 96), (64, 0)]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block4 = nn.Sequential(
            InceptionBlock(
                out_channels=[(192, 0), (96, 208), (16, 48), (64, 0)]),
            InceptionBlock(
                out_channels=[(160, 0), (112, 224), (24, 64), (64, 0)]),
            InceptionBlock(
                out_channels=[(128, 0), (128, 256), (24, 64), (64, 0)]),
            InceptionBlock(
                out_channels=[(112, 0), (114, 288), (32, 64), (64, 0)]),
            InceptionBlock(
                out_channels=[(256, 0), (160, 320), (32, 128), (128, 0)]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block5 = nn.Sequential(
            InceptionBlock(
                out_channels=[(256, 0), (160, 320), (32, 128), (128, 0)]),
            InceptionBlock(
                out_channels=[(384, 0), (192, 384), (48, 128), (128, 0)]),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        self.net = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4, self.block5,
            nn.LazyLinear(num_classes)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)
