import mytorch.func as func
import torch
from torch import nn, Tensor
from mytorch.net.cnn import MyAvgPool2d, MyConv2d


class MyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder Block, conv layer 1
            # 5x5 Conv2d(1, 6), pad = 2
            MyConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            # activation: sigmoid, sigmod会作用在所有元素上
            # https://pytorch.org/docs/stable/special.html#torch.special.expit
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            MyAvgPool2d(in_channels=6, kernel_size=2, stride=2),

            # conv layer2
            # 5x5 Conv(6, 16)
            MyConv2d(in_channels=6, out_channels=16, kernel_size=5),
            # activation: sigmoid
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            MyAvgPool2d(in_channels=16, kernel_size=2, stride=2),

            # now the output shape is (b, c, h, w), but fc layer can only accept (b, f)
            # so we just flatten it
            nn.Flatten(start_dim=1, end_dim=-1),

            # Dense Block, fc layer1
            nn.LazyLinear(out_features=120),
            # activation
            nn.Sigmoid(),

            # fc layer2
            nn.LazyLinear(out_features=84),
            nn.Sigmoid(),

            # fc layer3
            nn.LazyLinear(out_features=10),
        )

    def forward(self, input: Tensor) -> Tensor:
        b, ci, h, w = input.shape
        return self.net(input)


class LeNet(nn.Module):
    # 调库性能就没有问题 果然是我们实现的太垃圾了
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder Block, conv layer 1
            # 5x5 Conv2d(1, 6), pad = 2
            # MyConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2),
            # activation: sigmoid, sigmod会作用在所有元素上
            # https://pytorch.org/docs/stable/special.html#torch.special.expit
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            # MyAvgPool2d(in_channels=6, kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # conv layer2
            # 5x5 Conv(6, 16)
            # MyConv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.LazyConv2d(out_channels=16, kernel_size=5),
            # activation: sigmoid
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            # MyAvgPool2d(in_channels=16, kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # now the output shape is (b, c, h, w), but fc layer can only accept (b, f)
            # so we just flatten it
            nn.Flatten(start_dim=1, end_dim=-1),

            # Dense Block, fc layer1
            nn.LazyLinear(out_features=120),
            # activation
            nn.Sigmoid(),

            # fc layer2
            nn.LazyLinear(out_features=84),
            nn.Sigmoid(),

            # fc layer3
            nn.LazyLinear(out_features=10),
        )

    def forward(self, input: Tensor) -> Tensor:
        b, ci, h, w = input.shape
        return self.net(input)


class BNLeNet(nn.Module):
    # 调库性能就没有问题 果然是我们实现的太垃圾了
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder Block, conv layer 1
            # 5x5 Conv2d(1, 6), pad = 2
            # MyConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2),
            # add batch norm
            nn.LazyBatchNorm2d(),
            # activation: sigmoid, sigmod会作用在所有元素上
            # https://pytorch.org/docs/stable/special.html#torch.special.expit
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            # MyAvgPool2d(in_channels=6, kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # conv layer2
            # 5x5 Conv(6, 16)
            # MyConv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.LazyConv2d(out_channels=16, kernel_size=5),
            nn.LazyBatchNorm2d(),
            # activation: sigmoid
            nn.Sigmoid(),
            # downsampling: 2x2 AvgPool2d, stride = 2
            # MyAvgPool2d(in_channels=16, kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # now the output shape is (b, c, h, w), but fc layer can only accept (b, f)
            # so we just flatten it
            nn.Flatten(start_dim=1, end_dim=-1),

            # Dense Block, fc layer1
            nn.LazyLinear(out_features=120),
            nn.LazyBatchNorm1d(),
            # activation
            nn.Sigmoid(),

            # fc layer2
            nn.LazyLinear(out_features=84),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            # fc layer3
            nn.LazyLinear(out_features=10),
        )

    def forward(self, input: Tensor) -> Tensor:
        b, ci, h, w = input.shape
        return self.net(input)
