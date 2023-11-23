# 2023/11/21
# zhangzhong

import torch
from torch import nn

from mytorch.net.cnn import MyAvgPool2d, MyConv2d, MyMaxPool2d, MyConv2dWithGroups


def test_MyConv2d():
    conv = MyConv2d(in_channels=1, out_channels=1,
                    kernel_size=3, padding=1, stride=2)
    input = torch.rand(size=(2, 1, 8, 8))
    output = conv(input=input)
    assert output.shape == (2, 1, 4, 4)

    conv = MyConv2d(in_channels=1, out_channels=1,
                    kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    input = torch.rand(size=(1, 1, 8, 8))
    output = conv(input=input)
    assert output.shape == (1, 1, 2, 2)

    conv = MyConv2d(in_channels=2, out_channels=3,
                    kernel_size=2, padding=0, stride=1)
    input = torch.stack([torch.arange(start=0, end=9, step=1).reshape(
        3, 3), torch.arange(start=1, end=10, step=1).reshape(3, 3)])
    assert input.shape == (2, 3, 3)
    kernels = torch.tensor([
        [[0, 1], [2, 3], ],
        [[1, 2], [3, 4], ]
    ])
    kernels = torch.stack([kernels, kernels + 1, kernels + 2])
    # forward impl不考虑batch 所以
    output = conv.forward_impl(input=input, kernels=kernels)
    print(output)
    assert output.shape == (3, 2, 2)
    ground_truth = torch.tensor([
        [[56, 72],
         [104, 120], ],
        [
            [76, 100],
            [148, 172],
        ],
        [
            [96, 128],
            [192, 224],
        ]
    ])
    assert torch.all(output == ground_truth)


def test_MyAvgPool2d():
    x = torch.arange(start=0, end=9).reshape(3, 3)
    input = torch.stack([x, x]).unsqueeze(dim=0)
    y = torch.tensor([
        [2, 3],
        [5, 6],
    ])
    ground_truth = torch.stack([y, y]).unsqueeze(dim=1)
    in_channels = 2
    kernel_size = 2
    stride = 1
    padding = 0
    pool = MyAvgPool2d(
        in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    output = pool(input)
    assert torch.all(output == ground_truth)


def test_MyMaxPool2d():
    x = torch.arange(start=0, end=9).reshape(3, 3)
    input = torch.stack([x, x]).unsqueeze(dim=0)
    y = torch.tensor([
        [4, 5],
        [7, 8],
    ])
    ground_truth = torch.stack([y, y]).unsqueeze(dim=1)
    in_channels = 2
    kernel_size = 2
    stride = 1
    padding = 0
    pool = MyMaxPool2d(
        in_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    output = pool(input)
    assert torch.all(output == ground_truth)


def test_my_conv2d_with_groups():
    in_channels = 4
    out_channels = 6
    groups = 2
    conv = MyConv2dWithGroups(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                              groups=groups)
    batch_size = 8
    height = 16
    width = 16
    x = torch.randn(size=(batch_size, in_channels, height, width))
    y = conv(x)
    print(y.shape)
    assert y.shape == (batch_size, out_channels, height, width)

    torch_conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, padding=1, groups=groups)
    ground_truth = torch_conv(x)
    assert y.shape == ground_truth.shape
