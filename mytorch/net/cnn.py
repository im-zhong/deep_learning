# 2023/11/21
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch import func


class MyConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size
        self.stride = func.make_tuple(stride)
        self.padding = func.make_tuple(padding)
        self.kernel_size = func.make_tuple(kernel_size)
        self.use_bias = bias

        self.kernels = nn.Parameter(torch.normal(mean=0, std=1, size=(
            out_channels, in_channels, *self.kernel_size)), requires_grad=True)
        if self.use_bias:
            # BUG:FIX 这里不对吧，每一个outputchannel都有一个bias
            self.bias = nn.Parameter(torch.zeros(
                self.out_channels), requires_grad=True)

    def calculate_output_shape(self, input: Tensor) -> tuple[int, int]:
        h, w = input.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        return (h + 2*ph - kh)//sh + 1, (w + 2*pw - kw)//sw + 1

    def multiin(self, inputs: Tensor, kernels: Tensor) -> Tensor:
        ci, h, w = inputs.shape
        ci, kh, kw = kernels.shape
        # for input, kernel in zip(inputs, kernels):
        #     corr2dv2(input=add_padding(input, padding=self.padding), kernel=kernel, stride=self.stride)

        return sum([func.corr2d(input=input, padding=self.padding, kernel=kernel, stride=self.stride) for input, kernel in zip(inputs, kernels)])
        # return torch.tensor([])

    def forward(self, input: Tensor) -> Tensor:
        # 现在我们要实现batch
        b, ci, h, w = input.shape
        outputs = []
        for batch in input:
            # 每个batch单独做 这样虽然效率很低 但是实现起来很简单
            ci, h, w = batch.shape
            batch_output = self.forward_impl(input=batch, kernels=self.kernels)
            outputs.append(batch_output)

        output = torch.stack(outputs)
        b, co, oh, ow = output.shape
        return output

    def forward_impl(self, input: Tensor, kernels: Tensor) -> Tensor:
        ci, h, w = input.shape

        output = torch.stack(
            [self.multiin(inputs=input, kernels=kernel) for kernel in kernels])
        assert output.shape == (self.out_channels, *
                                self.calculate_output_shape(input=input[0]))
        return output


class MyAvgPool2d(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def multiin_multiout(self, input: Tensor) -> Tensor:
        ci, h, w = input.shape
        outputs = [func.avg_pool2d(input=x, kernel_size=self.kernel_size,
                                   padding=self.padding, stride=self.stride) for x in input]
        return torch.stack(outputs)

    def forward(self, input: Tensor) -> Tensor:
        b, ci, h, w = input.shape
        outputs = [self.multiin_multiout(input=x) for x in input]
        return torch.stack(outputs)

# MaxPool唯一的不同就是把avgPool改成maxpool 所以我们应该复用


class MyMaxPool2d(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def multiin_multiout(self, input: Tensor) -> Tensor:
        ci, h, w = input.shape
        outputs = [func.max_pool2d(input=x, kernel_size=self.kernel_size,
                                   padding=self.padding, stride=self.stride) for x in input]
        return torch.stack(outputs)

    def forward(self, input: Tensor) -> Tensor:
        b, ci, h, w = input.shape
        outputs = [self.multiin_multiout(input=x) for x in input]
        return torch.stack(outputs)
