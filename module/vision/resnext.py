# 2023/11/23
# zhangzhong

from torch import nn, Tensor


class ResNextBlock(nn.Module):
    def __init__(self, out_channels: int, groups: int, is_down_sampling: bool):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.is_down_sampling = is_down_sampling

        stride = 2 if is_down_sampling else 1
        self.net = nn.Sequential(
            # 1x1 conv, for transform channels
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),

            # 3x3 group conv, for down sampling maybe
            nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),

            # 1x1 conv, for connect groups
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1),
            nn.LazyBatchNorm2d()
        )

        self.bypass = nn.Sequential(
            nn.LazyConv2d(out_channels=out_channels, kernel_size=1,
                          stride=stride),
            nn.LazyBatchNorm2d()
        ) if is_down_sampling else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        bi, ci, hi, wi = x.shape
        res = self.net(x)
        x = x if self.bypass is None else self.bypass(x)
        y = self.relu(res + x)
        return y
