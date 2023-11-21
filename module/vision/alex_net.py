# 2023/11/21
# zhangzhong


from torch import nn, Tensor


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 11x11 conv
            nn.LazyConv2d(out_channels=96, kernel_size=11,
                          stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 5x5 conv
            nn.LazyConv2d(out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 3x3 conv
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 3x3 conv
            nn.LazyConv2d(out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            # 3x3 conv
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),

            # dense layer 1
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # dense layer 2
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # output layer
            nn.LazyLinear(out_features=1000),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)
