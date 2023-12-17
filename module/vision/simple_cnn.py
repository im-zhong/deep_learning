# 2023/12/17
# zhangzhong

import torch
from torch import nn, Tensor

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.LazyConv2d(out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.LazyLinear(out_features=128)
        self.fc2 = nn.LazyLinear(out_features=10)
        
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(start_dim=1, end_dim=-1),
            
            self.fc1,
            nn.ReLU(),
            
            self.fc2,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
