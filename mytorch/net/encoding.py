# 2023/11/21
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch import config


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.max_len: int = max_len
        # 感觉好多层都有dropout哦 为了提高generalization
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.position = torch.zeros(
            size=(max_len, hidden_size), dtype=torch.float32, device=config.conf['device'])

        # 列向量
        # i
        rows = torch.arange(start=0, end=max_len, dtype=torch.float32,
                            device=config.conf['device']).reshape(shape=(-1, 1))

        # 行向量
        # 2j/d
        cols = torch.arange(start=0, end=hidden_size,
                            step=2, dtype=torch.float32, device=config.conf['device']) / hidden_size
        # 现在行向量的长度是hidden_size/2
        # 现在这个就是频率
        # cols = 1.0 / (10000 ** cols)
        cols = 1.0 / torch.pow(10000, cols)

        # 然后进行一个boardcasting
        # BUG:FIX 这里应该是乘法 因为我们上面已经计算了倒数
        x = rows * cols
        # x = rows / torch.pow(10000, cols)
        assert x.shape == (max_len, hidden_size / 2)
        # 接下来就可以计算position matrix了
        # 所有的列按照 sin cos, sin cos 的顺序排列就行了
        # 从零开始，到最后的列，每隔一列 也就是 0, 2, 4, ... sin
        # 相应的， 1, 3, 5, ... 都是cos
        self.position[:, 0::2] = torch.sin(x)
        self.position[:, 1::2] = torch.cos(x)
        assert self.position.shape == (max_len, hidden_size)

        self.position = self.position.unsqueeze(dim=0)
        assert self.position.shape == (1, max_len, hidden_size)

    def forward(self, input: Tensor) -> Tensor:
        # embedding的结果是我们的输入
        # 也就是说我们的shape是 batch_size, seq_size, embed_size
        # 而我们的输出也要和输入一样
        # 所以我们的position 要在第一个维度上进行广播
        batch_size, seq_size, hidden_size = input.shape
        # 输入序列的长度必须小于我们预先计算的序列的最大长度
        # 在下面进行input+position的时候，只需要选择对应长度，即seq_size即可
        #    input.shape = (b, s, h)
        # position.shape = (1, s, h)
        # 刚好可以广播 每个seq的positional encoding是一样的
        assert seq_size < self.max_len
        # 这里应该在batch_size这个维度，也就是dim=0上进行广播
        # 但是最稳妥的实现方法就是让self.position的shape从(seq_size, hidden_size) -> (1, seq_size, hidden_size)
        x = input + self.position[:, :seq_size, :]
        assert x.shape == input.shape
        return self.dropout(x)
