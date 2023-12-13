# 2023/12/13
# zhangzhong
# Transformers for Vision

import torch
from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, stride: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.LazyConv2d(out_channels=hidden_size, kernel_size=kernel_size, stride=stride)

    def forward(self, images: Tensor):
        return self.conv(images).flatten(start_dim=2).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.positional_embedding = nn.Parameter(torch.randn(size=(1, max_len, hidden_size)))

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs.shape = (b, s, h)
        # _, seq_size, _ = inputs.shape
        # selected_positional_embedding = self.positional_embedding[:, :seq_size]
        # assert selected_positional_embedding.shape == (1, seq_size, self.hidden_size)
        return inputs + self.positional_embedding


class ViTMLP(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LazyLinear(out_features=output_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class ViTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, mlp_hidden_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_size)
        # TODO: maybe choose another hyper-parameters
        self.mlp = ViTMLP(hidden_size=mlp_hidden_size, output_size=hidden_size, dropout=dropout)

    def forward(self, inputs: Tensor):
        # residual connections 1: multihead attention
        x = self.ln1(inputs)
        # BUG:FIX, attention的参数中q,k,v应该拿到x的三个深拷贝的副本
        # 不对，没有任何区别
        attention_output, _ = self.attention(query=x, key=x, value=x)
        # attention_output, _ = self.attention(*([self.ln1(inputs)] * 3))
        inputs = inputs + attention_output
        # residual connection 2: mlp
        return inputs + self.mlp(self.ln2(inputs))


class ViT(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, stride: int,
                 max_len: int, dropout: float, num_blocks: int,
                 num_heads: int, mlp_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cls = nn.Parameter(torch.zeros(size=(1, 1, hidden_size)))
        self.patch_embedding = PatchEmbedding(hidden_size=hidden_size,
                                              kernel_size=kernel_size,
                                              stride=stride)
        self.positional_embedding = PositionalEmbedding(hidden_size=hidden_size,
                                                        max_len=max_len)
        # TODO: there are many dropout, maybe tune them
        self.dropout = nn.Dropout(p=dropout)
        # ViT blocks
        # 这样写torchinfo识别不了 还是得写成sequential？
        blocks = [ViTBlock(hidden_size=hidden_size,
                           num_heads=num_heads,
                           dropout=dropout,
                           mlp_hidden_size=mlp_hidden_size) for _ in range(num_blocks)]
        # special <cls> token
        self.blocks = nn.Sequential(*blocks)

    def forward(self, images: Tensor):
        # patch embedding
        batch_size, channels, height, weight = images.shape
        x = self.patch_embedding(images)
        # assert x.shape == (batch_size, seq_size, self.hidden_size)
        # batch_size1, seq_size, hidden_size = x.shape
        # assert batch_size == batch_size1
        # assert hidden_size == self.hidden_size

        # add cls
        # device = x.device
        # first copy cls to the device of x
        # clss = self.cls.repeat(batch_size, 1, 1).to(device)

        # y = torch.concat([clss, x], dim=1)
        # 不对，我理解错了，这里的y并不需要有required_grads=True
        # 因为
        self.cls = self.cls.to(x.device)
        y: Tensor = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), dim=1)
        # assert y.shape == (batch_size, seq_size + 1, hidden_size)
        # y.requires_grad_()

        # add positional embedding
        z = self.positional_embedding(y)

        # through dropout
        embedding = self.dropout(z)

        # through n vit blocks
        # for block in self.blocks:
        #     inputs = block(inputs)
        outputs = self.blocks(embedding)

        # get cls for outputs
        return outputs[:, 0]


class ViTClassifier(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, stride: int,
                 max_len: int, dropout: float, num_blocks: int,
                 num_heads: int, mlp_hidden_size: int, output_size: int):
        super().__init__()
        self.vit = ViT(hidden_size=hidden_size,
                       kernel_size=kernel_size,
                       stride=stride,
                       max_len=max_len,
                       dropout=dropout,
                       num_blocks=num_blocks,
                       num_heads=num_heads,
                       mlp_hidden_size=mlp_hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.linear = nn.LazyLinear(out_features=output_size)

    def forward(self, images: Tensor):
        return self.linear(self.ln(self.vit(images)))
