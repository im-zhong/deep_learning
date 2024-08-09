# 2023/11/21
# zhangzhong

import matplotlib.pyplot as plt
import torch

from mytorch import utils
from mytorch.net.encoding import PositionalEncoding


def test_PositionalEncoding():
    batch_size, seq_size, hidden_size = 4, 16, 32
    pe = PositionalEncoding(hidden_size=hidden_size)
    x = torch.rand(size=(batch_size, seq_size, hidden_size))
    y = pe(x)

    # 咱们画一张热力图
    positional_encoding = pe.position[0]
    plt.imshow(positional_encoding[:64, :])
    plt.colorbar()
    # plt.savefig('positional_encoding.png')
    utils.mysavefig("positional_encoding.png")

    max_len = 1024
    num_hiddens = hidden_size
    P = torch.zeros((1, max_len, num_hiddens))
    X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
    )
    P[:, :, 0::2] = torch.sin(X)
    P[:, :, 1::2] = torch.cos(X)
    p = P[0]
    plt.imshow(p[:64, :])
    plt.colorbar()
    # plt.savefig('gt_pe.png')
    utils.mysavefig("gt_pe.png")
