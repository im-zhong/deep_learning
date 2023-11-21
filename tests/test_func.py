# 2023/10.15
# zhangzhong
# test functional

import torch
import torch.nn.functional as F
from mytorch import func
from torch import nn

# BUG:FIX Ther are plenty of 'cpu or cuda' error
# we should find a way to handle that
# 我们应该遵循一个原则 我们在函数内部新产生的Tensor应该和输入的Tensor在同一个设备上
# 只有当我们无法确认的时候 才需要在参数中参数device 而且是必选的参数


def test_flatten():
    x = torch.tensor([[1]])
    # tensor([1])
    tx = torch.flatten(x)
    # tensor(1)
    mx = func.flatten(x)
    print(tx)
    print(mx)
    assert tx == mx


def test_one_host():
    ground_truth = F.one_hot(torch.arange(0, 5) % 3)
    x = func.one_hot(torch.arange(0, 5, dtype=torch.int64) % 3)
    print(x == ground_truth)
    print(ground_truth)
    print(x)
    # Tensor是不能作为all的一个迭代器输入的
    # 所以可以用torch.all
    # Tests if all elements in input evaluate to True.
    # https://pytorch.org/docs/stable/generated/torch.all.html
    assert torch.all(x == ground_truth)

    t = torch.tensor([[1, 2, 3], [3, 2, 1]])
    ground_truth = F.one_hot(t, num_classes=4)
    x = func.one_hot(t, num_classes=4)
    assert torch.all(x == ground_truth)


def test_embedding():
    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    embedding_matrix = torch.rand(10, 3)
    ground_truth = F.embedding(input, embedding_matrix)
    print(ground_truth)

    x = func.embedding(input, embedding_matrix)
    print(x)

    assert torch.all(x == ground_truth)


def test_batch_norm():
    num_feature = 2
    bn = nn.BatchNorm1d(num_features=num_feature)
    batch_size = 4
    batch = torch.ones(size=(batch_size, num_feature))
    y = bn(batch)
    print(y)
    gamma = torch.ones(size=[num_feature])
    beta = torch.zeros(size=[num_feature])
    y = func.batch_norm(input=batch, gamma=gamma, beta=beta)
    print(y)


def test_layer_norm():
    # seq
    batch_size, seq_size, feature_size = 2, 2, 4
    ln = nn.LayerNorm(normalized_shape=feature_size)
    # x = torch.rand(size=(batch_size, seq_size, feature_size))
    x = torch.tensor([[[1, 2, 3, 4], [6, 7, 3, 1]],
                     [[5, 3, 6, 7], [4, 3, 8, 3]]], dtype=torch.float32)
    print(x)
    y = ln(x)
    print(y)
    gamma = torch.ones(size=[feature_size])
    beta = torch.zeros(size=[feature_size])
    y = func.layer_norm(input=x, gamma=gamma, beta=beta)
    print(y)


def transpose_qkv(X, num_heads: int):
    """Transposition for parallel computation of multiple attention heads.
    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    # torch.permute vs torch.transpose
    # Transpose is a special case of permute, use it with 2d tensors
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    X = X.permute(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads: int):
    """Reverse the operation of transpose_qkv.
    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def test_multi_head():
    # 我们不仅要验证程序没有runtime error
    # 还要验证满足可逆运算
    batch_size, num_head, hidden_size = 4, 8, 32
    x = torch.rand(size=(batch_size, num_head, hidden_size))

    y = func.split_multi_head(input=x, num_head=num_head)
    # z = func.concat_multi_head(
    #     input=y, batch_size=batch_size, num_head=num_head)
    z = func.concat_multi_head(
        input=y, num_head=num_head)
    # 卧槽 一遍过 太对了！！！
    assert torch.all(x == z)

    # 还要验证和ground truth的实现是一样的
    # 卧槽 还是一遍过 太对啦！！！
    gt_y = transpose_qkv(X=x, num_heads=num_head)
    assert torch.all(y == gt_y)

    # 卧槽 又是一遍过 太对辣！！！
    gt_z = transpose_output(X=gt_y, num_heads=num_head)
    assert torch.all(z == gt_z)


def test_make_source_valid_lens():
    batch_size, num_seq = 3, 4
    source = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]])
    gt = torch.tensor([2, 3, 4])
    valid_lens = func.make_source_valid_lens(source, pad=0)
    assert torch.all(valid_lens == gt)


def test_target_valid_lens():
    batch_size, max_len = 2, 4
    # TODO: device should be default
    target_valid_lens = func.make_target_valid_lens(
        batch_size=batch_size, max_len=max_len, device='cpu')

    print(target_valid_lens)

    gt = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert torch.all(gt == target_valid_lens)


def test_corr2d():
    input = torch.ones(size=(6, 8))
    input[:, 2:6] = 0
    kernel = torch.tensor([[1.0, -1.0]])
    output = func.corr2d(input=input, kernel=kernel)
    # ground truth
    ground_truth = torch.tensor([[0.,  1.,  0.,  0.,  0., -1.,  0.],
                                 [0.,  1.,  0.,  0.,  0., -1.,  0.],
                                 [0.,  1.,  0.,  0.,  0., -1.,  0.],
                                 [0.,  1.,  0.,  0.,  0., -1.,  0.],
                                 [0.,  1.,  0.,  0.,  0., -1.,  0.],
                                 [0.,  1.,  0.,  0.,  0., -1.,  0.]])

    assert torch.all(output == ground_truth)


def test_corr2d_case2():
    # test padding
    input = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ])
    kernel = torch.tensor([
        [0, 1],
        [2, 3],
    ])
    ground_truth = torch.tensor([
        [0, 3, 8, 4],
        [9, 19, 25, 10],
        [21, 37, 43, 16],
        [6, 7, 8, 0],
    ])

    output = func.corr2d(input=input, padding=1,
                         kernel=kernel, stride=1)
    assert torch.all(output == ground_truth)


def test_corr2d_case3():
    input = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ])
    kernel = torch.tensor([
        [0, 1],
        [2, 3],
    ])
    ground_truth = torch.tensor([
        [0, 8],
        [6, 8],
    ])

    output = func.corr2d(input=input, padding=1, kernel=kernel, stride=(3, 2))
    assert torch.all(output == ground_truth)


def test_max_pool2d():
    input = torch.arange(start=0, end=9).reshape(3, 3)
    ground_truth = torch.tensor(data=[
        [4, 5],
        [7, 8],
    ])

    output = func.max_pool2d(input=input, kernel_size=2, padding=0, stride=1)
    assert torch.all(output == ground_truth)


def test_avg_pool2d():
    input = torch.arange(start=0, end=9).reshape(3, 3)
    ground_truth = torch.tensor(data=[
        [2, 3],
        [5, 6],
    ])

    output = func.avg_pool2d(input=input, kernel_size=2, padding=0, stride=1)
    assert torch.all(output == ground_truth)
