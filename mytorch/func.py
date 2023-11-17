# 2023/10/15
# zhangzhong
# functional

import torch
# https://pytorch.org/docs/stable/tensors.html
# LongTensor: dtype = 64bit signed integer
from torch import Tensor
# from torch import LongTensor
from typing import Callable
from . import config

# Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes)
# that have zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor,
# in which case it will be 1.
# https://en.wikipedia.org/wiki/One-hot


# BUG: flatten([[1]]) != [1]
def flatten(input: Tensor) -> Tensor:
    # input.reshape(shape=(-1, 1))[:, 0]
    # return input.reshape(shape=(-1, 1)).squeeze()
    return input.reshape(shape=(-1, 1))[:, 0]


def one_hot(input: Tensor, num_classes: int = -1) -> Tensor:
    # input could be any shape
    # num_classes: total number of classes
    # num_classes = int(torch.max(input))
    # 我们需要保证Tensor的数据类型是整形
    assert not torch.is_floating_point(
        input=input) and not torch.is_complex(input=input)

    shape = input.shape
    num_classes = num_classes if num_classes != -1 else int(torch.max(input))+1
    result = torch.zeros(size=(*input.shape, num_classes), dtype=torch.long, device=input.device
                         ).reshape(shape=(-1, num_classes))
    # 我们遍历result的最后一个维度
    # flatten input
    # input = input.reshape(shape=(-1, 1)).squeeze()
    input = flatten(input)
    # 只考虑result的最后一个维度，因为result被初始化为零向量
    # 我们只需要将最后一个维度的对应位置置为1即可表示对应的one hot vector
    # result[:, input] = 1
    # for i in input:
    #     # 不对呀 这样确实会选择这一行的所有元素
    #     # 我只是想遍历所有的行 我该怎么办呢
    #     result[:, i] = 1
    # 想遍历所有的行，直接arange
    result[torch.arange(len(input)), input] = 1
    return result.reshape(shape=(*shape, num_classes))


def embedding(input: Tensor, weight: Tensor) -> Tensor:
    vocab_size, embed_size = weight.shape
    # shape = input.shape
    # result.shape = shape + (embed_size)
    # BUG: one_hot's result will back to cpu, why?
    result = one_hot(
        input=input, num_classes=vocab_size).float().reshape(-1, vocab_size)
    result = result @ weight
    return result.reshape(shape=(*input.shape, embed_size))

# https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
# 这个函数应该只负责batch norm的计算而已


def batch_norm(input: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5, momentum: float = 0.1) -> Tensor:
    # 目前我们只处理二维的数据
    assert len(input.shape) == 2
    batch_size, feature_size = input.shape
    # mean和sigma的计算方法有两种
    # 1. 在训练时，使用一个batch的数据进行计算
    # 2. 在eval时，使用running average计算，也就是统计所有的输入，滚动计算x = (1-m)x + mx' x'表示新的输入，也就是新数据只占0.1
    mean = input.mean(dim=0)
    # sigma = input.norm(p='2', dim=0)
    # https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    sigma = torch.linalg.vector_norm(input, ord=2, dim=0)
    assert mean.shape == (feature_size,)
    assert sigma.shape == (feature_size,)

    norm = (input - mean) / (sigma + eps)
    assert norm.shape == input.shape
    assert gamma.shape == (feature_size,)
    assert beta.shape == (feature_size,)
    return norm * gamma + beta

# 我还是不理解layer norm 为什么要对一个向量的各个分量求均值和方差
# https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

# γ and β are learnable affine transform parameters of normalized_shape
# 目前我们只处理nomalized_shape是一个维度的 也就是一个向量

# 虽然和pytorch的数值还是有些不同 不过已经非常接近了


def layer_norm(input: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    # layernorm对input的形状不做假设 他就是对最后一个维度的向量的所有分量之间做标准化
    batch_size, seq_size, normalized_shape = input.shape
    shape = input.shape
    input = torch.reshape(input, shape=(-1, normalized_shape))
    mean = input.mean(dim=-1, keepdim=True)
    # sigma = input(dim=-1)
    # sigma = torch.linalg.vector_norm(input, ord=2, dim=-1)
    var = input.var(dim=-1, keepdim=True)
    assert mean.shape == (batch_size * seq_size, 1)
    assert var.shape == (batch_size * seq_size, 1)

    input = (input - mean) / torch.sqrt(var + eps)
    assert gamma.shape == (normalized_shape,)
    assert beta.shape == (normalized_shape,)
    input = input * gamma + beta
    # input.shape 应该保持不变
    input = input.reshape(shape)
    return input


def split_multi_head(input: Tensor, num_head: int) -> Tensor:
    # 数据的内容并没有发生变化，只是数据的形式发生了变化
    batch_size, num_seq, hidden_size = input.shape
    # num_head必须可以整除hidden_size
    assert hidden_size % num_head == 0
    head_hidden_size = int(hidden_size / num_head)
    # 1. 首先将每一个hidden vecotr切分成num_head块，每块的长度为 head_hidden_size
    # BUG: RuntimeError: only one dimension can be inferred
    # x = input.reshape(shape=(-1, -1, num_head, head_hidden_size))
    x = input.reshape(shape=(batch_size, num_seq, num_head, head_hidden_size))
    assert x.shape == (batch_size, num_seq, num_head, head_hidden_size)

    # 2. 交换num_seq和num_head对应的维度 这样后面两个维度就和之前的一个普通的seq所对应的矩阵没有区别
    # 而前两个维度合起来可以看作是更过的batch 更多的seq
    y = x.permute(0, 2, 1, 3)
    assert y.shape == (batch_size, num_head, num_seq, head_hidden_size)

    # 3. 然后将batch_size和num_head进行合并
    z = y.reshape(-1, num_seq, head_hidden_size)
    assert z.shape == (batch_size*num_head, num_seq, head_hidden_size)
    return z


# 这里有一个改进，可以把batch_size去掉
# 因为input的第一个维度是 batch_size * num_head
# 所以只要给定num_head 我们就可以根据input的第一个维度计算出batch_size
# def concat_multi_head(input: Tensor, batch_size, num_head: int) -> Tensor:
def concat_multi_head(input: Tensor, num_head: int) -> Tensor:
    # 这个函数完全是split的逆操作
    # 不过得先split再concat x == concat(split(x))
    # 首先将第一个维度分离呗
    batch_size_times_num_head, num_seq, head_hidden_size = input.shape
    # assert batch_size * num_head == batch_size_times_num_head
    assert batch_size_times_num_head % num_head == 0
    batch_size = int(batch_size_times_num_head / num_head)
    x = input.reshape(batch_size, num_head, num_seq, head_hidden_size)

    # 然后交换 1 2 维度
    y = x.permute(0, 2, 1, 3)
    assert y.shape == (batch_size, num_seq, num_head, head_hidden_size)

    # 然后合并后两个维度
    z = y.reshape(batch_size, num_seq, num_head * head_hidden_size)
    assert z.shape == (batch_size, num_seq, num_head * head_hidden_size)

    return z


def make_source_valid_lens(source: Tensor, pad: int):
    batch_size, num_seq = source.shape
    # 为了和我们的代码不冲突 我们必须返回一个一维的向量
    valid_lens = torch.zeros(size=[batch_size], device=source.device)
    # 遍历source 构造一个valid_lens
    for i in range(batch_size):
        seq = source[i]
        valid_len = num_seq
        for j in range(num_seq):
            if seq[j] == pad:
                valid_len = j
                break
        valid_lens[i] = valid_len
    return valid_lens


def make_target_valid_lens(batch_size: int, max_len: int, device: str = 'cpu') -> Tensor:
    # target的self attention的valid lens跟encoder是不同的
    # 在encoder做训练时 我们希望位于 xt 出的token 只能attend到 xt之前的token
    # 所以这里生成的valid_lens就是一个二维的矩阵
    # target_valid_lens.shape = (batch_size, seq_len)
    # 而且不同的target输入的seq_len是不同的 所以这个东西还需要动态生成
    # TIP: valid_lens的类型应该是int
    lens = torch.arange(start=1, end=max_len+1,
                        dtype=torch.int, device=torch.device(device))
    # lens是行向量
    # 然后纵向扩展到batch_size
    target_valid_lens = lens.repeat(
        batch_size, 1)
    assert target_valid_lens.shape == (batch_size, max_len)
    return target_valid_lens


def make_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        return x
    else:
        assert False


def add_padding(input: Tensor, padding: int | tuple[int, int]) -> Tensor:
    # padding要怎么算呢 是两边都算 还是只算一边
    # 长宽都一样吗？
    h, w = input.shape
    # if isinstance(padding, int):
    #     print('int')
    #     output = torch.zeros(size=(h+padding*2, w+padding*2))
    #     # 然后选取input上对应位置的窗口
    #     output[padding:padding+h, padding:padding+w] = input
    # elif isinstance(padding, tuple):
    #     ph, pw = padding
    #     # 我们只考虑奇数吧 不然实现起来不好看也没有意义
    #     print('tuple')

    #     output = torch.zeros(size=(h+ph*2, w + pw*2))
    #     output[ph:ph+h, pw:pw+w] = input
    # else:
    #     assert False

    padding = make_tuple(padding)
    ph, pw = padding

    output = torch.zeros(size=(h+ph*2, w + pw*2), device=input.device)
    output[ph:ph+h, pw:pw+w] = input
    return output


def corr2d_impl(input: Tensor, kernel_size: int | tuple[int, int], op: Callable[[Tensor], Tensor], padding: int | tuple[int, int] = 0, stride: int | tuple[int, int] = 1) -> Tensor:
    assert len(input.shape) == 2
    # assert len(kernel.shape) == 2
    # 现在去掉这个假设 我们的代码需要作出什么改动？
    # 假设kernel的长和宽都是奇数的话 实现回简单很多
    # assert kernel.shape[0] == kernel.shape[1]
    # assert kernel.shape[0] % 2 == 1
    # BUG: add_padding make input from cuda to cpu!
    input = add_padding(input=input, padding=padding)

    # 不对 我们的输出图像也是一个全新的tensor
    # 而且大小是小一号的
    # 输出图像的新的长宽是这样计算的 h - (kh - 1), w - (kw - 1)
    # h = input.shape[0]
    # w = input.shape[1]
    h, w = input.shape
    kh, kw = make_tuple(kernel_size)
    sh, sw = make_tuple(stride)

    oh, ow = (h - kh)//sh + 1, (w - kw)//sw + 1
    output = torch.zeros(size=(oh, ow), device=input.device)

    # 现在就是确定pad
    # 对奇数来讲 就是 / 2
    # 但是如果是偶数 比如kw = 2
    # 我们也可以先/2试一试
    # pad = kernel / 2
    # row_pad = kh / 2
    # col_pad = kw / 2

    # for row in range(pad, input.shape[0]-pad):
    #     for col in range(pad, input.shape[1]-pad):
    #         # use slice to pick the window
    #         window =
    # 那这样我们就直接遍历output就行了 然后根据当前的row col
    # 在input上选取合适的窗口window即可
    for row in range(oh):
        for col in range(ow):
            # 想象input和output叠在一起
            # intput比output大一圈 周围都可以看作由kernel引起的padding
            # 所以output中的坐标(row, col)在input上对应于(row + pad, col + pad)

            # 在input上 以(row+pad, col+pad)为中心，选取变长为kernel_size
            # 刚好窗口的起始位置是row+pad-pad=row 即(row, pad) on input
            # perfect!
            window = input[row*sh:row*sh+kh, col*sw:col*sw+kw]
            output[row, col] = op(window)

    return output


def corr2d(input: Tensor, kernel: Tensor, padding: int | tuple[int, int] = 0, stride: int | tuple[int, int] = 1) -> Tensor:
    h, w = input.shape
    assert len(kernel.shape) == 2
    kh, kw = kernel.shape
    return corr2d_impl(input, op=lambda window: (kernel * window).sum(), kernel_size=(kh, kw), padding=padding, stride=stride)

# https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html


# pool和corr一样 他们期望输入是二维的
# 但是实际上的输入是四维的 所以我们需要做和Conv2d一样的处理
def avg_pool2d(input: Tensor, kernel_size: int | tuple[int, int], padding: int | tuple[int, int] = 0, stride: int | tuple[int, int] = 1) -> Tensor:
    return corr2d_impl(input, op=torch.mean, kernel_size=kernel_size, padding=padding, stride=stride)


def max_pool2d(input: Tensor, kernel_size: int | tuple[int, int], padding: int | tuple[int, int] = 0, stride: int | tuple[int, int] = 1) -> Tensor:
    return corr2d_impl(input, op=torch.max, kernel_size=kernel_size, padding=padding, stride=stride)
