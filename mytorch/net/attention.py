# 2023/11/20
# zhangzhong

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from mytorch import config, func

# Attention


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        batch_size1, query_size, hidden_size1 = queries.shape
        batch_size2, key_size, hidden_size2 = keys.shape
        batch_size3, value_size, hidden_size3 = values.shape
        assert batch_size1 == batch_size2 == batch_size3
        assert key_size == value_size
        assert hidden_size1 == hidden_size2 == hidden_size3
        # 目前的实现里面，query_size是1
        assert query_size == 1

        # 接下来就是计算attention weight矩阵
        # shape = (b, query_size, key_size)
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # Returns a tensor that is a transposed version of input
        attention_weights = torch.bmm(queries, keys.transpose(1, 2))
        assert attention_weights.shape == (batch_size1, query_size, key_size)

        # 然后对attention_weight进行归一化
        # 也就是每个batch，或者说attention_weight的最后一个维度需要进行softmax归一化
        attention_weights = F.softmax(attention_weights, dim=-1)

        check_attention1 = torch.sum(attention_weights, dim=-1)
        # print(check_attention1)
        check_attention2 = torch.ones_like(
            check_attention1, dtype=torch.float32)
        # print(check_attention2)
        # # assert all(check_attention1 == check_attention2)
        # print(torch.abs(check_attention1 - check_attention2) < 1e-6)
        assert all(torch.abs(check_attention1 - check_attention2) < 1e-6)

        # 然后应用此attention_weight 对 values进行一个加权平均
        # 输出每一个batch对应的context_varialbe
        contexts = torch.bmm(attention_weights, values)
        return attention_weights, contexts

# TODO: 这个要改名字 应该叫 Bah...Encoder


class MaskedDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor = None) -> tuple[Tensor, Tensor]:
        batch_size1, query_size, hidden_size1 = queries.shape
        batch_size2, key_size, hidden_size2 = keys.shape
        batch_size3, value_size, hidden_size3 = values.shape
        assert batch_size1 == batch_size2 == batch_size3
        assert key_size == value_size
        assert hidden_size1 == hidden_size2 == hidden_size3
        # 目前的实现里面，query_size是1
        # 解开封印
        # assert query_size == 1

        # 接下来就是计算attention weight矩阵
        # shape = (b, query_size, key_size)
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # Returns a tensor that is a transposed version of input
        # BUG: FIX： bmm / sqrt(h) 老是忘了还要除以根号h 目的是修正方差
        attention_weights = torch.bmm(
            queries, keys.transpose(1, 2)) / math.sqrt(hidden_size1)
        assert attention_weights.shape == (batch_size1, query_size, key_size)
        if valid_lens is not None:
            # 但是有一些weight我们是不要的
            # 需要mask掉 其实就是赋予一个很大的负数 这样softmax就会变成零
            # 列向量
            # 我觉得这里不应该是assert而是先把valid_lensreshape一下
            # valid_lens = valid_lens.reshape(-1, 1)
            # assert valid_lens.shape == (batch_size1, 1)
            # # boardcast
            # # 我带着最新的研究回来了
            # repeat_valid_lens = valid_lens.repeat_interleave(
            #     query_size*key_size).reshape(-1, query_size, key_size)
            # assert repeat_valid_lens.shape == (
            #     batch_size1, query_size, key_size)
            repeat_valid_lens = self.make_repeat_valid_lens(
                valid_lens=valid_lens, batch_size=batch_size1, query_size=query_size, key_size=key_size)

            # 行向量
            mask = torch.arange(end=key_size, device=config.conf['device']).reshape(
                1, -1).repeat(query_size, 1)
            assert mask.shape == (query_size, key_size)

            mask = mask < repeat_valid_lens
            assert mask.shape == (batch_size1, query_size, key_size)
            self.mask = mask
            # print(mask)
            # attention_weights = attention_weights.squeeze()
            # 赋予一个巨大的负数，这样softmax之后对应的值就变成零
            attention_weights[~mask] = -1e6
            assert attention_weights.shape == (
                batch_size1, query_size, key_size)
            # print(attention_weights)
            # mask = mask.unsqueeze(dim=1)
            # assert mask.shape == (batch_size1, query_size, key_size)
            # attention_weights[~mask] = 1e-6
            # attention_weights = attention_weights.unsqueeze(dim=1)

        # 然后对attention_weight进行归一化
        # 也就是每个batch，或者说attention_weight的最后一个维度需要进行softmax归一化
        attention_weights = F.softmax(attention_weights, dim=-1)

        check_attention1 = torch.sum(attention_weights, dim=-1)
        # print(check_attention1)
        check_attention2 = torch.ones_like(
            check_attention1, dtype=torch.float32)
        # print(check_attention2)
        # # assert all(check_attention1 == check_attention2)
        # print(torch.abs(check_attention1 - check_attention2) < 1e-6)
        assert torch.all(torch.abs(check_attention1 - check_attention2) < 1e-6)

        # print(attention_weights)
        # if valid_lens is not None:
        #     # check valid lens
        #     for b in range(batch_size1):
        #         weights = attention_weights[b]
        #         assert weights.shape == (query_size, key_size)
        #         # weights = weights.squeeze()
        #         valid_lens = valid_lens.squeeze()
        #         valid_len = valid_lens[b]
        #         for l in range(key_size):
        #             if l >= valid_len:
        #                 assert torch.all(torch.abs(weights[:, l]) < 1e-6)

        # 然后应用此attention_weight 对 values进行一个加权平均
        # 输出每一个batch对应的context_varialbe
        contexts = torch.bmm(attention_weights, values)
        return attention_weights, contexts

    # 这里的逻辑操作太复杂了 所以必须拆成两个函数来处理
    # 实际上我发现只需要生成不同的valid_lens即可
    def make_repeat_valid_lens(self, valid_lens: Tensor, batch_size: int, query_size: int, key_size: int) -> Tensor:
        if len(valid_lens.shape) == 1 or valid_lens.shape[1] == 1:
            valid_lens = valid_lens.reshape(-1, 1).repeat_interleave(
                query_size*key_size).reshape(-1, query_size, key_size)
        elif len(valid_lens.shape) == 2:
            valid_lens = valid_lens.repeat_interleave(
                key_size, dim=-1).reshape(batch_size, query_size, key_size)
        else:
            assert False
        assert valid_lens.shape == (batch_size, query_size, key_size)
        return valid_lens


# TODO: MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head: int, hidden_size: int):
        super().__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        # 每一个head只拿hidden state的 hidden_size/num_head 个分量的向量

        # 有四个权重矩阵 分别对应 q k v 和最终合并的output
        # 先过权重矩阵，再分多头
        # 所以这里的权重矩阵的输出维度和我们的输入维度是一样的
        self.wq = nn.LazyLinear(out_features=hidden_size)
        self.wk = nn.LazyLinear(out_features=hidden_size)
        self.wv = nn.LazyLinear(out_features=hidden_size)
        # 结果所有的权重矩阵都是一样的...
        self.wo = nn.LazyLinear(out_features=hidden_size)
        self.attention = MaskedDotProductAttention()

    # 我们还要定义两个互逆的操作 就是分解多头和合并多头
    # 感觉这个可以写在func模块里面
    # def split_multi_head()

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens: Tensor) -> Tensor:
        # 首先 我们这里做的是self attention
        # 那么三个人的shape应该是一样的
        # 实际上这个测试是没法做的 因为在decoder里面qkv是不一样长的
        # assert queries.shape == keys.shape == values.shape
        batch_size, num_seq, hidden_size = queries.shape
        queries = func.split_multi_head(
            input=self.wq(queries), num_head=self.num_head)
        keys = func.split_multi_head(
            input=self.wk(keys), num_head=self.num_head)
        values = func.split_multi_head(
            input=self.wv(values), num_head=self.num_head)
        # 完成head的split之后 三个人的shape应该还是一样的
        # assert queries.shape == keys.shape == values.shape
        assert queries.shape == (
            batch_size*self.num_head, num_seq, int(hidden_size/self.num_head))

        batch_size1, *_ = valid_lens.shape
        assert batch_size == batch_size1

        # 接下来就是最关键的一步 体现你对multihead是否真正的理解
        # 之前我们的valid_lens对应的每个batch 也就是每个seq
        # 但是现在我们的q k v的第一个维度都变成了 batch_size*num_head
        # 所以我们的valid_lens实际上也要拉长
        #      q k v.shape = (batch_size*num_head, ...)
        # valid_lens.shape = (batch_size, ...)
        # 我们要让向量沿着第一个维度的方向重复 num_head次
        # 这样因为我们的所有head的seq其实来自于原始的同一个seq 他们的valid lens当然都是一样的
        # 在encoder的时候就出错啦？？
        # TODO：这里需要考虑两种情况 对应两种validlens
        valid_lens = valid_lens.repeat_interleave(self.num_head, dim=0)
        assert valid_lens.shape[0] == batch_size*self.num_head

        # attention前后shape不应该发生变化
        # 我操了 我们的attention会返回两个东西 这得多少错误卧槽
        _, attention_output = self.attention(
            queries=queries, keys=keys, values=values, valid_lens=valid_lens)
        assert attention_output.shape == (batch_size*self.num_head,
                                          num_seq, int(hidden_size/self.num_head))

        # 然后我们进行concat
        multihead_attention_output = func.concat_multi_head(
            attention_output, self.num_head)
        assert multihead_attention_output.shape == (
            batch_size, num_seq, hidden_size)

        # 然后再过一个output矩阵
        output = self.wo(multihead_attention_output)
        # 整个MultiHeadAttention都不应该改变数据的shape
        assert output.shape == (batch_size, num_seq, hidden_size)
        return output
