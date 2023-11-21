# 2023/11/20
# zhangzhong

import torch
from mytorch.net.attention import DotProductAttention, MaskedDotProductAttention
import matplotlib.pyplot as plt
from mytorch import config, func
import math
from mytorch import utils


def test_DotProductAttention():
    batch_size = 4
    hidden_size = 16
    seq_size = 8
    queries = torch.randn(size=(batch_size, 1, hidden_size))
    keys = torch.randn(size=(batch_size, seq_size, hidden_size))
    values = keys.clone()
    attention = DotProductAttention()
    attention_weights, contexts = attention(
        queries=queries, keys=keys, values=values)
    # plt.plot(attention_weights)
    plt.imshow(attention_weights[0])
    # plt.legend()
    # plt.tight_layout()
    plt.colorbar()
    # plt.savefig('attention.png')
    utils.mysavefig('attention.png')


def test_MaskedDotProductAttention():
    batch_size = 4
    hidden_size = 16
    seq_size = 8
    queries = torch.randn(size=(batch_size, 1, hidden_size))
    keys = torch.randn(size=(batch_size, seq_size, hidden_size))
    values = keys.clone()
    # valid_lens = torch.rand(size=(batch_size, 1)) * seq_size
    valid_lens = torch.tensor(
        [1, 2, 3, 4], device=config.conf['device']).reshape(shape=(-1, 1))
    attention = MaskedDotProductAttention()
    attention_weights, contexts = attention(
        queries=queries, keys=keys, values=values, valid_lens=valid_lens)
    # plt.plot(attention_weights)
    plt.imshow(attention_weights[0])
    # plt.legend()
    # plt.tight_layout()
    plt.colorbar()
    # plt.savefig('attention.png')
    utils.mysavefig('attention.png')


# TODO: 我们不能在每个tensor的创建的时候都指定device 我们最终是需要把config文件去掉的
def test_MaskedDotProductAttention_2():
    # 测试的思路
    # 给定几个qkv矩阵 最好都是1 这样算出来的attention比较好识别
    # 然后给定不同的valid lens 其实主要就是测试这个valid lens
    # 然后看生成的attention weight是不是对的
    attention = MaskedDotProductAttention()
    batch_size = 4
    query_size = 16
    kv_size = 16
    hidden_size = 8
    queries = torch.ones(size=(batch_size, query_size,
                         hidden_size),  device=config.conf['device'])
    keys = torch.ones(size=(batch_size, kv_size, hidden_size),
                      device=config.conf['device'])
    values = torch.ones(size=(batch_size, kv_size, hidden_size),
                        device=config.conf['device'])

    # case 1: valid_len is None
    valid_lens1 = None
    qk, _ = attention(queries=queries, keys=keys,
                      values=values, valid_lens=valid_lens1)
    assert qk.shape == (batch_size, query_size, kv_size)
    qk = qk.cpu()
    plt.imshow(qk[0])
    plt.colorbar()
    # plt.savefig('none.png')
    utils.mysavefig('none.png')

    # case 2: len(valid_len.shape) = 1
    valid_lens2 = torch.arange(
        start=5, end=batch_size+5, device=config.conf['device'])
    qk, _ = attention(queries=queries, keys=keys,
                      values=values, valid_lens=valid_lens2)
    assert qk.shape == (batch_size, query_size, kv_size)
    qk = qk.cpu()
    plt.imshow(qk[3])
    plt.colorbar()
    # plt.savefig('source_attention.png')
    utils.mysavefig('source_attention.png')

    # 感觉应该是对的 但是还是要和ground truth做对比
    X = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(hidden_size)
    qk = func.masked_softmax(X, valid_lens=valid_lens2)
    qk = qk.cpu()
    plt.imshow(qk[3])
    plt.colorbar()
    # plt.savefig('gt_source_attention.png')
    utils.mysavefig('gt_source_attention.png')

    # case 3. len(valid_len.shape) == 2
    valid_lens3 = torch.arange(
        start=1, end=query_size+1, device=config.conf['device']).reshape(1, -1).repeat(batch_size, 1)
    qk, _ = attention(queries=queries, keys=keys,
                      values=values, valid_lens=valid_lens3)
    assert qk.shape == (batch_size, query_size, kv_size)
    qk = qk.cpu()
    plt.imshow(qk[3])
    plt.colorbar()
    # plt.savefig('target_attention.png')
    utils.mysavefig('target_attention.png')

    my_mask = attention.mask.reshape(-1, kv_size)

    valid_lens3 = torch.arange(
        start=1, end=query_size+1, device=config.conf['device']).reshape(1, -1).repeat(batch_size, 1)
    # 卧槽 是我的问题！！！
    # mask_softmax会修改X 所以必须要重新计算一次!!!
    # 现在对了！！！
    X = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(hidden_size)
    qk = func.masked_softmax(X, valid_lens=valid_lens3, my_mask=my_mask)
    qk = qk.cpu()
    plt.imshow(qk[3])
    plt.colorbar()
    # plt.savefig('gt_target_attention.png')
    utils.mysavefig('gt_target_attention.png')
