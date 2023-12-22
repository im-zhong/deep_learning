# 2023/12/21
# zhangzhong
# Pretrain BERT

import torch
from torch import nn, Tensor
from mytorch.data.wikitext2 import WikiText2, WikiText2Example
from collections import OrderedDict
from mytorch.net.transformer import AddNorm, FeedForwardNetwork


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_len = max_len

        # 1. token embedding
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=hidden_size)
        # 2. segment embedding
        self.segment_embedding = nn.Embedding(num_embeddings=2,
                                              embedding_dim=hidden_size)
        # 3. positional embedding
        self.positional_embedding = nn.Parameter(
            data=torch.randn(size=(1, max_len, hidden_size)))

    def forward(self, x: WikiText2Example) -> tuple[Tensor, Tensor]:
        # check shape
        # result = token + segment + positional
        # sentences.shape = [batch_size, seq_size]
        sentences = x.sentences
        # segments.shape = [batch_size, seq_size]
        segments = x.segments
        # valid_lens.shape = [batch_size]
        # valid_lens = x.valid_lens
        mask = x.mask
        batch_size, seq_size = sentences.shape
        # 卧槽 写成[]是不行的 必须写成()
        assert segments.shape == (batch_size, seq_size)
        assert mask.shape == (batch_size, seq_size)

        # token embedding
        token_embedding: Tensor = self.token_embedding(sentences)
        assert token_embedding.shape == (
            batch_size, seq_size, self.hidden_size)

        # segment embedding
        segment_embedding: Tensor = self.segment_embedding(segments)
        assert segment_embedding.shape == (
            batch_size, seq_size, self.hidden_size)

        # positional embedding
        assert seq_size < self.max_len
        result: Tensor = token_embedding + segment_embedding + \
            self.positional_embedding[:, :seq_size, :]
        assert result.shape == (batch_size, seq_size, self.hidden_size)
        return result, mask

# 还是根据我们的只是重新实现一个TransformerEncoderBlock吧
# 这次需要提供mask而不是validlens


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_head, batch_first=True)
        # 因为LayerNorm里面有参数 所以必须有两个不同的add norm
        self.add_norm1 = AddNorm(feature_size=hidden_size)
        self.ffn = FeedForwardNetwork(
            hidden_size=ffn_hidden_size, output_size=hidden_size)
        self.add_norm2 = AddNorm(feature_size=hidden_size)

    # TODO: 目前Embedding的输出和Encoder的输入还没有对齐
    def forward(self, x: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        input, mask = x
        # 1. self multihead attention
        # https://www.zhihu.com/question/455164736
        output, weights = self.attention(
            query=input, key=input, value=input, key_padding_mask=mask,
            need_weights=True, average_attn_weights=True)
        # 2. add norm
        x = self.add_norm1(input=input, output=output)

        # 3. ffn
        output = self.ffn(x)

        # 4. add norm
        return self.add_norm2(input=x, output=output), mask


class BERTEncoder(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int, num_blocks: int):
        # 不对，不能直接用Encoder，因为我们有自己的Embedding
        # 那就用一系列的Block就行了
        # Dictionary that remembers insertion order
        super().__init__()
        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_blocks):
            blocks[f'block{i}'] = TransformerEncoderBlock(
                num_head=num_head, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        self.net = nn.Sequential(blocks)

    def forward(self, x: tuple[Tensor, Tensor]):
        return self.net(x)


class NextSentencePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(out_features=2)

    def forward(self, x: tuple[Tensor, Tensor]):
        outputs, _ = x
        batch_size, seq_size, hidden_size = outputs.shape
        # get <cls>
        clss = outputs[:, 0, :].flatten(start_dim=1)
        assert clss.shape == (batch_size, hidden_size)
        return self.linear(clss)

# TODO: MaskedLanguageModel


class BERT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int,
                 num_head: int, ffn_hidden_size: int, num_blocks: int):
        super().__init__()
        # 1. Embedding
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, hidden_size=hidden_size, max_len=max_len)
        # 2. BERTEncoder
        self.encoder = BERTEncoder(num_head=num_head, hidden_size=hidden_size,
                                   ffn_hidden_size=ffn_hidden_size, num_blocks=num_blocks)
        # 3. NextSentencePrediction + MaskedLanguageModel
        self.nsp = NextSentencePrediction()
        self.net = nn.Sequential(OrderedDict([
            ('embedding', self.embedding),
            ('encoder', self.encoder),
            ('nsp', self.nsp)
        ]))

    def forward(self, x: WikiText2Example):
        return self.net(x)
