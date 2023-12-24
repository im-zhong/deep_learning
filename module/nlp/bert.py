# 2023/12/21
# zhangzhong
# Pretrain BERT

import torch
from torch import nn, Tensor
from mytorch.data.wikitext2v2 import WikiText2, WikiText2Sample, WikiText2Label
from collections import OrderedDict
from mytorch.net.transformer import AddNorm, FeedForwardNetwork
from dataclasses import dataclass


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

    # 其实这样也不好啊， 因为我们传入了很多本模块用不到的数据
    # 因为目前我们传入的输入是一个2个元素的tuple 还可以接受
    def forward(self, sentences: Tensor, segments: Tensor) -> Tensor:
        # check shape
        # result = token + segment + positional
        # sentences.shape = [batch_size, seq_size]
        # sentences = x.sentences
        # segments.shape = [batch_size, seq_size]
        # segments = x.segments
        # valid_lens.shape = [batch_size]
        # valid_lens = x.valid_lens
        # mask = x.mask
        batch_size, seq_size = sentences.shape
        # 卧槽 写成[]是不行的 必须写成()
        assert segments.shape == (batch_size, seq_size)
        # assert mask.shape == (batch_size, seq_size)

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
        embedding: Tensor = token_embedding + segment_embedding + \
            self.positional_embedding[:, :seq_size, :]
        assert embedding.shape == (batch_size, seq_size, self.hidden_size)
        return embedding

# 还是根据我们的只是重新实现一个TransformerEncoderBlock吧
# 这次需要提供mask而不是validlens


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_head, batch_first=True, dropout=0.2)
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
        return self.net(x)[0]


class BERTEncoderV2(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int, num_blocks: int):
        # 不对，不能直接用Encoder，因为我们有自己的Embedding
        # 那就用一系列的Block就行了
        # Dictionary that remembers insertion order
        super().__init__()
        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_head,
                                                   dim_feedforward=ffn_hidden_size,
                                                   dropout=0.1, batch_first=True)
        # for i in range(num_blocks):
        #     blocks[f'block{i}'] = nn.TransformerEncoder(
        #         num_head=num_head, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        # self.net = nn.Sequential(blocks)
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_blocks)

    def forward(self, x: tuple[Tensor, Tensor]):
        input, mask = x
        return self.encoder(input, src_key_padding_mask=mask)


class NextSentencePrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(2)
        )
        # self.linear = nn.LazyLinear(out_features=2)

    def forward(self, encoder_output: Tensor) -> Tensor:
        # outputs, _ = x
        batch_size, seq_size, hidden_size = encoder_output.shape
        # get <cls>
        clss = encoder_output[:, 0, :].flatten(start_dim=1)
        assert clss.shape == (batch_size, hidden_size)
        return self.net(clss)


class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            # 或许应该用LayerNorm吧 在NLP问题上
            # 但是LayzerNorm应该用在fc层前面还是后面？
            nn.LayerNorm(normalized_shape=1024),
            nn.ReLU(),
            nn.LazyLinear(vocab_size)
        )
        # self.linear = nn.LazyLinear(out_features=2)

    def forward(self, encoder_output: Tensor, masked_indices: Tensor) -> Tensor:
        batch_size, seq_size, hidden_size = encoder_output.shape
        batch_size, masked_token_size = masked_indices.shape
        # 接下来我们要把mask位置的hidden vector选出来
        index0 = torch.arange(batch_size).repeat_interleave(
            masked_token_size)
        masked_tokens = encoder_output[index0, masked_indices.flatten()]
        assert masked_tokens.shape[1] == hidden_size
        return self.net(masked_tokens)


@dataclass
class BERTOutput:
    nsp: Tensor
    mlm: Tensor
    mlm_mask: Tensor


class BERTLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.unreduced_loss = nn.CrossEntropyLoss(reduce=None)

    def __call__(self, bert_output: BERTOutput, labels: WikiText2Label) -> Tensor:
        # nsp.shape = (batch_size, 2)
        nsp_loss = self.loss(bert_output.nsp, labels.nsp)
        # TODO: mlm loss
        # mlm.shape  = (batch_size*n, n, vocab_size)
        mlm_output = bert_output.mlm.reshape(-1, bert_output.mlm.shape[-1])
        mlm_labels = labels.mlm.flatten()
        mlm_mask = bert_output.mlm_mask.flatten()
        mlm_loss = self.unreduced_loss(mlm_output, mlm_labels) * mlm_mask
        return nsp_loss + mlm_loss.mean()


class NextSentencePredictionV2(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.LayerNorm(normalized_shape=1024),
            nn.ReLU(),
            # 这里的输出大小好像是不固定的？不对 是vocab_size
            nn.LazyLinear(vocab_size)
        )
        # self.linear = nn.LazyLinear(out_features=2)

    def forward(self, x: Tensor):
        outputs = x
        batch_size, seq_size, hidden_size = outputs.shape
        # get <cls>
        clss = outputs[:, 0, :].flatten(start_dim=1)
        assert clss.shape == (batch_size, hidden_size)
        return self.net(clss)

# TODO: MaskedLanguageModel


class BERT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int,
                 num_head: int, ffn_hidden_size: int, num_blocks: int, use_pytorch_encoder: bool = True):
        super().__init__()
        # 1. Embedding
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, hidden_size=hidden_size, max_len=max_len)
        # 2. BERTEncoder
        if use_pytorch_encoder:
            self.encoder = BERTEncoderV2(num_head=num_head, hidden_size=hidden_size,
                                         ffn_hidden_size=ffn_hidden_size, num_blocks=num_blocks)
        else:
            self.encoder = BERTEncoder(num_head=num_head, hidden_size=hidden_size,  # type: ignore
                                       ffn_hidden_size=ffn_hidden_size, num_blocks=num_blocks)
        # 3. NextSentencePrediction + MaskedLanguageModel
        self.nsp = NextSentencePrediction()
        # self.net = nn.Sequential(OrderedDict([
        #     ('embedding', embedding),
        #     ('encoder', encoder),
        #     ('nsp', nsp)
        # ]))
        # 4. MaskedLanguageModel
        self.mlm = MaskedLanguageModel(vocab_size=vocab_size)

    def forward(self, sample: WikiText2Sample) -> BERTOutput:
        # 主要就是这个地方的代码会变得复杂一些
        # 但是也没有很复杂 但是其他地方的代码变得更易读 更合理
        # return self.net(x)
        # 1. embedding
        embedding = self.embedding(sentences=sample.sentences,
                                   segments=sample.segments)
        # 2. encoder
        encoder_output = self.encoder(x=(embedding, sample.padding_mask))
        # 3. nsp
        nsp_output = self.nsp(encoder_output=encoder_output)
        # 4. mlm
        mlm_output = self.mlm(
            encoder_output=encoder_output, masked_indices=sample.masked_indices)
        return BERTOutput(nsp=nsp_output, mlm=mlm_output, mlm_mask=sample.mlm_mask)
