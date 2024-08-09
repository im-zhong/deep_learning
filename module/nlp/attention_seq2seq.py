# 2023/11/20
# zhangzhong

import torch
from torch import Tensor, nn

from mytorch.data.seq import VocabularyV2
from mytorch.net.attention import MaskedDotProductAttention

# TODO: 把这些net的子模块都隐藏起来，这样我就可以像这样import
# from mytorch.net import MyDeepGRU, MyEmbedding, ...
# from mytorch.data import DatasetA, DatasetB, ...
from mytorch.net.rnn import MyDeepGRU
from mytorch.net.seq2seq import MyEmbedding


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        vocab: VocabularyV2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.embedding = MyEmbedding(vocab_size, embed_size)
        self.rnn = MyDeepGRU(
            input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers
        )

    # TODO:
    # 其实encoder几乎没有变化
    # 应用了快速原型的思想
    # 为了实现上的简单，我们可以暂时不考虑masked_attention
    def forward(self, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # source.shape: batch_size, num_seq
        batch_size, num_seq = source.shape
        valid_lens = torch.zeros(size=(batch_size, 1), device=source.device)
        # 遍历source 构造一个valid_lens
        for i in range(batch_size):
            seq = source[i]
            valid_len = num_seq
            for j in range(num_seq):
                if seq[j] == self.vocab.pad():
                    valid_len = j
            valid_lens[i][0] = valid_len

        embed = self.embedding(source)
        assert embed.shape == (num_seq, batch_size, self.embed_size)
        output, state = self.rnn(embed, None)
        assert output.shape == (num_seq, batch_size, self.hidden_size)
        assert state.shape == (self.num_layers, batch_size, self.hidden_size)
        return output, state, valid_lens


class AttentionDecoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = MyEmbedding(vocab_size, embed_size)
        # self.attention = DotProductAttention()
        self.attention = MaskedDotProductAttention()
        # 注意这里decoder的输入是embed + context variable
        # 而且attention版本的contxext variable是变化的 每次迭代都需要去attention
        self.rnn = MyDeepGRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    # TODO: to remove this
    def clear_history(self):
        pass

    def forward(
        self,
        target: Tensor,
        encoder_output: Tensor,
        encoder_state: Tensor,
        valid_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_seqtgt = target.shape
        srcseq_size, batch_size, hidden_size = encoder_output.shape
        num_layers, batch_size, hidden_size = encoder_state.shape
        # 最开始的query是encoder_output的
        # context = self.attention()
        queries = encoder_output.transpose(0, 1)[:, -1, :]
        queries = queries.unsqueeze(1)
        assert queries.shape == (batch_size, 1, hidden_size)
        keys = encoder_output.transpose(0, 1)
        values = encoder_output.transpose(0, 1)

        # 1. embedding
        embed = self.embedding(target)
        assert embed.shape == (num_seqtgt, batch_size, self.embed_size)
        # 这里就和之前的encoder不一样了
        # 之前我们拿到context之后，会把他和seq里面的所有向量连接起来
        # 但是在这里我们每次迭代 都需要连接新的context

        # 2. rnn
        # 所以这里的rnn循环需要自己写了
        # 最开始decoder_state = encoder_state
        decoder_state = encoder_state
        outputs: list[Tensor] = []
        for xt in embed:
            # 每个token都需要重新计算context
            _, contexts = self.attention(
                queries=queries, keys=keys, values=values, valid_lens=valid_lens
            )
            assert contexts.shape == (batch_size, 1, hidden_size)
            contexts = contexts.squeeze(1)
            assert contexts.shape == (batch_size, hidden_size)

            assert xt.shape == (batch_size, self.embed_size)
            # xt concat context
            xt = torch.concat([xt, contexts], dim=-1)
            assert xt.shape == (batch_size, self.embed_size + hidden_size)
            # 但是rnn只接受 (num_seq, batch_size, input_size)这样类型的输入
            # 所以我们需要给xt在第一个维度上添加1

            decoder_output, decoder_state = self.rnn(xt.unsqueeze(0), decoder_state)
            # decoder_output需要不断累积起来
            # 最后一个stack就可以了 实现起来和rnn内部的实现差不多
            assert decoder_output.shape == (1, batch_size, hidden_size)
            outputs.append(decoder_output.squeeze(0))
            # 然后这里需要更新context
            # 之后的每一次的query 其实就是之前的decoder的输出

            queries = decoder_output.transpose(0, 1)
            assert queries.shape == (batch_size, 1, hidden_size)

        decoder_output = torch.stack(outputs)
        assert decoder_output.shape == (num_seqtgt, batch_size, hidden_size)
        assert decoder_state.shape == (num_layers, batch_size, hidden_size)

        # 3. output layer
        outputs.clear()
        for state in decoder_output:
            output = self.output_layer(state)
            assert output.shape == (batch_size, self.vocab_size)
            outputs.append(output)
        assert len(outputs) == num_seqtgt

        outputs_tensor = torch.stack(outputs, dim=1)
        assert outputs_tensor.shape == (batch_size, num_seqtgt, self.vocab_size)
        # outputs要和label做cross entropy
        # 我们需要确认label的shape = (batch_size, num_seq)
        # 完全正确
        return outputs_tensor, decoder_state
