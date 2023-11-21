# 2023/11/20
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch.net.norm import AddNorm
from mytorch.net.attention import MultiHeadAttention
from mytorch.net.encoding import PositionalEncoding
from mytorch.net.seq2seq import MyEmbedding
from mytorch.data.seq import VocabularyV2
from mytorch import config, func
import math


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        # 哎 不对呀 不是定义一个net的变量 直接自动生成forward吗
        self.net = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            nn.ReLU(),
            nn.LazyLinear(out_features=output_size),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.attention = MultiHeadAttention(
            num_head=num_head, hidden_size=hidden_size)
        # 因为LayerNorm里面有参数 所以必须有两个不同的add norm
        self.add_norm1 = AddNorm(feature_size=hidden_size)
        self.ffn = FeedForwardNetwork(
            hidden_size=ffn_hidden_size, output_size=hidden_size)
        self.add_norm2 = AddNorm(feature_size=hidden_size)

    def forward(self, input: Tensor, valid_lens: Tensor) -> Tensor:
        # 我们的输入是embeding的输出 或者是其他encoderblock的输出
        # 所以形状是
        batch_size, num_seq, hidden_size = input.shape
        # 1. 直接对input做multi head self attention
        output: Tensor = self.attention(queries=input, keys=input,
                                        values=input, valid_lens=valid_lens)
        assert output.shape == input.shape

        # 2. add norm
        x = self.add_norm1(input=input, output=output)
        assert x.shape == input.shape

        # 3. 过ffn
        input = x
        output = self.ffn(input)
        assert output.shape == input.shape

        # 4. add norm
        return self.add_norm2(input=input, output=output)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, ffn_hidden_size, num_head: int, num_blocks: int, vocab: VocabularyV2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = MyEmbedding(
            vocab_size=vocab_size, embed_size=hidden_size, transpose=False)
        self.positional_encoding = PositionalEncoding(hidden_size=hidden_size)
        # 然后我们会有n个encoder block串联起来
        self.encoders = nn.Sequential(*[
            TransformerEncoderBlock(
                num_head=num_head, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size) for _ in range(num_blocks)
        ])
        self.vocab = vocab

    def forward(self, source: Tensor):
        # TIP: 这里我们的实现和书上的实现有所不同
        # 虽然书上的实现是更为优秀的实现 但是我们想要改成那个样子需要修改大量的代码
        # 所以valid_lens不会以参数的形式传到这个函数里面
        # 而是我们在这个函数内部计算出这次batch数据的valid_lens
        # 和之前的AttentionEncoder的实现一样

        batch_size, num_seq = source.shape
        valid_lens = func.make_source_valid_lens(
            source=source, pad=self.vocab.pad())

        embed = self.embedding(source)
        assert embed.shape == (batch_size, num_seq, self.hidden_size)

        # TODO: 为什么要乘上这个数??
        input = self.positional_encoding(embed * math.sqrt(self.hidden_size))
        # input = self.positional_encoding(embed)
        assert input.shape == embed.shape

        # 不对！这个valid_lens正是decoder需要的
        for encoder in self.encoders:
            input = encoder(input, valid_lens)

        encoder_output = input
        return encoder_output, valid_lens


class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_head: int, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        # DecoderBlock一共有六层
        # 1. self attention
        self.self_attention = MultiHeadAttention(
            num_head=num_head, hidden_size=hidden_size)
        # 2. add norm1
        self.add_norm1 = AddNorm(feature_size=hidden_size)
        # 3. encoder decoder attention
        self.cross_attention = MultiHeadAttention(
            num_head=num_head, hidden_size=hidden_size)
        # 4. add norm2
        self.add_norm2 = AddNorm(feature_size=hidden_size)
        # 5. ffn
        self.ffn = FeedForwardNetwork(
            hidden_size=ffn_hidden_size, output_size=hidden_size)
        # 6. add norm3
        self.add_norm3 = AddNorm(feature_size=hidden_size)

        # 我们需要保存历史信息
        self.history: Tensor | None = None

    def clear_history(self):
        self.history = None
        pass

    def make_target_valid_lens(self, batch_size: int, max_len: int) -> Tensor:
        # target的self attention的valid lens跟encoder是不同的
        # 在encoder做训练时 我们希望位于 xt 出的token 只能attend到 xt之前的token
        # 所以这里生成的valid_lens就是一个二维的矩阵
        # target_valid_lens.shape = (batch_size, seq_len)
        # 而且不同的target输入的seq_len是不同的 所以这个东西还需要动态生成
        # TIP: valid_lens的类型应该是int
        device = next(self.parameters()).device
        lens = torch.arange(start=0, end=max_len,
                            dtype=torch.int, device=device).reshape(1, -1)
        # lens是行向量
        # 然后纵向扩展到batch_size
        target_valid_lens = lens.repeat(
            batch_size).reshape(batch_size, max_len)
        return target_valid_lens

    def forward(self, input: Tensor, encoder_output: Tensor, source_valid_lens: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor | None]:
        batch_size, num_seq, hidden_size = input.shape

        # 那么问题来了 我们怎么知道目前是做训练合适预测呢？？
        if not config.conf['predict']:
            # 不行 不能这样写 eval会退出training模式的
            # 所以必须是predict指定一个变量才行
            # 我们让predict指定一个全局变量吧
            keys = input
            values = input
        else:
            if self.history is None:
                self.history = input.clone()
            else:
                self.history = torch.concat([self.history, input], dim=1)
            keys = self.history.clone()
            values = keys.clone()

        # 1. self attention
        output: Tensor = self.self_attention(
            queries=input, keys=keys, values=values, valid_lens=func.make_target_valid_lens(batch_size=batch_size, max_len=num_seq, device=input.device))
        assert output.shape == input.shape

        # 2. add norm 1
        x = self.add_norm1(input=input, output=output)
        assert x.shape == input.shape

        # 3. encoder decoder attention
        input = x
        # 现在非常的关键 我们在这里的query是我们上一层额输出
        # 而我们的qv是endoer的输出
        # 而valid_lens是source的valid_lens
        output = self.cross_attention(
            queries=input, keys=encoder_output, values=encoder_output, valid_lens=source_valid_lens)
        # 这里就不能assert了 因为q和kv的seq长度是不一样的
        # 但是batch_size 和 hidden_size应该是一样的
        # 要不然也做不了DorProcutAttention了

        # 4. add norm 2
        y = self.add_norm2(input=input, output=output)
        assert y.shape == input.shape

        # 5. 过FFN
        input = y
        output = self.ffn(output)
        assert output.shape == input.shape

        # 6. add norm 3
        return self.add_norm3(input=input, output=output), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, ffn_hidden_size, num_head: int, num_blocks: int, vocab: VocabularyV2):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = MyEmbedding(
            vocab_size=vocab_size, embed_size=hidden_size, transpose=False)
        self.positional_encoding = PositionalEncoding(hidden_size=hidden_size)
        # 然后我们会有n个encoder block串联起来
        self.decoders = nn.Sequential(*[
            TransformerDecoderBlock(
                num_head=num_head, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size) for _ in range(num_blocks)
        ])
        self.output_layer = nn.LazyLinear(out_features=vocab_size)

    def clear_history(self):
        for decoder in self.decoders:
            decoder.clear_history()  # type: ignore

    def forward(self, target: Tensor, encoder_output: Tensor, source_valid_lens: Tensor):
        # Encoder和Decoder的返回值和调用参数必须匹配
        # 这样Seq2Seq才不会出问题
        # TODO：但是这样的结构显然不符合软件工程，应该重构一下
        batch_size1, num_tgtseq = target.shape
        batch_size2, num_srcseq, hidden_size = encoder_output.shape
        assert batch_size1 == batch_size2
        batch_size = batch_size1
        assert source_valid_lens.shape == (batch_size,)

        # embedding
        embed = self.embedding(target)
        assert embed.shape == (batch_size, num_tgtseq, self.hidden_size)

        # positional encoding
        input = self.positional_encoding(embed * math.sqrt(self.hidden_size))
        # input = self.positional_encoding(embed)
        assert input.shape == embed.shape

        # decoder blocks
        x: Tensor = input
        state = None
        for decoder in self.decoders:
            x, state = decoder(x, encoder_output, source_valid_lens, state)
        assert x.shape == (batch_size, num_tgtseq, hidden_size)

        # final output layer
        # ! 在这里反而要反转x的第一和第二维度 因为outputlayer一次只能处理一个token
        x = x.transpose(0, 1)
        output: Tensor = self.output_layer(x)
        assert output.shape == (num_tgtseq, batch_size, self.vocab_size)
        output = output.transpose(0, 1)
        assert output.shape == (batch_size, num_tgtseq, self.vocab_size)
        # 为了和seq2seq一致，我们返回一个none
        # 不行 必须返回source_valid_lens
        # 哎 架构实在是太乱了
        # TODO：需要重构
        return output, source_valid_lens
