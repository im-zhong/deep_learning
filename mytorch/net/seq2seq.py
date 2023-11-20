# 2023/11/20
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch import func
import torch.nn.functional as F
from mytorch import config
from .rnn import MyDeepGRU
from mytorch import losses
from mytorch.data import seq


# TODO: embeding不应该放在这里，并不只有seq2seq才会用到embeding 或者他应该和positional encoding放在一起
# 他们都是对数据的一次编码，从这样的角度来理解合适吗？
class MyEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, transpose: bool = True):
        super().__init__()
        self.transpose = transpose
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weight = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(vocab_size, embed_size), device=config.conf['device']))

    # dataloader的输出会直接送到embedding这里
    def forward(self, input: Tensor) -> Tensor:
        # input.shape = (batch_size, num_seq)
        batch_size, num_seq = input.shape
        # embedding.shape = (num_seq, batch_size, embed_size)

        # BUG: 忘了给input做转置
        if self.transpose:
            e = func.embedding(input.T, self.weight)
            assert e.shape == (num_seq, batch_size, self.embed_size)
        else:
            e = func.embedding(input, self.weight)
            assert e.shape == (batch_size, num_seq, self.embed_size)

        return e


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        # 1. embedding
        # 2. multilayer GRU
        super().__init__()
        self.vocab_size, self.embed_size, self.hidden_size, self.num_layers = vocab_size, embed_size, hidden_size, num_layers
        self.embedding = MyEmbedding(vocab_size, embed_size)
        self.rnn = MyDeepGRU(input_size=embed_size,
                             hidden_size=hidden_size, num_layers=num_layers)

    # TODO: 修改Encoder以输出valid_lens
    def forward(self, source: Tensor) -> tuple[Tensor, Tensor]:
        # source.shape = batch_size, num_seq
        batch_size, num_seq = source.shape
        embed = self.embedding(source)
        assert embed.shape == (num_seq, batch_size, self.embed_size)
        output, state = self.rnn(embed, None)
        assert output.shape == (num_seq, batch_size, self.hidden_size)
        # 所以 context_variable = output[-1]. shape: (batch_size, self.hidden_size)
        assert state.shape == (self.num_layers, batch_size, self.hidden_size)
        return output, state


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        # 1. embedding
        # 2. multilayer GRU
        # 3. output layer
        super().__init__()
        self.vocab_size, self.embed_size, self.hidden_size, self.num_layers = vocab_size, embed_size, hidden_size, num_layers
        self.embedding = MyEmbedding(vocab_size, embed_size)
        self.rnn = MyDeepGRU(input_size=embed_size+hidden_size,
                             hidden_size=hidden_size, num_layers=num_layers)
        self.output_layer = nn.Linear(
            in_features=hidden_size, out_features=vocab_size)

    # define a empty method
    def clear_history(self):
        pass

    # BUG:FIX context应该由decoder自己计算出来
    # 否则的话和AttentionDecoder冲突，因为AttentionDecoder的context是动态的
    # 不同的decoder的context的计算方法不同，不应该由seq2seq来计算
    def forward(self, target: Tensor, encoder_output: Tensor, encoder_state: Tensor):
        batch_size, num_seq = target.shape
        context = encoder_output[-1]
        assert context.shape == (batch_size, self.hidden_size)
        assert encoder_state.shape == (
            self.num_layers, batch_size, self.hidden_size)

        # 1. embedding
        embed = self.embedding(target)
        assert embed.shape == (num_seq, batch_size, self.embed_size)
        # make context from (batch_size, hidden_size) to (num_seq, batch_size, hidden_size)
        # so we can concat embed and context
        # pytorch.repeat pytorch.cat
        # https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
        # Repeats this tensor along the specified dimensions.
        context = context.repeat(num_seq, 1, 1)
        assert context.shape == (num_seq, batch_size, self.hidden_size)
        embed = torch.cat([embed, context], dim=-1)
        assert embed.shape == (num_seq, batch_size,
                               self.embed_size + self.hidden_size)

        # 2. rnn
        decoder_output, decoder_state = self.rnn(embed, encoder_state)
        assert decoder_output.shape == (num_seq, batch_size, self.hidden_size)
        assert decoder_state.shape == (
            self.num_layers, batch_size, self.hidden_size)
        # 3. output layer
        outputs = []
        for state in decoder_output:
            output = self.output_layer(state)
            assert output.shape == (batch_size, self.vocab_size)
            outputs.append(output)
        assert len(outputs) == num_seq

        outputs = torch.stack(outputs, dim=1)
        assert outputs.shape == (batch_size, num_seq, self.vocab_size)
        # outputs要和label做cross entropy
        # 我们需要确认label的shape = (batch_size, num_seq)
        # 完全正确
        return outputs, decoder_state

# TODO: loss还需要走mask


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # TODO: 重构
    # 这样的接口对Trainer的泛用性提出了挑战
    # 或者 我们可以重新思考我们的框架设计
    # 我们的Dataset不是一定只能返回一对 x, y 了
    # 而是返回一列tensor
    # 而具体要怎么处理这些tensor 由具体的模型自行决定
    # 也就是数据集和模型是要匹配的 但是这些都是内部的实现 可以被很好的封装起来
    # 我们要搞清楚Dataset和Dataloader的职责
    # dataset就要规定数据的格式，dataloader只不过是每次取一个batch出来而已
    # 设计实现新模型的时候，我们要努力做到只需要实现一个Dataset和Model 剩下的应该全部复用！
    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        # source.shape: batch_size, num_src_seq
        # target.shape: batch_size, num_tag_seq
        # 为了适配多种情况，我们不对encoder的输出做假设了
        # encoder_output, encoder_state = self.encoder(source)
        # 这样encoder和decoder就必须适配
        encoder_results = self.encoder(source)
        # context = encoder_output[-1]
        decoder_output, _ = self.decoder(
            target, *encoder_results)
        return decoder_output

    def predict(self, trans: seq.TranslationDataManager, prompt: str) -> str:
        with torch.no_grad():
            # 这里会出BUG 之前的代码就没法用了 但是没办法 我们目前的代码太烂了
            # 重构会花费大量的时间
            config.conf['predict'] = True
            self.decoder.clear_history()
            return self.predict_impl(trans=trans, prompt=prompt)

    def predict_impl(self, trans: seq.TranslationDataManager, prompt: str) -> str:
        # 1. preprocess
        source_vocab = trans.source_vocab
        target_vocab = trans.target_vocab
        source = source_vocab.tokenize(prompt)
        source = torch.tensor(
            source, device=torch.device(config.conf['device']))
        # 2. encoder
        encoder_output, encoder_state, *_ = self.encoder(source)
        # context = encoder_output[-1]
        # 3. decoder
        # 和我们的forward不同的是，我们现在没有target作为输入了
        # 我们必须拿decoder的前一次的输入作为本次decoder的输入
        # 在最开始的时候，用bos
        index = target_vocab.bos()

        decoder_state = encoder_state
        result: list[int] = []
        max_len = 32
        while len(result) < max_len and index != target_vocab.eos():
            # 在最开始的时候，用encoder_state作为decoder的初始state
            target = torch.tensor([[index]], device=config.conf['device'])
            assert target.shape == (1, 1)
            decoder_output, decoder_state = self.decoder(
                target, encoder_output, decoder_state
            )
            assert decoder_output.shape == (1, 1, len(target_vocab))
            output: Tensor = decoder_output.squeeze()
            index = torch.argmax(output)
            result.append(index)

        return target_vocab.to_string(result)


# 这里显然是需要自定义一个Loss函数的
class Seq2SeqLoss(losses.Loss):
    def __init__(self, vocab: seq.VocabularyV2):
        self.vocab = vocab

    def __call__(self, y_hat: Tensor, y: Tensor):
        batch_size, num_seq = y.shape
        loss_fn = losses.CrossEntropyLoss(calculate_mean=False)
        loss = loss_fn(y_hat, y)
        assert loss.shape == (batch_size*num_seq,)
        y = y.flatten()
        assert y.shape == loss.shape

        # mask = torch.ones_like(y)
        # for i, _ in enumerate(y):
        #     if y[i] == self.vocab.pad():
        #         mask[i] = 0

        mask = (y != self.vocab.pad()).float()

        # mask = (y[y != self.vocab.pad()]).float()
        return (mask * loss).sum() / mask.sum()
