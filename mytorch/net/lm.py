# 2023/11/20
# zhangzhong
# Language Model

import torch
from torch import nn, Tensor
from mytorch import func
import torch.nn.functional as F
from mytorch.data import seq
from mytorch.net.rnn import RNNScratch
from typing import Any
from mytorch import config


class LanguageModelScratch(nn.Module):
    def __init__(self, vocab: seq.Vocabulary, rnn: RNNScratch):
        super().__init__()
        self.vocab = vocab
        self.rnn = rnn

        # Ot = Ht@Wo + Bo
        # Ht.shape = (batch_size, num_hidden)
        # Wo.shape = (num_hidden, vocab_size)
        # Bo.shape = (vocab_size)
        # 也就是我们每个样本的输出向量是一个vocab_size大小的向量，
        # 代表了对应于每个token的可能性分布 logits
        self.Wo = nn.Parameter(torch.normal(mean=0, std=0.01, size=(
            rnn.num_hidden, rnn.vocab_size), requires_grad=True))
        self.Bo = nn.Parameter(torch.zeros(rnn.vocab_size, requires_grad=True))

    def forward(self, x: Tensor, show_hidden: bool = False) -> Tensor | tuple[Tensor, Tensor | None]:
        # x.shape = (batch_size, num_seq)
        batch_size, num_seq = x.shape[0], x.shape[1]
        vocab_size = len(self.vocab)

        # # step 1. 先交换前两个维度，这样更容易理解
        # 我们希望的是一次处理同一时刻的多个batch的样本
        # 也就是从之前的横着遍历，变成竖着遍历
        # x_transpose.shape = (num_seq, batch_size)
        x_transpose = torch.transpose(x, 0, 1)
        assert x_transpose.shape == (num_seq, batch_size)

        # step 2. 将token_index转换成对应的one_hot vector
        # x_one_hot.shape = (num_seq, batch_size, vocab_size)
        x_one_host = F.one_hot(x_transpose, vocab_size).float()

        assert x_one_host.shape == (num_seq, batch_size, vocab_size)

        state = None
        states: list[Tensor] = []
        outputs: list[Tensor] = []
        for batch in x_one_host:
            # batch.shape = (batch_size, vocab_size)
            # 每次遍历
            assert batch.shape == (batch_size, vocab_size)

            # rnn layer
            # state.shape = (batch_size, num_hidden)
            state = self.rnn(batch, state)
            states.append(state)

            # output layer
            # output.shape = (batch_size, vocab_size)
            output = state @ self.Wo + self.Bo
            outputs.append(output)

        # 最后我们的结果的维度
        # len(outputs) = num_seq
        # torch.tensor(outputs).shape = (num_seq, batch_size, vocab_size)
        # 最后的输出就是对所有样本的预测，其预测向量一个一个vocab_size大小的向量，表示下一个token的概率分布
        # 和最后一次rnn网络的输出的state向量
        assert len(outputs) == num_seq
        # TIP: use stack() to convert a list of tensor to a whole tensor!
        # output = torch.tensor(outputs)
        # output = torch.stack(outputs)
        # assert output.shape == (num_seq, batch_size, vocab_size)
        # # 因为我们之前将输入矩阵的前两个维度倒转了一下
        # # 为了之后计算的loss的时候 位置能够对应 我们需要将前两个维度倒转回去
        # return output.transpose(0, 1), state
        # 重大发现，stack.stack(outputs, dim=1)刚好就是上面两句话的效果
        output = torch.stack(outputs, dim=1)
        assert output.shape == (batch_size, num_seq, vocab_size)
        # TODO: 其实我这里的输出和pytorch的RNN的输出不一样
        # 我这里相当于只输出了所有的batch的sequence中最后一个token的state向量
        # 人家输出了所有的token的state向量
        # 也可以把所有的state向量stack起来
        # BUG:FIX: trainer只能接受返回一个tensor的network
        if show_hidden:
            return output, state
        else:
            return output

    # def parameters(self):
    #     yield from self.rnn.parameters()
    #     yield self.Wo
    #     yield self.Bo

    # TODO: predict需要修改以适应cuda
    def predict(self, prompt: str, num_seq: int) -> list[str]:
        state = None
        states = []
        outputs: list[Tensor] = []
        results: list[str] = []

        def yield_token():
            for token in prompt:
                yield token

            # output的最后一个是我们预测的符号
            # 然后我们拿着这个token 继续执行下面的流程
            for _ in range(num_seq - len(prompt)):
                index = torch.argmax(outputs[-1].squeeze())
                token = self.vocab.to_token(index)
                yield token

        for token in yield_token():
            index = self.vocab.to_index(token)
            # 包装成RNN能用的样子
            x = torch.tensor([index])
            vocab_size = len(self.vocab)

            # rnn
            x_one_host = F.one_hot(x, vocab_size).float()
            assert x_one_host.shape == (1, vocab_size)

            x_one_host = x_one_host.to(torch.device(config.conf['device']))
            state = self.rnn(x_one_host, state)
            assert state.shape == (1, self.rnn.num_hidden)
            states.append(state)

            # output
            # 但是在最开始我们是不去考虑output的
            # 我们只计算state
            output = state @ self.Wo + self.Bo
            assert output.shape == (1, vocab_size)
            outputs.append(output)

            # 把rnn预测的结果存放起来
            results.append(token)

        return results


class LanguageModel(nn.Module):
    def __init__(self, vocab: seq.Vocabulary, rnn: nn.Module, hidden_size: int):
        super().__init__()
        self.vocab = vocab
        self.rnn = rnn
        self.hidden_size = hidden_size

        # TODO: 不对，忘了还有输出层了。。。
        # Ot = Ht@Wo + Bo
        # Ht.shape = (batch_size, num_hidden)
        # Wo.shape = (num_hidden, vocab_size)
        # Bo.shape = (vocab_size)
        # 也就是我们每个样本的输出向量是一个vocab_size大小的向量，
        # 代表了对应于每个token的可能性分布 logits
        # self.output_layer = torch.nn.LazyLinear(len(vocab))
        self.output_layer = torch.nn.Linear(
            in_features=hidden_size, out_features=len(vocab))

    def forward(self, x: Tensor) -> Any:
        # x.shape = (batch_size, num_seq)
        batch_size, num_seq = x.shape
        vocab_size = len(self.vocab)

        # 这一句代码，顶之前的实现里面的两步
        embedding = F.one_hot(x.T, vocab_size).float()
        assert embedding.shape == (num_seq, batch_size, vocab_size)

        states, _ = self.rnn(embedding, None)
        # 这里是不对的 因为我们不知道hidden_size 其实也就没法做assert
        assert states.shape == (num_seq, batch_size, self.hidden_size)
        # 你会发i西安embedding和outputs的shape是一样的 这不是巧合
        # 对于每个字符 rnn都会预测他的下一个字符 当然输入和输出的形状是一样的
        # assert final_state.shape == (batch_size, hidden_size)

        # 不对不对，还要过output Layer
        # 交换维度，这样和groud truch做cross entropy loss才是对的
        # outputs = outputs.transpose(0, 1)
        # assert outputs.shape == (batch_size, num_seq, vocab_size)
        # return outputs
        outputs = []
        for state in states:
            # assert output.shape == (batch_size, hidden_size)
            output = self.output_layer(state)
            assert output.shape == (batch_size, vocab_size)
            outputs.append(output)
        assert len(outputs) == num_seq

        outputs = torch.stack(outputs, dim=1)
        assert outputs.shape == (batch_size, num_seq, vocab_size)
        return outputs

    def predict(self, prompt: str, max_len: int) -> str:
        with torch.no_grad():
            return self.predict_impl(prompt, max_len)

    def predict_impl(self, prompt: str, max_len: int) -> str:
        len_seq, vocab_size = len(prompt), len(self.vocab)
        # 1. prompt -> rnn.input -> final_state
        embedding = self.vocab.build_input(
            prompt=prompt).to(torch.device(config.conf['device']))
        assert embedding.shape == (len_seq, 1, vocab_size)
        # TODO: 或许应该创建一个全局的config对象，很多配置是全局的 应该放到里面
        # BUG:FIX. pytorch.rnn的final_state的shape = (num_layers, batch_size, hidden_size)
        outputs, state = self.rnn(embedding, None)
        # 这个assert是不能再用了
        # 因为不同类型的cell的返回的state是不一样的
        # assert state.shape == (1, 1, self.hidden_size)
        # TIP: 但是所有类型的cell的outputs都是一样的
        # shape = (len_seq, batch_size, hidden_size)

        # BUG:FIX.
        # For batched 3-D input, hx should also be 3-D but got 2-D tensor
        # rnn可以有多层的，每一层都有一个初始状态，所以state也应该是三维矩阵
        # 所以这里我们不应该squeeze
        # state = state.squeeze(0)
        # assert state.shape == (1, self.hidden_size)

        # 2.
        # get final ouputs
        result = prompt
        for _ in range(max_len - len_seq):
            # 拿到prompt的最后一个state做预测作为下一个token
            output = self.output_layer(outputs[-1])
            assert output.shape == (1, vocab_size)

            index = torch.argmax(output.squeeze())
            token = self.vocab.to_token(index)
            result += token

            # 把token转换成rnn的输入
            # TODO:DONE. 算了 好多重复的 还是写到Vocabulary里面吧
            embedding = self.vocab.build_input(
                prompt=token).to(torch.device(config.conf['device']))
            outputs, state = self.rnn(embedding, state)

        return result
