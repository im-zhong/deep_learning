# 2023/11/20
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch import func
import torch.nn.functional as F


class RNNScratch(nn.Module):
    def __init__(self, vocab_size: int, num_hidden: int):
        super().__init__()
        # 也就是每个one_hot vector转换成的hidden state vecotr的维度
        self.num_hidden = num_hidden
        self.vocab_size = vocab_size
        # H(t) = activation(X@Wx + H(t-1)@Wh + Bh)
        # X.shape = (batch_size, vocab_size)
        # Wx.shape = (vocab_size, num_hidden)
        # H(t-1).shape = (batch_size, num_hidden)
        # Wh.shape = (num_hidden, num_hidden)
        # Bh.shape = (num_hidden,) for boardcasting
        self.Wx = nn.Parameter(torch.normal(mean=0, std=0.01, size=(
            vocab_size, num_hidden), requires_grad=True))
        self.Wh = nn.Parameter(torch.normal(mean=0, std=0.01, size=(
            num_hidden, num_hidden), requires_grad=True))
        self.Bh = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
        # TIP:
        # 使用nn.Parameter就不需要自己在定义一个parameters() 函数了
        # 卧槽 真的神奇，这些参数被放到cuda上了
        # 也就是说，rnn是LM的成员，在LM上调用to device，rnn作为一个成员，他的参数也可以被放到cuda上

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        # input.shape = (batch_size, vocab_size), x.shape
        batch_size, vocab_size = input.shape
        if state is None:
            # 这个要怎么放到GPU上呢??
            state = torch.zeros(size=(batch_size, self.num_hidden))

        # 放到和input一样的device上
        state = state.to(input.device)
        Ht = func.relu_layer(input @ self.Wx + state @ self.Wh + self.Bh)
        return Ht

    # def parameters(self):
    #     yield self.Wx
    #     yield self.Wh
    #     yield self.Bh

# TODO: LM应该是通用的
# 我们的RNN的接口行为需要和pytorch.RNN保持一致
# 只考虑 batch_fist=False的情况 也就是默认情况

# TODO: 修改名字
# 这个名字不好 太长了
# 应该改成 MyLSTM, MyRNN, MyGRU
# 而且这样编辑器可以帮助你做补全
# 前缀 要 好于 后缀


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # forget gate
        self.weight_xf = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hf = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_f = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        # input gate
        self.weight_xi = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hi = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_i = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        # output gate
        self.weight_xo = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_ho = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_o = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        # input node
        self.weight_xn = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hn = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_n = nn.Parameter(torch.zeros(size=(1, hidden_size)))

    def forward(self, input: Tensor, initial_state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # input.shape = (num_seq, batch_size, input_size)
        _, batch_size, _ = input.shape
        if initial_state is not None:
            memory, state = initial_state
        else:
            memory = torch.zeros(
                size=(batch_size, self.hidden_size), device=input.device)
            state = torch.zeros(
                size=(batch_size, self.hidden_size), device=input.device)

        states: list[Tensor] = []
        for xt in input:
            forget_gate = F.sigmoid(
                xt @ self.weight_xf + state @ self.weight_hf + self.bias_f)
            input_gate = F.sigmoid(
                xt @ self.weight_xi + state @ self.weight_hi + self.bias_i)
            output_gate = F.sigmoid(
                xt @ self.weight_xo + state @ self.weight_ho + self.bias_o)
            input_node = F.tanh(
                xt @ self.weight_xn + state @ self.weight_hn + self.bias_n)
            # update memory, input node可以看作之前的RNN cell
            memory = forget_gate * memory + input_gate * input_node
            # update state
            state = output_gate * F.tanh(memory)
            states.append(state)

        return torch.stack(states), (memory, state)


class MyRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_x = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_h = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_h = nn.Parameter(torch.zeros(size=(1, hidden_size)))

    def forward(self, input: Tensor, initial_state: Tensor | None) -> tuple[Tensor, Tensor]:
        len_seq, batch_size, input_size = input.shape
        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros(
                size=(batch_size, self.hidden_size), device=input.device)

        states: list[Tensor] = []
        for xt in input:
            state = F.tanh(xt @ self.weight_x + state @
                           self.weight_h + self.bias_h)
            states.append(state)

        return torch.stack(states), state


class MyGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        # reset gate: short term sequence
        self.weight_xr = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hr = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_r = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        # update gate: long sequence
        self.weight_xu = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hu = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_u = nn.Parameter(torch.zeros(size=(1, hidden_size)))
        # candidate state, RNN cell
        self.weight_xh = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.normal(
            mean=0, std=0.01, size=(hidden_size, hidden_size)))
        self.bias_h = nn.Parameter(torch.zeros(size=(1, hidden_size)))

    def forward(self, input: Tensor, initial_state: Tensor) -> tuple[Tensor, Tensor]:
        len_seq, batch_size, input_size = input.shape
        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros(
                size=(batch_size, self.hidden_size), device=input.device)

        states: list[Tensor] = []
        for xt in input:
            reset_gate = F.sigmoid(
                xt @ self.weight_xr + state @ self.weight_hr + self.bias_r)
            update_gate = F.sigmoid(
                xt @ self.weight_xu + state @ self.weight_hu + self.bias_u)
            candidate_state = F.tanh(
                xt @ self.weight_xh + (reset_gate * state) @ self.weight_hh + self.bias_h)
            state = update_gate * state + (1 - update_gate) * candidate_state
            states.append(state)

        return torch.stack(states), state


class MyBiGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        hidden_size = int(hidden_size / 2)
        self.hidden_size = hidden_size
        self.forward_net = MyGRU(
            input_size=input_size, hidden_size=hidden_size)
        self.backward_net = MyGRU(
            input_size=input_size, hidden_size=hidden_size)

    def forward(self, input: Tensor, initial_state: Tensor | None) -> tuple[Tensor, Tensor]:
        len_seq, batch_size, input_size = input.shape
        # initial_state 应该是一个tuple 类似LSTM那样
        # 其实这个东西的处理过程和RNN非常类似
        if initial_state is not None:
            forward_state = initial_state[:, :self.hidden_size]
            backward_state = initial_state[:, self.hidden_size:]

        else:
            # 可以用None代替呀 实现上更加简单
            forward_state, backward_state = None, None

        forward_outputs, forward_state = self.forward_net(
            input, forward_state)
        backward_outputs, backward_state = self.backward_net(
            # use torch.flip instead
            torch.flip(input, [0]), backward_state)

        assert forward_outputs.shape == (len_seq, batch_size, self.hidden_size)
        assert backward_outputs.shape == (
            len_seq, batch_size, self.hidden_size)
        if forward_state is not None:
            assert forward_state.shape == (batch_size, self.hidden_size)
        if backward_state is not None:
            assert backward_state.shape == (batch_size, self.hidden_size)

        outputs = torch.cat(
            (forward_outputs, torch.flip(backward_outputs, [0])), dim=-1)
        # 这里不能用stack，需要使用cat才能和DeepGRU协同使用 否则就需要重新实现了
        # return outputs, torch.stack((forward_state, backward_state))
        assert forward_state is not None and backward_state is not None
        state = torch.cat((forward_state, backward_state), -1)
        assert outputs.shape == (len_seq, batch_size, int(2*self.hidden_size))
        assert state.shape == (batch_size, int(2*self.hidden_size))
        return outputs, state


class MyDeepGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 名字是net的话 连forward都不用写
        # 我的评价是：不如不用Sequential，因为你反正也没法自动实现forward()
        # 不过，或许模型的参数parameters()会自动实现？！大概吧 还是用sequential吧
        if bidirectional:
            self.net = nn.Sequential(*[
                MyBiGRU(input_size=input_size if l == 0 else hidden_size, hidden_size=hidden_size) for l in range(num_layers)
            ])
        else:
            self.net = nn.Sequential(*[
                MyGRU(input_size=input_size if l == 0 else hidden_size, hidden_size=hidden_size) for l in range(num_layers)
            ])

    def forward(self, input: Tensor, initial_states: Tensor | None = None) -> tuple[Tensor, Tensor]:
        len_seq, batch_size1, input_size = input.shape
        assert input_size == self.input_size
        if initial_states is not None:
            num_layers, batch_size2, hidden_size = initial_states.shape
            assert batch_size1 == batch_size2
            assert num_layers == self.num_layers
            assert hidden_size == self.hidden_size
        else:
            initial_states = [None] * self.num_layers  # type: ignore

        states: list[Tensor] = []
        outputs = input
        for l in range(self.num_layers):
            # 每一层我们都可以拿到一个state
            # 最终这些state也要stack起来
            # 反而是outputs，我们只要最后一层的
            assert initial_states is not None
            outputs, state = self.net[l](outputs, initial_states[l])
            states.append(state)

        # TODO:DONE. 这个应该是可以复用的
        # 就似乎stack无法stack tuple 所以BiGRU只需要修改一下state的返回格式即可
        state = torch.stack(states)
        assert state.shape == (self.num_layers, batch_size1, self.hidden_size)
        return outputs, torch.stack(states)
