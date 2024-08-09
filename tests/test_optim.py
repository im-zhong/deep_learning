# 2023/9/9
# zhangzhong

from mytorch import optim
import torch


def test_SGD():
    w = torch.tensor([2.0, -3.4], requires_grad=True)
    b = torch.tensor(4.2, requires_grad=True)

    # BUG:FIX
    # The error message you're seeing is because the optim.SGD function is expecting an argument of type Iterator[Tensor] for its params parameter, but you're providing a list[Tensor].
    # An Iterator[Tensor] is a type that implements the iterator protocol, which requires the __next__ method. A list[Tensor] does not have this method, hence the incompatibility.
    # To fix this, you can convert your list to an iterator using the iter() function:
    optimizer = optim.MySGD(iter([w, b]), lr=0.01)
    optimizer.zero_grad()
    x = torch.randn((1, 2))
    y_hat = torch.matmul(x, w) + b
    y_hat.backward()
    # 这就很明显了 我们的optimizer并没有更新参数
    # 为什么呢??
    # tensor([ 2.0000, -3.4000], requires_grad=True) tensor(4.2000, requires_grad=True)
    # 我大概了解为什么了
    # python里面的= 实际上是alias
    # a = 1
    # a = a + 1 实际上会创建一个新的对象，然后将a指向这个新的对象
    # 所以我们自然无法更新w和b
    optimizer.step()

    print(w, b)


def test_grad_clip():
    w = torch.tensor([2.0, -3.4], requires_grad=True)
    b = torch.tensor(4.2, requires_grad=True)
    optimizer = optim.MySGD(iter([w, b]), lr=0.01)
    optimizer.zero_grad()
    x = torch.randn((1, 2))
    y_hat = torch.matmul(x, w) + b
    y_hat.backward()

    m = optim.magnitude([w, b])
    clip = 1.0
    print(w.grad)
    print(b.grad)
    print(optim.magnitude([w, b]))
    optim.grad_clip([w, b], clip=clip)
    print(optim.magnitude([w, b]))
    assert abs(optim.magnitude([w, b]) - clip) < 1e-6

    # TODO: 如何测试grad的方向没有发生改变呢??
    # 平行，向量叉积为零
    if w.grad is not None:
        w.grad *= m / clip
        print(w.grad)
    if b.grad is not None:
        b.grad *= m / clip
        print(b.grad)


def test_SGD_with_clip():
    clip = 0.5
    w = torch.tensor([2.0, -3.4], requires_grad=True)
    b = torch.tensor(4.2, requires_grad=True)

    optimizer = optim.MySGD(iter([w, b]), lr=0.01, gradient_clip=clip)
    optimizer.zero_grad()
    x = torch.randn((1, 2))
    y_hat = torch.matmul(x, w) + b
    y_hat.backward()
    optimizer.step()
    assert abs(optimizer.magnitude() - clip) < 1e-6

    print(w, b)
