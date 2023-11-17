# 2023/9/9
# zhangzhong

from mytorch import losses
import torch
import math


def test_MSELoss():
    # 模拟生成两组数据
    y_hat = torch.randn((10, 1))
    y = torch.randn((10, 1))

    loss = losses.MSELoss()
    loss = loss(y_hat, y)
    assert loss == torch.mean((y_hat - y)**2) / 2


def test_CrossEntropyLoss():
    # 首先我们随机生成一组logits 模拟线性层的输出
    # 模拟一整个minibatch的输出
    batch_size = 32
    num_labels = 10
    # Returns a tensor filled with random numbers
    # from a uniform distribution on the interval [0, 1)
    logits = torch.rand((batch_size, num_labels))
    assert logits.shape == torch.Size([batch_size, num_labels])

    # 生成一个minibatch的labels
    # Returns a tensor filled with random integers generated uniformly
    # between low (inclusive) and high (exclusive)
    labels = torch.randint(low=0, high=num_labels, size=(batch_size,))
    assert labels.shape == torch.Size([batch_size])

    # TODO: 测试就要像这样，做交叉验证，同种算法的多种实现，多个库的实现
    # 都可以辅助我们进行代码测试，尤其是做测试的时候要找到ground truth

    # 我们再对比一样pytorch的CrossEntropyLoss的实现
    correct_loss_fn = torch.nn.CrossEntropyLoss()
    correct_loss = correct_loss_fn(logits, labels)

    naive_loss_fn = losses.NaiveCrossEntropyLoss()
    naive_loss = naive_loss_fn(logits, labels)

    loss_fn = losses.CrossEntropyLoss()
    loss = loss_fn(logits, labels)

    print("loss: ", loss.item())
    print("naive_loss: ", naive_loss.item())
    print("correct_loss: ", correct_loss.item())

    assert abs(correct_loss.item() - naive_loss.item()) < 1e-6
    assert abs(correct_loss.item() - loss.item()) < 1e-6


def test_CrossEntropyLossShape():
    # 首先我们随机生成一组logits 模拟线性层的输出
    # 模拟一整个minibatch的输出
    batch_size = 32
    num_labels = 10
    # Returns a tensor filled with random numbers
    # from a uniform distribution on the interval [0, 1)
    logits = torch.rand((batch_size, num_labels))
    assert logits.shape == torch.Size([batch_size, num_labels])

    # 生成一个minibatch的labels
    # Returns a tensor filled with random integers generated uniformly
    # between low (inclusive) and high (exclusive)
    labels = torch.randint(low=0, high=num_labels, size=(batch_size,))
    assert labels.shape == torch.Size([batch_size])

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    correct_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    correct_loss = correct_loss_fn(logits, labels)
    # shape = (32,), shape = (batch_size,), shape = labels.shape
    print(correct_loss.shape)
    assert correct_loss.shape == labels.shape

    # BUG: 我们的loss虽然返回值是对的 但是形状不对
    loss_fn = losses.CrossEntropyLoss(calculate_mean=False)
    loss = loss_fn(logits, labels)
    print(loss.shape)
    assert loss.shape == labels.shape
