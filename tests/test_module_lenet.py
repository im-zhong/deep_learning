# 2023/11/21
# zhangzhong

import torch
from torch import nn, Tensor
from module.vision.lenet import LeNet, MyLeNet, BNLeNet
from mytorch.data.mnist import FashionMNISTDataset
from mytorch import training


def test_MyLeNet():
    # 好消息：跑起来了
    # 坏消息：跑的非常非常非常慢
    lenet = MyLeNet()
    # make some random input
    # input = torch.randn(size=(4, 1, 28, 28))
    # output = lenet(input)

    # hyper parameters
    lr = 0.01
    batch_size = 32
    epochs = 0

    # now let's prepare the dataset, just use MNIST
    mnist = FashionMNISTDataset()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=lenet.parameters(), lr=lr)
    train_dataloader = mnist.get_train_dataloader(
        batch_size=batch_size, shuffle=True)
    val_dataloader = mnist.get_val_dataloader(
        batch_size=batch_size, shuffle=False)

    trainer = training.Trainer(model=lenet, loss_fn=loss_fn, optimizer=optimizer,
                               num_epochs=epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    trainer.train(tag='MyLeNet')


def test_LeNet():
    # 好消息：跑起来了
    # 坏消息：跑的非常非常非常慢
    lenet = LeNet()
    # make some random input
    # input = torch.randn(size=(4, 1, 28, 28))
    # output = lenet(input)

    # hyper parameters
    lr = 0.01
    batch_size = 32
    epochs = 1

    # now let's prepare the dataset, just use MNIST
    mnist = FashionMNISTDataset()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=lenet.parameters(), lr=lr)
    train_dataloader = mnist.get_train_dataloader(
        batch_size=batch_size, shuffle=True)
    val_dataloader = mnist.get_val_dataloader(
        batch_size=batch_size, shuffle=False)

    trainer = training.Trainer(model=lenet, loss_fn=loss_fn, optimizer=optimizer,
                               num_epochs=epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    trainer.train(tag='MyLeNet')


def test_BNLeNet():
    # 好消息：跑起来了
    # 坏消息：跑的非常非常非常慢
    lenet = BNLeNet()
    # make some random input
    # input = torch.randn(size=(4, 1, 28, 28))
    # output = lenet(input)

    # hyper parameters
    lr = 0.01
    batch_size = 32
    epochs = 1

    # now let's prepare the dataset, just use MNIST
    mnist = FashionMNISTDataset()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=lenet.parameters(), lr=lr)
    train_dataloader = mnist.get_train_dataloader(
        batch_size=batch_size, shuffle=True)
    val_dataloader = mnist.get_val_dataloader(
        batch_size=batch_size, shuffle=False)

    trainer = training.Trainer(model=lenet, loss_fn=loss_fn, optimizer=optimizer,
                               num_epochs=epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    trainer.train(tag='MyLeNet')
