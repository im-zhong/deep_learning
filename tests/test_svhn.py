# 2023/12/14
# zhangzhong

from torch import Tensor

from mytorch.data.svhn import SVHNDataset


def test_svhn():
    svhn = SVHNDataset()
    batch_size = 4
    train_dataloader = svhn.get_train_dataloader(batch_size)
    test_dataloader = svhn.get_test_dataloader(batch_size)
    print(len(train_dataloader))
    print(len(test_dataloader))
    x, y = next(iter(train_dataloader))
    print(type(x), type(y))
    # x y 都是Tensor，因为我们在transform里面用了ToTensor
    # 其实就是图片，和MNIST CIFAR一样
    # x.shape == (batch_size, channels=3, height=32, width=32)
    print(x.shape)
    # y.shape == (batch_size,) 其实就是每张图片一个标签
    print(y.shape)
    # 我们最好随机生成一批图像和他对应的标签，这样好让我们对数据有一个整体的认识
    # 论文里也可以放这样的图片进去
    # 这个就只能用jupyter来做了
