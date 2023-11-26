# 2023/11/26
# zhangzhong

from module.vision.resnet import ResNet18, SmallResNet18
import torchsummary  # type: ignore
from mytorch.data.cifar10 import CIFAR10Dataset
import torch
from mytorch import training, utils


def test_simple_resnet() -> None:
    net = ResNet18()
    torchsummary.summary(net, input_size=(3, 96, 96),
                         batch_size=1, device='cpu')


def test_small_resnet() -> None:
    net = SmallResNet18()
    torchsummary.summary(net, input_size=(3, 32, 32),
                         batch_size=1, device='cpu')

    batch_size = 128
    cifar10 = CIFAR10Dataset()
    # cifar10 = FashionMNISTDataset()
    train_dataloader = cifar10.get_train_dataloader(batch_size=batch_size)
    val_dataloader = cifar10.get_val_dataloader(batch_size=batch_size)
    test_dataloader = cifar10.get_test_dataloader(batch_size=batch_size)

    # 或许可以用环境变量来决定做测试还是做训练
    # 过拟合非常非常严重啊
    # 或许是batch_size太大了 数据太少了
    # 改了batch_size也没用
    # 或许是cifar10对lenet来说太难了
    # 果然是这样 同样的网络结果 换一个数据集 结果就好很多了 看来还是得上resnet呀
    # 实在是太容易过拟合了 数据增强 启动！
    lr: float = 0.01
    num_epochs = 100
    net = SmallResNet18()
    trainer = training.Trainer(
        model=net,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(params=net.parameters(), lr=lr),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=utils.get_device(),
        is_test=False)

    trainer.train(tag='SmallResNet18', calculate_accuracy=True)
