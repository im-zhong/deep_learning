# 2023/11/24
# zhangzhong


from module.vision.lenet import LeNet, BNLeNet
from mytorch.data.cifar10 import CIFAR10Dataset
from mytorch.data.mnist import FashionMNISTDataset
from mytorch import func, training, utils
import torch.nn
import torch.optim


def test_le_net() -> None:

    batch_size = 128
    # cifar10 = CIFAR10Dataset()
    cifar10 = FashionMNISTDataset()
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
    lr: float = 0.001
    num_epochs = 10
    alex_net = BNLeNet()
    trainer = training.Trainer(
        model=alex_net,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(params=alex_net.parameters(), lr=lr),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=utils.get_device(),
        is_test=False)

    trainer.train(tag='AlexNet', calculate_accuracy=True)
