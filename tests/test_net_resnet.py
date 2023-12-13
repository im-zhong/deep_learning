# 2023/11/26
# zhangzhong

from module.vision.resnet import ResNet18, SmallResNet18
import torchsummary  # type: ignore
from mytorch.data.cifar10 import CIFAR10Dataset, cifar10_predict
import torch
from mytorch import training, utils
import json
from torch import nn


def test_simple_resnet() -> None:
    net = ResNet18()
    torchsummary.summary(net, input_size=(3, 96, 96),
                         batch_size=1, device='cpu')


def test_small_resnet() -> None:

    batch_size = 128
    num_workers = 16
    cifar10 = CIFAR10Dataset(num_workers=num_workers)
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
    lr: float = 0.1
    num_epochs = 100
    tag = 'resnet18_2'
    net = SmallResNet18()

    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs)

    trainer = training.TrainerV2(
        model=net,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=utils.get_device())

    trainer.train(tag=tag)

def test_small_resnet_no_split() -> None:

    batch_size = 128
    num_workers = 16
    cifar10 = CIFAR10Dataset(num_workers=num_workers, splits=[0.99, 0.01])
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
    lr: float = 0.1
    num_epochs = 0
    tag = 'resnet18_3'
    net = SmallResNet18()

    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs)

    device = utils.get_device()
    trainer = training.TrainerV2(
        model=net,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device)

    trainer.train(tag=tag)

    # predict
    pretrained_model: nn.Module = trainer.model
    images = ['imgs/pattern_recognition/J20.jpg',
              'imgs/pattern_recognition/M7.png',
              'imgs/pattern_recognition/101.png'
              ]
    for image in images:
        cifar10_predict(model=pretrained_model, device=device, path=image)
 
    
    

def test_json():
    result = training.Result()
    s = json.dumps(result.to_dict())
    print(s)
    r = json.loads(s)
    print(r)
    # 我们还要提供一个函数 用来将json转换为Result对象
    rr = training.Result.from_dict(r)
    print(rr.epoch, rr.train_loss, rr.val_loss)


def test_pytorch() -> None:
    if torch.cuda.is_available():
        print('cuda is avaliable')
    else:
        print('cuda is not avaliable')
