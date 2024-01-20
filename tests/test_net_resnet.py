# 2023/11/26
# zhangzhong

from module.vision.resnet import ResNet18, SmallResNet18
# import torchsummary  # type: ignore
from mytorch.data.cifar10 import CIFAR10Dataset, cifar10_predict
from mytorch.data.svhn import SVHNDataset
import torch
from mytorch import training, utils
import json
from torch import nn, Tensor
from mytorch import utils
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from mytorch.data.mnist import MNISTDataset, FashionMNISTDataset

# def test_simple_resnet() -> None:
#     net = ResNet18()
#     torchsummary.summary(net, input_size=(3, 96, 96),
#                          batch_size=1, device='cpu')


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
    num_epochs = 100
    tag = 'resnet18_4'
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

def test_small_resnet_on_svhn() -> None:
    batch_size = 128
    num_workers = 16
    svhn = SVHNDataset(num_workers=num_workers)
    # cifar10 = FashionMNISTDataset()
    train_dataloader = svhn.get_train_dataloader(batch_size=batch_size)
    val_dataloader = svhn.get_val_dataloader(batch_size=batch_size)
    test_dataloader = svhn.get_test_dataloader(batch_size=batch_size)

    tag = 'resnet18_svhn_6'
    net = SmallResNet18()

    lr: float = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    warmup_epochs = 5
    num_epochs = 50
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs-warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )
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
    pretrained_model = trainer.model

    layers = ['net.last.AdaptiveMaxPool2d', 
              'net.last.LazyLinear1', 
              'net.last.BatchNorm', 
              'net.last.ReLU',
              'net.last.Dropout']
    hook = utils.RegisterIntermediateOutputHook(model=pretrained_model, layers=layers)

    # forward
    intermediate_outputs: dict[str, list[Tensor]] = {}
    for x, y in tqdm(svhn.get_test_dataloader(batch_size=batch_size)):
        x = x.to(device)
        y = pretrained_model(x)
        
        output = hook.get_intermediate_output()
        for layer in layers:
            if layer not in intermediate_outputs:
                intermediate_outputs[layer] = []
            intermediate_outputs[layer].append(output[layer])
   
    for layer in layers:
        outputs = intermediate_outputs[layer]
        outputs = torch.cat(outputs, dim=0)
        utils.draw_tsne(data=outputs, labels=svhn.test_dataset.labels, name=f'tsne_{layer}_resnet18_svhn_3.png')

def small_resnet_tsne_impl(dataset, tag: str, device, data, labels, train_data) -> None:
    batch_size = 128
    train_dataloader = dataset.get_train_dataloader(batch_size=batch_size)
    val_dataloader = dataset.get_val_dataloader(batch_size=batch_size)
    test_dataloader = dataset.get_test_dataloader(batch_size=batch_size)

    model = SmallResNet18()

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 5
    num_epochs = 0
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs-warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    trainer = training.TrainerV2(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device)

    trainer.train(tag=tag)
    pretrained_model = trainer.model

    layers = ['net.last.AdaptiveMaxPool2d', 
              'net.last.LazyLinear1', 
              'net.last.BatchNorm', 
              'net.last.ReLU',
              # 'net.last.Dropout'
              ]
    hook = utils.RegisterIntermediateOutputHook(model=pretrained_model, layers=layers)
    
    # # forward
    intermediate_outputs: dict[str, list[Tensor]] = {}
    for x, y in tqdm(dataset.get_test_dataloader(batch_size=batch_size)):
        x = x.to(device)
        y = pretrained_model(x)
        
        output = hook.get_intermediate_output()
        for layer in layers:
            if layer not in intermediate_outputs:
                intermediate_outputs[layer] = []
            intermediate_outputs[layer].append(output[layer])
    # draw tsne
    # for layer in layers:
    #     outputs = intermediate_outputs[layer]
    #     outputs = torch.cat(outputs, dim=0)
    #     utils.draw_tsne(data=data, labels=labels, name=f'tsne_{layer}_resnet18_{tag}.png')

    # 算一下欧氏距离 然后输出前五个最相似的图片
    train_index = 4514
    # 不行 不能从dataloader中拿图片 因为那样是随机的 我们不知道这张图片到底是那个index
    # images, labels = next(iter(train_dataloader))
    # assert images.shape == (batch_size, 3, 32, 32)
    
    
    image = train_data[train_index][0].unsqueeze(0).to(device)
    y = pretrained_model(image)
    output = hook.get_intermediate_output()['net.last.ReLU']
    # 然后我们和整个测试的fc1的输出计算出他们的欧氏距离
    # 然后取top5
    test_fc1_output = intermediate_outputs['net.last.ReLU']
    test_fc1_output = torch.concat(test_fc1_output, dim=0)
    norms = torch.linalg.vector_norm(test_fc1_output - output, ord=2, dim=1) 
    assert norms.shape == (10000,)
    values, indicies = (-norms).topk(k=5)
    print(-values, indicies)


def test_small_resnet_tsne_on_mnist():
    mnist=MNISTDataset(num_workers=8)
    small_resnet_tsne_impl(dataset=mnist, 
                           tag='resnet_tsne_mnist_1', 
                           device=torch.device('cuda:0'),
                           data=mnist.testing_data.data,
                           labels=mnist.testing_data.targets,
                           train_data=mnist.training_data)

def test_small_resnet_tsne_on_fashion_mnist():
    fashion_mnist = FashionMNISTDataset(num_workers=8)
    small_resnet_tsne_impl(dataset=fashion_mnist, 
                           tag='resnet_tsne_fashion_mnist_1', 
                           device=torch.device('cuda:1'),
                           data=fashion_mnist.testing_data.data,
                           labels=fashion_mnist.testing_data.targets,
                           train_data=fashion_mnist.training_data)
    
def test_small_resnet_tsne_on_cifar10():
    # 我笑了，根本就分不开啊，怪不得正确率上不去，resnet18对于这个问题来说太浅了，应该在深一点
    # 比如resnet110, 但是我现在就不浪费时间去训练了，该谢谢论文搞BERT了
    cifar10 = CIFAR10Dataset(num_workers=8)
    small_resnet_tsne_impl(dataset=cifar10, 
                           tag='resnet_tsne_cifar10_2', 
                           device=torch.device('cuda:2'),
                           data=torch.tensor(cifar10.cifar_test.data),
                           labels=torch.tensor(cifar10.cifar_test.targets),
                           train_data=cifar10.cifar_train)

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


def test_tsne() -> None:
    # load the model
    batch_size = 128
    num_workers = 16
    svhn = SVHNDataset(num_workers=num_workers)
    # cifar10 = FashionMNISTDataset()
    train_dataloader = svhn.get_train_dataloader(batch_size=batch_size)
    val_dataloader = svhn.get_val_dataloader(batch_size=batch_size)
    test_dataloader = svhn.get_test_dataloader(batch_size=batch_size)

    # 或许可以用环境变量来决定做测试还是做训练
    # 过拟合非常非常严重啊
    # 或许是batch_size太大了 数据太少了
    # 改了batch_size也没用
    # 或许是cifar10对lenet来说太难了
    # 果然是这样 同样的网络结果 换一个数据集 结果就好很多了 看来还是得上resnet呀
    # 实在是太容易过拟合了 数据增强 启动！
    lr: float = 0.1
    num_epochs = 0
    tag = 'resnet18_svhn_2'
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
    
    model = trainer.model
    # 模型的输出是什么？
    # 我们随便拿一张图片
    # 然后看看这个图片的输出是什么
    
    # 从训练集中随机选取一张图片
    image = svhn.train_dataset[0][0]
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.to(device)
    y = model(image)
    print(y.shape)
    
