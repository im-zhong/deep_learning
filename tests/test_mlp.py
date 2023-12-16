# 2023/12/14
# zhangzhong

import torch
from torch import nn, Tensor
from module.mlp.mlp import MLP
from mytorch.data.cifar10 import CIFAR10Dataset, cifar10_predict
from mytorch.training import TrainerV2
from mytorch import utils
import torchvision  # type: ignore
import torch.utils.data

def test_train_mlp() -> None:
    arch: list[int] = [4096, 2048, 1024, 512, 256]
    output_size = 10
    dropout = 0.01

    model = MLP(
        arch=arch,
        output_size=output_size,
        dropout=dropout
    )
    
    lr: float = 0.1
    num_epochs = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs)
    
    device = utils.get_device()
    batch_size = 128
    num_workers = 16
    cifar10 = CIFAR10Dataset(num_workers=num_workers, splits=[0.99, 0.01])
    trainer = TrainerV2(model=model,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                        train_dataloader=cifar10.get_train_dataloader(batch_size=batch_size),
                        val_dataloader=cifar10.get_val_dataloader(batch_size=batch_size),
                        test_dataloader=cifar10.get_test_dataloader(batch_size=batch_size),
                        scheduler=scheduler,
                        device=device)
    
    tag = 'mlp_3'
    trainer.train(tag=tag)
    
    # predict
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
    ])
    
    pretrained_model: nn.Module = trainer.model
    images = ['imgs/pattern_recognition/J20.jpg',
              'imgs/pattern_recognition/M7.png',
              'imgs/pattern_recognition/101.png'
              ]
    for image in images:
        cifar10_predict(model=pretrained_model, device=device, path=image)