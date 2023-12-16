# 2023/12/13
# zhangzhong

import torch
from torch import nn

from module.vision import vit
from mytorch import utils
from mytorch.data.cifar10 import CIFAR10Dataset, cifar10_predict
from mytorch.training import TrainerV2


def test_patch_embedding():
    hidden_size = 64
    patch_size = 8
    kernel_size = patch_size
    stride = patch_size
    pe = vit.PatchEmbedding(hidden_size=hidden_size,
                            kernel_size=kernel_size,
                            stride=stride)
    batch_size = 4
    in_channels = 3
    height = 32
    width = 32
    seq_size = height * width / (patch_size * patch_size)
    images = torch.randn(size=(batch_size, in_channels, height, width))
    embeddings = pe(images)
    print(embeddings.shape)
    assert embeddings.shape == (batch_size, seq_size, hidden_size)


def test_positional_embedding():
    hidden_size = 64
    max_len = 17
    pe = vit.PositionalEmbedding(hidden_size=hidden_size,
                                 max_len=max_len)
    batch_size = 4
    seq_size = 17
    inputs = torch.randn(size=(batch_size, seq_size, hidden_size))
    outputs = pe(inputs)
    assert outputs.shape == inputs.shape


def test_vit_block():
    hidden_size = 64
    num_heads = 8
    dropout = 0.5
    mlp_hidden_size = 128
    block = vit.ViTBlock(hidden_size=hidden_size,
                         num_heads=num_heads,
                         dropout=dropout,
                         mlp_hidden_size=mlp_hidden_size)
    batch_size = 4
    seq_size = 17
    inputs = torch.randn(size=(batch_size, seq_size, hidden_size))
    outputs = block(inputs)
    assert inputs.shape == outputs.shape


def test_vit():
    hidden_size = 64
    patch_size = 8
    kernel_size = patch_size
    stride = patch_size
    num_heads = 8
    dropout = 0.5
    mlp_hidden_size = 128
    batch_size = 4
    in_channels = 3
    height = 32
    width = 32
    max_len = 17
    num_blocks = 2
    seq_size = height * width / (patch_size * patch_size)
    images = torch.randn(size=(batch_size, in_channels, height, width))

    myvit = vit.ViT(hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    max_len=max_len,
                    dropout=dropout,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    mlp_hidden_size=mlp_hidden_size)

    outputs = myvit(images)
    assert outputs.shape == (batch_size, hidden_size)


def test_vit_classifier():
    hidden_size = 64
    patch_size = 8
    kernel_size = patch_size
    stride = patch_size
    num_heads = 8
    dropout = 0.5
    mlp_hidden_size = 128
    batch_size = 4
    in_channels = 3
    height = 32
    width = 32
    max_len = 17
    num_blocks = 2
    output_size = 10

    images = torch.randn(size=(batch_size, in_channels, height, width))
    classifier = vit.ViTClassifier(hidden_size=hidden_size,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   max_len=max_len,
                                   dropout=dropout,
                                   num_blocks=num_blocks,
                                   num_heads=num_heads,
                                   mlp_hidden_size=mlp_hidden_size,
                                   output_size=output_size)

    outputs = classifier(images)
    assert outputs.shape == (batch_size, output_size)


def test_train_vit_classifier() -> None:
    hidden_size = 256
    patch_size = 8
    kernel_size = patch_size
    stride = patch_size
    num_heads = 8
    dropout = 0.01
    mlp_hidden_size = 1024
    max_len = 17
    num_blocks = 10
    output_size = 10
    model = vit.ViTClassifier(hidden_size=hidden_size,
                              kernel_size=kernel_size,
                              stride=stride,
                              max_len=max_len,
                              dropout=dropout,
                              num_blocks=num_blocks,
                              num_heads=num_heads,
                              mlp_hidden_size=mlp_hidden_size,
                              output_size=output_size)

    lr: float = 0.1
    num_epochs = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler is better than Adam
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs)

    
    device = utils.get_device()
    batch_size = 128
    num_workers = 16
    cifar10 = CIFAR10Dataset(num_workers=num_workers)
    trainer = TrainerV2(model=model,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                        train_dataloader=cifar10.get_train_dataloader(batch_size=batch_size),
                        val_dataloader=cifar10.get_val_dataloader(batch_size=batch_size),
                        test_dataloader=cifar10.get_test_dataloader(batch_size=batch_size),
                        scheduler=scheduler,
                        device=device)

    tag = 'vit4'
    trainer.train(tag=tag)
    
    # predict
    pretrained_model: nn.Module = trainer.model
    images = ['imgs/pattern_recognition/J20.jpg',
              'imgs/pattern_recognition/M7.png',
              'imgs/pattern_recognition/101.png'
              ]
    for image in images:
        cifar10_predict(model=pretrained_model, device=device, path=image)
    

    
