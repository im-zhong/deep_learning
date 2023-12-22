# 2023/12/22
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch.data.wikitext2 import WikiText2, WikiText2Example
from mytorch.training import TrainerV2
from module.nlp.bert import BERT, BERTV2
from mytorch import utils
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def test_train_bert() -> None:
    batch_size = 128
    num_workers = 8
    wikitext2 = WikiText2(root='datasets/wikitext-2', split='train')
    train_dataloader = wikitext2.get_dataloader(
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = WikiText2(
        root='datasets/wikitext-2', split='valid').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = WikiText2(
        root='datasets/wikitext-2', split='test').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hidden_size = 512
    max_len = 256
    num_head = 16
    ffn_hidden_size = 1024
    num_blocks = 2
    model = BERT(
        vocab_size=len(wikitext2.vocabulary),
        hidden_size=hidden_size,
        max_len=max_len,
        num_head=num_head,
        ffn_hidden_size=ffn_hidden_size,
        num_blocks=num_blocks
    )

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 100
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            # start_factor就是最开始的时候的学习率 = optimizer.lr*start_factor
            # 在total_iters之前，学习率线性增加，直到学习达到lr
            LinearLR(optimizer=optimizer, start_factor=0.1,
                     total_iters=warmup_epochs),
            # 学习率从lr开始余弦下降
            CosineAnnealingLR(optimizer=optimizer,
                              T_max=num_epochs-warmup_epochs)
        ],
        # milestones就是当epoch到达warmup_epochs的时候切换到下一个scheduler
        milestones=[warmup_epochs]
    )

    device = utils.get_device()
    trainer = TrainerV2(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device,
    )

    trainer.train(tag='bert_1', summary=False)


def test_train_bert_2() -> None:
    batch_size = 256
    num_workers = 8
    max_len = 256
    wikitext2 = WikiText2(root='datasets/wikitext-2', split='train')
    train_dataloader = wikitext2.get_dataloader(
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = WikiText2(
        root='datasets/wikitext-2', split='valid').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = WikiText2(
        root='datasets/wikitext-2', split='test').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hidden_size = 512

    num_head = 16
    ffn_hidden_size = 1024
    num_blocks = 2
    model = BERTV2(
        vocab_size=len(wikitext2.vocabulary),
        hidden_size=hidden_size,
        max_len=max_len,
        num_head=num_head,
        ffn_hidden_size=ffn_hidden_size,
        num_blocks=num_blocks
    )

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 100
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            # start_factor就是最开始的时候的学习率 = optimizer.lr*start_factor
            # 在total_iters之前，学习率线性增加，直到学习达到lr
            LinearLR(optimizer=optimizer, start_factor=0.1,
                     total_iters=warmup_epochs),
            # 学习率从lr开始余弦下降
            CosineAnnealingLR(optimizer=optimizer,
                              T_max=num_epochs-warmup_epochs)
        ],
        # milestones就是当epoch到达warmup_epochs的时候切换到下一个scheduler
        milestones=[warmup_epochs]
    )

    device = utils.get_device()
    trainer = TrainerV2(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device,
    )

    trainer.train(tag='bert_2', summary=False)
