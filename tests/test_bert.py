# 2023/12/22
# zhangzhong

import torch
from torch import nn, Tensor
from mytorch.data.wikitext2 import WikiText2, WikiText2Example
from mytorch.training import TrainerV2
from module.nlp.bert import BERT
from mytorch import utils


def test_train_bert():
    batch_size = 256
    num_workers = 8
    wikitext2 = WikiText2(root='datasets/wikitext-2', split='train')
    train_dataloader = wikitext2.get_dataloader(
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = WikiText2(
        root='datasets/wikitext-2', split='valid').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = WikiText2(
        root='datasets/wikitext-2', split='test').get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hidden_size = 128
    max_len = 512
    num_head = 8
    ffn_hidden_size = 256
    num_blocks = 2
    model = BERT(
        vocab_size=len(wikitext2.vocabulary),
        hidden_size=hidden_size,
        max_len=max_len,
        num_head=num_head,
        ffn_hidden_size=ffn_hidden_size,
        num_blocks=num_blocks
    )

    device = utils.get_device()
    trainer = TrainerV2(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        num_epochs=100,
        train_dataloader=train_dataloader,

        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=None,
        device=device
    )

    trainer.train(tag='bert_1', summary=False)
