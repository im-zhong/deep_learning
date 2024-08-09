# 2023/12/22
# zhangzhong

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from module.nlp.bert import BERT, BERTEvaluator, BERTLoss
from mytorch import utils
from mytorch.data.seq import VocabularyV3 as Vocabulary
from mytorch.data.wikitext2v2 import Preprocessor, WikiText2, WikiText2Sample
from mytorch.training import TrainerV2, TrainerV3


def test_train_bert() -> None:
    batch_size = 512
    num_workers = 8
    max_len = 64

    preprocessor = Preprocessor(root="datasets/wikitext-2")
    train_paragraphs = preprocessor(split="train")
    val_paragraphs = preprocessor(split="valid")
    test_paragraphs = preprocessor(split="test")

    vocabulary = Vocabulary(
        text=" ".join([" ".join(paragraph) for paragraph in train_paragraphs]),
        reserved_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<cls>", "<seq>"],
        min_frequency=5,
    )

    train_dataloader = WikiText2(
        paragraphs=train_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = WikiText2(
        paragraphs=val_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = WikiText2(
        paragraphs=test_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hidden_size = 256
    num_head = 4
    ffn_hidden_size = 1024
    num_blocks = 8
    model = BERT(
        vocab_size=len(vocabulary),
        hidden_size=hidden_size,
        max_len=max_len,
        num_head=num_head,
        ffn_hidden_size=ffn_hidden_size,
        num_blocks=num_blocks,
    )

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 100
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )

    # device = utils.get_device()
    device = torch.device("cuda:1")
    trainer = TrainerV2(
        model=model,
        loss_fn=BERTLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device,
    )

    trainer.train(tag="bert_5", summary=False)


def test_train_bert_2() -> None:
    batch_size = 192
    num_workers = 8
    max_len = 64

    preprocessor = Preprocessor(root="datasets/wikitext-2")
    train_paragraphs = preprocessor(split="train")
    val_paragraphs = preprocessor(split="valid")
    test_paragraphs = preprocessor(split="test")

    vocabulary = Vocabulary(
        text=" ".join([" ".join(paragraph) for paragraph in train_paragraphs]),
        reserved_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<cls>", "<seq>"],
        min_frequency=5,
    )

    train_dataloader = WikiText2(
        paragraphs=train_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = WikiText2(
        paragraphs=val_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = WikiText2(
        paragraphs=test_paragraphs, max_len=max_len, vocabulary=vocabulary
    ).get_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)

    hidden_size = 768
    num_head = 12
    ffn_hidden_size = 2048
    num_blocks = 12
    model = BERT(
        vocab_size=len(vocabulary),
        hidden_size=hidden_size,
        max_len=max_len,
        num_head=num_head,
        ffn_hidden_size=ffn_hidden_size,
        num_blocks=num_blocks,
    )

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 100
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )

    evaluator = BERTEvaluator()
    # device = utils.get_device()
    device = torch.device("cuda:1")
    trainer = TrainerV3(
        model=model,
        loss_fn=BERTLoss(),
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        scheduler=scheduler,
        device=device,
        evaluator=evaluator,
    )

    trainer.train(tag="bert_6", summary=False)
