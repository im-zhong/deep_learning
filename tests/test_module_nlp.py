# 2023/11/20
# zhangzhong


from module.nlp.attention_seq2seq import AttentionDecoder, AttentionEncoder
from mytorch import optim, training
from mytorch.data.seq import TranslationDataManager
from mytorch.net.seq2seq import Seq2Seq, Seq2SeqLoss


def test_seq2seqwithattention():
    trans = TranslationDataManager()
    source_vocab = trans.source_vocab
    target_vocab = trans.target_vocab

    # encoder, decoder两者的embed_size是可以不一样的
    embed_size = 512
    hidden_size = 256
    num_layers = 3
    encoder = AttentionEncoder(
        vocab_size=len(source_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab=source_vocab,
    )
    decoder = AttentionDecoder(
        vocab_size=len(target_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)  # type: ignore

    batch_size = 32
    num_epochs = 2
    learning_rate = 0.01
    gradient_clip = 1.0

    train_dl, val_dl = trans.get_dataloader(batch_size=batch_size)
    trainer = training.Trainer(
        model=seq2seq,
        # loss_fn=losses.CrossEntropyLoss(),
        loss_fn=Seq2SeqLoss(target_vocab),
        optimizer=optim.MySGD(
            params=seq2seq.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
    )

    trainer.train(tag="Seq2Seq")

    source = "hello, world!"
    target = seq2seq.predict(trans=trans, prompt=source)
    print(f"source: {source}, target: {target}")
