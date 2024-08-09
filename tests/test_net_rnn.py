# 2023/11/20
# zhangzhong

import torch.nn

from mytorch import losses, optim, training
from mytorch.data.seq import TimeMachineDataset
from mytorch.net.lm import LanguageModel, LanguageModelScratch
from mytorch.net.rnn import MyBiGRU, MyDeepGRU, MyGRU, MyLSTM, MyRNN, RNNScratch


def test_RNN_scratch():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    rnn = RNNScratch(vocab_size=len(vocab), num_hidden=64)
    lm = LanguageModelScratch(vocab=vocab, rnn=rnn)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="RNN Scratch")


def test_language_model():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = torch.nn.RNN(input_size=len(vocab), hidden_size=64, nonlinearity="relu")
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=64)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="RNN LM")
    print(lm.predict(prompt="hello", max_len=32))


def test_mylstm():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyLSTM(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_mygru():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyGRU(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 1
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_myrnn():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyRNN(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 1
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_lstm():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    rnn = torch.nn.LSTM(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_gru():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = torch.nn.GRU(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_my_deep_GRU():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyDeepGRU(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_my_bi_GRU():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 64
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyBiGRU(input_size=len(vocab), hidden_size=hidden_size)
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 32
    num_epochs = 0
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LSTM")

    print(lm.predict(prompt="hello", max_len=32))


def test_my_deep_bi_GRU():
    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    hidden_size = 128
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    rnn = MyDeepGRU(
        input_size=len(vocab), hidden_size=hidden_size, num_layers=2, bidirectional=True
    )
    lm = LanguageModel(vocab=vocab, rnn=rnn, hidden_size=hidden_size)

    batch_size = 64
    num_epochs = 2
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size
    )

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.MySGD(
            params=lm.parameters(), lr=learning_rate, gradient_clip=gradient_clip
        ),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="DeepBiGRU")
