# 2023/11/20
# zhangzhong

from mytorch.data.seq import TimeMachineDataset
from mytorch.net.rnn import RNNScratch, MyLSTM, MyRNN, MyDeepGRU, MyBiGRU, MyGRU
from mytorch.net.lm import LanguageModelScratch, LanguageModel
from mytorch import training, losses, optim
import torch.nn


def test_RNNLM():

    time_machine = TimeMachineDataset(num_seq=16)
    vocab = time_machine.get_vocabulary()

    rnn = RNNScratch(vocab_size=len(vocab), num_hidden=64)
    lm = LanguageModelScratch(vocab=vocab, rnn=rnn)

    batch_size = 32
    num_epochs = 2
    learning_rate = 0.01
    gradient_clip = 1.0
    train_dataloader, val_dataloader = time_machine.get_dataloader(
        batch_size=batch_size)

    trainer = training.Trainer(
        model=lm,
        loss_fn=losses.CrossEntropyLoss(),
        optimizer=optim.SGD(params=lm.parameters(),
                            lr=learning_rate, gradient_clip=gradient_clip),
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )

    trainer.train(tag='RNN Scratch')

    print(lm.predict('hello', num_seq=10))
