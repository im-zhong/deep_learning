# 2023/11/21
# zhangzhong

from mytorch.data.seq import TimeMachineDataset, TranslationDataManager
from mytorch.net.rnn import RNNScratch, MyLSTM, MyRNN, MyDeepGRU, MyBiGRU, MyGRU
from mytorch.net.lm import LanguageModelScratch, LanguageModel
from mytorch import training, losses, optim
import torch.nn
from mytorch.net.seq2seq import Encoder, Decoder, MyEmbedding, Seq2Seq, Seq2SeqLoss
from mytorch.net.transformer import TransformerEncoder, TransformerDecoder


def test_Transformer():
    # TODO:
    # 1.DONE 目前的实现肯定是不对的，至少MaskedDotProductAttention对valid_lens的处理肯定不对
    # 我们需要处理valid_lens的两种形式，分别对应encoder和decoder
    # 2.DONE 看看要不要想方法修改一下书上的transformer代码 可以接受一个参数矩阵 这样可以验证我们的实现是否正确
    # 但是这样太花时间了 还是尽可能的测试组件是否正确 然后认真检查对比代码的实现吧
    # 3. 当然还有predict 也就是decoder中对state的处理 实际上非常简单 就是在state中保存每一层的历史输入即可
    # 然后呢 这些历史输入有什么用？应该是用在第一层的self attention，因为在predict的时候，decoder一次只能拿到一个
    # token的向量，但是我们需要历史输入来做啊attention 这样每次把当前的xt作为query，历史数据作为kv 就欧克了
    trans = TranslationDataManager()
    source_vocab = trans.source_vocab
    target_vocab = trans.target_vocab

    # encoder, decoder两者的embed_size是可以不一样的
    embed_size = 512
    hidden_size = 256
    ffn_hidden_size = 64
    # num_layers = 3
    num_blocks = 2
    num_head = 4
    encoder = TransformerEncoder(vocab_size=len(
        source_vocab), hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, num_head=num_head, num_blocks=num_blocks, vocab=source_vocab)
    decoder = TransformerDecoder(vocab_size=len(
        target_vocab), hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, num_head=num_head, num_blocks=num_blocks, vocab=target_vocab)
    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder)  # type: ignore

    batch_size = 128
    num_epochs = 2
    learning_rate = 0.0015
    gradient_clip = 1.0

    train_dl, val_dl = trans.get_dataloader(batch_size=batch_size)
    trainer = training.Trainer(
        model=seq2seq,
        # loss_fn=losses.CrossEntropyLoss(),
        loss_fn=Seq2SeqLoss(target_vocab),
        # optimizer=optim.SGD(params=seq2seq.parameters(),
        #                     lr=learning_rate, gradient_clip=gradient_clip),
        optimizer=torch.optim.Adam(
            params=seq2seq.parameters(), lr=learning_rate),
        num_epochs=num_epochs,
        train_dataloader=train_dl,
        val_dataloader=val_dl
    )

    trainer.train(tag='Transformer')

    source = 'hello, world!'
    target = seq2seq.predict(trans=trans, prompt=source)
    print(f'source: {source}, target: {target}')
