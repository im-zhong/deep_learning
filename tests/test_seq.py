# 2023/11/20
# zhangzhong


from mytorch.data import seq

# TODO:
# I need download cmn.txt, which is the dataset


def test_translation():
    dm = seq.TranslationDataManager()
    train_dl, val_dl = dm.get_dataloader(batch_size=32)
    for batch in train_dl:
        source, target, label = batch
        # target, label 的num_seq 必须一样
        batch_size, num_source = source.shape
        _, num_target = target.shape
        _, num_label = label.shape
        assert num_target == num_label

        # 所有的token必须都在0 - vocab_size之间
        source_vocab_size = len(dm.source_vocab)
        target_vocab_size = len(dm.target_vocab)
        # TODO: 让特殊的字符在vocab中占据一样的位置
        for line in source:
            for token in line:
                assert 0 <= token < source_vocab_size
        for line in target:
            assert int(line[0]) == dm.target_vocab.bos()
            for token in line:
                assert 0 <= token < target_vocab_size
        for line in label:
            for token in line:
                assert 0 <= token < target_vocab_size


def test_vocabulary():
    vocabulary = seq.VocabularyV3(text='hello world', reserved_tokens=[
                                  '<pad>', '<unk>', '<bos>', '<eos>', '<cls>', '<seq>'], min_frequency=1)
    print(len(vocabulary))
    for token in vocabulary:
        print(token)

    print(vocabulary.to_index('hello'))
