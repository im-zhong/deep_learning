# 2023/11/20
# zhangzhong

import torch
import re
from torch import Tensor
import torch.utils.data
import torch.nn.functional as F
import copy
import random
from . import data


class Vocabulary:
    def __init__(self, raw_text: str, reserved_tokens=['<unk>']):
        # 1. preprocess, 去掉标点符号，转换为小写
        self.text: str = re.sub('[^A-Za-z]+', ' ', raw_text).lower()

        # 2. tokennize, 将单词拆分成单个字母
        self.tokens: list[str] = reserved_tokens + list(self.text)

        # 3. 建立: index -> token, token -> index 的映射
        self.idx_to_token = list(set(self.tokens))
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.idx_to_token)}
        self.corpus = self.build_corpus()

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, indicies):
        return self.idx_to_token[indicies]

    def to_token(self, indicies):
        return self.idx_to_token[indicies]

    def to_index(self, tokens):
        return self.token_to_idx[tokens]

    def build_corpus(self):
        self.corpus = [self.token_to_idx[token] for token in self.tokens]
        return self.corpus

    # embedding layer
    def build_input(self, prompt: str) -> Tensor:

        vocab_size = len(self)
        len_seq = len(prompt)
        # Vocabulary不应该做onehot
        # 这一步应该在LM里面做 内聚性更高
        input = torch.tensor([self.to_index(token)
                             for token in prompt]).reshape(-1, 1)
        assert input.shape == (len_seq, 1)
        embedding = F.one_hot(input, vocab_size).float()
        assert embedding.shape == (len_seq, 1, vocab_size)
        return embedding


class TimeMachineDataset(data.DataManager):  # 继承DataModule蛋用没有
    def __init__(self, num_seq: int = 32):
        # 设计: 这个函数的参数有两个选择，一个是给定原始文件的路径，一个是给出文件的内容
        # 从调用者的角度来看 直接给出文件路径更为简单
        # 因为这个类的设计不存在通用性，所以直接给出文件路径更为简单
        # TODO:
        self.num_seq = num_seq
        filename = 'datasets/timemachine/timemachine.txt'
        self.filename = filename

        with open(self.filename) as f:
            self.raw_text = f.read()
        self.vocab = Vocabulary(self.raw_text)
        self.tokens = self.vocab.tokens
        self.corpus = self.vocab.build_corpus()
        self.initialized = False

    # TODO: 这样设计是不对的，可以参考pytorch的实现，他将不同类型的数据集分开了
    # 比如 train_dataset, val_dataset, 这样再去创建DataLoader就更加合理了
    # def get_train_dataloader(self, batch_size, shuffle=True):
    #     subseqs = torch.tensor([self.corpus[i: i + self.num_seq + 1] for i in range(len(self.corpus) - self.num_seq)])
    #     num_subseqs: int = subseqs.shape[0]
    #     # 70% 作为 train，30% 作为 val
    #     indicies = torch.split(torch.arange(0, len(subseqs)), int(0.7 * len(subseqs)))
    #     train_indicies = indicies[0]
    #     val_indicies = indicies[1]

    #     # corpus = h e l l o, num_seq = 4
    #     #      X = h e l lc
    #     #      y = e l l o
    #     # 所以我们拿num_seq + 1长度的序列，然后X不要最后一列，y不要第一列即可
    #     # 一定要仔细分析，不要越界
    #     self.X, self.y = subseqs[train_indicies, :-1], subseqs[train_indicies, 1:]
    #     return DataLoaderV2(self, tag='train', batch_size=batch_size, shuffle=True)

    # def get_val_dataloader(self, batch_size):
    #     subseqs = torch.tensor([self.corpus[i: i + self.num_seq + 1] for i in range(len(self.corpus) - self.num_seq)])
    #     num_subseqs: int = subseqs.shape[0]
    #     # 70% 作为 train，30% 作为 val
    #     indicies = torch.split(torch.arange(0, len(subseqs)), int(0.7 * len(subseqs)))
    #     train_indicies = indicies[0]
    #     val_indicies = indicies[1]

    #     # corpus = h e l l o, num_seq = 4
    #     #      X = h e l l
    #     #      y = e l l o
    #     # 所以我们拿num_seq + 1长度的序列，然后X不要最后一列，y不要第一列即可
    #     # 一定要仔细分析，不要越界
    #     self.X, self.y = subseqs[val_indicies, :-1], subseqs[val_indicies, 1:]
    #     return DataLoaderV2(self, tag='val', batch_size=batch_size, shuffle=True)

    def get_train_dataset(self) -> data.Dataset:
        self.get_dataset()
        return self.train_dataset

    def get_val_dataset(self) -> data.Dataset:
        self.get_dataset()
        return self.val_dataset

    def get_vocabulary(self) -> Vocabulary:
        return self.vocab

    def get_dataset(self, ratio: float = 0.7):
        if not self.initialized:
            subseqs = torch.tensor([self.corpus[i: i + self.num_seq + 1]
                                   for i in range(len(self.corpus) - self.num_seq)])
            # num_subseqs: int = subseqs.shape[0]
            # 70% 作为 train，30% 作为 val
            indicies = torch.split(torch.arange(
                0, len(subseqs)), int(0.7 * len(subseqs)))
            train_indicies = indicies[0]
            val_indicies = indicies[1]
            self.train_dataset = data.Dataset(
                x=subseqs[train_indicies, :-1], y=subseqs[train_indicies, 1:])
            self.val_dataset = data.Dataset(
                x=subseqs[val_indicies, :-1], y=subseqs[val_indicies, 1:])
            self.initialized = True
        # return train_dataset, val_dataset

    def get_dataloader(self, batch_size: int) -> tuple[data.DataLoaderV2, data.DataLoaderV2]:
        # train_dataset, val_dataset = self.get_dataset()
        train_dataloader = data.DataLoaderV2(
            datamgr=self, tag='train', batch_size=batch_size, shuffle=True)
        val_dataloader = data.DataLoaderV2(
            datamgr=self, tag='val', batch_size=batch_size, shuffle=False)
        return train_dataloader, val_dataloader


def preprocess_english(raw_text: list[str]) -> list[str]:
    text: list[str] = []
    for line in raw_text:
        # r'' stand for raw string, usually use in regex
        # () stand for match group, \1 will match the first group
        # 找到[]内部的所有符号，并且在其前后插入空格
        line = re.sub(r'([?.!,"])', r' \1 ', line)
        # 将所有连续的空格替换为一个空格
        line = re.sub(r'[ ]+', ' ', line)
        # strip: Return a copy of the string with leading and trailing whitespace removed.
        line = line.lower().strip()
        text.append(line)
    return text


def preprocess_chinese(raw_text: list[str]) -> list[str]:
    text: list[str] = []
    for line in raw_text:
        # 中文就在每个符号中间插入空格即可
        # 现在不需要加bos 在后面加
        # line = '<bos> ' + line
        line = ' '.join(line)
        line = line.lower().strip()
        text.append(line)
    return text


class VocabularyV2:
    def __init__(self, text: list[str], reversed_tokens: list[str] = ['<pad>', '<bos>', '<eos>', '<unk>'], min_frequency: int = 2, is_target: bool = False):
        # example:
        # text -> token
        # "hello ." -> ["hello" '.' '<eos>']
        # "你 好 。" -> ["你" "好" "。" '<eos>']
        self.text = text
        # append eos to every line
        # 不应该在这里加eos 否则下面统计词频会加到里面
        # self.tokenized_text: list[list[str]] = [
        #     f'{line} <eos>'.split(' ') for line in text]
        self.tokenized_text = [line.split(' ') for line in text]

        # 统计词频
        # 统计所有的token
        tokens: dict[str, int] = {}
        for line in self.tokenized_text:
            for token in line:
                tokens[token] = tokens.get(token, 0) + 1

        # tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
        # print(tokens)

        filtered_tokens = set()
        for token, freq in tokens.items():
            if freq >= min_frequency:
                filtered_tokens.add(token)

        # print(filtered_tokens)

        # 保证reversed_tokens都不在filtered_tokens中
        for token in reversed_tokens:
            assert token not in filtered_tokens

        # token -> index
        self.index_to_token: list[str] = reversed_tokens + \
            list(filtered_tokens)
        self.token_to_index: dict[str, int] = {
            token: index for index, token in enumerate(self.index_to_token)}
        # print(self.index_to_token)
        # print(self.token_to_index)

        self.corpus: list[list[int]] = self.build_corpus(is_target)

    def to_token(self, index: int) -> str:
        return self.index_to_token[index] if index < len(self.index_to_token) else '<unk>'

    def to_index(self, token: str) -> int:
        return self.token_to_index.get(token, self.unk())

    # TODO: 其实这个函数是可以复用的 你看在构造函数里面和这个函数的逻辑是一样的
    # input: source prompt
    # output: list[list[int]]
    def tokenize(self, prompt: str) -> list[list[int]]:
        text = preprocess_english([prompt])
        text = [line.split(' ') for line in text]  # type: ignore
        return [[self.to_index(token) for token in line] + [self.eos()]
                for line in text]

    def build_corpus(self, is_target: bool) -> list[list[int]]:
        # 我们在这里处理对于 bos eos的增加
        if is_target:
            return [[self.bos()] + [self.to_index(token) for token in line] + [self.eos()]
                    for line in self.tokenized_text]
        else:
            return [[self.to_index(token) for token in line] + [self.eos()]
                    for line in self.tokenized_text]

    def __len__(self):
        return len(self.index_to_token)

    def __getitem__(self, token: str):
        return self.token_to_index[token]

    def bos(self) -> int:
        return self['<bos>']

    def eos(self) -> int:
        return self['<eos>']

    def unk(self) -> int:
        return self['<unk>']

    def pad(self) -> int:
        return self['<pad>']

    def to_string(self, indicies: list[int]) -> str:
        result = [self.to_token(index) for index in indicies]
        return ''.join(result)


def dynamic_padding(seqs: list[list[int]],  max_length: int, vocab: VocabularyV2) -> Tensor:
    # 我们不应该修改seqs，因为seqs会包含对原始数据集的引用
    # 我们应该创建新的aligned_seqs
    ctx_max_length: int = max_length
    ctx_vocab: VocabularyV2 = vocab
    aligned_seqs: list[list[int]] = copy.deepcopy(seqs)
    # 第一步 先简单的实现，不考虑max_length
    max_length = min(ctx_max_length, max([len(seq) for seq in seqs]))
    max_len = max_length
    padding = ctx_vocab.to_index('<pad>')
    # # 将所有长度小于max_length的seq都补上一个特殊的符号: '<pad>'
    # for i, seq in enumerate(aligned_seqs):
    #     if len(seq) < max_length:
    #         # dynamic padding 显然是需要一个额外的对象做支持的
    #         # 我有一个办法 我让dynamic_padding变成Vocabulary的一个成员函数
    #         # 那我要如何在dataloader里面调用呢？？
    #         # TIP: list.append(list) != list += list
    #         # 1. l1.append(l2), len(l1) += 1, l1 = [..., [l2]]
    #         # 2. l1 += l2, len(l1) += l3n(l2), l1 = [..., *l2]
    #         seq += [ctx_vocab.to_index('<pad>')] * (max_length-len(seq))

    #         aligned_seqs[i] = seq + [ctx_vocab.to_index('<pad>')]*(max_length-len(seq))

    #     # elif len(seq) > max_length:
    #     else:
    #         # 这样做是没用的 相当于创建了一个新的变量 没法对seq一开始引用的变量做修改
    #         #
    #         aligned_seqs[i] = seq[:max_length]

    # list comprehension 会创建新的副本 就不需要我们显式的调用deepcopy了
    aligned_seqs = [seq[:max_len] if len(seq) > max_len
                    else seq + [padding]*(max_len-len(seq))
                    for seq in seqs]

    # 检查所有的sequence的长度都等于 max_length
    for seq in aligned_seqs:
        assert len(seq) == max_length

    # 将整个seqs变成一个tensor
    # shape = (batch_size, num_seq)
    # TIP: data_collator返回的vector需要时整形的，这样可以和embedding进行配合
    return torch.tensor(aligned_seqs, dtype=torch.long)


class DynamicPadding:
    def __init__(self, source_vocab: VocabularyV2, target_vocab: VocabularyV2, max_len: int):
        # self.vocab = vocab
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __call__(self, seqs: tuple[list[list[int]], list[list[int]]]):
        # return dynamic_padding(seqs, self.max_len, self.vocab)
        source, target = seqs

        padded_source = dynamic_padding(
            source, self.max_len, self.source_vocab)
        padded_target = dynamic_padding(
            target, self.max_len, self.target_vocab)
        return padded_source, padded_target[:, :-1], padded_target[:, 1:]


class TranslationDataset(data.Dataset):
    def __init__(self, source: list[list[int]], target: list[list[int]]):
        self.source = source
        # encoder的输入是target的去掉最后一个token的seq
        # encoder的ground truch 是target去掉第一个token的seq
        self.target = target
        self.shuffle_indicies = list(range(len(self.source)))

    def __len__(self) -> int:
        assert len(self.source) == len(self.target)
        return len(self.source)

    # 这个的返回也是不对的
    # Dataset返回的东西是没有经过padding的 所以实际上这个时候label还没有生成
    # 真正的label是由data_collator生成的
    # get item返回的东西会直接给到data collator 也就是 dynamic padding
    def __getitem__(self, indicies):
        # return self.source[indicies], self.target[indicies]
        source = [self.source[i] for i in self.shuffle_indicies[indicies]]
        target = [self.target[i] for i in self.shuffle_indicies[indicies]]
        return source, target
    # def data(self):
    #     return self.source, self.target, self.label

    def shuffle(self):
        # This function shuffles the elements of a list in place,
        # 卧槽 不对！
        # 我们必须让两者的shuffle indicies一样
        # 所以必须指定同一个seed
        # TODO: 我们最好验证一下
        # seed = random.random()
        # random.shuffle(self.source, seed)
        # random.shuffle(self.target, seed)
        self.shuffle_indicies = random.sample(
            list(range(len(self))), len(self))


class TranslationDataManager(data.DataManager):
    def __init__(self) -> None:
        super().__init__()
        # under the pytest, this file should be found at the root dir
        # so we could download it and put it in the data/ folder
        self.filename = 'datasets/translation/cmn.txt'
        self.source: list[str] = []
        self.target: list[str] = []
        self.initialized = False

        with open(self.filename) as f:
            for line in f:
                seqs = line.split('\t')
                self.source.append(seqs[0])
                self.target.append(seqs[1])

        # 因为数据的preprocess和data_collator是在不同的地方对数据所处理
        # 那么我们就需要分析，数据在这些地方分别要处理到什么程度
        # 那么从行为上分析，preprocess这个过程只会发生一次
        # 所以那些需要对数据做一次处理的过程应该放在preprocess里面
        # 而data_collator发生在每个batch里面，所以应该尽可能的少做处理，或者说只做必要的处理
        self.source = preprocess_english(self.source)
        self.target = preprocess_chinese(self.target)

        self.source_vocab = VocabularyV2(self.source)
        self.target_vocab = VocabularyV2(self.target)

        self.source_corpus = self.source_vocab.build_corpus(is_target=False)
        self.target_corpus = self.target_vocab.build_corpus(is_target=True)

        pass

    def get_dataset(self, ratio: float = 0.7):
        if not self.initialized:
            # num_subseqs: int = subseqs.shape[0]
            # 70% 作为 train，30% 作为 val

            # indicies = torch.split(torch.arange(
            #     0, len(self.source_corpus)), int(0.7 * len(self.source_corpus)))
            # train_indicies = indicies[0]
            # val_indicies = indicies[1]
            split = int(0.7 * len(self.source_corpus))
            self.train_dataset = TranslationDataset(
                source=self.source_corpus[:split], target=self.target_corpus[:split])
            self.val_dataset = TranslationDataset(
                source=self.source_corpus[split:], target=self.target_corpus[split:])
            self.initialized = True

    def get_dataloader(self, batch_size: int) -> tuple[data.DataLoaderV2, data.DataLoaderV2]:
        # train_dataset, val_dataset = self.get_dataset()
        train_dataloader = data.DataLoaderV2(
            datamgr=self, tag='train', batch_size=batch_size, shuffle=True, data_collator=DynamicPadding(self.source_vocab, self.target_vocab, 128))
        val_dataloader = data.DataLoaderV2(
            datamgr=self, tag='val', batch_size=batch_size, shuffle=False, data_collator=DynamicPadding(self.source_vocab, self.target_vocab, 128))
        return train_dataloader, val_dataloader

    def get_train_dataset(self) -> data.Dataset:
        self.get_dataset()
        return self.train_dataset

    def get_val_dataset(self) -> data.Dataset:
        self.get_dataset()
        return self.val_dataset
