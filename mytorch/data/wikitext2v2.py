# 2023/12/24
# zhangzhong
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import os
import random
from mytorch.data.seq import VocabularyV3 as Vocabulary
from dataclasses import dataclass
from mytorch import func
from mytorch.func import PaddingResult

# CHECK: 所有的train valid test应该公用一个字典，所以tokenize应该发生在最后一步


@dataclass
class WikiText2Sample:
    sentences: Tensor
    segments: Tensor
    padding_mask: Tensor
    masked_indices: Tensor
    # labels不应该出现在sample中 应该在Label里面
    # nsp_labels: Tensor
    # mlm_labels: Tensor
    mlm_mask: Tensor

    def to(self, device: torch.device) -> 'WikiText2Sample':
        return WikiText2Sample(sentences=self.sentences.to(device),
                               segments=self.segments.to(device),
                               padding_mask=self.padding_mask.to(device),
                               masked_indices=self.masked_indices.to(device),
                               #  nsp_labels=self.nsp_labels.to(device),
                               #  mlm_labels=self.mlm_labels.to(device),
                               mlm_mask=self.mlm_mask.to(device))

    # TODO: 正确实现
    def shape(self) -> torch.Size:
        return self.sentences.shape


@dataclass
class WikiText2Label:
    nsp: Tensor
    mlm: Tensor

    def to(self, device: torch.device) -> 'WikiText2Label':
        return WikiText2Label(nsp=self.nsp.to(device),
                              mlm=self.mlm.to(device))


# 模仿pytorch的module
# 我们设计一个DataModule
# 同样我们有forward
# 我们做的就是拿到一个输入 对数据进行处理
# 然后返回处理过的数据


# 我们应该遵从lazy的原则, 不到最后时刻不应该提供多余的信息
# root和split是在构造的时候就需要直到的吗？显然不是
# 因为真正的文件读取信息处理等等都是等到__call__的时候才会发生
# 所以这些参数应该放在__call__里面
# 而且既然我们做的是管线，那么最好前面的输出就是后面的输入
class Preprocessor:
    def __init__(self, root: str):
        self.root = root

    def read_wiki(self, path: str) -> list[str]:
        with open(path, 'r') as f:
            # f.readlines() will read all lines and seperate them with '\n'
            lines: list[str] = f.readlines()
        return lines

    def gen_paragraphs(self, lines: list[str]) -> list[list[str]]:
        return [line.strip().lower().split(' . ')
                for line in lines if len(line.split(' . ')) > 1]

    def __call__(self, split: str) -> list[list[str]]:
        assert split in ["train", "valid", "test"]
        lines: list[str] = self.read_wiki(os.path.join(
            self.root, f'wiki.{split}.tokens'))
        paragraphs: list[list[str]] = self.gen_paragraphs(lines)
        return paragraphs


@dataclass
class NSPItem:
    sentence: list[str]
    segment: list[int]
    label: bool


class NextSentencePrediction:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def gen_next_sentence(self, sentence, next_sentence, paragraphs) -> tuple[str, str, bool]:
        # Choose a random element from a non-empty sequence.
        if random.choice([True, False]):
            return sentence, next_sentence, True
        else:
            return sentence, random.choice(random.choice(paragraphs)), False

    # 这个函数的返回同样很复杂 那么为什么这里不把返回结果组织成一个dataclass呢
    # 因为这个函数只在内部使用 而且返回和使用的代码靠的很近 就不需要
    # 但是__call__函数的返回值需要在外部使用 那我们就无法保证他们靠的很近 就需要组织成一个dataclass
    def gen_nsp(self, paragraphs: list[list[str]]):
        nsp = []
        for paragraph in paragraphs:
            for i in range(len(paragraph) - 1):
                sentence = paragraph[i]
                next_sentence = paragraph[i + 1]
                sentence, next_sentence, label = self.gen_next_sentence(
                    sentence, next_sentence, paragraphs)

                nsp.append((sentence, next_sentence, label))
        return nsp

    def add_segment_and_make_token(self, nsp: list[tuple[str, str, bool]]) -> list[NSPItem]:
        # nsp_examples: list[tuple[list[str], list[int]]] = []
        # nsp_labels: list[bool] = []
        items: list[NSPItem] = []
        for sentence, next_sentence, label in nsp:
            # nsp_labels.append(label)
            sentence1 = sentence.split()
            sentence2 = next_sentence.split()
            tokens: list[str] = ['<cls>'] + sentence1 + \
                ['<sep>'] + sentence2 + ['<eos>']
            segment = [0] + [0] * len(sentence1) + \
                [0] + [1] * len(sentence2) + [1]
            assert len(tokens) == len(segment)
            # 去掉长度过长的句子
            if len(tokens) > self.max_len:
                continue
            items.append(
                NSPItem(sentence=tokens, segment=segment, label=label))
        return items

    # 因为这里的返回元素数量超过了两个 过于复杂 所以应该单独写一个dataclass
    def __call__(self, paragraphs: list[list[str]]) -> list[NSPItem]:
        nsp = self.gen_nsp(paragraphs=paragraphs)
        return self.add_segment_and_make_token(nsp=nsp)


@dataclass
class MLMItem(NSPItem):
    masked_indices: list[int]
    masked_tokens: list[str]


class MaskedLanguageModel:
    def __init__(self, vocabulary: Vocabulary, mask_prob: float = 0.15):
        self.mask_prob = mask_prob
        self.vocabulary = vocabulary

    def __call__(self, paragraphs: list[list[str]], nsp_items: list[NSPItem], vocabulary: Vocabulary) -> list[MLMItem]:
        items: list[MLMItem] = []
        for nsp_item in nsp_items:
            # 这里的主要任务就是给sentence加上mask
            # 首先我们需要随机找到15%的位置用于mask
            masked_indices = self.gen_masked_indices(
                sentence=nsp_item.sentence)

            sentence, masked_tokens = self.mask_sentence(
                sentence=nsp_item.sentence,
                masked_indices=masked_indices,
                vocabulary=vocabulary)

            items.append(MLMItem(
                sentence=sentence,
                segment=nsp_item.segment,
                label=nsp_item.label,
                masked_tokens=masked_tokens,
                masked_indices=masked_indices
            ))
        return items

    def gen_masked_indices(self, sentence: list[str]) -> list[int]:
        masked_token_size: int = round(len(sentence) * self.mask_prob)
        indices_without_special_token: list[int] = [index for index in range(
            len(sentence)) if not self.vocabulary.is_special_token(sentence[index])]
        random.shuffle(indices_without_special_token)
        masked_indices: list[int] = indices_without_special_token[:masked_token_size]
        return masked_indices

    def mask_sentence(self, sentence: list[str], masked_indices: list[int], vocabulary: Vocabulary) -> tuple[list[str], list[str]]:
        masked_tokens: list[str] = []
        for index in masked_indices:
            masked_tokens.append(sentence[index])
            # random() -> x in the interval [0, 1).
            if random.random() < 0.8:
                token = '<mask>'
            elif random.random() < 0.5:
                token = sentence[index]
            else:
                token = random.choice(vocabulary)
            sentence[index] = token
        return sentence, masked_tokens


@dataclass
class WikiText2Item:
    sentence: list[int]
    segment: list[int]
    masked_indices: list[int]
    nsp_label: bool
    mlm_labels: list[int]


class Tokenizer:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, mlm_items: list[MLMItem]) -> list[WikiText2Item]:
        return [
            WikiText2Item(
                sentence=self.vocabulary.tokenize(mlm_item.sentence),
                segment=mlm_item.segment,
                masked_indices=mlm_item.masked_indices,
                nsp_label=mlm_item.label,
                mlm_labels=self.vocabulary.tokenize(mlm_item.masked_tokens)
            ) for mlm_item in mlm_items
        ]


class DynamicPadding:
    def __init__(self, vocabulary: Vocabulary, max_len: int):
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.segment_vocabulary = Vocabulary(
            text='', reserved_tokens=['<tmp>', '<pad>', '<unk>'], min_frequency=0)

    # 我懂了，collate_fn的输出是整个dataset的输出 也就是一个tuple
    def __call__(self, batch: list[WikiText2Item]) -> tuple[WikiText2Sample, WikiText2Label]:
        # 你觉得这样的代码写出来看得懂吗？
        # 你怎么能直到0是哪个1是哪个？万一后面咱们调换了顺序 换了名字 加了东西
        # 你要怎么改？
        # what we should do is 用一个结构体把dataset的返回类型给包装起来
        # 应该用python的 dataclass
        sentences: list[list[int]] = [item.sentence for item in batch]
        segments: list[list[int]] = [item.segment for item in batch]
        nsp_labels = torch.tensor(
            [item.nsp_label for item in batch], dtype=torch.long)

        padded_sentences: PaddingResult = func.dynamic_padding(seqs=sentences, max_len=self.max_len,
                                                               pad=self.vocabulary.pad())
        padded_segments: PaddingResult = func.dynamic_padding(seqs=segments, max_len=self.max_len,
                                                              pad=self.segment_vocabulary.pad())
        # 两者的shape必须一致
        assert padded_sentences.padded_seqs.shape == padded_segments.padded_seqs.shape
        # change labels to tensor
        # tlabels = torch.tensor(labels, dtype=torch.long)
        # 在返回数据之前尽可能检查数据的正确性

        assert padded_sentences.padded_seqs.shape[0] == nsp_labels.shape[0]
        # 这里必须返回一对数据 (x, y)
        # 而且x也应该用dataclass包装起来
        # 从valid_lens生成mask
        batch_size, seq_size = padded_segments.padded_seqs.shape

        # 然后我们同样需要对masked tokens进行padding 因为不同的句子被mask的token个数也是不同的
        masked_indices: list[list[int]] = [
            item.masked_indices for item in batch]
        mlm_labels: list[list[int]] = [item.mlm_labels for item in batch]
        padded_masked_indices: PaddingResult = func.dynamic_padding(seqs=masked_indices, max_len=round(self.max_len * 0.15),
                                                                    pad=0)
        padded_mlm_labels: PaddingResult = func.dynamic_padding(seqs=mlm_labels, max_len=round(self.max_len * 0.15),
                                                                pad=0)
        assert padded_masked_indices.padded_seqs.shape == padded_mlm_labels.padded_seqs.shape

        return WikiText2Sample(
            sentences=padded_sentences.padded_seqs,
            segments=padded_segments.padded_seqs,
            padding_mask=padded_sentences.padding_mask,
            masked_indices=padded_masked_indices.padded_seqs,
            mlm_mask=padded_mlm_labels.padding_weight_mask,
        ), WikiText2Label(nsp=nsp_labels, mlm=padded_mlm_labels.padded_seqs)


class WikiText2(Dataset):
    def __init__(self, paragraphs: list[list[str]], max_len: int, vocabulary: Vocabulary):
        # data pipeline
        # 1. preprocessing
        # preprocessing = Preprocessing()
        # paragraphs: list[list[str]] = preprocessing(root=root, split=split)
        # # 我们应该在这里单独构建这个vocabulary
        # if vocabulary is not None:
        #     self.vocabulary = vocabulary
        # else:
        #     self.vocabulary = Vocabulary(text=' '.join([' '.join(paragraph) for paragraph in paragraphs]),
        #                                  reserved_tokens=[
        #         '<pad>', '<unk>', '<bos>', '<eos>', '<cls>', '<seq>'],
        #         min_frequency=5)
        self.vocabulary = vocabulary
        # 2. next sentence prediction
        nsp = NextSentencePrediction(max_len=max_len)
        nsp_items: list[NSPItem] = nsp(paragraphs=paragraphs)
        # 3. masked language model
        mlm = MaskedLanguageModel(vocabulary=self.vocabulary)
        mlm_items: list[MLMItem] = mlm(paragraphs=paragraphs,
                                       nsp_items=nsp_items,
                                       vocabulary=self.vocabulary)
        # 4. tokenizer
        tokenizer = Tokenizer(vocabulary=self.vocabulary)
        self.items: list[WikiText2Item] = tokenizer(mlm_items=mlm_items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> WikiText2Item:
        # 在转成tokenize吧
        # 但是不是在这里转 还是在数据处理哪里转
        # len和getitem就只是简单的返回数据而已
        return self.items[index]

    def get_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=DynamicPadding(vocabulary=self.vocabulary, max_len=512))  # type: ignore
