# 2023/12/21
# zhangzhong
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import os
import random
from mytorch.data.seq import VocabularyV3
from dataclasses import dataclass


# TODO: 这个函数应该放到 func.py 里面
# 重构dynamic padding


@dataclass
class WikiText2Item:
    sentence: list[int]
    segment: list[int]
    label: bool


# TODO: 想一个更贴切的名字 example, observation, instance, sample, feature,
@dataclass
class WikiText2Example:
    sentences: Tensor
    segments: Tensor
    valid_lens: Tensor


# 应该在这里返回valid_lens
def dynamic_padding(seqs: list[list[int]],  max_len: int, pad: int) -> tuple[Tensor, Tensor]:
    max_len = min(max_len, max([len(seq) for seq in seqs]))
    valid_lens = [len(seq) if len(seq) < max_len else max_len
                  for seq in seqs]
    aligned_seqs = [seq[:max_len] if len(seq) > max_len
                    else seq + [pad]*(max_len-len(seq))
                    for seq in seqs]

    for seq in aligned_seqs:
        assert len(seq) == max_len
    assert len(valid_lens) == len(aligned_seqs)
    return torch.tensor(aligned_seqs, dtype=torch.long), torch.tensor(valid_lens, dtype=torch.long)


class DynamicPadding:
    def __init__(self, vocabulary: VocabularyV3, max_len: int):
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.segment_vocabulary = VocabularyV3(
            text='', reversed_tokens=['<tmp>', '<pad>'], min_frequency=0)

    # TODO: 还缺一个东西，valid_lens
    # 我懂了，collate_fn的输出是整个dataset的输出 也就是一个tuple
    def __call__(self, batch: list[WikiText2Item]) -> tuple[WikiText2Example, Tensor]:
        # 你觉得这样的代码写出来看得懂吗？
        # 你怎么能直到0是哪个1是哪个？万一后面咱们调换了顺序 换了名字 加了东西
        # 你要怎么改？
        # what we should do is 用一个结构体把dataset的返回类型给包装起来
        # 应该用python的 dataclass
        sentences: list[list[int]] = [item.sentence for item in batch]
        segments: list[list[int]] = [item.segment for item in batch]
        labels: list[bool] = [item.label for item in batch]
        padded_sentences, valid_lens = dynamic_padding(seqs=sentences, max_len=self.max_len,
                                                       pad=self.vocabulary.pad())
        padded_segments, _ = dynamic_padding(seqs=segments, max_len=self.max_len,
                                             pad=self.segment_vocabulary.pad())
        # 两者的shape必须一致
        assert padded_sentences.shape == padded_segments.shape
        # change labels to tensor
        tlabels = torch.tensor(labels, dtype=torch.long)
        # 在返回数据之前尽可能检查数据的正确性
        assert padded_sentences.shape[0] == tlabels.shape[0]
        # 这里必须返回一对数据 (x, y)
        # 而且x也应该用dataclass包装起来
        return WikiText2Example(sentences=padded_sentences,
                                segments=padded_segments,
                                valid_lens=valid_lens), tlabels


class WikiText2(Dataset):
    def __init__(self, root: str, split: str, num_workers: int = 0):
        self.root = root
        assert split in ["train", "valid", "test"]
        self.split = split
        lines = self.read_wiki(os.path.join(root, f'wiki.{split}.tokens'))
        paragraphs = self.gen_paragraphs(lines)
        nsp = self.gen_nsp(paragraphs)
        self.nsp_examples, self.nsp_labels = self.add_segment_and_make_token(
            nsp)

        # make paragraphs to str
        self.vocabulary = VocabularyV3(text=' '.join([' '.join(paragraph) for paragraph in paragraphs]),
                                       reversed_tokens=[
                                           '<pad>', '<unk>', '<bos>', '<eos>', '<cls>', '<seq>'],
                                       min_frequency=5)

    def __getitem__(self, index) -> WikiText2Item:
        # TODO: 需要转成Tensor吗？如果要转成Tensor就需要实现Vocabulary和tokenize
        sentence = self.nsp_examples[index][0]
        # 我们需要在这里对x进行tokenize
        sentence = self.vocabulary.tokenize(sentence)
        segment = self.nsp_examples[index][1]
        label = self.nsp_labels[index]
        return WikiText2Item(sentence=sentence, segment=segment, label=label)
        # return self.nsp_examples[index], self.nsp_labels[index]

    def __len__(self) -> int:
        # The __len__ function returns the number of samples in our dataset.
        return len(self.nsp_examples)

    # 我们让这个类可以直接返回dataloader 这样就更简单了
    def get_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        # 重点关注 data collator 因为我们需要指定dynamic padding
        # !!! collate_fn的输入需要和datasets[indicies]的输出一致 然后collate_fn需要返回一系列tensor
        # 作为整个dataloader的输出
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=DynamicPadding(vocabulary=self.vocabulary, max_len=512))  # type: ignore

    def read_wiki(self, path: str) -> list[str]:
        with open(path, 'r') as f:
            # f.readlines() will read all lines and seperate them with '\n'
            lines = f.readlines()
        return lines

    def gen_paragraphs(self, lines: list[str]) -> list[list[str]]:
        return [line.strip().lower().split(' . ')
                for line in lines if len(line.split(' . ')) > 1]

    def gen_next_sentence(self, sentence, next_sentence, paragraphs) -> tuple[str, str, bool]:
        # Choose a random element from a non-empty sequence.
        if random.choice([True, False]):
            return sentence, next_sentence, True
        else:
            return sentence, random.choice(random.choice(paragraphs)), False

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

    def add_segment_and_make_token(self, nsp: list[tuple[str, str, bool]]) -> tuple[list[tuple[list[str], list[int]]], list[bool]]:
        nsp_examples: list[tuple[list[str], list[int]]] = []
        nsp_labels: list[bool] = []
        for sentence, next_sentence, label in nsp:
            nsp_labels.append(label)
            sentence = sentence.split()
            next_sentence = next_sentence.split()
            tokens = ['<cls>'] + sentence + \
                ['<sep>'] + next_sentence + ['<eos>']
            segments = [0] + [0] * len(sentence) + \
                [0] + [1] * len(next_sentence) + [1]
            assert len(tokens) == len(segments)
            nsp_examples.append((tokens, segments))
        return nsp_examples, nsp_labels
