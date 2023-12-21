# 2023/12/21
# zhangzhong
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import os
import random

# TODO: 为了做tokennize，数据库本身必须提供copurs
# 但是这样这个类的职责太多了，Copurs由另外一个类来提供


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

    def __getitem__(self, index):
        # TODO: 需要转成Tensor吗？如果要转成Tensor就需要实现Vocabulary和tokenize
        x = self.nsp_examples[index][0]
        z = self.nsp_examples[index][1]
        y = self.nsp_labels[index]
        return x, z, y
        # return self.nsp_examples[index], self.nsp_labels[index]

    def __len__(self) -> int:
        # The __len__ function returns the number of samples in our dataset.
        return len(self.nsp_examples)

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
