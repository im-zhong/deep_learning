# 2023/12/16
# zhangzhong

import torch
from torch import Tensor, nn

from mytorch import utils


def test_top_one_error_rate():
    batch_size = 4
    num_classes = 5
    logits = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    ground_truth = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([1, 1, 2, 4])
    error = utils.top1_error_rate(logits=logits, labels=labels)
    assert error == 2


def test_top_five_error_rate():
    batch_size = 8
    num_classes = 10
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0, 1, 2, 3, 4
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # 1, 2, 3, 4, 5
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0],  # 2, 3, 4, 5, 6
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],  # 3, 4, 5, 6, 7
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0],  # 4, 5, 6, 7, 8
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # 5, 6, 7, 8, 9
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0, 1, 2, 3, 4
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # 1, 2, 3, 4, 5
        ]
    )
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    error = utils.top5_error_rate(logits=logits, labels=labels)
    assert error == 2


def test_topk_err():
    batch_size = 4
    num_classes = 5
    logits = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    ground_truth = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([1, 1, 2, 4])
    error = utils.topk_err(k=1, logits=logits, labels=labels)
    assert error == 2 / 4

    batch_size = 8
    num_classes = 10
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0, 1, 2, 3, 4
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # 1, 2, 3, 4, 5
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0],  # 2, 3, 4, 5, 6
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],  # 3, 4, 5, 6, 7
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0],  # 4, 5, 6, 7, 8
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # 5, 6, 7, 8, 9
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0, 1, 2, 3, 4
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # 1, 2, 3, 4, 5
        ]
    )
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    error = utils.topk_err(k=5, logits=logits, labels=labels)
    assert error == 2 / batch_size
