# 2023/11/21
# zhangzhong


import matplotlib.pyplot as plt
import os.path
import torch
from torch import device, Tensor
import random


def mysavefig(filename: str) -> None:
    os.makedirs(name='imgs', exist_ok=True)
    filename = os.path.join('imgs', filename)
    plt.savefig(filename)


def get_device() -> device:
    # 不行，你不能保证大家都在同一个设备上，代码在不同的地方使用了get_device()
    # 所以所有的训练代码都不能够调用 get_device() 他们只能使用参数传入的device！
    if torch.cuda.is_available():
        # random select a gpu
        # Return random integer in range [a, b], including both end points.
        gpu_id = random.randint(0, torch.cuda.device_count() - 1)
        return device(f'cuda:{gpu_id}')
    else:
        return device('cpu')

def top1_error_rate(logits: Tensor, labels: Tensor):
    # 假设输入是batch的，这是合理的假设
    batch_size, num_classes = logits.shape
    predict_labels = logits.argmax(dim=1)
    assert predict_labels.shape == labels.shape
    # count the error
    return (predict_labels != labels).int().sum().item()

def top5_error_rate(logits: Tensor, labels: Tensor):
    # https://pytorch.org/docs/stable/generated/torch.topk.html
    # Returns the k largest elements of the given input tensor along a given dimension.
    batch_size, num_classes = logits.shape
    assert num_classes >= 5
    _, top5_labels = logits.topk(k=5, dim=1)
    assert top5_labels.shape == (batch_size, 5)
    # correct = torch.isin(labels, top5_labels).int().sum().item()
    # correct = 0
    # for i in range(batch_size):
    #     if labels[i] in top5_labels[i]:
    #         correct += 1
    
    # correct = torch.any(top5_labels.T == labels, dim=0).sum().item()
    # return batch_size - correct
    return torch.all(top5_labels.T != labels, dim=0).sum().item()
