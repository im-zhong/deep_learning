# 2023/11/21
# zhangzhong


import matplotlib.pyplot as plt
import os.path
import torch
from torch import device
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
