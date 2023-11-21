# 2023/11/21
# zhangzhong


import matplotlib.pyplot as plt
import os.path


def mysavefig(filename: str):
    os.makedirs('imgs', exist_ok=True)
    filename = os.path.join('imgs', filename)
    plt.savefig(filename)
