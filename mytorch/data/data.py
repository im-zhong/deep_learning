# 2023/11/17
# zhangzhong
# 作为最基本的框架文件 定义所有其他数据集和dataloder的接口

# 2023/9/9
# zhangzhong

import torch
from torch import Tensor
from typing import Any
import torch.utils.data

# TODO: 优化Dataset和DataLoader之间的结构，使得dataloder每次遍历的时候可以动态获取dataset的数据
# 从而可以支持 RNN 中在每个epoch开头的时候随机discard整个文本开头的数个字符，从而在多次迭代中尽可能
# 的看到所有的n-sequence字符
# 目前的接口设计可以说是完全没有设计，DataLoader直接获取底层dataset.X dataset.y ... 令人无语的垃圾实现


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    # generator, use yield to yield a batch of data
    def __iter__(self):
        self.X = self.dataset.X
        self.y = self.dataset.y
        if self.shuffle:
            # Returns a random permutation of integers from 0 to n - 1.
            indicies = torch.randperm(len(self.dataset))
            self.X = self.dataset.X[indicies]
            self.y = self.dataset.y[indicies]

        for i in range(0, len(self.dataset), self.batch_size):
            yield self.X[i:i+self.batch_size], self.y[i:i+self.batch_size]

    def __len__(self):
        return int(torch.ceil(torch.tensor(len(self.dataset) / self.batch_size)))

# 虽然如此，这里其实是简化了设计
# DataModule相当于 Dataset + DataLoader


class Dataset:
    def __init__(self, x: Tensor, y: Tensor):
        self.x = x
        self.y = y
        self.x_view = x
        self.y_view = y
        assert len(self.x) == len(self.y)

    def __len__(self) -> int:
        return len(self.x)

    # 这样我们的dataset就不一定会返回tuple
    # def data(self) -> tuple[Tensor, Tensor]:
    #     return self.x, self.y

    def __getitem__(self, indicies):
        # 提供一个默认的实现
        # 一般而言 数据的组织形式都是x, y这样的
        # 这个实现是对的
        # 对于train dataset 我们每个epoch都会shuffle
        # 但是对于val dataset 我们从不shuffle 读取到的数据都是对的
        return self.x_view[indicies], self.y_view[indicies]

    # def __getitem__(self, indicies):
    #     pass

    def shuffle(self):
        self.shuffle_indicies = torch.randperm(len(self))
        self.x_view = self.x[self.shuffle_indicies]
        self.y_view = self.y[self.shuffle_indicies]

# DataManager或者说每个项目的Data的实现是非常灵活的 和问题高度相关的
# 其实无法提取一个具体的接口 也没有必要 所以可以取消掉这个设计


class DataManager:
    def __init__(self):
        pass

    # 我们可以拿到不同的dataset
    # TODO: Fis this type error
    def get_train_dataset(self) -> Dataset:
        return Any

    def get_val_dataset(self) -> Dataset:
        return Any


class DataLoaderV2:
    # 实现一个默认的data_collator让实现更加精炼
    def __init__(self, datamgr: DataManager, tag: str, batch_size: int, shuffle: bool, data_collator=lambda x: x):
        self.datamgr = datamgr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tag = tag
        self.data_collator = data_collator
        pass

    def get_dataset(self) -> Dataset:
        if self.tag == 'train':
            return self.datamgr.get_train_dataset()
        elif self.tag == 'val':
            return self.datamgr.get_val_dataset()
        else:
            raise Exception(
                f'DataLoader get_dataset() bugs, no tag: {self.tag}')

    # def __iter__(self):
    #     dataset = self.get_dataset()
    #     # TODO: dataset不一定只会返回两个值
    #     x, y = dataset.data()
    #     if self.shuffle:
    #         # Returns a random permutation of integers from 0 to n - 1.
    #         indicies = torch.randperm(len(dataset))
    #         x = x[indicies]
    #         y = y[indicies]

    #     for i in range(0, len(dataset), self.batch_size):
    #         # 其实实现起来很简单，但是做类型注释就很麻烦了
    #         yield self.data_collator(x[i:i+self.batch_size]), self.data_collator(y[i:i+self.batch_size])

    def __iter__(self):
        dataset = self.get_dataset()
        if self.shuffle:
            # BUG: 这里做shuffle的结果是不对的
            # dataset做索引 返回的结果就不是Dataset类型了
            # 在下面通过索引进行访问就是不对的了
            # 所以randperm这个操作需要Dataset类本身进行支持
            # incidies = torch.randperm(len(dataset))
            # dataset = dataset[incidies]
            dataset.shuffle()

        for i in range(0, len(dataset), self.batch_size):
            # 卧槽！太对了
            # 我们直接让dataloader不知道dataset的任何信息
            # 我们不对dataset的返回格式做任何假定
            # 而且直接让data_collator接管 返回我们想要的任何形状的数据
            # 太对了！！！
            yield self.data_collator(dataset[i:i+self.batch_size])

    def __len__(self):
        dataset = self.get_dataset()
        return int(torch.ceil(torch.tensor(len(dataset) / self.batch_size)))


class DataModule:
    """The base class of data"""

    def __init__(self, root='../data', num_workers=4):
        pass

    # pytorch的batch_size在那个地方？
    # batch_size不在dataset里 而应该在dataloader里面
    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size, shuffle)
