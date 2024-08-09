# 2023/11/20
# zhangzhong
# mnist datasets

import torch.utils.data

# copilot: /explain Skipping analyzing "torchvision": module is installed, but missing library stubs or py.typed marker [import-untyped]
# The warning message you're seeing is from a static type checker for Python, such as mypy or Pyright. It's saying that the torchvision module is installed, but it's missing either library stubs or a py.typed marker.
# Library stubs are files that contain type annotations for Python modules. They're used by static type checkers to understand the types of the functions and classes in a module.
# A py.typed marker is a file that indicates a package has opted into type checking. If a package includes this file, it's promising that all its code is annotated with type hints.
# The torchvision package doesn't include either of these, so the type checker can't verify the types of the objects you're importing from torchvision. This isn't necessarily a problem, but it does mean that the type checker can't help you catch type errors in the parts of your code that use torchvision.
# If you want to suppress this warning, you can add a # type: ignore comment to the import line:
# This tells the type checker to ignore this line and not issue any warnings about it.
# https://github.com/pytorch/vision/issues/2025
import torchvision  # type: ignore

from . import data


class FashionMNISTDataset(data.DataModule):
    """FashionMNIST data set."""

    def __init__(self, num_workers=0, splits=[0.8, 0.2]):
        super().__init__()
        self.num_workers = num_workers
        self.training_data = torchvision.datasets.FashionMNIST(
            root="datasets",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        self.testing_data = torchvision.datasets.FashionMNIST(
            root="datasets",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        # 还要做一件事，就是把数据进行一个展平
        # torch.flatten(self.training_data.data, start_dim=1)
        # 或者说在网络结构里进行修改

        # split
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.training_data, splits
        )

    # 那我想知道 dataloader每次开始遍历的顺序都会不一样吗
    # 也就是train的每个epoch的shuffle都会不一样吗？？
    def get_train_dataloader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_val_dataloader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_test_dataloader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            self.testing_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )


class MNISTDataset:
    def __init__(self, num_workers=0, splits=[0.8, 0.2]) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.training_data = torchvision.datasets.MNIST(
            root="datasets",
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        self.testing_data = torchvision.datasets.MNIST(
            root="datasets",
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        # split
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            self.training_data, splits
        )

    def get_train_dataloader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_val_dataloader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_test_dataloader(self, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            self.testing_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
