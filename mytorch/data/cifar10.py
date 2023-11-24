# 2023/11/24
# zhangzhong

import torchvision  # type: ignore
import torch.utils.data


class CIFAR10Dataset:
    def __init__(self, splits: list[float] = [0.8, 0.2]) -> None:
        self.cifar_train = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True,
                                                        transform=torchvision.transforms.ToTensor(), download=True)
        self.cifar_test = torchvision.datasets.CIFAR10(
            root='datasets/cifar10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            dataset=self.cifar_train, lengths=splits
        )

    def get_train_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True)

    def get_val_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=False)

    def get_test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.cifar_test, batch_size=batch_size, shuffle=False)
