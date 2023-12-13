# 2023/11/24
# zhangzhong

import torchvision  # type: ignore
import torch.utils.data


class CIFAR10Dataset:
    def __init__(self, num_workers = 0, splits: list[float] = [0.8, 0.2]) -> None:

        self.num_workers = num_workers
        # 数据增强才是真的管用 给我使劲用！
        # 我要增加数据集了
        # 我突然明白了！其实我可以不断的重新训练，因为数据都是随机transform的，所以也相当于增加数据了
        # TIP: 数据处理太花时间了，导致GPU的利用率很低，因为数据处理发生在cpu上，怪不得cpu都满了
        train_transform = torchvision.transforms.Compose([
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            # # https://pytorch.org/vision/main/generated/torchvision.transforms.RandomCrop.html
            torchvision.transforms.RandomCrop(32, padding=4),
            # # https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
            # torchvision.transforms.RandomResizedCrop(size=(32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            # 这些数值是怎么计算出来的？是不是遍历了数据集然后算出来的？
            # 你在这里做batch normalize 和网络的第一层直接做normallze有什么区别?
            # resnet的第一层并不是norm 但是norm确实带来了性能提升
            # 这是否给我一些启示 也许我可以在网络的第一层做normalize
            torchvision.transforms.Normalize(
                 mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

        self.cifar_train = torchvision.datasets.CIFAR10(root='datasets/cifar10', train=True,
                                                        transform=train_transform,
                                                        download=True)

        transform_test = torchvision.transforms.Compose([
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
        ])

        self.cifar_test = torchvision.datasets.CIFAR10(
            root='datasets/cifar10', train=False, transform=transform_test, download=True)

        # 我知道为什么重启训练会有性能提升了，因为train和val的分配是随机的 所以重启训练的时候 有一些模型见过的图片被分到val里面了
        # 所以为了稳定的分配 我必须固定一个随机数种子
        # 其实为了更高的准确率 我应该把所有的数据都用来训练
        # generator = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            dataset=self.cifar_train, lengths=splits
            # , generator=generator
        )

    def get_train_dataloader(self, batch_size):
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation
        # runs significantly faster!!!
        return torch.utils.data.DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def get_val_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)

    def get_test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(dataset=self.cifar_test, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
