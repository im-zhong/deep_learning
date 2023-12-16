# 2023/12/14
# zhangzhong
# SVHN: http://ufldl.stanford.edu/housenumbers/
# https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN


import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision import transforms
import torch.utils.data

class SVHNDataset:
    def __init__(self, num_workers = 0, splits: list[float] = [0.8, 0.2]) -> None:
        self.num_workers = num_workers
        self.train_dataset = SVHN(root='datasets/svhn', split='train', transform=transforms.ToTensor(), download=True)
        self.test_dataset = SVHN(root='datasets/svhn', split='test', transform=transforms.ToTensor(), download=True)
        
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            dataset=self.train_dataset, lengths=splits
            # , generator=generator
        )
        
        
    def get_train_dataloader(self, batch_size):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
    
    def get_val_dataloader(self, batch_size):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_dataloader(self, batch_size):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
    

    