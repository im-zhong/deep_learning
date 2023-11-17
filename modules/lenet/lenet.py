import mytorch.func as func
import torch


def my_lenet():
    input = torch.tensor([1, 2])
    func.flatten(input=input)
