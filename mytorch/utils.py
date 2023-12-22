# 2023/11/21
# zhangzhong


import matplotlib.pyplot as plt
import os.path
import torch
from torch import device, Tensor, nn
import random
from tsnecuda import TSNE


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
    if num_classes < 5:
        return 0
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


activation = {}


def get_activation(name):
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    def hook(model, input, output):
        # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
        # Returns a new Tensor, detached from the current graph.
        # Returned Tensor shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks.
        activation[name] = output.detach().clone()
    return hook


def get_intermediate_output(model, layer_name):
    intermediate_output = {}
    getattr(model, layer_name).register_forward_hook(
        get_activation(layer_name))
    # do model forward
    # then get the model intermediate output
    return activation[layer_name]


class IntermediateOutputHook:
    def __init__(self):
        pass

    def __call__(self, model, input, output):
        self.output = output.detach().clone()


def get_nested_attr(obj, attr_path: str):
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


class RegisterIntermediateOutputHook:
    def __init__(self, model: nn.Module, layers: list[str]):
        self.model = model
        self.layers = layers
        self.hooks = {}
        self.handles = []
        for layer in layers:
            self.hooks[layer] = IntermediateOutputHook()
            self.handles.append(get_nested_attr(
                model, layer).register_forward_hook(self.hooks[layer]))

    def get_intermediate_output(self) -> dict[str, Tensor]:
        return {layer: self.hooks[layer].output for layer in self.layers}

    def __del__(self):
        for handle in self.handles:
            handle.remove()


def draw_tsne(data: Tensor, labels: Tensor, name: str | None = None):
    data = data.cpu().flatten(start_dim=1)
    batch_size, feature_size = data.shape
    assert len(labels.shape) == 1
    assert labels.shape[0] == batch_size

    # 原来t-SNE每次的输出是随机的
    tsne = TSNE(n_components=2, perplexity=15,
                learning_rate=10).fit_transform(data)
    figure, ax = plt.subplots()
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    scatter = ax.scatter(tsne[:, 0], tsne[:, 1], s=1, c=labels, cmap='tab10')
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html#matplotlib.axes.Axes.set_xticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(*scatter.legend_elements())
    if name is not None:
        mysavefig(name)
    else:
        plt.show()
