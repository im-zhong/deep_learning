# 2023/12/17
# zhangzhong

import torch
from torch import nn,Tensor
from module.vision.simple_cnn import SimpleCNN
from mytorch import utils
from mytorch.data.cifar10 import CIFAR10Dataset, cifar10_predict
from mytorch.data.mnist import MNISTDataset
from mytorch.training import TrainerV2
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from matplotlib import pyplot as plt
from tsnecuda import TSNE

def draw_tsne(intermediate_output: dict[str, list[Tensor]], layer: str, targets):
    last_output = intermediate_output[layer]
    # concat
    last_output = torch.concat(last_output, dim=0)
    # last_output = torch.stack(last_output, dim=0)
    last_output = last_output.flatten(start_dim=1, end_dim=-1)

    # 把last_output给画出来
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(last_output.cpu())
    scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=targets, cmap='tab10') 
    plt.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    utils.mysavefig(f'mnist_{layer}_tsne.png')    

def test_simple_cnn() -> None:
    model = SimpleCNN()
    
    lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 0
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs-warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )
    
    device = utils.get_device()
    batch_size = 128
    num_workers = 16
    mnist = MNISTDataset(num_workers=num_workers)
    trainer = TrainerV2(model=model,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                        train_dataloader=mnist.get_train_dataloader(batch_size=batch_size),
                        val_dataloader=mnist.get_val_dataloader(batch_size=batch_size),
                        test_dataloader=mnist.get_test_dataloader(batch_size=batch_size),
                        scheduler=scheduler,
                        device=device)
    
    tag = 'simple_cnn_1'
    trainer.train(tag=tag)
    
    # get the pretrained model
    pretrained_model = trainer.model
    # 其实最好是写成一个callable object 这样可以拿到所有的
    # 如果我想拿到一次forward里面的所有层的中间输出怎么办呢
    layers = ['conv1', 'conv2', 'fc1', 'fc2']
    # 那这样的话 显然我们需要一个统一的注册函数
    # 这里其实最好是构造一个对象 他的作用就是会根据我们提供的layers的名字
    # 给提供的model注册hook 然后我们就可以拿到所有的中间输出了
    # hook = utils.register_forward_hook(model, layers)
    hook = utils.RegisterIntermediateOutputHook(pretrained_model, layers)

    # forward
    intermediate_outputs: dict[str, list[Tensor]] = {}
    for x, y in tqdm(mnist.get_test_dataloader(batch_size=batch_size)):
        x = x.to(device)
        y = pretrained_model(x)
        
        output = hook.get_intermediate_output()
        for layer in layers:
            if layer not in intermediate_outputs:
                intermediate_outputs[layer] = []
            intermediate_outputs[layer].append(output[layer])
        
    # 1. conv层的向量没办法画tsne，因为向量太长了
    # 2. 最终输出的logits概率分布画出来的向量的tsne的效果不如倒数第二层的效果好
    # 3. conv2的长度是7000，还是可以画出来的，可以看出conv2的输出相比原始图像已经可以较好的区分了
    # 但是仍然不如fc1的输出好
    # for layer in ['fc1']:
    #     draw_tsne(intermediate_outputs, layer, targets=mnist.testing_data.targets)
    
    
    # search graph
    # random pick a graph from training set
    # then search the graph from the test set
    # then find the top 5 nearest graph
    # then show the top 5 nearest graph
    
    train_index = 233
    image = mnist.training_data[train_index][0].unsqueeze(0).to(device)
    y = pretrained_model(image)
    output = hook.get_intermediate_output()['fc1']
    # 然后我们和整个测试的fc1的输出计算出他们的欧氏距离
    # 然后取top5
    test_fc1_output = intermediate_outputs['fc1']
    test_fc1_output = torch.concat(test_fc1_output, dim=0)
    norms = torch.linalg.vector_norm(test_fc1_output - output, ord=2, dim=1) 
    assert norms.shape == (10000,)
    values, indicies = (-norms).topk(k=5)
    print(-values, indicies)
    
    # 我们看一下这个图像
    print(mnist.training_data[train_index][1])
    for i in range(5):
        print(mnist.testing_data[indicies[i]][1])
