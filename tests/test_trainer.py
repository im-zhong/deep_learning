# 2023/11/20
# zhangzhong

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import mytorch.net
import mytorch.optim as optim
import mytorch.training
from mytorch import utils
from mytorch.data import linear


def test_Trainer():
    w = torch.tensor([2.0, -3.4])
    b = torch.tensor(4.2)
    num_train = 1000
    batch_size = 32
    train_data = linear.SyntheticLinearRegressionData(w, b, num_train)
    train_dataloader = train_data.get_dataloader(batch_size, shuffle=True)

    num_val = num_train
    val_data = linear.SyntheticLinearRegressionData(w, b, num_val)
    val_dataloader = val_data.get_dataloader(batch_size)

    # out_feature = 1
    in_feature = len(w)
    model = mytorch.net.linear.LinearRegressionScratch(in_feature)

    # 那么loss和optimizer写在那个地方呢??

    # todo: 这里变量名字和模块名字重名了
    # 可以将trainer => training
    # 可以参考hugging face的trainer的定义
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets['train'],
    #     eval_dataset=tokenized_datasets['validation'],
    #     data_collator=data_collator,    # dynamic padding
    #     tokenizer=tokenizer
    # )
    # 可以看到，hugging face在trainer的定义的时候就给出了所有
    # 训练需要的东西，那么我们也可以在trainer的定义的时候就给出
    # loss和optimizer
    # traning_args定义了许多hyper parameter,比如 epoch, lr_schedule 等等
    trainer = mytorch.training.Trainer(
        model=model,
        # loss也没有问题
        loss_fn=torch.nn.MSELoss(),
        # 很明显 从validation的loss来看 我们没有能够成功的更新参数
        optimizer=optim.MySGD(model.parameters(), lr=0.01),
        num_epochs=20,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.train("ScratchLoss")

    # 训练完成之后 输出一下模型参数
    # tensor([ 2.0138, -3.3883], requires_grad=True) tensor(4.1900, requires_grad=True)
    print(model.w, model.b)


def test_scheduler() -> None:
    model = nn.Linear(10, 2)

    lr: float = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    warmup_epochs = 10
    num_epochs = 100
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            # start_factor就是最开始的时候的学习率 = optimizer.lr*start_factor
            # 在total_iters之前，学习率线性增加，直到学习达到lr
            LinearLR(optimizer=optimizer, start_factor=0.1, total_iters=warmup_epochs),
            # 学习率从lr开始余弦下降
            CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - warmup_epochs),
        ],
        # milestones就是当epoch到达warmup_epochs的时候切换到下一个scheduler
        milestones=[warmup_epochs],
    )

    # 把学习率画出来 看看对不对
    lrs = []
    for epoch in range(num_epochs):
        print(optimizer.param_groups[0]["lr"])
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # 太对了，多么漂亮的曲线！
    plt.plot(lrs)
    utils.mysavefig("test_scheduler.png")
