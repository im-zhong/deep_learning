# 2023/11/20
# zhangzhong

import torch

import mytorch.net
import mytorch.optim as optim
import mytorch.training as training
from mytorch import losses
from mytorch.data import linear, mnist


def test_LinearRegressionScratch():
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
    trainer = training.Trainer(
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


def test_pytorch_linear_regression():
    w = torch.tensor([2.0, -3.4])
    b = torch.tensor(4.2)
    num_train = 1000
    batch_size = 32
    train_data = linear.SyntheticLinearRegressionData(w, b, num_train)
    train_dataloader = train_data.get_dataloader(batch_size, shuffle=True)

    num_val = num_train
    val_data = linear.SyntheticLinearRegressionData(w, b, num_val)
    val_dataloader = val_data.get_dataloader(batch_size)

    model = mytorch.net.linear.LinearRegression()
    trainer = training.Trainer(
        model=model,
        # loss_fn=torch.nn.MSELoss(),
        # 按照weight_decay的计算公式 loss必须知道模型的weight啊
        # 所以这是怎么传进来的呢??
        loss_fn=torch.nn.MSELoss(),
        # optimizer的问题
        # model.parameters(): Returns an iterator over module parameters
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01),
        num_epochs=20,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.train("PytorchLoss")

    # tensor([[ 2.0001], [-3.4003]], requires_grad=True) tensor([4.2004], requires_grad=True)
    print(model.parameters())


# def test_weight_decay():
#     # w = torch.tensor([2.0, -3.4])
#     # 200维的向量，但是只有20个数据
#     # 模型相较于数据过于复杂，会导致过拟合
#     # weight_decay会缓解这个问题
#     # 尝试设置不同的weight_decay，观察loss曲线
#     w = torch.randn((200, 1))
#     b = torch.tensor(4.2)
#     num_train = 20
#     batch_size = 4
#     train_data = linear.SyntheticLinearRegressionData(w, b, num_train)
#     train_dataloader = train_data.get_dataloader(batch_size, shuffle=True)

#     num_val = num_train
#     val_data = linear.SyntheticLinearRegressionData(w, b, num_val)
#     val_dataloader = val_data.get_dataloader(batch_size)

#     # out_feature = 1
#     in_feature = len(w)
#     model = mytorch.net.linear.LinearRegressionScratch(in_feature)

#     # 那么loss和optimizer写在那个地方呢??

#     # todo: 这里变量名字和模块名字重名了
#     # 可以将trainer => training
#     # 可以参考hugging face的trainer的定义
#     # trainer = Trainer(
#     #     model=model,
#     #     args=training_args,
#     #     train_dataset=tokenized_datasets['train'],
#     #     eval_dataset=tokenized_datasets['validation'],
#     #     data_collator=data_collator,    # dynamic padding
#     #     tokenizer=tokenizer
#     # )
#     # 可以看到，hugging face在trainer的定义的时候就给出了所有
#     # 训练需要的东西，那么我们也可以在trainer的定义的时候就给出
#     # loss和optimizer
#     # traning_args定义了许多hyper parameter,比如 epoch, lr_schedule 等等

#     # with weight_decay we can beat overfit!
#     weight_decay = 0.5
#     trainer = training.Trainer(
#         model=model,
#         # loss也没有问题
#         loss_fn=losses.MSELossWithWeightDecay(
#             parameters=model.parameters(), weight_decay=weight_decay),
#         # 很明显 从validation的loss来看 我们没有能够成功的更新参数
#         optimizer=optim.SGD(model.parameters(), lr=0.01,
#                             weight_decay=weight_decay),
#         num_epochs=20,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader
#     )
#     trainer.train('ScratchWeightDecayLoss')

#     # 训练完成之后 输出一下模型参数
#     # tensor([[ 1.9531], [-3.3375]], requires_grad=True) tensor([4.1477], requires_grad=True)
#     print(model.w, model.b)


def test_linear_classifier_scratch():
    # step 1. Fashison-MNIST
    ds = mnist.FashionMNISTDataset()
    num_labels = 10
    in_features = 28 * 28

    # step 2. DataLoader
    batch_size = 32
    train_dataloader = ds.get_train_dataloader(batch_size=batch_size)
    val_dataloader = ds.get_val_dataloader(batch_size=batch_size)

    # step 3. model
    model = mytorch.net.linear.LinearClassifierScratch(
        in_features=in_features, out_features=num_labels
    )
    loss_fn = losses.CrossEntropyLoss()
    optimizer = optim.MySGD(model.parameters(), lr=0.01)

    # step 4. trainer
    num_epochs = 2
    trainer = training.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="LinearClassifierScratch")


def test_MLP_scratch():
    # step 1. Fashison-MNIST
    ds = mnist.FashionMNISTDataset()
    num_labels = 10
    in_features = 28 * 28

    # step 2. DataLoader
    batch_size = 32
    train_dataloader = ds.get_train_dataloader(batch_size=batch_size)
    val_dataloader = ds.get_val_dataloader(batch_size=batch_size)

    # step 3. model
    model = mytorch.net.linear.MLPScratch(
        in_features=in_features,
        out_features=num_labels,
        num_hidden_1=256,
        num_hidden_2=128,
        dropout_1=0.2,
        dropout_2=0.1,
    )
    loss_fn = losses.CrossEntropyLoss()
    optimizer = optim.MySGD(model.parameters(), lr=0.01)

    # step 4. trainer
    num_epochs = 2
    trainer = training.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="MLPScratch")


def test_MLP_Pytorch():
    # step 1. Fashison-MNIST
    ds = mnist.FashionMNISTDataset()
    num_labels = 10
    in_features = 28 * 28

    # step 2. DataLoader
    batch_size = 32
    train_dataloader = ds.get_train_dataloader(batch_size=batch_size)
    val_dataloader = ds.get_val_dataloader(batch_size=batch_size)

    # step 3. model
    model = mytorch.net.linear.MLP(
        out_features=num_labels,
        num_hidden_1=256,
        num_hidden_2=128,
        dropout_1=0.2,
        dropout_2=0.1,
    )
    loss_fn = losses.CrossEntropyLoss()
    optimizer = optim.MySGD(model.parameters(), lr=0.01)

    # step 4. trainer
    num_epochs = 2
    trainer = training.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    trainer.train(tag="MLP", calculate_accuracy=True)
