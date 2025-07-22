# 2025/7/22
# zhangzhong
# https://docs.pytorch.org/docs/main/notes/ddp.html


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP


# example(rank, world_size) 就是每个子进程的“主函数”。每个由 mp.spawn() 启动的子进程，都会执行这个函数。
def example(rank, world_size):
    # 直接设置当前的设置，那么就不需要在很多pytroch代码里面写那个 to了
    torch.cuda.set_device(rank)

    # create default process group
    # TODO: 这些参数都是啥意思？
    # world_size 表示总共有多少个进程（通常等于 GPU 数量），rank 表示当前进程的编号（从 0 到 world_size - 1）。
    # 每个进程都负责一部分数据和模型，并在每次反向传播后同步梯度。
    # DDP relies on c10d ProcessGroup for communications.
    # Hence, applications must create ProcessGroup instances before constructing DDP.
    # When using the default “env://” initialization, PyTorch will read the environment variables MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE.
    # 初始化分布式通信（读取上面两个变量） MASTER_ADDR MASTER_ADDR
    # https://docs.pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # create local model
    # ? to rank?
    # 是将模型放到对应的 GPU 上（rank 对应的 GPU 编号）。
    # 注意：如果你用的是单机多 GPU，一般 rank 就等于 GPU 编号。

    model = nn.Linear(10, 10).to(rank)
    # 如果你想要自定义初始化模型，必须在DDP之前初始化
    # model.apply(my_custom_init)

    # construct DDP model
    # To use DDP, you’ll need to spawn multiple processes and create a single instance of DDP per process.
    # deivce? rank ?
    # 那每个进程都构造了自己的model实例，DDP怎么保证每个gpu上的模型的初始参数是一致的呢？
    # 在你调用 dist.init_process_group() 初始化通信组之后，DistributedDataParallel(DDP) 会自动将 rank=0 的模型参数广播给其他所有进程（GPU），确保初始化参数完全一致。
    # When wrapping a model with DDP, the constructor performs a broadcast from rank 0 to ensure that all processes start with the same model parameters and buffers.
    ddp_model = DDP(module=model, device_ids=[rank])
    # DDP works with TorchDynamo.
    #  When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes.
    ddp_model = torch.compile(ddp_model)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # forward pass
    # to rank? 到底是什么意思
    # 同理，数据也要放在同一个设备上：
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)

    # backward pass
    loss_fn(outputs, labels).backward()

    # update parameters
    optimizer.step()


def main():
    # 什么又是world size呢？
    world_size = 2

    # Spawns nprocs processes that run fn with args.
    # Perform a blocking join on all processes.
    # 不对啊，这里咋没有提供rank参数呢？
    # 这个 spawn 会自动启动 nprocs 个子进程，并自动为每个子进程传入一个 rank 作为 fn 的第一个参数。
    # 也就是说 example(rank, world_size) 里的 rank 是 mp.spawn() 自动注入的。
    mp.spawn(fn=example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    #  这两行是给 PyTorch 分布式通信模块（c10d 后端）提供的信息，让多个进程之间能互相发现、连接，并建立通信组（ProcessGroup）。
    # 	•	MASTER_ADDR: 主进程所在的 IP 地址或主机名（在单机时通常是 localhost）
    # 	•	MASTER_PORT: 主进程监听的端口，用来等其他进程连上来。
    # 它们就像一个服务器的 IP 和端口，其他 GPU 的进程会根据这个地址去连接主进程，进行组网通信。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
