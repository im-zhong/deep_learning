# 2025/7/22
# zhangzhong
# https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
# https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html


import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# torch.multiprocessing is a PyTorch wrapper around Python’s native multiprocessing
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
# from torch.distributed import init_process_group, destroy_process_group


# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # First, before initializing the group process, call set_device,
    # which sets the default GPU for each process.
    # This is important to prevent hangs or excessive memory utilization on GPU:0
    torch.cuda.set_device(rank)

    # initialize the process group
    # The distributed process group contains all the processes that can communicate and synchronize with each other.
    dist.init_process_group(rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Distributing input data
dataset = torch.randn(100, 10)
train_data = torch.utils.data.DataLoader(
    dataset=dataset,
    # Each process will receive an input batch of 32 samples
    # The effective batch size is 32 * nprocs
    batch_size=32,
    shuffle=False,
    sampler=DistributedSampler(dataset),
)

# Calling the set_epoch() method on the DistributedSampler at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs.
# Otherwise, the same ordering will be used in each epoch.
# epoch = 1
# train_data.sampler.set_epoch(epoch)  # call this additional line at every epoch


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    # as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
    # you do not need to worry about different DDP processes starting from different initial model parameter values.
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    # Gradient synchronization communications take place during the backward pass and overlap with the backward computation
    # 我好像明白他说的overlap是什么意思了
    # 因为参数是按照bucket存的，不需要等整个backward计算完成，只要有一个bucket中的梯度计算完了
    # 就可以开始同步这个bucket中的梯度了
    # 这样就可以在计算梯度的同时进行通信，减少等待时间，提高
    # 所以overlap指的就是backward的计算过程和梯度的传递过程是overlap的
    loss_fn(outputs, labels).backward()
    # When the backward() returns, param.grad already contains the synchronized gradient tensor.
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    # rank is auto-allocated by DDP when calling mp.spawn.
    # world_size is the number of processes across the training job.
    #  For GPU training, this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # When using DDP, one optimization is to save the model in only one process and then load it on all processes, reducing write overhead.
    # This works because all processes start from the same parameters and gradients are synchronized in backward passes, and hence optimizers should keep setting parameters to the same values.
    CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), f"checkpoint_rank_{rank}.pt")
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # make sure no process starts loading before the saving is finished
    # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
    dist.barrier()

    # Additionally, when loading the module, you need to provide an appropriate map_location argument to prevent processes from stepping into others’ devices
    # If map_location is missing, torch.load will first load the module to CPU and then copy each parameter to where it was saved
    # 但是这个map location写成这个样子是什么意思？必须是一个字典吗？
    map_location = {"cuda:%d" % rank: "cuda:%d" % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")


if __name__ == "__main__":
    # The world size is the total number of processes that will be spawned.
    # For example, if you have 4 GPUs and want to use all of them, set world_size=4.
    world_size = 2  # Change this to the number of GPUs you want to use

    # 不行，必须像第二个教程那样来初始化
    # 其他例子会报错
    # ValueError: Default process group has not been initialized, please make sure to call init_process_group.
    # https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    # 用这里面的方法
    # setup(rank=0, world_size=world_size)
    # Run the basic DDP example
    run_demo(demo_basic, world_size)

    # Run the checkpoint example
    # run_demo(demo_checkpoint, world_size)
