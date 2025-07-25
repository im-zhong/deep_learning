import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size):
    setup(rank, world_size)

    local_data = {"rank": rank, "value": rank * 100}

    # rank 0 会收集所有人的数据，其它 rank 填 None
    gathered = None
    if rank == 0:
        gathered = [None for _ in range(world_size)]

    dist.gather_object(local_data, gathered, dst=0)

    if rank == 0:
        print(f"[Rank 0] Gathered all data: {gathered}")

    cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size)
