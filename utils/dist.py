import os
import torch.distributed as dist


def get_dist_info() -> str:
    """Check distributed training env variables"""
    # Tested on the instance (best.sh):
    # LOCAL_RANK = 2 | RANK = 2 | MASTER_ADDR = 127.0.0.1 | MASTER_PORT = 29500 | WORLD_SIZE = 4
    keys = [
        "NODE_RANK",
        "GROUP_RANK",
        "LOCAL_RANK",
        "RANK",
        "GLOBAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        # for now, torch.distributed.run env variables
        # https://github.com/pytorch/pytorch/blob/d69c22dd61/torch/distributed/run.py#L121
        "ROLE_RANK",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
    ]
    rs = []
    for key in keys:
        r = os.getenv(key)
        if r:
            s = f"{key} = {r}"
            rs.append(s)

    return " | ".join(rs)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if is_main_process():
        print(message, flush=True)
