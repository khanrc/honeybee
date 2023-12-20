import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def unwrap_ddp(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


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
