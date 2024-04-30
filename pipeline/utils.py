import math
import os
import random
from typing import List

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

import utils
from utils import print_rank_0

def get_cache_dir():
    DEFAULT_HF_HOME = "~/.cache/huggingface"
    cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_HOME)

    return cache_dir

def check_local_file(model_name_or_path):
    cache_dir = get_cache_dir()
    file_name = os.path.join(
        cache_dir, f"models--{model_name_or_path.replace('/', '--')}"
    )
    local_files_only = os.path.exists(file_name)
    file_name = file_name if local_files_only else model_name_or_path
    return local_files_only, file_name


def find_file(filename, directory="."):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_param_groups(modules, no_weight_decay_cond, scale_lr_cond, lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
    and learning rate scale condition (args.lr vs lr_mult * args.lr)
    scale_lr_cond is used during finetuning where head of the network requires a scaled
    version of the base learning rate.
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({"params": wd_no_scale_lr, "wd_mult": 1.0, "lr_mult": 1.0})
    if len(wd_scale_lr):
        param_groups.append({"params": wd_scale_lr, "wd_mult": 1.0, "lr_mult": lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({"params": no_wd_no_scale_lr, "wd_mult": 0.0, "lr_mult": 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({"params": no_wd_scale_lr, "wd_mult": 0.0, "lr_mult": lr_mult})

    return param_groups


def get_cosine_schedule_with_warmup(
    optimizer,
    lr,
    min_lr,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Cosine schedule with warmup & min_lr"""
    delta_min_lr = (lr - min_lr) / lr

    def cvt_mult_with_minlr(mult):
        """Convert multiplier considering min_lr"""
        return (1 - delta_min_lr) + delta_min_lr * mult

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Do not consider min_lr when warmup
            progress = float(current_step) / float(max(1, num_warmup_steps))
            return cvt_mult_with_minlr(progress)

        # [0, 1] progress
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # [0, 1] cosine multiplier
        cos_mult = max(0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return cvt_mult_with_minlr(cos_mult)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def seed_worker(worker_id):
    """
    Copied and slightly modified from https://github.com/Lightning-AI/lightning/blob/984f49f7195ddc67e961c7c498ee6e19fc0cecb5/src/lightning/fabric/utilities/seed.py#L81-L104
    Helper function to set worker seed during Dataloader initialization.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = torch.distributed.get_rank()
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np_seed = ss.generate_state(4)
    np.random.seed(np_seed)
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)


def set_trainable_parameters(model, module_to_update: List[str] = None):
    if module_to_update is None:
        return

    if "vision_model" in module_to_update:
        for param in model.vision_model.parameters():
            param.requires_grad = True

    if "abstractor" in module_to_update:
        for param in model.abstractor.parameters():
            param.requires_grad = True

    if "language_model" in module_to_update:
        for param in model.language_model.parameters():
            param.requires_grad = True


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print_rank_0(
        f"trainable params: {trainable_params} "
        f"|| all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    vision_params = utils.get_num_params(model.vision_model)
    proj_params = utils.get_num_params(model.abstractor)
    llm_params = utils.get_num_params(model.language_model)
    print_rank_0(
        f"\tTotal params: {all_param/1000/1000:.1f}M "
        f"(trainable: {trainable_params/1000/1000:.2f}M)"
    )
    print_rank_0(f"\tVision model params: {vision_params/1000/1000:.1f}M")
    print_rank_0(f"\tProjector (abstractor) params: {proj_params/1000/1000:.1f}M")
    print_rank_0(f"\tLanguage model params: {llm_params/1000/1000:.1f}M")
