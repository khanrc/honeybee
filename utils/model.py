import hashlib

import numpy as np
from peft.utils import ModulesToSaveWrapper
from torch.nn.parallel import DistributedDataParallel


def unwrap_ddp(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module


def unwrap_peft(layer):
    """ This function is designed for the purpose of checking dtype of model or fetching model configs. """
    if isinstance(layer, ModulesToSaveWrapper):
        return layer.original_module
    else:
        return layer


def get_num_params(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def module_fingerprint(module) -> str:
    """Get a fingerprint of a module by hashing its parameters
    """
    params = []
    for param in module.parameters():
        # to float32 if bf16 or fp16
        params.append(param.float().detach().numpy().flatten())

    params_concat = np.concatenate(params)
    params_bytes = params_concat.tobytes()
    hash_sha256 = hashlib.sha256(params_bytes)

    return hash_sha256.hexdigest()
