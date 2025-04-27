import random
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed


def setup_for_distributed(local_rank):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def cleanup():
    if torch.distributed.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False