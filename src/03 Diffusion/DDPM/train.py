import os
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from ddpm import UNetModel, GaussianDiffusion, Trainer

def parser_argument():
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-04)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="./")
    # Model params
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--channel_mult", type=tuple, default=(1,2,2,))
    parser.add_argument("--attention_resolutions", type=list, default=[])
    # Diffusion params
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    args = parser.parse_args()
    return args


def setup(rank):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def download_dataset(rank):
    if rank == 0:
        print("[Rank 0] Downloading MNIST...")
        transforms_ = transforms.Compose([transforms.ToTensor()])
        MNIST(root="./data", train=True, transform=transforms_, download=True)
        MNIST(root="./data", train=False, transform=transforms_, download=True)
    dist.barrier()


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup(local_rank)
    set_seed(args.seed + rank)
    
    download_dataset(rank)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])
        
    dataset = MNIST(root = "./data", train=True, transform=transform, download=False)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True)

    model = UNetModel(in_channels=args.in_channels,
                      model_channels=args.model_channels,
                      out_channels=args.out_channels,
                      channel_mult=args.channel_mult,
                      attention_resolutions=args.attention_resolutions)
    diffusion = GaussianDiffusion(args.timesteps, beta_schedule=args.beta_schedule)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(model,
                      diffusion,
                      dataloader,
                      optimizer,
                      rank,
                      world_size,
                      device=args.device,
                      save_dir=args.save_dir)
    trainer.train(epochs=args.epochs)

    cleanup()


if __name__ == "__main__":
    args = parser_argument()
    main(args)