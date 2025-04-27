import os
import argparse

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from distributed import *
from model import UNetModel, GaussianDiffusion, Trainer

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
    
    
def download_dataset():
    transforms_ = transforms.Compose([transforms.ToTensor()])
    MNIST(root="./data", train=True, transform=transforms_, download=True)
    MNIST(root="./data", train=False, transform=transforms_, download=True)


def main(args):
    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_for_distributed(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    set_seed(args.seed + rank)

    if rank == 0:
        print("Downloading Dataset...")
        download_dataset()
    if distributed:
        dist.barrier()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5])
    ])
        
    dataset = MNIST(root = "./data", train=True, transform=transform, download=False)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            shuffle = (sampler is None),
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            drop_last = (sampler is not None))

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
                      rank if distributed else None,
                      world_size if distributed else None,
                      device=args.device,
                      save_dir=args.save_dir)
    trainer.train(epochs=args.epochs)

    if distributed:
        cleanup()


if __name__ == "__main__":
    args = parser_argument()
    main(args)