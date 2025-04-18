import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from .loss import loss_function
from .model import BaseVAE, ConvolutionVAE


def parser_argument():
    parser = argparse.ArgumentParser()
    # Training params
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    # Model params
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--mode", type=str, default="base")
    args = parser.parse_args()
    return args


def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    total = 0
    total_loss = 0.

    model.train()
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        total += data.size(0)

        optimizer.zero_grad()
        recon_data, mu, logvar = model(data)
        loss = criterion(recon_data, data, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"[Train] Epoch {epoch} [{total}/{len(dataloader.dataset)}]\tAvg Loss: {total_loss/(batch_idx+1):.5f}")

    total_loss /= len(dataloader)
    return total_loss


@torch.no_grad()
def test_epoch(model, dataloader, criterion, device):
    total_loss = 0.

    model.eval()
    for data, _ in tqdm(dataloader):
        data = data.to(device)

        recon_data, mu, logvar = model(data)
        loss = criterion(recon_data, data, mu, logvar)

        total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss


if __name__ == "__main__":
    args = parser_argument()
    
    # set seed & hyper-params
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 1. Define dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root = ".", train = True, transform = transform, download = True)
    test_dataset = MNIST(root = ".", train = False, transform = transform, download = True)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

    # 2. Define model & optimizer
    if args.mode == "base":
        hidden_dim = 256
        model = BaseVAE(args.in_channels, args.z_dim, hidden_dim).to(args.device)
        save_path = os.path.join(args.save_path, "basevae.pth")
    elif args.mode == "conv":
        hidden_dim = [32, 64]
        model = ConvolutionVAE(args.in_channels, args.z_dim, hidden_dim).to(args.device)
        save_path = os.path.join(args.save_path, "convolutionvae.pth")
    optimizer = optim.Adam(model.paramters(), lr=args.lr)

    # 3. VAE train
    best_loss = 0.
    for epoch in range(1, args.epochs+1):
        _ = train_epoch(model,
                        train_loader,
                        loss_function,
                        epoch,
                        args.device)
        test_avg_loss = test_epoch(model,
                                   test_loader,
                                   loss_function,
                                   args.device)
        
        print(f"[Test] Epoch: {epoch}\tTotal Avg Loss: {test_avg_loss:.5f}")

        if epoch == 1:
            torch.save(model.state_dict(), args.save_path)
            print("save checkpoint!")
            best_loss = test_avg_loss
        else:
            if best_loss > test_avg_loss:
                torch.save(model.state_dict(), args.save_path)
                print("save checkpoint! [{best_loss:.5f} -> {avg_loss:.5f}]")
                best_loss = test_avg_loss
            else:
                print("no save checkpoint!")

    


