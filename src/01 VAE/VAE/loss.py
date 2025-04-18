import torch
import torch.nn.functional as F

def loss_function(pred, target, mu, logvar):
    recon_loss = F.binary_cross_entropy(pred, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss