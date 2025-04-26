import torch
import torch.nn.functional as F

def loss_function(pred, target, mu, logvar):
    """
    Loss = Reconstruction Loss + KL divergence
    """
    recon_loss = F.binary_cross_entropy(pred, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss