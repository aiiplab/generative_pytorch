import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256):
        super().__init__()
        input_size = in_channels * 28 * 28 # C x H x W

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_channels=1, z_dim=128, hidden_dim=256):
        super().__init__()
        input_size = in_channels * 28 * 28 # C x H x W

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class VAE(nn.Module):
    def __init__(self, in_channels=1, z_dim=128, hidden_dim=256):
        super().__init__()
        input_size = in_channels*28*28 # C x H x W

        # Encoder
        self.encoder = Encoder(in_channels, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        # Decoder
        self.decoder = Decoder(in_channels, z_dim, hidden_dim)
    
    def reparameterize(self, mu, logvar):
        """
        z = mean + std dev. x noise;   noise ~ N(0,I)
        """
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma).to(logvar.device)
        return mu + sigma * eps
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28) # [B, (C, H, W)] -> [B, C, H, W]
        out = torch.sigmoid(out) # value to [0, 1]
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar