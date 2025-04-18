import torch
import torch.nn as nn


# utils

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode="nearest"):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        return self.up(x)
    

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.down(x)
    
# model
class BaseVAE(nn.Module):
    def __init__(self, in_channels=1, z_dim=128, hidden_dim=256):
        super().__init__()
        input_size = in_channels*28*28

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
        )
    
    def reparameterize(self, mu, logvar):
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
        out = out.view(-1, 1, 28, 28)
        return out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


class ConvolutionVAE(nn.Module):
    def __init__(self, in_channels=1, z_dim=128, hidden_dim=[32, 64]):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            Downsample(in_channels, hidden_dim[0]),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.ReLU(),
            Downsample(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm2d(hidden_dim[1]),
            nn.ReLU())

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64*7*7, z_dim)
        self.fc_logvar = nn.Linear(64*7*7, z_dim)

        # Decoder: Recover input data from latent vectors
        self.decoder_input = nn.Linear(z_dim, 64*7*7)
        self.decoder = nn.Sequential(
            Upsample(hidden_dim[1], hidden_dim[0]),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.ReLU(),
            Upsample(hidden_dim[0], in_channels))

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma).to(logvar.device)
        return mu + sigma * eps

    def encode(self, x):
        """
        x shape: (1, 1, 28, 28)
        """
        x = self.encoder(x) # 
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 64, 7, 7)
        out = self.decoder(z)
        out = torch.sigmoid(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar