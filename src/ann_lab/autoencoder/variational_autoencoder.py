"""
Variational Autoencoder (VAE) for generative modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class VariationalAutoencoder(BaseModel):
    """
    Variational Autoencoder (VAE).
    
    Learns a probabilistic latent space where points can be sampled to
    generate new data. Encoder outputs mean and log-variance of latent distribution.
    
    Good for: generative modeling, learning smooth latent spaces, interpolation.
    
    Args:
        input_dim: Dimension of input
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 20,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor):
        """
        Encode input to latent distribution parameters.
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean
            logvar: Log variance
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            reconstruction: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate new samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            samples: Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, kl_weight: float = 1.0):
        """
        VAE loss = Reconstruction loss + KL divergence.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance
            kl_weight: Weight for KL term (for beta-VAE)
            
        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (binary cross-entropy or MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + kl_weight * kl_loss
        
        return loss, recon_loss, kl_loss


class ConvVAE(BaseModel):
    """
    Convolutional Variational Autoencoder for images.
    
    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        image_size: Size of input images (assumed square)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 20,
        image_size: int = 28
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.flatten_dim = 128 * (image_size // 8) * (image_size // 8)
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x: torch.Tensor):
        batch_size = x.size(0)
        h = self.encoder(x)
        h = h.view(batch_size, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 128, self.image_size // 8, self.image_size // 8)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
