"""
Denoising autoencoder that learns to reconstruct clean data from corrupted input.
"""

import torch
import torch.nn as nn
from .autoencoder_basic import BasicAutoencoder, ConvolutionalAutoencoder


class DenoisingAutoencoder(ConvolutionalAutoencoder):
    """
    Denoising autoencoder learns to remove noise from inputs.
    
    Trained by corrupting inputs with noise and learning to reconstruct
    the clean original. Forces the model to learn robust features.
    
    Good for: noise reduction, robust feature learning, preprocessing.
    
    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        image_size: Size of input images
        noise_factor: Amount of noise to add during training
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        image_size: int = 28,
        noise_factor: float = 0.3
    ):
        super().__init__(in_channels, latent_dim, image_size)
        self.noise_factor = noise_factor
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        return torch.clamp(noisy_x, 0., 1.)
    
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Args:
            x: Clean input images
            add_noise: Whether to add noise (True during training)
            
        Returns:
            reconstruction: Denoised reconstruction
        """
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
            
        z = self.encode(x_noisy)
        reconstruction = self.decode(z)
        return reconstruction
