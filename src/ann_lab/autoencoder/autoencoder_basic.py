"""
Basic autoencoder implementations.

Autoencoders learn compressed representations by encoding input to a latent
space and then decoding back to reconstruct the original input.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class BasicAutoencoder(BaseModel):
    """
    Simple fully-connected autoencoder.
    
    Encodes input to a lower-dimensional latent space, then reconstructs.
    Good for: dimensionality reduction, feature learning, anomaly detection.
    
    Args:
        input_dim: Dimension of input
        latent_dim: Dimension of latent space (bottleneck)
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            reconstruction: Reconstructed input (batch, input_dim)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction


class ConvolutionalAutoencoder(BaseModel):
    """
    Convolutional autoencoder for images.
    
    Uses convolutional layers for encoding and transposed convolutions
    for decoding. Better for spatial data like images.
    
    Good for: image compression, denoising, feature learning from images.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        latent_dim: Dimension of latent space
        image_size: Size of input images (assumed square)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        image_size: int = 28
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 7x7 -> 4x4 (for 28x28 input)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
        # Calculate flattened dimension after convolutions
        self.flatten_dim = 128 * (image_size // 8) * (image_size // 8)
        
        # Bottleneck
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent representation."""
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image."""
        x = self.fc_decode(z)
        x = x.view(-1, 128, self.image_size // 8, self.image_size // 8)
        reconstruction = self.decoder(x)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, channels, height, width)
            
        Returns:
            reconstruction: Reconstructed images (batch, channels, height, width)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
