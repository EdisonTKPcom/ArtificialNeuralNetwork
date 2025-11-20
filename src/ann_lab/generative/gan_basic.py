"""
Basic Generative Adversarial Network (GAN).

GANs consist of two networks: a generator that creates fake data and
a discriminator that tries to distinguish real from fake.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class Generator(nn.Module):
    """
    Generator network for GAN.
    
    Takes random noise and generates fake data.
    
    Args:
        latent_dim: Dimension of input noise vector
        output_dim: Dimension of generated output
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_dim: int = 784,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
        
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim
        
        layers.extend([
            nn.Linear(in_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Random noise (batch, latent_dim)
            
        Returns:
            generated: Generated samples (batch, output_dim)
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.
    
    Classifies inputs as real or fake.
    
    Args:
        input_dim: Dimension of input data
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        # Note: No sigmoid here if using BCEWithLogitsLoss
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input data (batch, input_dim)
            
        Returns:
            validity: Realness score (batch, 1)
        """
        return self.model(x)


class BasicGAN(BaseModel):
    """
    Basic Generative Adversarial Network.
    
    Wrapper combining generator and discriminator.
    Training is typically done separately for G and D.
    
    Good for: generating synthetic data, learning data distributions.
    
    Note: Training GANs is notoriously unstable. Consider using:
    - Label smoothing
    - Feature matching
    - Careful learning rate tuning
    - Gradient penalty (WGAN-GP)
    
    Args:
        latent_dim: Dimension of noise input to generator
        data_dim: Dimension of real data
        hidden_dims_g: Hidden dims for generator
        hidden_dims_d: Hidden dims for discriminator
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        data_dim: int = 784,
        hidden_dims_g: list = None,
        hidden_dims_d: list = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, data_dim, hidden_dims_g)
        self.discriminator = Discriminator(data_dim, hidden_dims_d)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples from noise."""
        return self.generator(z)
    
    def generate(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            samples: Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples
