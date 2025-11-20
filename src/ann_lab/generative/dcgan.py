"""
Deep Convolutional GAN (DCGAN) for image generation.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class DCGenerator(nn.Module):
    """
    DCGAN Generator with transposed convolutions.
    
    Projects noise to a small spatial size, then upsamples with
    transposed convolutions.
    
    Args:
        latent_dim: Dimension of noise vector
        img_channels: Number of output image channels
        feature_maps: Base number of feature maps
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            # Project and reshape: (batch, latent_dim) -> (batch, fm*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # Upsample: 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 32x32 -> 64x64 (or adjust for 28x28 MNIST)
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise vector (batch, latent_dim, 1, 1)
            
        Returns:
            img: Generated image (batch, channels, height, width)
        """
        return self.model(z)


class DCDiscriminator(nn.Module):
    """
    DCGAN Discriminator with strided convolutions.
    
    Args:
        img_channels: Number of input image channels
        feature_maps: Base number of feature maps
    """
    
    def __init__(
        self,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        
        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image (batch, channels, height, width)
            
        Returns:
            validity: Realness score (batch, 1, 1, 1)
        """
        return self.model(img)


class DCGAN(BaseModel):
    """
    Deep Convolutional GAN for image generation.
    
    Architecture guidelines from Radford et al., 2015:
    - Replace pooling with strided convolutions
    - Use batch normalization
    - Remove fully connected hidden layers
    - Use ReLU in generator, LeakyReLU in discriminator
    - Use Tanh in generator output
    
    Good for: generating images, especially on datasets like CIFAR-10, CelebA.
    
    Args:
        latent_dim: Dimension of noise vector
        img_channels: Number of image channels
        feature_maps: Base number of feature maps
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_channels: int = 1,
        feature_maps: int = 64
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.generator = DCGenerator(latent_dim, img_channels, feature_maps)
        self.discriminator = DCDiscriminator(img_channels, feature_maps)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise."""
        return self.generator(z)
    
    def generate(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples."""
        z = torch.randn(num_samples, self.latent_dim, 1, 1).to(device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples
