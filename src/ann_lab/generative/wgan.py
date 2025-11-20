"""
Wasserstein GAN (WGAN) with improved training stability.
"""

import torch
import torch.nn as nn
from .gan_basic import Generator
from ..core.base_model import BaseModel


class WGANDiscriminator(nn.Module):
    """
    WGAN Critic (not exactly a discriminator).
    
    Outputs a score (not probability) indicating realness.
    No sigmoid activation.
    
    Args:
        input_dim: Dimension of input
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),  # Layer norm instead of batch norm for WGAN
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        # No activation - output raw score
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WGAN(BaseModel):
    """
    Wasserstein GAN for more stable training.
    
    Uses Wasserstein distance instead of JS divergence.
    Key differences from standard GAN:
    - Critic (not discriminator) outputs unbounded scores
    - Weight clipping or gradient penalty to enforce Lipschitz constraint
    - Different loss: maximize D(real) - D(fake)
    
    Good for: more stable GAN training, avoiding mode collapse.
    
    Training tips:
    - Train critic more times than generator (e.g., 5:1 ratio)
    - Use RMSprop optimizer with small learning rate
    - Clip weights to [-0.01, 0.01] or use gradient penalty
    
    Args:
        latent_dim: Dimension of noise vector
        data_dim: Dimension of data
        hidden_dims_g: Generator hidden dimensions
        hidden_dims_c: Critic hidden dimensions
        clip_value: Weight clipping value (set to None if using GP)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        data_dim: int = 784,
        hidden_dims_g: list = None,
        hidden_dims_c: list = None,
        clip_value: float = 0.01
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        
        self.generator = Generator(latent_dim, data_dim, hidden_dims_g)
        self.critic = WGANDiscriminator(data_dim, hidden_dims_c)
        
    def clip_critic_weights(self):
        """Clip critic weights to enforce Lipschitz constraint."""
        if self.clip_value is not None:
            for p in self.critic.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate samples."""
        return self.generator(z)
    
    @staticmethod
    def gradient_penalty(critic, real_data, fake_data, device='cpu'):
        """
        Calculate gradient penalty for WGAN-GP.
        
        Enforces Lipschitz constraint via gradient penalty instead of weight clipping.
        """
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1).to(device)
        alpha = alpha.expand_as(real_data)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)
        
        # Critic scores for interpolates
        disc_interpolates = critic(interpolates)
        
        # Gradients w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # Penalty is (norm - 1)^2
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
