"""
Sparse autoencoder with sparsity regularization on latent activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class SparseAutoencoder(BaseModel):
    """
    Autoencoder with sparsity constraint on latent representations.
    
    Encourages most latent units to be inactive (near zero) for any input,
    forcing the model to learn more interpretable features.
    
    Good for: feature learning, finding sparse representations.
    
    Args:
        input_dim: Dimension of input
        latent_dim: Dimension of latent space
        hidden_dims: Hidden layer dimensions
        sparsity_target: Target average activation (rho)
        sparsity_weight: Weight of sparsity penalty (beta)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = None,
        sparsity_target: float = 0.05,
        sparsity_weight: float = 0.1
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        encoder_layers.extend([
            nn.Linear(in_dim, latent_dim),
            nn.Sigmoid()  # Activation in [0, 1] for sparsity
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
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
        
    def kl_divergence(self, rho_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between target and actual sparsity.
        
        KL(rho || rho_hat) = rho * log(rho / rho_hat) + (1-rho) * log((1-rho) / (1-rho_hat))
        """
        rho = self.sparsity_target
        kl = rho * torch.log(rho / (rho_hat + 1e-10)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-10))
        return kl.sum()
    
    def forward(self, x: torch.Tensor, return_sparsity_loss: bool = False):
        """
        Args:
            x: Input tensor (batch, input_dim)
            return_sparsity_loss: Whether to return sparsity penalty
            
        Returns:
            reconstruction: Reconstructed input
            sparsity_loss: (optional) Sparsity penalty term
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        
        if return_sparsity_loss:
            # Average activation of each latent unit across batch
            rho_hat = z.mean(dim=0)
            sparsity_loss = self.kl_divergence(rho_hat) * self.sparsity_weight
            return reconstruction, sparsity_loss
        
        return reconstruction
