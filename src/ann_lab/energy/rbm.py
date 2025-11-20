"""
Restricted Boltzmann Machine (RBM) for unsupervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class RestrictedBoltzmannMachine(BaseModel):
    """
    Restricted Boltzmann Machine (RBM).
    
    Two-layer energy-based model with visible and hidden units.
    No connections within layers (hence "restricted").
    
    Trained using Contrastive Divergence.
    
    Good for: feature learning, dimensionality reduction, pre-training.
    
    Args:
        num_visible: Number of visible units
        num_hidden: Number of hidden units
        learning_rate: Learning rate for CD
        k: Number of Gibbs sampling steps (CD-k)
    """
    
    def __init__(
        self,
        num_visible: int,
        num_hidden: int = 128,
        learning_rate: float = 0.01,
        k: int = 1
    ):
        super().__init__()
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.k = k
        
        # Weights and biases
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        
    def sample_hidden(self, v: torch.Tensor) -> tuple:
        """
        Sample hidden units given visible units.
        
        P(h_j = 1 | v) = sigmoid(b_j + sum_i(v_i * w_ij))
        
        Args:
            v: Visible units (batch, num_visible)
            
        Returns:
            h_prob: Activation probabilities (batch, num_hidden)
            h_sample: Binary samples (batch, num_hidden)
        """
        activation = F.linear(v, self.W.t(), self.h_bias)
        h_prob = torch.sigmoid(activation)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample
    
    def sample_visible(self, h: torch.Tensor) -> tuple:
        """
        Sample visible units given hidden units.
        
        P(v_i = 1 | h) = sigmoid(a_i + sum_j(h_j * w_ij))
        
        Args:
            h: Hidden units (batch, num_hidden)
            
        Returns:
            v_prob: Activation probabilities (batch, num_visible)
            v_sample: Binary samples (batch, num_visible)
        """
        activation = F.linear(h, self.W, self.v_bias)
        v_prob = torch.sigmoid(activation)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample
    
    def contrastive_divergence(self, v0: torch.Tensor) -> tuple:
        """
        Perform Contrastive Divergence (CD-k) update.
        
        Args:
            v0: Initial visible units (batch, num_visible)
            
        Returns:
            positive_grad: Positive phase gradient
            negative_grad: Negative phase gradient
            reconstruction_error: Mean reconstruction error
        """
        # Positive phase
        h0_prob, h0_sample = self.sample_hidden(v0)
        
        # Negative phase (k-step Gibbs sampling)
        v_sample = v0
        for _ in range(self.k):
            h_prob, h_sample = self.sample_hidden(v_sample)
            v_prob, v_sample = self.sample_visible(h_sample)
        
        # Gradients
        positive_grad = torch.mm(v0.t(), h0_prob)
        negative_grad = torch.mm(v_sample.t(), h_prob)
        
        # Reconstruction error
        reconstruction_error = torch.mean((v0 - v_prob) ** 2)
        
        return positive_grad, negative_grad, reconstruction_error
    
    def train_batch(self, v: torch.Tensor):
        """
        Train on a batch using CD.
        
        Args:
            v: Batch of visible units (batch, num_visible)
            
        Returns:
            reconstruction_error: Mean reconstruction error
        """
        batch_size = v.size(0)
        
        # Contrastive divergence
        pos_grad, neg_grad, recon_error = self.contrastive_divergence(v)
        
        # Update weights and biases
        self.W.data += self.learning_rate * (pos_grad - neg_grad) / batch_size
        self.v_bias.data += self.learning_rate * torch.mean(v - self.sample_visible(self.sample_hidden(v)[1])[0], dim=0)
        self.h_bias.data += self.learning_rate * torch.mean(self.sample_hidden(v)[0] - self.sample_hidden(self.sample_visible(self.sample_hidden(v)[1])[0])[0], dim=0)
        
        return recon_error.item()
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Encode visible to hidden (inference).
        
        Args:
            v: Visible units (batch, num_visible)
            
        Returns:
            h_prob: Hidden unit probabilities
        """
        h_prob, _ = self.sample_hidden(v)
        return h_prob
    
    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct visible from visible (v -> h -> v).
        
        Args:
            v: Visible units (batch, num_visible)
            
        Returns:
            v_recon: Reconstructed visible units
        """
        h_prob, h_sample = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_sample)
        return v_prob
