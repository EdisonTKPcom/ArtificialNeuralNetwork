"""
Deep Belief Network (DBN) - stack of RBMs.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel
from .rbm import RestrictedBoltzmannMachine


class DeepBeliefNetwork(BaseModel):
    """
    Deep Belief Network (stacked RBMs).
    
    Consists of multiple layers of RBMs stacked on top of each other.
    Can be pre-trained layer-wise, then fine-tuned with backpropagation.
    
    Good for: unsupervised pre-training, feature learning, generative modeling.
    
    Args:
        layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ...]
        learning_rate: Learning rate for RBM training
        k: CD-k steps
    """
    
    def __init__(
        self,
        layer_sizes: list,
        learning_rate: float = 0.01,
        k: int = 1
    ):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Create RBM layers
        self.rbm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            rbm = RestrictedBoltzmannMachine(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i+1],
                learning_rate=learning_rate,
                k=k
            )
            self.rbm_layers.append(rbm)
        
    def pretrain_layer(
        self,
        layer_idx: int,
        data: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Pre-train a single RBM layer.
        
        Args:
            layer_idx: Index of layer to train
            data: Training data for this layer
            num_epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        rbm = self.rbm_layers[layer_idx]
        num_samples = data.size(0)
        
        for epoch in range(num_epochs):
            epoch_error = 0
            num_batches = 0
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = data[batch_indices]
                
                error = rbm.train_batch(batch)
                epoch_error += error
                num_batches += 1
            
            avg_error = epoch_error / num_batches
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Layer {layer_idx}, Epoch {epoch+1}/{num_epochs}, Error: {avg_error:.4f}")
    
    def pretrain(
        self,
        data: torch.Tensor,
        num_epochs_per_layer: int = 10,
        batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Greedy layer-wise pre-training.
        
        Train each RBM layer sequentially, using outputs of previous layer.
        
        Args:
            data: Training data (num_samples, input_dim)
            num_epochs_per_layer: Epochs per layer
            batch_size: Batch size
            verbose: Whether to print progress
        """
        current_data = data
        
        for layer_idx in range(self.num_layers):
            if verbose:
                print(f"\nPre-training layer {layer_idx}/{self.num_layers}")
            
            # Train this layer
            self.pretrain_layer(
                layer_idx,
                current_data,
                num_epochs_per_layer,
                batch_size,
                verbose
            )
            
            # Transform data for next layer
            if layer_idx < self.num_layers - 1:
                with torch.no_grad():
                    current_data = self.rbm_layers[layer_idx](current_data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            x: Input (batch, input_dim)
            
        Returns:
            output: Final layer activations
        """
        h = x
        for rbm in self.rbm_layers:
            h = rbm(h)
        return h
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input (encode then decode).
        
        Args:
            x: Input (batch, input_dim)
            
        Returns:
            reconstruction: Reconstructed input
        """
        # Encode
        h = self.forward(x)
        
        # Decode (go back through layers)
        for rbm in reversed(self.rbm_layers):
            h = rbm.sample_visible(h)[0]
        
        return h
