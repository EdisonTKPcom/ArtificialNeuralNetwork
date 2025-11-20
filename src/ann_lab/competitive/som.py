"""
Self-Organizing Map (SOM) / Kohonen Network.

Unsupervised learning algorithm that produces a low-dimensional
representation of the input space.
"""

import torch
import torch.nn as nn
import numpy as np
from ..core.base_model import BaseModel


class SelfOrganizingMap(BaseModel):
    """
    Self-Organizing Map for unsupervised clustering and visualization.
    
    Creates a topological map where similar inputs activate nearby neurons.
    Uses competitive learning with neighborhood cooperation.
    
    Good for: dimensionality reduction, visualization, clustering.
    
    Args:
        input_dim: Dimension of input vectors
        map_height: Height of the 2D map
        map_width: Width of the 2D map
        learning_rate: Initial learning rate
        sigma: Initial neighborhood radius
    """
    
    def __init__(
        self,
        input_dim: int,
        map_height: int = 10,
        map_width: int = 10,
        learning_rate: float = 0.5,
        sigma: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.map_height = map_height
        self.map_width = map_width
        self.num_neurons = map_height * map_width
        
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        
        # Initialize weight vectors randomly
        self.weights = nn.Parameter(
            torch.randn(map_height, map_width, input_dim),
            requires_grad=False  # SOM uses custom learning rule, not backprop
        )
        
        # Normalize weights
        self.weights.data = nn.functional.normalize(self.weights.data, dim=2)
        
        # Pre-compute neuron positions for neighborhood calculation
        self.neuron_positions = self._compute_neuron_positions()
        
    def _compute_neuron_positions(self) -> torch.Tensor:
        """Compute 2D positions of all neurons."""
        positions = torch.zeros(self.map_height, self.map_width, 2)
        for i in range(self.map_height):
            for j in range(self.map_width):
                positions[i, j] = torch.tensor([i, j], dtype=torch.float32)
        return positions
    
    def find_bmu(self, x: torch.Tensor) -> tuple:
        """
        Find Best Matching Unit (BMU) for input x.
        
        Args:
            x: Input vector (input_dim,)
            
        Returns:
            bmu_idx: (i, j) indices of BMU
        """
        # Compute distances to all neurons
        distances = torch.sum((self.weights - x) ** 2, dim=2)
        
        # Find minimum
        bmu_idx = torch.argmin(distances)
        bmu_i = bmu_idx // self.map_width
        bmu_j = bmu_idx % self.map_width
        
        return bmu_i.item(), bmu_j.item()
    
    def update_weights(
        self,
        x: torch.Tensor,
        bmu_idx: tuple,
        learning_rate: float,
        sigma: float
    ):
        """
        Update weights based on BMU and neighborhood function.
        
        Args:
            x: Input vector
            bmu_idx: (i, j) indices of BMU
            learning_rate: Current learning rate
            sigma: Current neighborhood radius
        """
        bmu_i, bmu_j = bmu_idx
        bmu_pos = self.neuron_positions[bmu_i, bmu_j]
        
        # Compute neighborhood function (Gaussian)
        distances_sq = torch.sum((self.neuron_positions - bmu_pos) ** 2, dim=2)
        neighborhood = torch.exp(-distances_sq / (2 * sigma ** 2))
        
        # Update weights
        # w_new = w_old + lr * neighborhood * (x - w_old)
        neighborhood = neighborhood.unsqueeze(2)
        delta = learning_rate * neighborhood * (x - self.weights.data)
        self.weights.data += delta
        
    def train_som(
        self,
        data: torch.Tensor,
        num_epochs: int = 100,
        verbose: bool = False
    ):
        """
        Train the SOM on data.
        
        Args:
            data: Training data (num_samples, input_dim)
            num_epochs: Number of training epochs
            verbose: Whether to print progress
        """
        num_samples = data.size(0)
        
        for epoch in range(num_epochs):
            # Decay learning rate and sigma
            learning_rate = self.initial_learning_rate * (1 - epoch / num_epochs)
            sigma = self.initial_sigma * (1 - epoch / num_epochs)
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            for idx in indices:
                x = data[idx]
                
                # Find BMU
                bmu_idx = self.find_bmu(x)
                
                # Update weights
                self.update_weights(x, bmu_idx, learning_rate, sigma)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input to BMU coordinates.
        
        Args:
            x: Input (batch, input_dim) or (input_dim,)
            
        Returns:
            bmu_coords: BMU coordinates for each input
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        bmu_coords = torch.zeros(batch_size, 2)
        
        for i in range(batch_size):
            bmu_i, bmu_j = self.find_bmu(x[i])
            bmu_coords[i] = torch.tensor([bmu_i, bmu_j])
        
        return bmu_coords
    
    def get_u_matrix(self) -> torch.Tensor:
        """
        Compute U-matrix (unified distance matrix) for visualization.
        
        Shows average distance between each neuron and its neighbors.
        
        Returns:
            u_matrix: Distance matrix (map_height, map_width)
        """
        u_matrix = torch.zeros(self.map_height, self.map_width)
        
        for i in range(self.map_height):
            for j in range(self.map_width):
                neighbors = []
                
                # Get neighbors (4-connectivity)
                if i > 0:
                    neighbors.append(self.weights[i-1, j])
                if i < self.map_height - 1:
                    neighbors.append(self.weights[i+1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j-1])
                if j < self.map_width - 1:
                    neighbors.append(self.weights[i, j+1])
                
                if neighbors:
                    neighbors = torch.stack(neighbors)
                    distances = torch.norm(self.weights[i, j] - neighbors, dim=1)
                    u_matrix[i, j] = distances.mean()
        
        return u_matrix
