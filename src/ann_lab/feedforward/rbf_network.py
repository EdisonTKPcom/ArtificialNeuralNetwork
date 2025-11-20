"""
Radial Basis Function Network (RBFN).

RBFNs use radial basis functions (typically Gaussian) as activation functions
in the hidden layer, providing a different approach to non-linear mapping
compared to traditional MLPs.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModel


class RBFNetwork(BaseModel):
    """
    Radial Basis Function Network for classification or regression.
    
    Architecture:
    - Input layer → RBF hidden layer → Linear output layer
    
    The hidden layer uses Gaussian RBF activations:
        φ(x) = exp(-γ * ||x - c||²)
    
    where c are the center points and γ controls the width.
    
    Args:
        input_dim: Number of input features
        num_centers: Number of RBF centers
        output_dim: Number of outputs
        rbf_gamma: Width parameter for RBF (higher = narrower)
        init_centers: How to initialize centers ('random' or 'kmeans')
        
    Example:
        >>> model = RBFNetwork(input_dim=10, num_centers=20, output_dim=3)
        >>> x = torch.randn(32, 10)
        >>> output = model(x)  # Shape: (32, 3)
        
    Notes:
        - Centers can be initialized randomly or using k-means clustering
        - The gamma parameter is crucial for performance
        - Works well for function approximation and pattern recognition
    """
    
    def __init__(
        self,
        input_dim: int,
        num_centers: int,
        output_dim: int,
        rbf_gamma: float = 1.0,
        init_centers: Literal['random', 'fixed'] = 'random',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.rbf_gamma = rbf_gamma
        
        # RBF centers (learnable parameters)
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        
        # Optional: learnable gamma per center
        # self.gammas = nn.Parameter(torch.ones(num_centers) * rbf_gamma)
        
        # Output layer
        self.linear = nn.Linear(num_centers, output_dim)
        
        # Initialize centers
        if init_centers == 'random':
            nn.init.normal_(self.centers, mean=0.0, std=1.0)
        elif init_centers == 'fixed':
            # Spread centers uniformly in feature space
            nn.init.uniform_(self.centers, -2.0, 2.0)
    
    def rbf_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF activations for input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            RBF activations of shape (batch_size, num_centers)
        """
        # Compute squared Euclidean distances to all centers
        # x: (batch, input_dim), centers: (num_centers, input_dim)
        # distances: (batch, num_centers)
        
        x_expanded = x.unsqueeze(1)  # (batch, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_centers, input_dim)
        
        distances_squared = torch.sum((x_expanded - centers_expanded) ** 2, dim=2)
        
        # Apply Gaussian RBF
        activations = torch.exp(-self.rbf_gamma * distances_squared)
        
        return activations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Compute RBF activations
        rbf_out = self.rbf_activations(x)
        
        # Linear output layer
        output = self.linear(rbf_out)
        
        return output
    
    def fit_centers(self, X: torch.Tensor, method: str = 'kmeans') -> None:
        """
        Fit RBF centers to data using clustering.
        
        Args:
            X: Training data of shape (num_samples, input_dim)
            method: Clustering method ('kmeans' or 'random_sample')
        """
        if method == 'random_sample':
            # Randomly sample centers from data
            indices = torch.randperm(X.size(0))[:self.num_centers]
            self.centers.data = X[indices].clone()
        
        elif method == 'kmeans':
            # Simple k-means clustering
            # Note: For production, use sklearn or a proper implementation
            centers = X[torch.randperm(X.size(0))[:self.num_centers]]
            
            for _ in range(10):  # 10 iterations of k-means
                # Assign points to nearest center
                distances = torch.cdist(X, centers)
                assignments = distances.argmin(dim=1)
                
                # Update centers
                for k in range(self.num_centers):
                    mask = assignments == k
                    if mask.any():
                        centers[k] = X[mask].mean(dim=0)
            
            self.centers.data = centers.clone()


class RBFClassifier(BaseModel):
    """
    RBF Network specifically for classification.
    
    Adds softmax output and prediction methods.
    
    Args:
        input_dim: Number of input features
        num_centers: Number of RBF centers
        num_classes: Number of classes
        rbf_gamma: Width parameter for RBF
        
    Example:
        >>> model = RBFClassifier(input_dim=10, num_centers=20, num_classes=3)
        >>> x = torch.randn(32, 10)
        >>> logits = model(x)  # Shape: (32, 3)
        >>> preds = model.predict(x)  # Shape: (32,)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_centers: int,
        num_classes: int,
        rbf_gamma: float = 1.0,
    ):
        super().__init__()
        
        self.rbf_network = RBFNetwork(
            input_dim=input_dim,
            num_centers=num_centers,
            output_dim=num_classes,
            rbf_gamma=rbf_gamma,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.rbf_network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Class predictions of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = logits.argmax(dim=1)
        return predictions
    
    def fit_centers(self, X: torch.Tensor, method: str = 'kmeans') -> None:
        """Fit RBF centers to training data."""
        self.rbf_network.fit_centers(X, method)
