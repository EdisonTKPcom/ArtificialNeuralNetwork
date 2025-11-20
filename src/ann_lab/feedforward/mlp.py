"""
Multi-Layer Perceptron (MLP) - Deep feedforward neural networks.

MLPs consist of multiple layers of neurons with non-linear activations,
enabling them to learn complex non-linear decision boundaries.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class MLPClassifier(BaseModel):
    """
    Multi-Layer Perceptron for classification tasks.
    
    A fully-connected feedforward network with configurable hidden layers,
    activations, dropout, and batch normalization.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes
        num_classes: Number of output classes
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability (0 = no dropout)
        use_batch_norm: Whether to use batch normalization
        
    Example:
        >>> model = MLPClassifier(
        ...     input_dim=784,
        ...     hidden_dims=[128, 64],
        ...     num_classes=10
        ... )
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns class labels).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Class predictions of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = logits.argmax(dim=1)
        return predictions


class MLPRegressor(BaseModel):
    """
    Multi-Layer Perceptron for regression tasks.
    
    Similar to MLPClassifier but outputs continuous values
    instead of class logits.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes
        output_dim: Number of output values (default: 1)
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        dropout: Dropout probability (0 = no dropout)
        use_batch_norm: Whether to use batch normalization
        
    Example:
        >>> model = MLPRegressor(
        ...     input_dim=10,
        ...     hidden_dims=[64, 32],
        ...     output_dim=1
        ... )
        >>> x = torch.randn(32, 10)
        >>> output = model(x)  # Shape: (32, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.network(x)


class SingleLayerFeedforward(BaseModel):
    """
    Single-layer feedforward network (one hidden layer).
    
    The simplest non-linear neural network. Good baseline model
    and useful for understanding how hidden layers enable non-linear mappings.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer size
        num_classes: Number of output classes
        activation: Activation function
        
    Example:
        >>> model = SingleLayerFeedforward(input_dim=10, hidden_dim=32, num_classes=2)
        >>> x = torch.randn(8, 10)
        >>> output = model(x)  # Shape: (8, 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = 'relu',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Single hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
