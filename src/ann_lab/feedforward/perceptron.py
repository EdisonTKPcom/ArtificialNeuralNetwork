"""
Perceptron - The simplest neural network model.

A single-layer linear classifier that learns a decision boundary
to separate two classes. Historical importance as the first neural network model.
"""

from typing import Optional
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class Perceptron(BaseModel):
    """
    Single-layer perceptron for binary classification.
    
    The perceptron learns a linear decision boundary of the form:
        y = sign(w^T x + b)
    
    Args:
        input_dim: Number of input features
        activation: Activation function ('sign', 'sigmoid', or 'none')
        
    Example:
        >>> model = Perceptron(input_dim=2)
        >>> x = torch.randn(32, 2)  # Batch of 32 samples
        >>> output = model(x)  # Shape: (32, 1)
    """
    
    def __init__(self, input_dim: int, activation: str = 'sign'):
        super().__init__()
        
        self.input_dim = input_dim
        self.activation_type = activation
        
        # Single linear layer
        self.linear = nn.Linear(input_dim, 1)
        
        # Activation function
        if activation == 'sign':
            self.activation = torch.sign
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        out = self.linear(x)
        return self.activation(out)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns class labels).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Binary predictions {0, 1} of shape (batch_size,)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > 0).long().squeeze()
        return predictions


class MultiClassPerceptron(BaseModel):
    """
    Multi-class perceptron using one-vs-all approach.
    
    Extends the binary perceptron to handle multiple classes
    by learning a separate decision boundary for each class.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        
    Example:
        >>> model = MultiClassPerceptron(input_dim=10, num_classes=3)
        >>> x = torch.randn(32, 10)
        >>> output = model(x)  # Shape: (32, 3)
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Linear layer for multi-class classification
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.linear(x)
    
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
