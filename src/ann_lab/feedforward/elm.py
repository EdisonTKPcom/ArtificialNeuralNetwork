"""
Extreme Learning Machine (ELM).

ELM is a fast-training feedforward network where hidden layer weights
are randomly initialized and fixed, with only the output layer trained.
This makes training extremely fast compared to backpropagation.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class ExtremeLearningMachine(BaseModel):
    """
    Extreme Learning Machine for classification or regression.
    
    Key idea: Randomly initialize hidden layer weights and keep them fixed.
    Only train the output layer using least-squares or pseudo-inverse.
    
    This approach:
    - Trains much faster than backpropagation
    - No iterative optimization needed
    - Good generalization for many tasks
    - Works best with large hidden layers
    
    Args:
        input_dim: Number of input features
        hidden_dim: Number of hidden neurons (typically large, e.g., 500-1000)
        output_dim: Number of outputs
        activation: Activation function ('sigmoid', 'tanh', 'relu')
        C: Regularization parameter (higher = less regularization)
        
    Example:
        >>> model = ExtremeLearningMachine(input_dim=10, hidden_dim=500, output_dim=3)
        >>> # Train using fit() method instead of typical training loop
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        
    Notes:
        - Unlike typical neural networks, use fit() method for training
        - No gradient descent or backpropagation
        - Hidden weights are never updated after initialization
        - Output weights computed analytically
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: Literal['sigmoid', 'tanh', 'relu', 'linear'] = 'sigmoid',
        C: float = 1.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.C = C
        
        # Hidden layer weights (random, frozen)
        self.input_weights = nn.Parameter(
            torch.randn(input_dim, hidden_dim) * 0.5,
            requires_grad=False
        )
        self.hidden_bias = nn.Parameter(
            torch.randn(hidden_dim) * 0.5,
            requires_grad=False
        )
        
        # Output layer weights (to be computed analytically)
        self.output_weights = nn.Parameter(
            torch.zeros(hidden_dim, output_dim),
            requires_grad=False
        )
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.is_fitted = False
    
    def hidden_layer_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hidden layer activations.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Hidden activations of shape (batch_size, hidden_dim)
        """
        # H = activation(X @ W + b)
        h = torch.matmul(x, self.input_weights) + self.hidden_bias
        return self.activation(h)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Hidden layer
        H = self.hidden_layer_output(x)
        
        # Output layer
        output = torch.matmul(H, self.output_weights)
        
        return output
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        is_classification: bool = True,
    ) -> None:
        """
        Fit the ELM model using analytical solution.
        
        For classification: converts labels to one-hot encoding.
        For regression: uses targets as-is.
        
        Args:
            X: Training data of shape (num_samples, input_dim)
            y: Training targets of shape (num_samples,) or (num_samples, output_dim)
            is_classification: Whether this is a classification task
        """
        # Ensure tensors are on same device
        X = X.to(self._device)
        y = y.to(self._device)
        
        # Compute hidden layer output
        H = self.hidden_layer_output(X)  # (num_samples, hidden_dim)
        
        # Prepare targets
        if is_classification and y.dim() == 1:
            # Convert to one-hot encoding
            num_classes = self.output_dim
            T = torch.zeros(y.size(0), num_classes, device=self._device)
            T[torch.arange(y.size(0)), y.long()] = 1.0
        else:
            T = y if y.dim() == 2 else y.unsqueeze(1)
        
        # Compute output weights using regularized least squares
        # β = (H^T H + I/C)^(-1) H^T T
        
        # Add regularization
        identity = torch.eye(self.hidden_dim, device=self._device) / self.C
        
        # Solve: (H^T H + I/C) β = H^T T
        # Using pseudo-inverse for numerical stability
        try:
            HTH_reg = torch.matmul(H.t(), H) + identity
            HTT = torch.matmul(H.t(), T)
            self.output_weights.data = torch.linalg.solve(HTH_reg, HTT)
        except:
            # Fallback to pseudo-inverse if solve fails
            HTH_reg = torch.matmul(H.t(), H) + identity
            HTT = torch.matmul(H.t(), T)
            self.output_weights.data = torch.matmul(torch.linalg.pinv(HTH_reg), HTT)
        
        self.is_fitted = True
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        For classification: returns class labels.
        For regression: returns continuous values.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predictions
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            if self.output_dim > 1:
                # Classification: return argmax
                predictions = outputs.argmax(dim=1)
            else:
                # Regression: return values
                predictions = outputs.squeeze()
        
        return predictions


class ELMClassifier(BaseModel):
    """
    ELM specifically for classification with convenient interface.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Number of hidden neurons
        num_classes: Number of classes
        activation: Activation function
        C: Regularization parameter
        
    Example:
        >>> model = ELMClassifier(input_dim=784, hidden_dim=1000, num_classes=10)
        >>> model.fit(X_train, y_train)
        >>> accuracy = (model.predict(X_test) == y_test).float().mean()
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        activation: str = 'sigmoid',
        C: float = 1.0,
    ):
        super().__init__()
        
        self.elm = ExtremeLearningMachine(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            activation=activation,
            C=C,
        )
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the classifier."""
        self.elm.fit(X, y, is_classification=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.elm.forward(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        return self.elm.predict(x)
    
    def to_device(self, device: torch.device) -> "ELMClassifier":
        """Move to device."""
        self.elm.to_device(device)
        return super().to_device(device)
