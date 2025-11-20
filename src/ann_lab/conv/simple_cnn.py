"""
Simple Convolutional Neural Network architectures.

Basic CNNs that demonstrate core concepts: convolution, pooling, and fully-connected layers.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    Simple 2-layer CNN for image classification.
    
    Architecture:
    - Conv1 (3x3) → ReLU → MaxPool
    - Conv2 (3x3) → ReLU → MaxPool
    - Flatten → FC1 → ReLU → Dropout → FC2
    
    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        input_size: Input image size (height, width)
        
    Example:
        >>> model = SimpleCNN(input_channels=1, num_classes=10, input_size=(28, 28))
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: Tuple[int, int] = (28, 28),
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        h, w = input_size
        h, w = h // 4, w // 4  # After two pooling layers
        flatten_size = 64 * h * w
        
        # Fully connected layers
        self.fc1 = nn.Linear(flatten_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BasicCNN(BaseModel):
    """
    Configurable basic CNN with multiple conv blocks.
    
    Each block consists of: Conv → BatchNorm → ReLU → MaxPool
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        channels: List of channel sizes for each conv layer
        kernel_sizes: Kernel sizes for each layer (or single int for all)
        fc_hidden: Hidden size for fully-connected layer
        dropout: Dropout probability
        input_size: Input image size
        
    Example:
        >>> model = BasicCNN(
        ...     input_channels=3,
        ...     num_classes=10,
        ...     channels=[32, 64, 128],
        ...     input_size=(32, 32)
        ... )
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: List[int] = [32, 64, 128],
        kernel_sizes: int | List[int] = 3,
        fc_hidden: int = 256,
        dropout: float = 0.5,
        input_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Handle kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(channels)
        
        # Build convolutional blocks
        conv_layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            in_channels = out_channels
        
        self.conv_blocks = nn.Sequential(*conv_layers)
        
        # Calculate flatten size
        h, w = input_size
        h, w = h // (2 ** len(channels)), w // (2 ** len(channels))
        flatten_size = channels[-1] * h * w
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x
