"""
LeNet - Classic convolutional neural network for digit recognition.

LeNet-5 was one of the earliest CNNs, designed by Yann LeCun for handwritten
digit recognition (MNIST). It demonstrated the power of convolution and pooling.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class LeNet5(BaseModel):
    """
    LeNet-5 architecture (LeCun et al., 1998).
    
    Original architecture designed for 32x32 grayscale images.
    Modified to work with various input sizes.
    
    Architecture:
    - Conv1 (5x5, 6 filters) → Tanh → AvgPool
    - Conv2 (5x5, 16 filters) → Tanh → AvgPool
    - Conv3 (5x5, 120 filters) → Tanh
    - FC1 (84) → Tanh
    - FC2 (num_classes)
    
    Args:
        input_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes
        use_relu: Use ReLU instead of Tanh (modern variant)
        
    Example:
        >>> model = LeNet5(input_channels=1, num_classes=10)
        >>> x = torch.randn(32, 1, 32, 32)
        >>> output = model(x)  # Shape: (32, 10)
        
    Reference:
        LeCun et al. "Gradient-Based Learning Applied to Document Recognition"
        Proceedings of the IEEE, 1998
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        use_relu: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Activation function
        activation = nn.ReLU() if use_relu else nn.Tanh()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.act1 = nn.Tanh() if not use_relu else nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2 = nn.Tanh() if not use_relu else nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.act3 = nn.Tanh() if not use_relu else nn.ReLU()
        
        # Classifier
        self.fc1 = nn.Linear(120, 84)
        self.act4 = nn.Tanh() if not use_relu else nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.conv1(x)  # (B, 6, 28, 28)
        x = self.act1(x)
        x = self.pool1(x)  # (B, 6, 14, 14)
        
        x = self.conv2(x)  # (B, 16, 10, 10)
        x = self.act2(x)
        x = self.pool2(x)  # (B, 16, 5, 5)
        
        x = self.conv3(x)  # (B, 120, 1, 1)
        x = self.act3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, 120)
        
        # Classifier
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        
        return x


class ModernLeNet(BaseModel):
    """
    Modernized LeNet with ReLU, MaxPooling, and Dropout.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Example:
        >>> model = ModernLeNet(input_channels=1, num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),  # For 28x28 input
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(500, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x
