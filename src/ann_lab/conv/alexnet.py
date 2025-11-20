"""
AlexNet - Deep CNN with ReLU and Dropout.

AlexNet (Krizhevsky et al., 2012) popularized deep CNNs and won ImageNet 2012.
Key innovations: ReLU activations, dropout, GPU training, data augmentation.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class AlexNet(BaseModel):
    """
    AlexNet architecture for image classification.
    
    Original network designed for 227x227 images. This implementation
    is adapted for smaller images (e.g., 32x32 or 64x64).
    
    Architecture:
    - 5 convolutional layers
    - 3 fully-connected layers
    - ReLU activations
    - Dropout for regularization
    - Local Response Normalization (optional, replaced with BatchNorm)
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        dropout: Dropout probability
        use_batch_norm: Use BatchNorm instead of LRN
        small_input: Adapt for small images (32x32) instead of ImageNet size
        
    Example:
        >>> model = AlexNet(input_channels=3, num_classes=10, small_input=True)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)  # Shape: (16, 10)
        
    Reference:
        Krizhevsky et al. "ImageNet Classification with Deep CNN"
        NeurIPS 2012
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        small_input: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.small_input = small_input
        
        if small_input:
            # Adapted for 32x32 or 64x64 images
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.BatchNorm2d(384) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            fc_input_size = 256 * 4 * 4  # For 32x32 input
        else:
            # Original AlexNet for ImageNet-size images
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            fc_input_size = 256 * 6 * 6
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) if not small_input else nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
