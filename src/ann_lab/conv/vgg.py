"""
VGGNet - Very deep networks with small convolution filters.

VGG networks (Simonyan & Zisserman, 2014) demonstrated that depth matters.
Uses only 3x3 convolutions stacked deeply, achieving excellent performance.
"""

from typing import List
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class VGG(BaseModel):
    """
    VGG network with configurable depth.
    
    Key principles:
    - Use small (3x3) convolution filters
    - Stack convolutions to get effective large receptive fields
    - Use 2x2 max pooling
    - Double channels after each pooling
    
    Args:
        config: List specifying layers ('M' for MaxPool, int for conv channels)
        input_channels: Number of input channels
        num_classes: Number of output classes
        batch_norm: Use batch normalization
        dropout: Dropout probability
        small_input: Adapt for small images (32x32)
        
    Example:
        >>> config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
        >>> model = VGG(config, input_channels=3, num_classes=10)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        config: List,
        input_channels: int = 3,
        num_classes: int = 1000,
        batch_norm: bool = True,
        dropout: float = 0.5,
        small_input: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Build feature extraction layers
        self.features = self._make_layers(config, input_channels, batch_norm)
        
        # Adaptive pooling for flexibility with input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) if not small_input else nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        if small_input:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes),
            )
    
    def _make_layers(self, config: List, input_channels: int, batch_norm: bool) -> nn.Sequential:
        """Build VGG conv layers from config."""
        layers = []
        in_channels = input_channels
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# VGG configurations
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG11(VGG):
    """VGG-11 model."""
    def __init__(self, input_channels: int = 3, num_classes: int = 1000, 
                 batch_norm: bool = True, small_input: bool = False):
        super().__init__(VGG_CONFIGS['VGG11'], input_channels, num_classes, batch_norm, small_input=small_input)


class VGG13(VGG):
    """VGG-13 model."""
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 batch_norm: bool = True, small_input: bool = False):
        super().__init__(VGG_CONFIGS['VGG13'], input_channels, num_classes, batch_norm, small_input=small_input)


class VGG16(VGG):
    """VGG-16 model (most commonly used)."""
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 batch_norm: bool = True, small_input: bool = False):
        super().__init__(VGG_CONFIGS['VGG16'], input_channels, num_classes, batch_norm, small_input=small_input)


class VGG19(VGG):
    """VGG-19 model (deepest variant)."""
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 batch_norm: bool = True, small_input: bool = False):
        super().__init__(VGG_CONFIGS['VGG19'], input_channels, num_classes, batch_norm, small_input=small_input)
