"""
DenseNet - Densely Connected Convolutional Networks.

DenseNet (Huang et al., 2017) connects each layer to every subsequent layer
within a dense block, promoting feature reuse and gradient flow.
"""

from typing import List
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class DenseBlock(nn.Module):
    """
    Dense block where each layer receives features from all preceding layers.
    
    Each layer outputs k feature maps (growth rate), and all previous
    layer outputs are concatenated as input.
    
    Args:
        num_layers: Number of layers in the block
        in_channels: Number of input channels
        growth_rate: Number of output channels per layer (k)
        bn_size: Multiplicative factor for bottleneck layers
        drop_rate: Dropout probability
    """
    
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int = 32,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                self._make_dense_layer(layer_in_channels, growth_rate, bn_size, drop_rate)
            )
    
    def _make_dense_layer(
        self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> nn.Sequential:
        """Create a single dense layer with bottleneck."""
        layers = []
        
        # Bottleneck: 1x1 conv to reduce dimensions
        layers.extend([
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
        ])
        
        # 3x3 conv
        layers.extend([
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        ])
        
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through dense block, concatenating all layer outputs."""
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.
    
    Reduces feature map dimensions using 1x1 conv and pooling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)


class DenseNet(BaseModel):
    """
    DenseNet architecture with configurable depth.
    
    Args:
        block_config: Number of layers in each dense block
        growth_rate: Growth rate (k) - number of filters per layer
        num_init_features: Number of initial features after first conv
        bn_size: Bottleneck size multiplier
        drop_rate: Dropout rate
        input_channels: Number of input channels
        num_classes: Number of output classes
        small_input: Adapt for small images (32x32)
        
    Example:
        >>> model = DenseNet([6, 12, 24, 16], growth_rate=32, num_classes=10, small_input=True)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)
        
    Reference:
        Huang et al. "Densely Connected Convolutional Networks" (2017)
    """
    
    def __init__(
        self,
        block_config: List[int] = [6, 12, 24, 16],
        growth_rate: int = 32,
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        input_channels: int = 3,
        num_classes: int = 1000,
        small_input: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Initial convolution
        if small_input:
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, num_init_features, kernel_size=3,
                         stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, num_init_features, kernel_size=7,
                         stride=2, padding=3, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        
        # Build dense blocks and transition layers
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def DenseNet121(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> DenseNet:
    """DenseNet-121 model."""
    return DenseNet([6, 12, 24, 16], growth_rate=32, num_init_features=64,
                    input_channels=input_channels, num_classes=num_classes, small_input=small_input)


def DenseNet169(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> DenseNet:
    """DenseNet-169 model."""
    return DenseNet([6, 12, 32, 32], growth_rate=32, num_init_features=64,
                    input_channels=input_channels, num_classes=num_classes, small_input=small_input)


def DenseNet201(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> DenseNet:
    """DenseNet-201 model."""
    return DenseNet([6, 12, 48, 32], growth_rate=32, num_init_features=64,
                    input_channels=input_channels, num_classes=num_classes, small_input=small_input)


def DenseNet264(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> DenseNet:
    """DenseNet-264 model."""
    return DenseNet([6, 12, 64, 48], growth_rate=32, num_init_features=64,
                    input_channels=input_channels, num_classes=num_classes, small_input=small_input)
