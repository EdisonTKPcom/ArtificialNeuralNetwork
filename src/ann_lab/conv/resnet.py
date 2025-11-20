"""
ResNet - Residual Networks with skip connections.

ResNet (He et al., 2015) introduced residual connections (skip connections)
that enable training of very deep networks (100+ layers) by mitigating
vanishing gradient problems.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class ResidualBlock(nn.Module):
    """
    Basic residual block with skip connection.
    
    Implements: output = F(x) + x
    where F(x) is a series of conv-bn-relu layers
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution
        downsample: Optional downsampling layer for skip connection
    """
    
    expansion = 1  # Output channels multiplier
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block for deeper ResNets (ResNet-50+).
    
    Uses 1x1 → 3x3 → 1x1 convolutions to reduce parameters.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of bottleneck channels
        stride: Stride for 3x3 convolution
        downsample: Optional downsampling for skip connection
    """
    
    expansion = 4  # Output channels = out_channels * 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        # 1x1 conv (reduce dimensions)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv (expand dimensions)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(BaseModel):
    """
    ResNet architecture with configurable depth.
    
    Args:
        block: Type of residual block (ResidualBlock or BottleneckBlock)
        layers: Number of blocks in each stage
        input_channels: Number of input channels
        num_classes: Number of output classes
        small_input: Adapt for small images (32x32) vs ImageNet (224x224)
        
    Example:
        >>> model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10, small_input=True)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)
    """
    
    def __init__(
        self,
        block,
        layers: List[int],
        input_channels: int = 3,
        num_classes: int = 1000,
        small_input: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.in_channels = 64
        
        # Initial convolution
        if small_input:
            # For CIFAR-10 sized images (32x32)
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        else:
            # For ImageNet sized images (224x224)
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if not small_input else nn.Identity()
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a residual stage with multiple blocks."""
        downsample = None
        
        # Downsample skip connection if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def ResNet18(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> ResNet:
    """ResNet-18 model."""
    return ResNet(ResidualBlock, [2, 2, 2, 2], input_channels, num_classes, small_input)


def ResNet34(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> ResNet:
    """ResNet-34 model."""
    return ResNet(ResidualBlock, [3, 4, 6, 3], input_channels, num_classes, small_input)


def ResNet50(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> ResNet:
    """ResNet-50 model with bottleneck blocks."""
    return ResNet(BottleneckBlock, [3, 4, 6, 3], input_channels, num_classes, small_input)


def ResNet101(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> ResNet:
    """ResNet-101 model with bottleneck blocks."""
    return ResNet(BottleneckBlock, [3, 4, 23, 3], input_channels, num_classes, small_input)


def ResNet152(input_channels: int = 3, num_classes: int = 1000, small_input: bool = False) -> ResNet:
    """ResNet-152 model with bottleneck blocks."""
    return ResNet(BottleneckBlock, [3, 8, 36, 3], input_channels, num_classes, small_input)
