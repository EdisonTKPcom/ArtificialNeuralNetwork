"""
MobileNet - Efficient CNN for mobile and embedded devices.

MobileNets use depthwise separable convolutions to dramatically reduce
parameters and computations while maintaining accuracy.
"""

from typing import List
import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    
    Factorizes standard convolution into:
    1. Depthwise convolution (spatial filtering per channel)
    2. Pointwise convolution (1x1 conv for channel mixing)
    
    This reduces parameters from k²·C_in·C_out to k²·C_in + C_in·C_out
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for depthwise conv
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Depthwise convolution (one filter per input channel)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Pointwise convolution (1x1 conv)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block for MobileNetV2.
    
    Uses expansion → depthwise → projection structure with residual connection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for depthwise conv
        expand_ratio: Channel expansion factor
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection phase (linear bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(BaseModel):
    """
    MobileNetV2 architecture.
    
    Uses inverted residuals with linear bottlenecks for efficiency.
    Excellent for mobile and edge devices.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        width_mult: Width multiplier for channels (e.g., 0.5, 1.0, 1.5)
        
    Example:
        >>> model = MobileNetV2(input_channels=3, num_classes=10)
        >>> x = torch.randn(16, 3, 224, 224)
        >>> output = model(x)
        
    Reference:
        Sandler et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        width_mult: float = 1.0,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Building block configuration: [expand_ratio, channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Initial layer
        input_channel = int(32 * width_mult)
        self.features = [
            nn.Sequential(
                nn.Conv2d(input_channels, input_channel, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
            )
        ]
        
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        # Final conv layer
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True),
            )
        )
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV1(BaseModel):
    """
    MobileNetV1 using depthwise separable convolutions.
    
    Simpler than V2 but still very efficient.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        width_mult: Width multiplier
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        width_mult: float = 1.0,
    ):
        super().__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return DepthwiseSeparableConv(inp, oup, stride)
        
        # Build network
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * width_mult), 2),
            conv_dw(int(32 * width_mult), int(64 * width_mult), 1),
            conv_dw(int(64 * width_mult), int(128 * width_mult), 2),
            conv_dw(int(128 * width_mult), int(128 * width_mult), 1),
            conv_dw(int(128 * width_mult), int(256 * width_mult), 2),
            conv_dw(int(256 * width_mult), int(256 * width_mult), 1),
            conv_dw(int(256 * width_mult), int(512 * width_mult), 2),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(512 * width_mult), 1),
            conv_dw(int(512 * width_mult), int(1024 * width_mult), 2),
            conv_dw(int(1024 * width_mult), int(1024 * width_mult), 1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * width_mult), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
