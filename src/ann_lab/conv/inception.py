"""
Inception / GoogLeNet - Multi-scale feature extraction.

Inception modules process input at multiple scales simultaneously using
parallel convolutions of different sizes (1x1, 3x3, 5x5) plus pooling.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class InceptionModule(nn.Module):
    """
    Inception module with parallel multi-scale convolutions.
    
    Branches:
    1. 1x1 conv
    2. 1x1 → 3x3 conv
    3. 1x1 → 5x5 conv
    4. 3x3 maxpool → 1x1 conv
    
    All branches are concatenated along the channel dimension.
    
    Args:
        in_channels: Number of input channels
        ch1x1: Channels for 1x1 convolution branch
        ch3x3_reduce: Channels for 1x1 before 3x3
        ch3x3: Channels for 3x3 convolution
        ch5x5_reduce: Channels for 1x1 before 5x5
        ch5x5: Channels for 5x5 convolution
        pool_proj: Channels for 1x1 after pool
    """
    
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super().__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True),
        )
        
        # 1x1 → 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
        )
        
        # 1x1 → 5x5 conv branch (often replaced with two 3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
        )
        
        # MaxPool → 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all branches and concatenate."""
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class InceptionV1(BaseModel):
    """
    GoogLeNet / Inception V1 architecture.
    
    Uses Inception modules to efficiently capture multi-scale features.
    Much more parameter-efficient than VGGNet despite similar depth.
    
    Args:
        input_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        aux_logits: Use auxiliary classifiers for training (original paper)
        small_input: Adapt for small images (32x32)
        
    Example:
        >>> model = InceptionV1(input_channels=3, num_classes=10, small_input=True)
        >>> x = torch.randn(16, 3, 32, 32)
        >>> output = model(x)
        
    Reference:
        Szegedy et al. "Going Deeper with Convolutions" (2015)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1000,
        aux_logits: bool = False,
        small_input: bool = False,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.aux_logits = aux_logits
        
        if small_input:
            # Simplified for smaller images
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.conv2 = nn.Identity()
        
        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Inception blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
