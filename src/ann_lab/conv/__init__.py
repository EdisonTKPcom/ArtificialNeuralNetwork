"""
Convolutional Neural Networks (CNNs) for image and spatial data processing.
"""

from .simple_cnn import SimpleCNN, BasicCNN
from .lenet import LeNet5, LeNet5 as LeNet  # Alias for convenience
from .alexnet import AlexNet
from .vgg import VGG11, VGG13, VGG16, VGG19
from .inception import InceptionV1, InceptionModule
from .resnet import ResNet18, ResNet34, ResNet50, ResNet, ResidualBlock
from .densenet import DenseNet121, DenseBlock
from .mobilenet_like import MobileNetV2

__all__ = [
    "SimpleCNN",
    "BasicCNN",
    "LeNet5",
    "LeNet",  # Alias
    "AlexNet",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
    "InceptionV1",
    "InceptionModule",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet",  # Generic ResNet
    "ResidualBlock",
    "DenseNet121",
    "DenseBlock",
    "MobileNetV2",
]
