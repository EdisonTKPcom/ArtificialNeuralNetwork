"""
Dataset utilities and common dataset loaders.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = './data',
    val_split: float = 0.1,
    num_workers: int = 0,
    flatten: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size
        data_dir: Directory to store/load data
        val_split: Fraction of training data to use for validation
        num_workers: Number of data loading workers
        flatten: If True, flatten images to vectors
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Download datasets
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Split training into train/val
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 64,
    data_dir: str = './data',
    val_split: float = 0.1,
    num_workers: int = 0,
    augment: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size
        data_dir: Directory to store/load data
        val_split: Fraction of training data to use for validation
        num_workers: Number of data loading workers
        augment: If True, apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Normalization values for CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Download datasets
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split training into train/val
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


class SyntheticDataset(Dataset):
    """
    Generate synthetic data for testing.
    
    Useful for quick experiments without downloading real datasets.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 10,
        num_classes: int = 2,
        task: str = 'classification',
        noise: float = 0.1,
    ):
        """
        Args:
            num_samples: Number of samples
            input_dim: Input dimensionality
            num_classes: Number of classes (for classification)
            task: 'classification' or 'regression'
            noise: Noise level
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task
        
        # Generate random data
        self.data = torch.randn(num_samples, input_dim)
        
        if task == 'classification':
            # Generate labels based on linear combination + noise
            weights = torch.randn(input_dim, 1)
            logits = self.data @ weights + noise * torch.randn(num_samples, 1)
            self.targets = (logits.squeeze() > 0).long()
            
            if num_classes > 2:
                # Multi-class: use multiple decision boundaries
                self.targets = torch.randint(0, num_classes, (num_samples,))
        else:
            # Regression: linear relationship + noise
            weights = torch.randn(input_dim, 1)
            self.targets = self.data @ weights + noise * torch.randn(num_samples, 1)
            self.targets = self.targets.squeeze()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def get_synthetic_loaders(
    num_samples: int = 1000,
    batch_size: int = 32,
    input_dim: int = 10,
    num_classes: int = 2,
    task: str = 'classification',
    val_split: float = 0.2,
    test_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get synthetic data loaders.
    
    Args:
        num_samples: Total number of samples
        batch_size: Batch size
        input_dim: Input dimensionality
        num_classes: Number of classes
        task: 'classification' or 'regression'
        val_split: Fraction for validation
        test_split: Fraction for test
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = SyntheticDataset(num_samples, input_dim, num_classes, task)
    
    # Split into train/val/test
    test_size = int(num_samples * test_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
