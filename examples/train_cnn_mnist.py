"""
Example: Train CNN on MNIST or CIFAR-10 dataset.

Demonstrates training various CNN architectures on image classification tasks.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from ann_lab.conv import (
    SimpleCNN, BasicCNN, LeNet5, AlexNet, VGG11, VGG16,
    ResNet18, ResNet34, DenseNet121, MobileNetV2
)
from ann_lab.core.training_loops import train_model, evaluate_model
from ann_lab.core.datasets import get_mnist_loaders, get_cifar10_loaders


def get_model(model_name, dataset, num_classes):
    """Create model based on name and dataset."""
    small_input = (dataset == 'mnist')
    input_channels = 1 if dataset == 'mnist' else 3
    input_size = (28, 28) if dataset == 'mnist' else (32, 32)
    
    model_map = {
        'simple_cnn': lambda: SimpleCNN(input_channels, num_classes, input_size),
        'basic_cnn': lambda: BasicCNN(input_channels, num_classes, [32, 64, 128], input_size=input_size),
        'lenet': lambda: LeNet5(input_channels, num_classes, use_relu=True),
        'alexnet': lambda: AlexNet(input_channels, num_classes, small_input=True),
        'vgg11': lambda: VGG11(input_channels, num_classes, small_input=True),
        'vgg16': lambda: VGG16(input_channels, num_classes, small_input=True),
        'resnet18': lambda: ResNet18(input_channels, num_classes, small_input=True),
        'resnet34': lambda: ResNet34(input_channels, num_classes, small_input=True),
        'densenet121': lambda: DenseNet121(input_channels, num_classes, small_input=True),
        'mobilenetv2': lambda: MobileNetV2(input_channels, num_classes),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_map.keys())}")
    
    return model_map[model_name]()


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if history['train_acc']:
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    if history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")
    else:
        plt.show()


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            val_split=0.1,
            flatten=False,  # Keep as images for CNNs
        )
    elif args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            val_split=0.1,
            augment=args.augment,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    num_classes = 10
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, args.dataset, num_classes)
    model.print_summary()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    else:
        scheduler = None
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        val_loader=val_loader,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping if args.early_stopping > 0 else None,
        checkpoint_path=args.checkpoint if args.save_checkpoint else None,
        verbose=True,
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        verbose=True,
    )
    
    # Plot training history
    if args.plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f'{args.model}_{args.dataset}_training.png'
        plot_training_history(history, save_path=plot_path)
    
    print("\nTraining complete!")
    print(f"Final test accuracy: {test_metrics.get('accuracy', 0.0):.2f}%")
    print(f"Final test loss: {test_metrics.get('loss', 0.0):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN on image classification')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store/load data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation (for CIFAR-10)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['simple_cnn', 'basic_cnn', 'lenet', 'alexnet',
                               'vgg11', 'vgg16', 'resnet18', 'resnet34',
                               'densenet121', 'mobilenetv2'],
                       help='CNN architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['step', 'cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (0 to disable)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save outputs')
    parser.add_argument('--checkpoint', type=str, default='./output/cnn_best.pt',
                       help='Path to save best model checkpoint')
    parser.add_argument('--save-checkpoint', action='store_true',
                       help='Save model checkpoint')
    parser.add_argument('--plot', action='store_true',
                       help='Plot and save training curves')
    
    # Device
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    main(args)
