"""
Example: Train an MLP classifier on MNIST dataset.

This script demonstrates:
- Loading MNIST data
- Creating an MLP model
- Training with validation
- Evaluating on test set
- Plotting training curves
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from ann_lab.feedforward.mlp import MLPClassifier
from ann_lab.core.training_loops import train_model, evaluate_model
from ann_lab.core.datasets import get_mnist_loaders


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if history['train_acc']:
        axes[1].plot(history['train_acc'], label='Train Accuracy')
    if history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
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
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        val_split=0.1,
        flatten=True,  # Flatten 28x28 images to 784-dim vectors
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating MLP model...")
    model = MLPClassifier(
        input_dim=784,  # 28x28 = 784
        hidden_dims=args.hidden_dims,
        num_classes=10,
        activation=args.activation,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
    )
    
    model.print_summary()
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
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
        early_stopping_patience=args.early_stopping,
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
        plot_save_path = Path(args.output_dir) / 'training_curves.png'
        plot_save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_training_history(history, save_path=plot_save_path)
    
    print("\nTraining complete!")
    print(f"Final test accuracy: {test_metrics.get('accuracy', 0.0):.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP on MNIST')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store/load MNIST data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128],
                       help='List of hidden layer dimensions')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh', 'sigmoid', 'gelu'],
                       help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability')
    parser.add_argument('--use-batch-norm', action='store_true',
                       help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience (epochs)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save outputs')
    parser.add_argument('--checkpoint', type=str, default='./output/mlp_mnist_best.pt',
                       help='Path to save best model checkpoint')
    parser.add_argument('--save-checkpoint', action='store_true',
                       help='Save model checkpoint')
    parser.add_argument('--plot', action='store_true',
                       help='Plot and save training curves')
    
    # Device
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    main(args)
