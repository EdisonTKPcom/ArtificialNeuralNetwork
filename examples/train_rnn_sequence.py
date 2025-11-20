"""
Train an LSTM classifier on MNIST by treating images as sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ann_lab.recurrent import LSTMClassifier
from ann_lab.core.training_loops import train_epoch, evaluate


def main():
    # Configuration
    config = {
        'batch_size': 64,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'input_size': 28,  # Each row of pixels
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 60)
    print("Training LSTM on MNIST (treating images as sequences)")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print()
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create LSTM model
    # Treat 28x28 image as sequence of 28 vectors of length 28
    model = LSTMClassifier(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    ).to(config['device'])
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {model.num_parameters():,}")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Modify data for sequence input
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        # Reshape: (batch, 1, 28, 28) -> (batch, 28, 28)
        images = images.squeeze(1)
        return images, labels
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, collate_fn=collate_fn)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config['device']
        )
        
        # Print progress
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_lstm_mnist.pth')
    
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
