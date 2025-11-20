"""
Generic training and evaluation loops for neural network models.
"""

from typing import Optional, Callable, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "Training",
    clip_grad: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        desc: Description for progress bar
        clip_grad: Optional gradient clipping value
        
    Returns:
        Dictionary with 'loss' and optionally 'accuracy'
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=desc, leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
        else:
            inputs = batch
            targets = None
        
        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        if targets is not None:
            loss = criterion(outputs, targets)
        else:
            # For autoencoders or other self-supervised tasks
            loss = criterion(outputs, inputs)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        if targets is not None and outputs.dim() > 1 and outputs.size(1) > 1:
            # Classification accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    metrics = {'loss': total_loss / len(dataloader)}
    
    if total > 0:
        metrics['accuracy'] = 100.0 * correct / total
    
    return metrics


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Neural network model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to run on
        desc: Description for progress bar
        
    Returns:
        Dictionary with 'loss' and optionally 'accuracy'
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if targets is not None:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, inputs)
            
            # Metrics
            total_loss += loss.item()
            
            if targets is not None and outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    metrics = {'loss': total_loss / len(dataloader)}
    
    if total > 0:
        metrics['accuracy'] = 100.0 * correct / total
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Optional[torch.device] = None,
    val_loader: Optional[DataLoader] = None,
    scheduler: Optional[Any] = None,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Complete training loop with validation and early stopping.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to run on (auto-detect if None)
        val_loader: Optional validation data loader
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Stop if no improvement after N epochs
        checkpoint_path: Path to save best model
        verbose: Print progress
        
    Returns:
        Dictionary with training history (losses, accuracies, etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch+1}/{epochs} [Train]"
        )
        
        history['train_loss'].append(train_metrics['loss'])
        if 'accuracy' in train_metrics:
            history['train_acc'].append(train_metrics['accuracy'])
        
        # Validation
        if val_loader is not None:
            val_metrics = evaluate_epoch(
                model, val_loader, criterion, device,
                desc=f"Epoch {epoch+1}/{epochs} [Val]"
            )
            
            history['val_loss'].append(val_metrics['loss'])
            if 'accuracy' in val_metrics:
                history['val_acc'].append(val_metrics['accuracy'])
            
            # Print metrics
            if verbose:
                train_info = f"Train Loss: {train_metrics['loss']:.4f}"
                if 'accuracy' in train_metrics:
                    train_info += f", Acc: {train_metrics['accuracy']:.2f}%"
                
                val_info = f"Val Loss: {val_metrics['loss']:.4f}"
                if 'accuracy' in val_metrics:
                    val_info += f", Acc: {val_metrics['accuracy']:.2f}%"
                
                print(f"{train_info} | {val_info}")
            
            # Early stopping
            if early_stopping_patience is not None:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    if checkpoint_path is not None:
                        torch.save(model.state_dict(), checkpoint_path)
                        if verbose:
                            print(f"Saved best model to {checkpoint_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        else:
            # No validation - just print training metrics
            if verbose:
                train_info = f"Train Loss: {train_metrics['loss']:.4f}"
                if 'accuracy' in train_metrics:
                    train_info += f", Acc: {train_metrics['accuracy']:.2f}%"
                print(train_info)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'] if val_loader else train_metrics['loss'])
            else:
                scheduler.step()
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on (auto-detect if None)
        verbose: Print results
        
    Returns:
        Dictionary with test metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    test_metrics = evaluate_epoch(model, test_loader, criterion, device, desc="Testing")
    
    if verbose:
        print(f"\nTest Results:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        if 'accuracy' in test_metrics:
            print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    
    return test_metrics
