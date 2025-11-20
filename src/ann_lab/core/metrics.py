"""
Evaluation metrics for neural network models.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        
    Returns:
        Accuracy as percentage (0-100)
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        # Logits - get argmax
        _, predicted = predictions.max(1)
    else:
        predicted = predictions
    
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    
    return 100.0 * correct / total


def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as percentage (0-100)
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, top_k_pred = predictions.topk(k, dim=1, largest=True, sorted=True)
        top_k_pred = top_k_pred.t()
        correct = top_k_pred.eq(targets.view(1, -1).expand_as(top_k_pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        
        return 100.0 * correct_k.item() / batch_size


def confusion_matrix_stats(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate confusion matrix statistics.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Tuple of (precision, recall, f1_score, confusion_matrix)
        Each metric is per-class
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        _, predicted = predictions.max(1)
    else:
        predicted = predictions
    
    # Initialize confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(targets.view(-1), predicted.view(-1)):
        confusion[t.long(), p.long()] += 1
    
    # Calculate metrics per class
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    
    # Precision, recall, F1
    precision = tp.float() / (tp + fp).float().clamp(min=1e-7)
    recall = tp.float() / (tp + fn).float().clamp(min=1e-7)
    f1 = 2 * (precision * recall) / (precision + recall).clamp(min=1e-7)
    
    return precision, recall, f1, confusion


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate mean squared error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        MSE value
    """
    return F.mse_loss(predictions, targets).item()


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate mean absolute error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        MAE value
    """
    return F.l1_loss(predictions, targets).item()


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate R² (coefficient of determination) score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        R² score
    """
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    
    return (1 - ss_res / ss_tot).item()
