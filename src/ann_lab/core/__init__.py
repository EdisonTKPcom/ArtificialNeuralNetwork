"""
Core utilities and base classes for neural network models.
"""

from .base_model import BaseModel
from .training_loops import train_model, evaluate_model, train_epoch, evaluate_epoch
from .metrics import accuracy, top_k_accuracy, confusion_matrix_stats

__all__ = [
    "BaseModel",
    "train_model",
    "evaluate_model",
    "train_epoch",
    "evaluate_epoch",
    "accuracy",
    "top_k_accuracy",
    "confusion_matrix_stats",
]
