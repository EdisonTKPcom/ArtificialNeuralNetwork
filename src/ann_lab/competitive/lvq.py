"""
Learning Vector Quantization (LVQ) for supervised classification.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class LearningVectorQuantization(BaseModel):
    """
    Learning Vector Quantization for classification.
    
    Similar to K-means but supervised. Maintains prototype vectors
    for each class and updates them based on labeled examples.
    
    Good for: classification with prototype-based representation.
    
    Args:
        input_dim: Dimension of input vectors
        num_classes: Number of classes
        prototypes_per_class: Number of prototype vectors per class
        learning_rate: Learning rate for prototype updates
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        prototypes_per_class: int = 1,
        learning_rate: float = 0.01
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        
        total_prototypes = num_classes * prototypes_per_class
        
        # Initialize prototype vectors
        self.prototypes = nn.Parameter(
            torch.randn(total_prototypes, input_dim),
            requires_grad=False
        )
        
        # Normalize
        self.prototypes.data = nn.functional.normalize(self.prototypes.data, dim=1)
        
        # Class labels for each prototype
        self.prototype_labels = torch.arange(num_classes).repeat_interleave(prototypes_per_class)
        
    def find_closest_prototype(self, x: torch.Tensor) -> int:
        """
        Find index of closest prototype to input x.
        
        Args:
            x: Input vector (input_dim,)
            
        Returns:
            idx: Index of closest prototype
        """
        distances = torch.sum((self.prototypes - x) ** 2, dim=1)
        return torch.argmin(distances).item()
    
    def update_prototype(self, x: torch.Tensor, proto_idx: int, correct: bool):
        """
        Update prototype based on whether it correctly classified x.
        
        LVQ1 update rule:
        - If correct: move prototype toward x
        - If incorrect: move prototype away from x
        
        Args:
            x: Input vector
            proto_idx: Index of winning prototype
            correct: Whether classification was correct
        """
        if correct:
            # Move toward x
            self.prototypes.data[proto_idx] += self.learning_rate * (x - self.prototypes.data[proto_idx])
        else:
            # Move away from x
            self.prototypes.data[proto_idx] -= self.learning_rate * (x - self.prototypes.data[proto_idx])
        
        # Normalize
        self.prototypes.data[proto_idx] = nn.functional.normalize(
            self.prototypes.data[proto_idx].unsqueeze(0), dim=1
        ).squeeze(0)
    
    def train_lvq(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 100,
        verbose: bool = False
    ):
        """
        Train LVQ on labeled data.
        
        Args:
            data: Training data (num_samples, input_dim)
            labels: Class labels (num_samples,)
            num_epochs: Number of training epochs
            verbose: Whether to print progress
        """
        num_samples = data.size(0)
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)
            correct_count = 0
            
            for idx in indices:
                x = data[idx]
                true_label = labels[idx].item()
                
                # Find closest prototype
                proto_idx = self.find_closest_prototype(x)
                pred_label = self.prototype_labels[proto_idx].item()
                
                # Update prototype
                correct = (pred_label == true_label)
                self.update_prototype(x, proto_idx, correct)
                
                if correct:
                    correct_count += 1
            
            accuracy = correct_count / num_samples
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input(s).
        
        Args:
            x: Input (batch, input_dim) or (input_dim,)
            
        Returns:
            predictions: Predicted class labels
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        predictions = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            proto_idx = self.find_closest_prototype(x[i])
            predictions[i] = self.prototype_labels[proto_idx]
        
        return predictions
