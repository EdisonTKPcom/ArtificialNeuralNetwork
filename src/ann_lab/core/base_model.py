"""
Base model class providing common functionality for all neural network architectures.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all neural network models in the lab.
    
    Provides common utilities like parameter counting, device management,
    and standardized initialization patterns.
    
    All models in this repository should inherit from this class.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, only count trainable parameters
            
        Returns:
            Total number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device
    
    def to_device(self, device: torch.device) -> "BaseModel":
        """
        Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            self for chaining
        """
        self._device = device
        return self.to(device)
    
    def print_summary(self) -> None:
        """Print a summary of the model architecture."""
        print(f"\n{'='*60}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Total parameters: {self.num_parameters(trainable_only=False):,}")
        print(f"Trainable parameters: {self.num_parameters(trainable_only=True):,}")
        print(f"Device: {self._device}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None, loss: Optional[float] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Current epoch number
            loss: Current loss value
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, 
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[Optional[int], Optional[float]]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            
        Returns:
            Tuple of (epoch, loss) if available, else (None, None)
        """
        checkpoint = torch.load(path, map_location=self._device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        
        return epoch, loss
    
    def freeze_parameters(self) -> None:
        """Freeze all model parameters (disable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self) -> None:
        """Unfreeze all model parameters (enable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_parameter_groups(self, lr: float, weight_decay: float = 0.0) -> list:
        """
        Get parameter groups for optimizer with different settings.
        
        By default, applies weight decay to all parameters except biases and norms.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            List of parameter group dictionaries
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and normalization layers
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0}
        ]
