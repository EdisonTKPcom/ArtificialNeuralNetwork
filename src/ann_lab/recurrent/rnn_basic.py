"""
Basic Recurrent Neural Network (RNN) implementations.

RNNs process sequential data by maintaining hidden state across time steps.
Useful for time series, text, and any sequential data.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.base_model import BaseModel


class SimpleRNN(BaseModel):
    """
    Basic RNN for sequence classification.
    
    Processes sequences and outputs a prediction based on the final hidden state.
    Good for: sentiment analysis, sequence classification tasks.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of the hidden state
        num_layers: Number of stacked RNN layers
        num_classes: Number of output classes
        dropout: Dropout probability between layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            output: Predictions of shape (batch, num_classes)
            hidden: Final hidden state
        """
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        
        # Take the last time step output
        # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Classification head
        out = self.fc(out)  # (batch, num_classes)
        
        return out, hidden


class ManyToManyRNN(BaseModel):
    """
    Many-to-many RNN for sequence-to-sequence tasks.
    
    Outputs a prediction at each time step.
    Good for: sequence labeling, time series prediction.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of the hidden state
        output_size: Number of output features per time step
        num_layers: Number of stacked RNN layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 10,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            output: Predictions of shape (batch, seq_len, output_size)
            hidden: Final hidden state
        """
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        
        # Apply output layer to all time steps
        # out: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = out.shape
        out = out.reshape(-1, self.hidden_size)  # (batch * seq_len, hidden_size)
        out = self.fc(out)  # (batch * seq_len, output_size)
        out = out.reshape(batch_size, seq_len, -1)  # (batch, seq_len, output_size)
        
        return out, hidden
