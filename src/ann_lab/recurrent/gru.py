"""
Gated Recurrent Unit (GRU) implementations.

GRUs are similar to LSTMs but with fewer parameters (no separate cell state).
Often perform comparably to LSTMs while being more efficient.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.base_model import BaseModel


class GRUClassifier(BaseModel):
    """
    GRU-based sequence classifier.
    
    Similar to LSTM but with a simpler gating mechanism and no separate cell state.
    Generally faster to train than LSTM with comparable performance.
    
    Good for: sentiment analysis, text classification, time series classification.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of the hidden state
        num_layers: Number of stacked GRU layers
        num_classes: Number of output classes
        dropout: Dropout probability between layers
        bidirectional: Whether to use bidirectional GRU
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        
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
        out, hidden = self.gru(x, hidden)
        
        # Take the last time step output
        # out: (batch, seq_len, hidden_size * num_directions)
        out = out[:, -1, :]  # (batch, hidden_size * num_directions)
        
        # Classification head
        out = self.fc(out)  # (batch, num_classes)
        
        return out, hidden


class GRUSeq2Seq(BaseModel):
    """
    GRU-based sequence-to-sequence model.
    
    Outputs predictions at each time step.
    Good for: sequence labeling, video frame prediction, music generation.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of the hidden state
        output_size: Number of output features per time step
        num_layers: Number of stacked GRU layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
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
        out, hidden = self.gru(x, hidden)
        
        # Apply output layer to all time steps
        batch_size, seq_len, _ = out.shape
        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)
        out = out.reshape(batch_size, seq_len, -1)
        
        return out, hidden
