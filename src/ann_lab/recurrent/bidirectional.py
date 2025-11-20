"""
Bidirectional RNN variants.

Bidirectional RNNs process sequences in both forward and backward directions,
capturing context from both past and future time steps.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.base_model import BaseModel


class BiLSTM(BaseModel):
    """
    Bidirectional LSTM for sequence classification.
    
    Processes the sequence in both directions and combines the information.
    Particularly effective when the entire sequence is available at once.
    
    Good for: named entity recognition, part-of-speech tagging, sentiment analysis.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of hidden state in each direction
        num_layers: Number of stacked BiLSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability
        pooling: How to combine sequence outputs ('last', 'mean', 'max')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        pooling: str = 'last'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Output is concatenation of forward and backward hidden states
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden states
            
        Returns:
            output: Predictions of shape (batch, num_classes)
            hidden: Final hidden states
        """
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        # out: (batch, seq_len, hidden_size * 2)
        
        # Pool sequence outputs
        if self.pooling == 'last':
            out = out[:, -1, :]  # (batch, hidden_size * 2)
        elif self.pooling == 'mean':
            out = out.mean(dim=1)  # (batch, hidden_size * 2)
        elif self.pooling == 'max':
            out, _ = out.max(dim=1)  # (batch, hidden_size * 2)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        out = self.fc(out)  # (batch, num_classes)
        
        return out, hidden


class BiGRU(BaseModel):
    """
    Bidirectional GRU for sequence classification.
    
    Similar to BiLSTM but uses GRU cells for efficiency.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of hidden state in each direction
        num_layers: Number of stacked BiGRU layers
        num_classes: Number of output classes
        dropout: Dropout probability
        pooling: How to combine sequence outputs ('last', 'mean', 'max')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        pooling: str = 'last'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
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
        out, hidden = self.gru(x, hidden)
        
        # Pool sequence outputs
        if self.pooling == 'last':
            out = out[:, -1, :]
        elif self.pooling == 'mean':
            out = out.mean(dim=1)
        elif self.pooling == 'max':
            out, _ = out.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        out = self.fc(out)
        
        return out, hidden


class BiRNNTagger(BaseModel):
    """
    Bidirectional RNN for sequence tagging (many-to-many).
    
    Outputs a prediction at each time step, using bidirectional context.
    Good for: NER, POS tagging, any token-level prediction task.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of hidden state in each direction
        num_tags: Number of output tags per position
        num_layers: Number of stacked layers
        dropout: Dropout probability
        cell_type: Type of RNN cell ('lstm', 'gru', 'rnn')
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_tags: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = 'lstm'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type.lower()
        
        # Choose RNN cell type
        rnn_class = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN
        }[self.cell_type]
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_tags)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            output: Tag logits of shape (batch, seq_len, num_tags)
        """
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # out: (batch, seq_len, hidden_size * 2)
        
        # Apply tagging layer to each time step
        batch_size, seq_len, _ = out.shape
        out = out.reshape(-1, self.hidden_size * 2)
        out = self.fc(out)
        out = out.reshape(batch_size, seq_len, -1)
        
        return out
