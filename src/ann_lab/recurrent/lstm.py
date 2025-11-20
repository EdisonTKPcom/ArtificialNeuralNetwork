"""
Long Short-Term Memory (LSTM) implementations.

LSTMs address the vanishing gradient problem in RNNs through gating mechanisms.
Better at learning long-term dependencies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.base_model import BaseModel


class LSTMClassifier(BaseModel):
    """
    LSTM-based sequence classifier.
    
    Uses LSTM cells which have gates to control information flow,
    making them better at learning long-term dependencies than vanilla RNNs.
    
    Good for: sentiment analysis, text classification, any sequential classification.
    
    Args:
        input_size: Number of input features per time step
        hidden_size: Size of the hidden state and cell state
        num_layers: Number of stacked LSTM layers
        num_classes: Number of output classes
        dropout: Dropout probability between layers
        bidirectional: Whether to use bidirectional LSTM
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
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional hidden states
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional tuple of (h_0, c_0) initial states
            
        Returns:
            output: Predictions of shape (batch, num_classes)
            hidden: Tuple of (h_n, c_n) final states
        """
        # x: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        
        # Take the last time step output
        # out: (batch, seq_len, hidden_size * num_directions)
        out = out[:, -1, :]  # (batch, hidden_size * num_directions)
        
        # Classification head
        out = self.fc(out)  # (batch, num_classes)
        
        return out, hidden


class LSTMLanguageModel(BaseModel):
    """
    LSTM-based language model for sequence generation.
    
    Predicts the next token at each position in the sequence.
    Good for: text generation, language modeling.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_size: Size of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input token indices of shape (batch, seq_len)
            hidden: Optional tuple of (h_0, c_0) initial states
            
        Returns:
            output: Logits of shape (batch, seq_len, vocab_size)
            hidden: Tuple of (h_n, c_n) final states
        """
        # Embed tokens
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        emb = self.dropout(emb)
        
        # LSTM forward pass
        out, hidden = self.lstm(emb, hidden)  # (batch, seq_len, hidden_size)
        out = self.dropout(out)
        
        # Project to vocabulary
        batch_size, seq_len, _ = out.shape
        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)  # (batch * seq_len, vocab_size)
        out = out.reshape(batch_size, seq_len, -1)  # (batch, seq_len, vocab_size)
        
        return out, hidden
