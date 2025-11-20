"""
Sequence-to-Sequence (Seq2Seq) encoder-decoder models.

Classic architecture for machine translation, text summarization,
and other sequence transformation tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..core.base_model import BaseModel


class Seq2SeqEncoder(nn.Module):
    """
    Encoder for Seq2Seq models.
    
    Encodes the input sequence into a context vector.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        cell_type: str = 'lstm'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        rnn_class = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN
        }[self.cell_type]
        
        self.rnn = rnn_class(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            
        Returns:
            outputs: All hidden states (batch, seq_len, hidden_size)
            hidden: Final hidden state(s)
        """
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class Seq2SeqDecoder(nn.Module):
    """
    Decoder for Seq2Seq models.
    
    Generates the output sequence autoregressively.
    """
    
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        cell_type: str = 'lstm'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        
        rnn_class = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN
        }[self.cell_type]
        
        self.rnn = rnn_class(
            output_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, hidden):
        """
        Args:
            x: Input for current step (batch, 1, output_size)
            hidden: Hidden state from previous step
            
        Returns:
            output: Prediction (batch, 1, output_size)
            hidden: Updated hidden state
        """
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


class Seq2Seq(BaseModel):
    """
    Basic Sequence-to-Sequence model with encoder-decoder architecture.
    
    The encoder processes the input sequence and produces a context representation.
    The decoder generates the output sequence based on this context.
    
    Good for: machine translation, text summarization, dialogue systems.
    
    Note: This is a basic implementation without attention.
    For better performance on longer sequences, consider adding attention mechanisms.
    
    Args:
        input_size: Size of input vocabulary or features
        output_size: Size of output vocabulary or features
        hidden_size: Size of hidden states
        num_layers: Number of RNN layers in encoder and decoder
        dropout: Dropout probability
        cell_type: Type of RNN cell ('lstm', 'gru', 'rnn')
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = 'lstm'
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        
        self.encoder = Seq2SeqEncoder(
            input_size, hidden_size, num_layers, dropout, cell_type
        )
        self.decoder = Seq2SeqDecoder(
            output_size, hidden_size, num_layers, dropout, cell_type
        )
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (batch, src_len, input_size)
            tgt: Target sequence (batch, tgt_len, output_size)
            teacher_forcing_ratio: Probability of using ground truth vs prediction
            
        Returns:
            outputs: Predicted sequence (batch, tgt_len, output_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        output_size = tgt.size(2)
        
        # Encode
        _, hidden = self.encoder(src)
        
        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, output_size).to(src.device)
        
        # First input to decoder is first token of target
        dec_input = tgt[:, 0:1, :]  # (batch, 1, output_size)
        
        # Decode step by step
        for t in range(tgt_len):
            output, hidden = self.decoder(dec_input, hidden)
            outputs[:, t:t+1, :] = output
            
            # Teacher forcing: use ground truth or prediction
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing and t < tgt_len - 1:
                dec_input = tgt[:, t+1:t+2, :]
            else:
                dec_input = output
                
        return outputs
    
    def encode(self, src: torch.Tensor):
        """Encode source sequence."""
        return self.encoder(src)
    
    def decode_step(self, x: torch.Tensor, hidden):
        """Single decoding step."""
        return self.decoder(x, hidden)


class Seq2SeqWithEmbedding(BaseModel):
    """
    Seq2Seq model with embedding layers for discrete tokens.
    
    More suitable for text-based tasks where inputs/outputs are token IDs.
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        embedding_dim: Dimension of embeddings
        hidden_size: Size of hidden states
        num_layers: Number of RNN layers
        dropout: Dropout probability
        cell_type: Type of RNN cell ('lstm', 'gru', 'rnn')
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell_type: str = 'lstm'
    ):
        super().__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        
        self.encoder = Seq2SeqEncoder(
            embedding_dim, hidden_size, num_layers, dropout, cell_type
        )
        self.decoder = Seq2SeqDecoder(
            embedding_dim, hidden_size, num_layers, dropout, cell_type
        )
        
        self.fc_out = nn.Linear(embedding_dim, tgt_vocab_size)
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target token IDs (batch, tgt_len)
            teacher_forcing_ratio: Probability of teacher forcing
            
        Returns:
            outputs: Logits for target vocabulary (batch, tgt_len, tgt_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.fc_out.out_features
        
        # Embed and encode
        src_emb = self.src_embedding(src)
        _, hidden = self.encoder(src_emb)
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        
        # Start with <sos> token
        dec_input = self.tgt_embedding(tgt[:, 0:1])
        
        for t in range(tgt_len):
            output, hidden = self.decoder(dec_input, hidden)
            output = self.fc_out(output)
            outputs[:, t:t+1, :] = output
            
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing and t < tgt_len - 1:
                dec_input = self.tgt_embedding(tgt[:, t+1:t+2])
            else:
                top1 = output.argmax(2)
                dec_input = self.tgt_embedding(top1)
                
        return outputs
