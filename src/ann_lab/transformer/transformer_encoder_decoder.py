"""
Transformer encoder-decoder architecture.

The classic Transformer from "Attention Is All You Need".
Uses self-attention and feedforward layers for sequence-to-sequence tasks.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from ..core.base_model import BaseModel
from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Positional encoding adds position information to embeddings.
    
    Uses sinusoidal functions of different frequencies.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Contains multi-head self-attention followed by feedforward network,
    with residual connections and layer normalization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            output: Encoded representation (batch, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    
    Contains self-attention, cross-attention to encoder, and feedforward,
    with residual connections and layer normalization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, tgt_len, embed_dim)
            encoder_output: Encoder output (batch, src_len, embed_dim)
            tgt_mask: Self-attention mask (causal mask)
            src_mask: Cross-attention mask
            
        Returns:
            output: Decoded representation (batch, tgt_len, embed_dim)
        """
        # Self-attention (masked)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention to encoder
        attn_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class TransformerEncoderDecoder(BaseModel):
    """
    Full Transformer model for sequence-to-sequence tasks.
    
    Classic architecture from "Attention Is All You Need" (Vaswani et al., 2017).
    Uses stacked encoder and decoder layers with multi-head attention.
    
    Good for: machine translation, text summarization, sequence transduction.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_dim: Embedding dimension (d_model in the paper)
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        ff_dim: Feedforward network hidden dimension
        max_len: Maximum sequence length for positional encoding
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ff_dim: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
        
        # Scale embeddings
        self.scale = math.sqrt(embed_dim)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Optional source mask
            
        Returns:
            encoder_output: Encoded representations (batch, src_len, embed_dim)
        """
        # Embed and add positional encoding
        x = self.src_embedding(src) * self.scale
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target token IDs (batch, tgt_len)
            encoder_output: Encoder output (batch, src_len, embed_dim)
            tgt_mask: Causal mask for target
            src_mask: Mask for cross-attention
            
        Returns:
            output: Decoder output (batch, tgt_len, embed_dim)
        """
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target token IDs (batch, tgt_len)
            src_mask: Optional source padding mask
            tgt_mask: Optional target causal mask
            
        Returns:
            output: Logits (batch, tgt_len, tgt_vocab_size)
        """
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        output = self.fc_out(decoder_output)
        
        return output
