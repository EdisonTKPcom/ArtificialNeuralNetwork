"""
GPT-style decoder-only transformer model.

Decoder-only transformers are good for generation tasks:
- Text generation
- Language modeling
- Code generation
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from ..core.base_model import BaseModel
from .transformer_encoder_decoder import PositionalEncoding
from .attention import MultiHeadAttention


class GPTDecoderLayer(nn.Module):
    """
    Single GPT-style decoder layer with causal self-attention.
    
    Similar to transformer decoder but only has masked self-attention
    (no cross-attention to encoder).
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
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # GPT uses GELU instead of ReLU
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
            mask: Causal attention mask
            
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        # Masked self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class GPTLikeDecoder(BaseModel):
    """
    Simplified GPT-style decoder-only transformer.
    
    Uses causal (autoregressive) self-attention for language modeling.
    Each position can only attend to previous positions.
    
    Good for: text generation, language modeling, code generation.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feedforward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            GPTDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Generate causal mask so each position can only attend to earlier positions.
        
        Returns:
            mask: Lower triangular mask (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            logits: Next-token prediction logits (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Generate causal mask
        causal_mask = self.generate_causal_mask(seq_len).to(input_ids.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len)
            # Expand to (batch, 1, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            # Combine with causal mask
            mask = causal_mask.unsqueeze(0) + attention_mask
        else:
            mask = causal_mask.unsqueeze(0)  # (1, seq_len, seq_len)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            
        Returns:
            generated: Generated token IDs (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if sequence is too long
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_len else input_ids[:, -self.max_len:]
            
            # Forward pass
            logits = self.forward(input_ids_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
