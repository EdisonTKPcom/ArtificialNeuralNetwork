"""
Attention mechanisms for neural networks.

Self-attention allows the model to weigh the importance of different parts
of the input when processing each element.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SelfAttention(nn.Module):
    """
    Basic scaled dot-product self-attention.
    
    Computes attention weights and applies them to values.
    Core building block of Transformer architectures.
    
    Args:
        embed_dim: Dimension of input embeddings
        dropout: Dropout probability
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask (batch, seq_len, seq_len)
            
        Returns:
            output: Attention-weighted values (batch, seq_len, embed_dim)
        """
        # Compute attention scores
        # (batch, seq_len, embed_dim) @ (batch, embed_dim, seq_len)
        # -> (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, seq_len, seq_len) @ (batch, seq_len, embed_dim)
        # -> (batch, seq_len, embed_dim)
        output = torch.matmul(attn_weights, value)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Runs multiple attention operations in parallel and concatenates results.
    Allows the model to jointly attend to information from different
    representation subspaces.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attention output (batch, seq_len, embed_dim)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        # (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for all heads
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention for attending to encoder outputs from decoder.
    
    Similar to multi-head attention but queries come from one sequence
    and keys/values from another (e.g., decoder attending to encoder).
    
    Args:
        embed_dim: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query from decoder (batch, tgt_len, embed_dim)
            key_value: Keys and values from encoder (batch, src_len, embed_dim)
            mask: Optional cross-attention mask
            
        Returns:
            output: Cross-attended output (batch, tgt_len, embed_dim)
        """
        return self.attention(query, key_value, key_value, mask)
