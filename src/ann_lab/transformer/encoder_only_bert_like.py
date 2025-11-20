"""
BERT-style encoder-only transformer model.

Encoder-only transformers are good for understanding tasks:
- Text classification
- Named entity recognition
- Question answering
"""

import torch
import torch.nn as nn
from typing import Optional
from ..core.base_model import BaseModel
from .transformer_encoder_decoder import TransformerEncoderLayer, PositionalEncoding


class BERTLikeEncoder(BaseModel):
    """
    Simplified BERT-style encoder-only transformer.
    
    Uses bidirectional self-attention to build contextualized representations.
    Suitable for classification and token-level prediction tasks.
    
    Good for: text classification, NER, question answering, semantic similarity.
    
    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ff_dim: Feedforward hidden dimension
        max_len: Maximum sequence length
        num_classes: Number of output classes (for classification)
        dropout: Dropout probability
        pooling: How to pool for classification ('cls', 'mean', 'max')
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        max_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = 'cls'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling = pooling
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)
            
        Returns:
            logits: Classification logits (batch, num_classes)
        """
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Convert attention mask to appropriate format if provided
        if attention_mask is not None:
            # Expand mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        
        # Pool sequence representation
        if self.pooling == 'cls':
            # Use first token ([CLS] in BERT)
            pooled = x[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pooling over sequence
            pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            # Max pooling over sequence
            pooled, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get contextualized embeddings without classification head.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            embeddings: Contextualized token embeddings (batch, seq_len, embed_dim)
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        
        return x


class BERTForTokenClassification(BaseModel):
    """
    BERT-style model for token-level classification (e.g., NER, POS tagging).
    
    Outputs a prediction for each token in the sequence.
    
    Args:
        vocab_size: Vocabulary size
        num_labels: Number of labels per token
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
        num_labels: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Token classification head
        self.classifier = nn.Linear(embed_dim, num_labels)
        
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
            logits: Token-level logits (batch, seq_len, num_labels)
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        x = self.dropout(x)
        
        # Classify each token
        logits = self.classifier(x)
        
        return logits
